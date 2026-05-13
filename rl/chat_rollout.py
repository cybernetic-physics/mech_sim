"""Multi-turn chat-completion rollouts against an OpenAI-compatible
SGLang server.

Why this exists:
    The worldlines backend's base-model sampler is a stub unless
    `WORLDLINES_SGLANG_BASE_URL` is set. Even with sglang wired in,
    we want full chat semantics — including the model's own
    reasoning span and tool-call format — and the ability to feed
    verifier feedback back as a tool / user turn between
    assistant turns. The OpenAI `/v1/chat/completions` endpoint
    that SGLang exposes is the cleanest way to get that.

What it does per rollout:
    1. Send (system, user) to /v1/chat/completions, max_tokens=N,
       temperature=T.
    2. Score the assistant message via ``mech_env.score``. If the
       hard gate passes (or we used the last turn), return.
    3. Otherwise append (assistant, <verifier feedback as user>)
       to the message list and try again, up to ``max_turns``.
    4. Capture the FINAL assistant tokens (and the cumulative
       token count across turns) so the GRPO trainer can apply
       advantage-weighted CE on the closing assistant span.

For RL we keep the final scoring + the per-step prompt/completion
token ids so a downstream trainer can build worldlines ``Datum``s
exactly the same way ``train_grpo.py`` already does. The token ids
come back from SGLang's `usage.prompt_tokens` / `usage.completion_tokens`
and from re-tokenizing the assistant text locally (SGLang doesn't
return per-token ids in the chat endpoint).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

try:
    import requests  # type: ignore[import-untyped]
except ImportError as _e:
    requests = None  # type: ignore[assignment]


@dataclass
class TurnTrace:
    """One assistant turn worth of evidence."""
    turn_idx: int
    assistant_text: str
    score: float          # verified score (0 unless hard gate + valid)
    dense_pct: float      # mean(per-probe score)*100 — always defined
    passed: bool
    parsed_ok: bool
    failure_codes: list[str]
    completion_tokens: int
    stop_reason: str


@dataclass
class Rollout:
    """End-to-end multi-turn rollout summary."""
    task_id: str
    messages: list[dict[str, str]]  # final conversation
    turns: list[TurnTrace] = field(default_factory=list)
    best_turn: int = -1
    best_score: float = 0.0
    best_dense_pct: float = 0.0
    final_score: float = 0.0
    final_dense_pct: float = 0.0
    final_passed: bool = False
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    wall_clock_s: float = 0.0


# --------------------------------------------------------------------- #
# Verifier-feedback formatting                                          #
# --------------------------------------------------------------------- #


def _format_verifier_feedback(turn: TurnTrace) -> str:
    """Produce a compact human-+-LLM-readable critique."""
    bits = [
        f"score={turn.score:.1f}/100",
        "passed=true" if turn.passed else "passed=false",
        "parsed=true" if turn.parsed_ok else "parsed=false",
    ]
    if turn.failure_codes:
        bits.append("codes=" + ",".join(turn.failure_codes))
    if turn.stop_reason and turn.stop_reason != "stop":
        bits.append(f"stop_reason={turn.stop_reason}")
    hint = ""
    if not turn.parsed_ok:
        hint = (
            "Your reply did not contain a valid fenced ```python ... ``` "
            "block, so the verifier could not extract design.py. "
            "Re-emit ONE fenced block, no prose outside it."
        )
    elif "wrong_topology" in turn.failure_codes:
        hint = (
            "wrong_topology: at least one grounded port references a "
            "part that is not `fixed=True`, or a joint/port refers to "
            "a missing id. Re-check the IR."
        )
    elif "missing_port" in turn.failure_codes:
        hint = (
            "missing_port: a required port was absent. Re-read the "
            "task's `requirements.required_ports` and ensure every "
            "one appears in `ports`."
        )
    elif "wrong_ratio" in turn.failure_codes:
        hint = (
            "wrong_ratio: a declared scalar in `params` did not "
            "match the closed-form target. Re-derive it."
        )
    elif "invalid_artifact" in turn.failure_codes:
        hint = (
            "invalid_artifact: the IR shape was malformed — most "
            "often a non-finite value, missing schema_version, or a "
            "port pointing at a non-existent joint."
        )
    payload = "  ".join(bits)
    if hint:
        payload += "\n\n" + hint
    return (
        "VERIFIER FEEDBACK\n"
        f"{payload}\n\n"
        "Emit ONE corrected design.py inside a single fenced "
        "```python ... ``` block. No prose outside the block."
    )


# --------------------------------------------------------------------- #
# OpenAI-compatible chat call                                           #
# --------------------------------------------------------------------- #


def _chat_completion(
    *,
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    top_p: float,
    timeout_s: float,
) -> dict[str, Any]:
    """Send one /v1/chat/completions request. Returns the JSON body."""
    if requests is None:
        raise RuntimeError("`requests` is not installed in this venv")
    url = base_url.rstrip("/") + "/v1/chat/completions"
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "stream": False,
    }
    r = requests.post(url, json=body, timeout=timeout_s,
                      headers={"Authorization": "Bearer dummy"})
    r.raise_for_status()
    return r.json()


# --------------------------------------------------------------------- #
# Multi-turn rollout                                                    #
# --------------------------------------------------------------------- #


def run_rollout(
    *,
    base_url: str,
    model: str,
    task,  # rl.mech_env.TaskInfo, late import to avoid cycle
    system_prompt: str,
    user_prompt: str,
    max_turns: int = 4,
    max_tokens_per_turn: int = 4096,
    temperature: float = 0.8,
    top_p: float = 0.95,
    timeout_s: float = 240.0,
    parse_bonus: float = 5.0,
) -> Rollout:
    """One multi-turn rollout on *task* with feedback in the loop.

    Stops early when the verifier's hard gate passes. Always plays
    the final turn so the trainer has a clean closing assistant
    span to apply CE on.
    """
    # Local imports to keep this module importable without the full
    # mech_bench stack at import time.
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import mech_env as env  # noqa: E402  pyright: ignore

    started = time.perf_counter()
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    rollout = Rollout(task_id=task.task_id, messages=messages)

    for turn_idx in range(max_turns):
        try:
            resp = _chat_completion(
                base_url=base_url, model=model, messages=messages,
                max_tokens=max_tokens_per_turn,
                temperature=temperature, top_p=top_p,
                timeout_s=timeout_s,
            )
        except Exception as e:  # noqa: BLE001 — sampler firewall
            rollout.turns.append(TurnTrace(
                turn_idx=turn_idx,
                assistant_text=f"[sampler_error: {type(e).__name__}: {e}]",
                score=0.0, passed=False, parsed_ok=False,
                failure_codes=["sampler_error"],
                completion_tokens=0,
                stop_reason="error",
            ))
            break

        choice = (resp.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        assistant_text = str(msg.get("content") or "")
        stop_reason = choice.get("finish_reason") or "stop"
        usage = resp.get("usage") or {}
        rollout.total_tokens_in += int(usage.get("prompt_tokens") or 0)
        rollout.total_tokens_out += int(usage.get("completion_tokens") or 0)

        # Score the assistant message via mech_bench.
        ep = env.score(task, assistant_text, parse_bonus=parse_bonus)
        ep.completion_tokens = int(usage.get("completion_tokens") or 0)

        turn = TurnTrace(
            turn_idx=turn_idx,
            assistant_text=assistant_text,
            score=ep.score,
            dense_pct=ep.dense_pct,
            passed=ep.passed,
            parsed_ok=ep.parsed_ok,
            failure_codes=ep.failure_codes,
            completion_tokens=ep.completion_tokens,
            stop_reason=stop_reason,
        )
        rollout.turns.append(turn)
        if ep.score > rollout.best_score:
            rollout.best_score = ep.score
            rollout.best_turn = turn_idx
        if ep.dense_pct > rollout.best_dense_pct:
            rollout.best_dense_pct = ep.dense_pct

        # Append assistant message and decide whether to continue.
        messages.append({"role": "assistant", "content": assistant_text})
        if ep.passed:
            break
        if turn_idx == max_turns - 1:
            break
        messages.append({
            "role": "user",
            "content": _format_verifier_feedback(turn),
        })

    if rollout.turns:
        last = rollout.turns[-1]
        rollout.final_score = last.score
        rollout.final_dense_pct = last.dense_pct
        rollout.final_passed = last.passed
    rollout.wall_clock_s = time.perf_counter() - started
    rollout.messages = messages
    return rollout


# --------------------------------------------------------------------- #
# CLI for quick smoke testing                                           #
# --------------------------------------------------------------------- #


def _main() -> int:
    import argparse
    import sys
    from pathlib import Path

    p = argparse.ArgumentParser(prog="chat_rollout")
    p.add_argument("--base-url", default="http://127.0.0.1:30000")
    p.add_argument("--model",
                   default="NousResearch/DeepHermes-3-Llama-3-3B-Preview")
    p.add_argument("--task", required=True,
                   help="path to a tasks/<id>/ directory")
    p.add_argument("--max-turns", type=int, default=4)
    p.add_argument("--max-tokens-per-turn", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.95)
    args = p.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import mech_env as env  # noqa

    task_dir = Path(args.task).resolve()
    tasks = env.list_tasks()
    matches = [t for t in tasks if t.task_dir == task_dir]
    if not matches:
        print(f"error: no task at {task_dir}", file=sys.stderr)
        return 2
    task = matches[0]

    system_prompt = (
        Path(__file__).resolve().parent / "agent_prompt_rl.md"
    ).read_text()
    user_prompt = (
        f"Solve mech_bench task **{task.task_id}**.\n\n"
        f"## prompt.md\n{task.prompt}\n\n## task.toml\n```toml\n"
        f"{task.task_toml}\n```\n\nEmit ONE Python file as a single "
        "fenced ```python ... ``` block. No prose outside the block."
    )

    rollout = run_rollout(
        base_url=args.base_url, model=args.model,
        task=task,
        system_prompt=system_prompt, user_prompt=user_prompt,
        max_turns=args.max_turns,
        max_tokens_per_turn=args.max_tokens_per_turn,
        temperature=args.temperature, top_p=args.top_p,
    )

    print(json.dumps({
        "task_id": rollout.task_id,
        "best_score": rollout.best_score,
        "best_turn": rollout.best_turn,
        "final_score": rollout.final_score,
        "final_passed": rollout.final_passed,
        "n_turns": len(rollout.turns),
        "tokens_in": rollout.total_tokens_in,
        "tokens_out": rollout.total_tokens_out,
        "wall_clock_s": rollout.wall_clock_s,
        "per_turn": [
            {"turn": t.turn_idx, "score": t.score, "passed": t.passed,
             "parsed_ok": t.parsed_ok, "codes": t.failure_codes,
             "completion_tokens": t.completion_tokens,
             "stop_reason": t.stop_reason}
            for t in rollout.turns
        ],
    }, indent=2))
    return 0 if rollout.final_passed else 1


if __name__ == "__main__":
    import sys
    sys.exit(_main())
