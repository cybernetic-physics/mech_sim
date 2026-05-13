"""Multi-turn GRPO on mech_bench: SGLang rollouts → worldlines training.

Architecture (single-host, two GPUs):

  GPU 1 ─ SGLang OpenAI server (:30000)         ← samples per turn
  GPU 0 ─ Worldlines backend (:18100)           ← PEFT LoRA trainer
            ├─ ServiceClient.create_lora_training_client
            ├─ forward_backward(loss_fn="cross_entropy")
            └─ optim_step(AdamParams)

Per round:
    1. Pick ``tasks_per_round`` tasks (uniform, curriculum is TBD).
    2. For each task, run ``samples_per_task`` multi-turn rollouts
       through SGLang via ``rl.chat_rollout.run_rollout`` — each
       rollout is up to ``max_turns`` assistant turns with verifier
       feedback in between. Reward = best score across turns + parse
       bonus.
    3. Compute group-relative advantages over the K rollouts.
    4. For each kept rollout, tokenise (system, user, ...turns_until_last_assistant,
       final_assistant) and build a worldlines Datum: prompt tokens
       weighted 0, final-assistant tokens weighted by advantage.
    5. forward_backward + optim_step on worldlines. Periodic save_state.

Logs land under ``runs/<run_name>/`` in the same shape as
rl-spark/worldlines-engdesign — ``history.jsonl``,
``task_scores.jsonl``, ``heartbeat.json``.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))
sys.path.insert(0, str(THIS.parent))

import mech_env as env  # noqa: E402
import chat_rollout as cr  # noqa: E402


SYSTEM_PROMPT_PATH = THIS / "agent_prompt_rl.md"
USER_PROMPT_TEMPLATE = """Solve mech_bench task **{task_id}**.

## prompt.md
{prompt_md}

## task.toml
```toml
{task_toml}
```

Emit ONE Python file as a single fenced ```python ... ``` block.
No prose outside the block.
"""


# --------------------------------------------------------------------- #
# Logging helpers (mirror rl-spark)                                     #
# --------------------------------------------------------------------- #


def heartbeat(runs_dir: Path, **fields: Any) -> None:
    fields.setdefault("ts", time.time())
    path = runs_dir / "heartbeat.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(fields, default=str))
    tmp.replace(path)


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, default=str) + "\n")


# --------------------------------------------------------------------- #
# Advantage math                                                        #
# --------------------------------------------------------------------- #


def _adjust_reward(score: float, parse_bonus: float,
                   completion_tokens: int, length_alpha: float,
                   *, kl: float = 0.0, kl_coef: float = 0.0) -> float:
    r = score + parse_bonus
    if length_alpha > 0:
        r -= length_alpha * math.log1p(max(0, int(completion_tokens)))
    if kl_coef > 0:
        r -= kl_coef * float(kl)
    return float(r)


def _group_advantages(
    rewards: list[float],
    mask: list[bool],
    *,
    adv_clip: float = 5.0,
    drop_zero_var: bool = True,
) -> list[float]:
    if not rewards:
        return []
    if drop_zero_var and len(set(rewards)) <= 1:
        return [0.0] * len(rewards)
    valid = [r for r, m in zip(rewards, mask) if m]
    if not valid:
        return [0.0] * len(rewards)
    mu = sum(valid) / len(valid)
    sigma = statistics.pstdev(valid) if len(valid) > 1 else 0.0
    advs: list[float] = []
    for r, m in zip(rewards, mask):
        if not m:
            advs.append(0.0)
            continue
        a = (r - mu) / (sigma + 1e-6) if sigma > 0 else 0.0
        if adv_clip > 0:
            a = max(-adv_clip, min(adv_clip, a))
        advs.append(float(a))
    return advs


# --------------------------------------------------------------------- #
# Tokenise (messages-up-to-final-assistant) + (final-assistant)         #
# --------------------------------------------------------------------- #


def _split_into_prompt_and_final_assistant(
    tok,
    messages: list[dict[str, str]],
) -> tuple[list[int], list[int]]:
    """Return (prompt_ids, final_assistant_ids).

    ``prompt_ids`` is the chat-templated tokenisation of all messages
    up to (but not including) the LAST assistant message, with the
    generation prompt appended — exactly what the model saw when it
    produced the final completion. ``final_assistant_ids`` is the
    content of that last assistant message.
    """
    # Find the last assistant.
    last_assistant_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]["role"] == "assistant":
            last_assistant_idx = i
            break
    if last_assistant_idx is None:
        # No assistant turn — return whole thing as prompt, empty completion.
        prompt_text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        prompt_ids = tok.encode(prompt_text, add_special_tokens=False)
        return [int(x) for x in prompt_ids], []

    prompt_msgs = messages[:last_assistant_idx]
    prompt_text = tok.apply_chat_template(
        prompt_msgs, tokenize=False, add_generation_prompt=True)
    prompt_ids = tok.encode(prompt_text, add_special_tokens=False)

    final_content = messages[last_assistant_idx]["content"] or ""
    final_ids = tok.encode(final_content, add_special_tokens=False)
    return [int(x) for x in prompt_ids], [int(x) for x in final_ids]


# --------------------------------------------------------------------- #
# main                                                                  #
# --------------------------------------------------------------------- #


def main() -> int:
    p = argparse.ArgumentParser(prog="train_grpo")
    p.add_argument("--backend-url", default="http://127.0.0.1:18100",
                   help="worldlines backend (PEFT trainer)")
    p.add_argument("--api-key", default="wld-local")
    p.add_argument("--sglang-url", default="http://127.0.0.1:30000",
                   help="sglang OpenAI server (rollout sampler)")
    p.add_argument(
        "--base-model",
        default="NousResearch/DeepHermes-3-Llama-3-3B-Preview")
    p.add_argument("--tokenizer", default=None,
                   help="defaults to --base-model")
    p.add_argument("--run-name", default="mech-grpo")
    p.add_argument("--runs-root", default="runs")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--tasks-per-round", type=int, default=4)
    p.add_argument("--samples-per-task", type=int, default=4)
    p.add_argument("--max-turns", type=int, default=4,
                   help="max assistant turns per rollout")
    p.add_argument("--max-tokens-per-turn", type=int, default=4096)
    p.add_argument("--max-context-tokens", type=int, default=16384,
                   help="hard cap on full prompt+completion length")
    p.add_argument("--rollout-temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--length-alpha", type=float, default=0.0)
    p.add_argument("--parse-bonus", type=float, default=5.0)
    p.add_argument("--pass-bonus", type=float, default=25.0,
                   help="extra reward when the final-turn hard gate "
                        "passes (on top of dense_pct)")
    p.add_argument("--adv-clip", type=float, default=5.0)
    p.add_argument("--drop-zero-var", action="store_true", default=True)
    p.add_argument("--mask-truncated", action="store_true", default=True)
    p.add_argument("--checkpoint-every", type=int, default=10)
    p.add_argument("--families", default=None)
    p.add_argument("--tiers", default=None)
    p.add_argument("--score-timeout", type=float, default=60.0)
    p.add_argument("--rollout-timeout", type=float, default=300.0)
    args = p.parse_args()

    rng = random.Random(args.seed)
    repo_root = Path(__file__).resolve().parent.parent
    runs_dir = repo_root / args.runs_root / args.run_name
    runs_dir.mkdir(parents=True, exist_ok=True)

    families = (
        {s.strip() for s in args.families.split(",") if s.strip()}
        if args.families else None
    )
    tiers = (
        {s.strip() for s in args.tiers.split(",") if s.strip()}
        if args.tiers else None
    )
    tasks = env.list_tasks(
        root=repo_root / "tasks",
        families=families, tiers=tiers,
    )
    if not tasks:
        print("error: no tasks matched", file=sys.stderr)
        return 2
    print(f"loaded {len(tasks)} tasks")

    import worldlines as wl  # type: ignore[import-not-found]
    from transformers import AutoTokenizer  # type: ignore[import-not-found]

    svc = wl.ServiceClient(base_url=args.backend_url, api_key=args.api_key)
    print(f"worldlines @ {args.backend_url}")
    print("creating LoRA training client ...")
    train = svc.create_lora_training_client(
        base_model=args.base_model,
        rank=args.lora_rank,
        seed=args.seed,
    )
    print(f"training client created. model_id={train.model_id}")

    tokenizer_name = args.tokenizer or args.base_model
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    system_prompt = SYSTEM_PROMPT_PATH.read_text()

    step = 0
    t0 = time.time()
    heartbeat(runs_dir, phase="starting", round=0, step=0,
              base_model=args.base_model, algo="grpo-multi-turn",
              tokenizer=tokenizer_name,
              n_tasks=len(tasks),
              sglang_url=args.sglang_url,
              backend_url=args.backend_url)

    for round_idx in range(args.rounds):
        batch = rng.sample(
            tasks, min(args.tasks_per_round, len(tasks)))
        print(
            f"\n=== round {round_idx} step={step} "
            f"tasks={[t.task_id for t in batch]} ===")
        heartbeat(runs_dir, phase="rollout", round=round_idx, step=step,
                  tasks=[t.task_id for t in batch],
                  rollouts_done=0,
                  rollouts_target=len(batch) * args.samples_per_task)

        groups: list[dict] = []
        for ti in batch:
            user_prompt = USER_PROMPT_TEMPLATE.format(
                task_id=ti.task_id,
                prompt_md=ti.prompt,
                task_toml=ti.task_toml,
            )
            rollouts: list[dict] = []
            for k in range(args.samples_per_task):
                heartbeat(runs_dir, phase="rollout",
                          round=round_idx, step=step,
                          current_task=ti.task_id, sample_idx=k,
                          rollouts_target=len(batch) * args.samples_per_task)
                try:
                    r = cr.run_rollout(
                        base_url=args.sglang_url,
                        model=args.base_model,
                        task=ti,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        max_turns=args.max_turns,
                        max_tokens_per_turn=args.max_tokens_per_turn,
                        temperature=args.rollout_temperature,
                        top_p=args.top_p,
                        timeout_s=args.rollout_timeout,
                        parse_bonus=args.parse_bonus,
                    )
                except Exception as e:  # noqa: BLE001
                    print(f"  [warn] rollout {ti.task_id} k={k}: {e}")
                    continue
                # The "completion" we train on is the final assistant turn.
                last = r.turns[-1] if r.turns else None
                if last is None:
                    continue
                prompt_ids, final_ids = (
                    _split_into_prompt_and_final_assistant(tok, r.messages)
                )
                if not final_ids:
                    continue
                if len(prompt_ids) + len(final_ids) > args.max_context_tokens:
                    # Skip rollouts that overflow the trainer's context
                    # window. (Worldlines' default max_train_tokens is
                    # 16384.)
                    print(f"  [skip] {ti.task_id} k={k} "
                          f"full_len={len(prompt_ids)+len(final_ids)} > "
                          f"{args.max_context_tokens}")
                    continue
                rollouts.append({
                    "task": ti,
                    "ep_score": last.score,
                    "dense_pct": last.dense_pct,
                    "best_dense_pct": r.best_dense_pct,
                    "passed": last.passed,
                    "parsed_ok": last.parsed_ok,
                    "completion_tokens": last.completion_tokens,
                    "stop_reason": last.stop_reason,
                    "failure_codes": last.failure_codes,
                    "best_score": r.best_score,
                    "n_turns": len(r.turns),
                    "prompt_ids": prompt_ids,
                    "final_ids": final_ids,
                })

            if not rollouts:
                continue
            best_pass = max(r["ep_score"] for r in rollouts)
            best_dense = max(r["best_dense_pct"] for r in rollouts)
            mean_dense = (
                sum(r["best_dense_pct"] for r in rollouts) / len(rollouts)
            )
            n_passed = sum(1 for r in rollouts if r["passed"])
            n_parsed = sum(1 for r in rollouts if r["parsed_ok"])
            print(
                f"  {ti.task_id:48}  pass={best_pass:5.1f}  "
                f"dense_best={best_dense:5.1f}  "
                f"dense_avg={mean_dense:5.1f}  "
                f"parsed={n_parsed}/{len(rollouts)}  "
                f"passed={n_passed}/{len(rollouts)}"
            )
            for r in rollouts:
                append_jsonl(runs_dir / "task_scores.jsonl", {
                    "ts": time.time(),
                    "task_id": ti.task_id,
                    "tier": ti.tier, "family": ti.family,
                    "score": r["ep_score"],
                    "best_score": r["best_score"],
                    "dense_pct": r["dense_pct"],
                    "best_dense_pct": r["best_dense_pct"],
                    "passed": r["passed"],
                    "parsed_ok": r["parsed_ok"],
                    "failure_codes": r["failure_codes"],
                    "completion_tokens": r["completion_tokens"],
                    "stop_reason": r["stop_reason"],
                    "n_turns": r["n_turns"],
                    "round": round_idx, "step": step,
                })
            groups.append({"task": ti, "rollouts": rollouts})

        heartbeat(runs_dir, phase="scoring", round=round_idx, step=step,
                  n_groups=len(groups))

        # Build advantage-weighted Datums.
        data: list = []
        kept = 0
        all_rewards: list[float] = []
        for g in groups:
            rewards: list[float] = []
            for r in g["rollouts"]:
                # Dense reward = best-across-turns dense_pct (mean of
                # per-probe scores) + verified-pass bonus on the
                # final turn. Continuous in [0, 100+]; gives RL
                # signal even when the hard gate fails.
                base = max(r["dense_pct"], r["best_dense_pct"])
                if r["passed"]:
                    base += float(args.pass_bonus)
                rewards.append(_adjust_reward(
                    base,
                    args.parse_bonus if r["parsed_ok"] else 0.0,
                    r["completion_tokens"], args.length_alpha,
                ))
            mask = [True] * len(rewards)
            if args.mask_truncated:
                mask = [
                    r["stop_reason"] != "length" for r in g["rollouts"]
                ]
            advs = _group_advantages(
                rewards, mask,
                adv_clip=args.adv_clip,
                drop_zero_var=args.drop_zero_var,
            )
            all_rewards.extend(rewards)
            for j, r in enumerate(g["rollouts"]):
                adv = advs[j]
                if adv == 0.0:
                    continue
                pl = len(r["prompt_ids"])
                cl = len(r["final_ids"])
                full_ids = list(r["prompt_ids"]) + list(r["final_ids"])
                target_ids = full_ids[1:] + [0]
                if pl >= 1:
                    weights = [0.0] * (pl - 1) + [adv] * cl + [0.0]
                else:
                    weights = [adv] * cl + [0.0]
                if len(weights) != len(full_ids):
                    weights = (
                        weights + [0.0] * len(full_ids)
                    )[: len(full_ids)]
                loss_inputs = {
                    "target_tokens": wl.TensorData(
                        data=target_ids, dtype="int64",
                        shape=[len(target_ids)],
                    ),
                    "weights": wl.TensorData(
                        data=weights, dtype="float32",
                        shape=[len(weights)],
                    ),
                }
                data.append(wl.Datum(
                    model_input=wl.ModelInput.from_ints(full_ids),
                    loss_fn_inputs=loss_inputs,
                ))
                kept += 1

        if groups:
            top_by_task = {
                g["task"].task_id: max((r["ep_score"] for r in g["rollouts"]),
                                       default=0.0)
                for g in groups
            }
            n_passed_round = sum(
                1 for g in groups for r in g["rollouts"]
                if r["passed"]
            )
            n_total_round = sum(len(g["rollouts"]) for g in groups)
            append_jsonl(runs_dir / "history.jsonl", {
                "ts": time.time(), "kind": "rollout",
                "step": step, "round": round_idx,
                "n_rollouts": n_total_round, "n_kept": kept,
                "pass_rate": n_passed_round / max(1, n_total_round),
                "score_pct": (
                    sum(top_by_task.values()) /
                    (100.0 * max(1, len(top_by_task)))
                ) * 100.0,
                "reward_mean": (
                    sum(all_rewards) / max(1, len(all_rewards))
                ) if all_rewards else 0.0,
                "base_model": args.base_model,
                "algo": "grpo-multi-turn",
            })

        if data:
            try:
                heartbeat(runs_dir, phase="optim", round=round_idx,
                          step=step, n_kept=kept, n_data=len(data))
                fb = train.forward_backward(
                    data=data, loss_fn="cross_entropy",
                ).result()
                opt = train.optim_step(
                    wl.AdamParams(learning_rate=args.lr),
                ).result()
                _ = opt
                step += 1
                # ``ForwardBackwardOutput`` has ``loss_fn_outputs:
                # list[dict[str, TensorData]]`` where each entry's
                # "loss" TensorData carries the per-datum scalar
                # loss. Mean across the kept datums.
                losses: list[float] = []
                for out in getattr(fb, "loss_fn_outputs", []) or []:
                    td = out.get("loss") if isinstance(out, dict) else None
                    if td is None:
                        continue
                    vals = getattr(td, "data", None) or []
                    if vals:
                        try:
                            losses.append(float(vals[0]))
                        except (TypeError, ValueError):
                            pass
                loss = sum(losses) / len(losses) if losses else 0.0
                heartbeat(runs_dir, phase="optim_done", round=round_idx,
                          step=step, n_kept=kept, last_loss=loss)
                append_jsonl(runs_dir / "history.jsonl", {
                    "ts": time.time(), "kind": "optim",
                    "step": step, "round": round_idx,
                    "loss": loss, "n_losses": len(losses),
                    "lr": args.lr, "n_kept": kept,
                    "algo": "grpo-multi-turn",
                    "base_model": args.base_model,
                })
                print(
                    f"  [train] loss={loss:+.4f} (n={len(losses)}) "
                    f"kept={kept} step={step}"
                )
                if (args.checkpoint_every
                        and step % args.checkpoint_every == 0):
                    try:
                        name = (
                            f"mech-grpo-{train.model_id}-step{step}"
                        )
                        sv = train.save_state(name=name).result()
                        path = getattr(sv, "path", "?")
                        print(f"  [ckpt] step={step} saved -> {path}")
                        append_jsonl(runs_dir / "history.jsonl", {
                            "ts": time.time(), "kind": "checkpoint",
                            "step": step, "round": round_idx,
                            "name": name, "path": str(path),
                        })
                    except Exception as e:  # noqa: BLE001
                        print(f"  [warn] save_state: {e}")
            except Exception as e:  # noqa: BLE001
                print(
                    f"  [warn] train step: {e}\n{traceback.format_exc()}")
        else:
            print(f"  [train] skipped — 0 rollouts with non-zero advantage")
            heartbeat(runs_dir, phase="skip_optim", round=round_idx,
                      step=step, reason="no rollouts above threshold")

    print(
        f"\n[done] {args.rounds} rounds, {step} optim steps, "
        f"{(time.time() - t0) / 60:.1f} min total"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
