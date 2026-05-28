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
import re
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

## verifier contract
{contract}

## prompt.md
{prompt_md}

## task.toml
```toml
{task_toml}
```

Emit ONE Python file as a single fenced ```python ... ``` block.
No prose outside the block.

Use the exact required port ids and exact `params.*` keys shown in
the verifier contract or prompt.md. Do not add a `declared_` prefix
unless the contract or prompt itself uses that exact key. Legal port
kinds are only `frame`,
`revolute_joint`, and `prismatic_joint`. The task's explicit
`requirements.expected_mobility`, prompt mobility statement, and
required port kinds override the task title and tier name.
"""


_PARAM_RE = re.compile(r"params\.([A-Za-z_][A-Za-z0-9_]*)")


def _collect_param_paths(raw: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(raw, dict):
        for value in raw.values():
            found.update(_collect_param_paths(value))
    elif isinstance(raw, list):
        for value in raw:
            found.update(_collect_param_paths(value))
    elif isinstance(raw, str) and raw.startswith("params."):
        found.add(raw)
    return found


def _format_param_constraint(probe: dict[str, Any]) -> str | None:
    path = probe.get("path")
    if not isinstance(path, str) or not path.startswith("params."):
        return None
    if "expected" not in probe:
        return None
    comparator = str(probe.get("comparator", "eq"))
    op = {
        "eq": "==",
        "le": "<=",
        "lt": "<",
        "ge": ">=",
        "gt": ">",
    }.get(comparator, comparator)
    expected = probe.get("expected")
    bits = [f"{path} {op} {expected}"]
    if "tolerance_abs" in probe:
        bits.append(f"abs_tol={probe['tolerance_abs']}")
    if "tolerance_pct" in probe:
        bits.append(f"pct_tol={probe['tolerance_pct']}")
    if len(bits) > 1:
        return f"{bits[0]} ({', '.join(bits[1:])})"
    return bits[0]


def _collect_public_param_constraints(raw: Any) -> list[str]:
    if not isinstance(raw, dict):
        return []
    out: list[str] = []
    for probe in raw.get("probes", []) or []:
        if not isinstance(probe, dict):
            continue
        if probe.get("type") != "analytic_param_check":
            continue
        constraint = _format_param_constraint(probe)
        if constraint:
            out.append(constraint)
    return out


def _contract_from_task(
    prompt_md: str,
    task_toml: str,
    eval_config_toml: str = "",
) -> str:
    """Extract the visible contract into a small, hard-to-miss block."""
    ports: list[str] = []
    mobility: int | None = None
    try:
        import tomllib
        blob = tomllib.loads(task_toml)
        req = blob.get("requirements", {})
        raw_ports = req.get("required_ports") or []
        if isinstance(raw_ports, list):
            ports = [str(p) for p in raw_ports]
        raw_mobility = req.get("expected_mobility")
        if raw_mobility is not None:
            mobility = int(raw_mobility)
    except Exception:  # noqa: BLE001 - prompt helper must not kill training
        pass

    params = {f"params.{m}" for m in _PARAM_RE.findall(prompt_md)}
    constraints: list[str] = []
    if eval_config_toml:
        try:
            import tomllib
            eval_blob = tomllib.loads(eval_config_toml)
            params.update(_collect_param_paths(eval_blob))
            constraints = _collect_public_param_constraints(eval_blob)
        except Exception:  # noqa: BLE001
            pass
    params = sorted(params)
    lines: list[str] = []
    if mobility is not None:
        lines.append(f"- expected_mobility: {mobility}")
    if ports:
        lines.append("- required_ports: " + ", ".join(f"`{p}`" for p in ports))
    if params:
        lines.append("- required_params: " + ", ".join(f"`{p}`" for p in params))
    for constraint in constraints:
        lines.append(f"- public_param_constraint: `{constraint}`")
    if mobility == 0:
        lines.append(
            "- static topology: use one fixed carrier part or fixed joints "
            "only; do not add revolute/prismatic joints unless the prompt "
            "explicitly requires them"
        )
    if mobility == 1:
        lines.append(
            "- mobility-1 topology: create exactly the required moving joint "
            "and point joint-kind ports at the joint id"
        )
    return "\n".join(lines) if lines else "- no extracted contract"


def _build_user_prompt(ti: env.TaskInfo) -> str:
    eval_config = ""
    eval_config_path = ti.task_dir / "eval_config.public.toml"
    if not eval_config_path.exists():
        eval_config_path = ti.task_dir / "eval_config.toml"
    if eval_config_path.exists():
        eval_config = eval_config_path.read_text()
    return USER_PROMPT_TEMPLATE.format(
        task_id=ti.task_id,
        contract=_contract_from_task(ti.prompt, ti.task_toml, eval_config),
        prompt_md=ti.prompt,
        task_toml=ti.task_toml,
    )


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


def _sample_tasks(
    tasks: list[env.TaskInfo],
    k: int,
    rng: random.Random,
    *,
    family_balanced: bool = False,
) -> list[env.TaskInfo]:
    if k <= 0:
        return []
    if not family_balanced:
        return rng.sample(tasks, min(k, len(tasks)))

    by_family: dict[str, list[env.TaskInfo]] = {}
    for task in tasks:
        by_family.setdefault(task.family, []).append(task)
    families = list(by_family)
    out: list[env.TaskInfo] = []
    used: set[str] = set()
    while len(out) < min(k, len(tasks)):
        rng.shuffle(families)
        progressed = False
        for family in families:
            choices = [
                task for task in by_family[family]
                if task.task_id not in used
            ]
            if not choices:
                continue
            task = rng.choice(choices)
            out.append(task)
            used.add(task.task_id)
            progressed = True
            if len(out) >= min(k, len(tasks)):
                break
        if not progressed:
            break
    return out


def _sample_anchor_items(
    items: list[dict[str, Any]],
    k: int,
    rng: random.Random,
    *,
    family_balanced: bool = False,
) -> list[dict[str, Any]]:
    if k <= 0 or not items:
        return []
    if not family_balanced:
        return rng.sample(items, min(k, len(items)))

    by_family: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        task = item["task"]
        by_family.setdefault(task.family, []).append(item)
    families = list(by_family)
    out: list[dict[str, Any]] = []
    used_sources: set[str] = set()
    while len(out) < min(k, len(items)):
        rng.shuffle(families)
        progressed = False
        for family in families:
            choices = [
                item for item in by_family[family]
                if str(item.get("source", id(item))) not in used_sources
            ]
            if not choices:
                continue
            item = rng.choice(choices)
            out.append(item)
            used_sources.add(str(item.get("source", id(item))))
            progressed = True
            if len(out) >= min(k, len(items)):
                break
        if not progressed:
            break
    return out


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
    p.add_argument("--rollout-backend", default="sglang_chat",
                   choices=["sglang_chat", "worldlines_sampling"],
                   help="sampler used for rollouts; worldlines_sampling can "
                        "consume saved LoRA adapter checkpoints")
    p.add_argument(
        "--base-model",
        default="NousResearch/DeepHermes-3-Llama-3-3B-Preview")
    p.add_argument("--tokenizer", default=None,
                   help="defaults to --base-model")
    p.add_argument("--run-name", default="mech-grpo")
    p.add_argument("--runs-root", default="runs")
    p.add_argument("--tasks-root", default="tasks",
                   help="task suite root, relative to repo root unless "
                        "absolute")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--tasks-per-round", type=int, default=4)
    p.add_argument("--samples-per-task", type=int, default=4)
    p.add_argument("--family-balanced-task-sampler", action="store_true",
                   help="sample at most one task per family per pass before "
                        "revisiting a family; reduces small-split overfit")
    p.add_argument("--max-turns", type=int, default=4,
                   help="max assistant turns per rollout")
    p.add_argument("--max-tokens-per-turn", type=int, default=4096)
    p.add_argument("--max-context-tokens", type=int, default=16384,
                   help="hard cap on full prompt+completion length")
    p.add_argument("--rollout-temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--init-training-state", default=None,
                   help="optional worldlines:// training state path to "
                        "continue from instead of creating a fresh LoRA")
    p.add_argument("--length-alpha", type=float, default=0.0)
    p.add_argument("--parse-bonus", type=float, default=5.0)
    p.add_argument("--pass-bonus", type=float, default=25.0,
                   help="extra reward when the final turn has non-zero "
                        "verified score (on top of dense_pct)")
    p.add_argument("--pass-score-threshold", type=float, default=100.0,
                   help="0-100 verified score required before a rollout is "
                        "treated as a pass for pass bonuses, pass-rate logs, "
                        "and --positive-only-passes")
    p.add_argument("--invalid-penalty", type=float, default=20.0,
                   help="reward penalty when mech_bench marks the "
                        "artifact evaluation invalid")
    p.add_argument("--critical-code-penalty", type=float, default=5.0,
                   help="additional reward penalty per critical verifier "
                        "code such as schema_error or invalid_artifact")
    p.add_argument("--positive-only-passes", action="store_true",
                   help="train only on verifier-passing final turns, using "
                        "a fixed positive weight instead of group-relative "
                        "negative updates")
    p.add_argument("--positive-pass-weight", type=float, default=1.0,
                   help="total sequence weight for each passing rollout when "
                        "--positive-only-passes is enabled")
    p.add_argument("--positive-min-score", type=float, default=100.0,
                   help="minimum 0-100 verified score for a rollout to be "
                        "used by --positive-only-passes")
    p.add_argument("--max-train-datums-per-step", type=int, default=0,
                   help="optional deterministic cap on kept training datums "
                        "per optimizer step; 0 keeps all datums")
    p.add_argument("--reference-sft-weight", type=float, default=0.0,
                   help="optional total sequence weight for supervised "
                        "reference-solution anchors added to each train step")
    p.add_argument("--reference-sft-split-file", default=None,
                   help="newline-delimited task ids whose reference_solution "
                        "design.py files may be used as positive anchors")
    p.add_argument("--reference-sft-per-step", type=int, default=0,
                   help="number of reference anchors to add per optimizer "
                        "step; defaults to tasks-per-round when weight > 0")
    p.add_argument("--sample-sft-summary-file", default=None,
                   help="optional sample_and_score smoke_summary.json whose "
                        "strict-passing completions are used as verified "
                        "self-imitation anchors")
    p.add_argument("--sample-sft-weight", type=float, default=0.0,
                   help="optional total sequence weight for each verified "
                        "sample anchor added to a train step")
    p.add_argument("--sample-sft-per-step", type=int, default=0,
                   help="number of verified sample anchors to add per "
                        "optimizer step; defaults to tasks-per-round")
    p.add_argument("--sample-sft-min-score", type=float, default=1.0,
                   help="minimum 0-1 verified_score from sample_and_score "
                        "for a completion to be eligible")
    p.add_argument("--sft-warmup-rounds", type=int, default=0,
                   help="initial rounds that skip rollouts and train only on "
                        "reference-solution anchors; useful before online "
                        "RL when the base sampler rarely emits valid IR")
    p.add_argument("--adv-clip", type=float, default=2.0,
                   help="absolute clip on per-rollout advantage; "
                        "advantage * completion-len-many-tokens "
                        "produces big gradients when clip is loose")
    p.add_argument("--drop-zero-var", action="store_true", default=True)
    p.add_argument("--mask-truncated", action="store_true", default=True)
    p.add_argument("--checkpoint-every", type=int, default=10)
    p.add_argument("--refresh-sampler-every", type=int, default=0,
                   help="with --rollout-backend=worldlines_sampling, save "
                        "current LoRA weights and use that adapter for future "
                        "rollouts every N optimizer steps")
    p.add_argument("--save-final-sampler-name", default=None,
                   help="optional name for final sampler weights exported via "
                        "save_weights_for_sampler")
    p.add_argument("--families", default=None)
    p.add_argument("--tiers", default=None)
    p.add_argument("--split-file", default=None,
                   help="optional newline-delimited task_id allowlist")
    p.add_argument("--score-timeout", type=float, default=60.0)
    p.add_argument("--rollout-timeout", type=float, default=300.0)
    args = p.parse_args()

    rng = random.Random(args.seed)
    repo_root = Path(__file__).resolve().parent.parent
    runs_dir = repo_root / args.runs_root / args.run_name
    runs_dir.mkdir(parents=True, exist_ok=True)
    (runs_dir / "train_config.json").write_text(
        json.dumps(
            {
                "argv": sys.argv,
                "args": vars(args),
                "repo_root": str(repo_root),
                "started_ts": time.time(),
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n"
    )

    families = (
        {s.strip() for s in args.families.split(",") if s.strip()}
        if args.families else None
    )
    tiers = (
        {s.strip() for s in args.tiers.split(",") if s.strip()}
        if args.tiers else None
    )
    tasks_root = Path(args.tasks_root)
    if not tasks_root.is_absolute():
        tasks_root = repo_root / tasks_root

    tasks = env.list_tasks(
        root=tasks_root,
        families=families, tiers=tiers,
        split_file=Path(args.split_file) if args.split_file else None,
    )
    if not tasks:
        print("error: no tasks matched", file=sys.stderr)
        return 2
    print(f"loaded {len(tasks)} tasks")
    task_by_id = {t.task_id: t for t in tasks}

    reference_tasks: list[env.TaskInfo] = []
    if args.reference_sft_weight > 0:
        reference_tasks = env.list_tasks(
            root=tasks_root,
            families=families, tiers=tiers,
            split_file=(
                Path(args.reference_sft_split_file)
                if args.reference_sft_split_file else
                (Path(args.split_file) if args.split_file else None)
            ),
        )
        reference_tasks = [
            t for t in reference_tasks
            if (t.task_dir / "reference_solution" / "design.py").exists()
        ]
        if not reference_tasks:
            print("error: reference SFT requested but no reference_solution "
                  "files matched", file=sys.stderr)
            return 2
        print(
            f"loaded {len(reference_tasks)} reference anchor tasks "
            f"(weight={args.reference_sft_weight})"
        )

    sample_sft_items: list[dict[str, Any]] = []
    if args.sample_sft_weight > 0:
        if not args.sample_sft_summary_file:
            print("error: --sample-sft-weight requires "
                  "--sample-sft-summary-file", file=sys.stderr)
            return 2
        summary_path = Path(args.sample_sft_summary_file).resolve()
        try:
            summary = json.loads(summary_path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            print(f"error: cannot read sample SFT summary {summary_path}: {e}",
                  file=sys.stderr)
            return 2
        rows = summary.get("all_samples") or summary.get("tasks") or []
        for row in rows:
            if not row.get("strict_passed"):
                continue
            try:
                score = float(
                    row.get("verified_score", row.get("score", 0.0)) or 0.0
                )
            except (TypeError, ValueError):
                score = 0.0
            if score < float(args.sample_sft_min_score):
                continue
            task_id = str(row.get("task_id", ""))
            task = task_by_id.get(task_id)
            if task is None:
                continue
            try:
                sample_idx = int(row.get("sample_idx", 0) or 0)
            except (TypeError, ValueError):
                sample_idx = 0
            completion_path = (
                summary_path.parent / f"sample_{sample_idx}" /
                task_id / "completion.txt"
            )
            if not completion_path.exists():
                continue
            assistant = completion_path.read_text().rstrip()
            if not assistant:
                continue
            sample_sft_items.append({
                "task": task,
                "assistant": assistant,
                "score": score,
                "sample_idx": sample_idx,
                "source": str(completion_path),
            })
        if not sample_sft_items:
            print("error: sample SFT requested but no strict-passing "
                  "completion files matched the active task split",
                  file=sys.stderr)
            return 2
        print(
            f"loaded {len(sample_sft_items)} verified sample anchors "
            f"from {summary_path} (weight={args.sample_sft_weight})"
        )

    import worldlines as wl  # type: ignore[import-not-found]
    from transformers import AutoTokenizer  # type: ignore[import-not-found]

    svc = wl.ServiceClient(base_url=args.backend_url, api_key=args.api_key)
    print(f"worldlines @ {args.backend_url}")
    if args.init_training_state:
        print(f"loading LoRA training client from state "
              f"{args.init_training_state} ...")
        train = svc.create_training_client_from_state(
            args.init_training_state,
        )
    else:
        print("creating LoRA training client ...")
        train = svc.create_lora_training_client(
            base_model=args.base_model,
            rank=args.lora_rank,
            seed=args.seed,
        )
    print(f"training client ready. model_id={train.model_id}")

    tokenizer_name = args.tokenizer or args.base_model
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    system_prompt = SYSTEM_PROMPT_PATH.read_text()
    rollout_sampling_client = None
    rollout_sampling_client_needs_init_adapter = bool(args.init_training_state)
    if args.rollout_backend == "worldlines_sampling":
        print("worldlines rollout sampling client will be created lazily")

    step = 0
    t0 = time.time()
    heartbeat(runs_dir, phase="starting", round=0, step=0,
              base_model=args.base_model, algo="grpo-multi-turn",
              tokenizer=tokenizer_name,
              n_tasks=len(tasks),
              sglang_url=args.sglang_url,
              backend_url=args.backend_url,
              rollout_backend=args.rollout_backend)

    for round_idx in range(args.rounds):
        batch = _sample_tasks(
            tasks, args.tasks_per_round, rng,
            family_balanced=args.family_balanced_task_sampler,
        )
        print(
            f"\n=== round {round_idx} step={step} "
            f"tasks={[t.task_id for t in batch]} ===")
        heartbeat(runs_dir, phase="rollout", round=round_idx, step=step,
                  tasks=[t.task_id for t in batch],
                  rollouts_done=0,
                  rollouts_target=len(batch) * args.samples_per_task)

        groups: list[dict] = []
        warmup_only = (
            args.sft_warmup_rounds > 0
            and round_idx < args.sft_warmup_rounds
            and (
                (args.reference_sft_weight > 0 and reference_tasks)
                or (args.sample_sft_weight > 0 and sample_sft_items)
            )
        )
        if warmup_only:
            print(
                f"  [sft] warmup round {round_idx + 1}/"
                f"{args.sft_warmup_rounds}: skipping rollouts"
            )

        for task_idx, ti in enumerate(batch):
            if warmup_only:
                continue
            user_prompt = _build_user_prompt(ti)
            rollouts: list[dict] = []
            for k in range(args.samples_per_task):
                heartbeat(runs_dir, phase="rollout",
                          round=round_idx, step=step,
                          current_task=ti.task_id, sample_idx=k,
                          rollouts_target=len(batch) * args.samples_per_task)
                try:
                    if args.rollout_backend == "worldlines_sampling":
                        if rollout_sampling_client is None:
                            if rollout_sampling_client_needs_init_adapter:
                                name = f"{args.run_name}-rollout-init"
                                print(
                                    "exporting initialized adapter for "
                                    f"rollout -> {name}"
                                )
                                rollout_sampling_client = (
                                    train.save_weights_and_get_sampling_client(
                                        name=name
                                    )
                                )
                                print("initialized adapter sampling client ready")
                            else:
                                print("creating base sampling client for rollout ...")
                                rollout_sampling_client = svc.create_sampling_client(
                                    base_model=args.base_model,
                                )
                                print("base sampling client ready")
                        r = cr.run_rollout_with_sampling_client(
                            sampling_client=rollout_sampling_client,
                            tokenizer=tok,
                            task=ti,
                            system_prompt=system_prompt,
                            user_prompt=user_prompt,
                            max_turns=args.max_turns,
                            max_tokens_per_turn=args.max_tokens_per_turn,
                            temperature=args.rollout_temperature,
                            top_p=args.top_p,
                            timeout_s=args.rollout_timeout,
                            parse_bonus=args.parse_bonus,
                            seed=(
                                args.seed
                                + round_idx * 100000
                                + task_idx * 1000
                                + k
                            ),
                        )
                    else:
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
                            seed=(
                                args.seed
                                + round_idx * 100000
                                + task_idx * 1000
                                + k
                            ),
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
                    "passed": (
                        last.passed
                        and float(last.score) >= args.pass_score_threshold
                    ),
                    "hard_gate_nonzero": last.passed,
                    "parsed_ok": last.parsed_ok,
                    "evaluation_valid": last.evaluation_valid,
                    "completion_tokens": last.completion_tokens,
                    "stop_reason": last.stop_reason,
                    "failure_codes": last.failure_codes,
                    "feedback": last.feedback,
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
            n_hard_gate_nonzero = sum(
                1 for r in rollouts if r["hard_gate_nonzero"]
            )
            n_parsed = sum(1 for r in rollouts if r["parsed_ok"])
            print(
                f"  {ti.task_id:48}  pass={best_pass:5.1f}  "
                f"dense_best={best_dense:5.1f}  "
                f"dense_avg={mean_dense:5.1f}  "
                f"parsed={n_parsed}/{len(rollouts)}  "
                f"passed={n_passed}/{len(rollouts)}  "
                f"gate_nonzero={n_hard_gate_nonzero}/{len(rollouts)}"
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
                    "hard_gate_nonzero": r["hard_gate_nonzero"],
                    "parsed_ok": r["parsed_ok"],
                    "evaluation_valid": r["evaluation_valid"],
                    "failure_codes": r["failure_codes"],
                    "feedback": r["feedback"],
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
        n_reference_sft = 0
        n_sample_sft = 0

        def _add_ids_datum(
            prompt_ids: list[int],
            final_ids: list[int],
            total_weight: float,
        ) -> None:
            nonlocal kept
            pl = len(prompt_ids)
            cl = len(final_ids)
            full_ids = list(prompt_ids) + list(final_ids)
            target_ids = full_ids[1:] + [0]
            # Per-token weight = advantage / completion_len so the
            # total weight contributed by ONE rollout is |adv| rather
            # than |adv| * completion_len. Without this the gradient
            # scales linearly with sequence length and blows up.
            per_tok = float(total_weight) / max(1, cl)
            if pl >= 1:
                weights = [0.0] * (pl - 1) + [per_tok] * cl + [0.0]
            else:
                weights = [per_tok] * cl + [0.0]
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

        def _add_weighted_datum(r: dict, adv: float) -> None:
            _add_ids_datum(r["prompt_ids"], r["final_ids"], adv)

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
                if not r["evaluation_valid"]:
                    base -= float(args.invalid_penalty)
                critical_codes = {
                    "schema_error",
                    "invalid_artifact",
                    "invalid_mass_properties",
                    "missing_port",
                    "wrong_topology",
                }
                n_critical = sum(
                    1 for code in r["failure_codes"]
                    if code in critical_codes
                )
                base -= float(args.critical_code_penalty) * n_critical
                rewards.append(_adjust_reward(
                    base,
                    args.parse_bonus if r["parsed_ok"] else 0.0,
                    r["completion_tokens"], args.length_alpha,
                ))
            if args.positive_only_passes:
                all_rewards.extend(rewards)
                for r in g["rollouts"]:
                    if (
                        not r["passed"]
                        or float(r["ep_score"]) < args.positive_min_score
                    ):
                        continue
                    _add_weighted_datum(
                        r, float(args.positive_pass_weight))
                continue
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
                _add_weighted_datum(r, adv)

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

        rl_data_len = len(data)
        if args.reference_sft_weight > 0 and reference_tasks:
            n_ref = (
                args.reference_sft_per_step
                if args.reference_sft_per_step > 0
                else len(batch)
            )
            ref_batch = _sample_tasks(
                reference_tasks, n_ref, rng, family_balanced=True,
            )
            for rt in ref_batch:
                src = (
                    rt.task_dir / "reference_solution" / "design.py"
                ).read_text().rstrip()
                assistant = f"```python\n{src}\n```"
                user_prompt = _build_user_prompt(rt)
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": assistant},
                ]
                prompt_ids, final_ids = (
                    _split_into_prompt_and_final_assistant(tok, messages)
                )
                if not final_ids:
                    continue
                if len(prompt_ids) + len(final_ids) > args.max_context_tokens:
                    print(f"  [skip-sft] {rt.task_id} "
                          f"full_len={len(prompt_ids)+len(final_ids)} > "
                          f"{args.max_context_tokens}")
                    continue
                _add_ids_datum(
                    prompt_ids, final_ids,
                    float(args.reference_sft_weight),
                )
                n_reference_sft += 1
            if n_reference_sft:
                print(
                    f"  [sft] added {n_reference_sft} reference anchors "
                    f"weight={args.reference_sft_weight}"
                )

        if args.sample_sft_weight > 0 and sample_sft_items:
            n_sample = (
                args.sample_sft_per_step
                if args.sample_sft_per_step > 0
                else len(batch)
            )
            sample_batch = _sample_anchor_items(
                sample_sft_items, n_sample, rng, family_balanced=True,
            )
            for item in sample_batch:
                task = item["task"]
                user_prompt = _build_user_prompt(task)
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": item["assistant"]},
                ]
                prompt_ids, final_ids = (
                    _split_into_prompt_and_final_assistant(tok, messages)
                )
                if not final_ids:
                    continue
                if len(prompt_ids) + len(final_ids) > args.max_context_tokens:
                    print(f"  [skip-sample-sft] {task.task_id} "
                          f"full_len={len(prompt_ids)+len(final_ids)} > "
                          f"{args.max_context_tokens}")
                    continue
                _add_ids_datum(
                    prompt_ids, final_ids, float(args.sample_sft_weight),
                )
                n_sample_sft += 1
            if n_sample_sft:
                print(
                    f"  [sft] added {n_sample_sft} verified sample anchors "
                    f"weight={args.sample_sft_weight}"
                )

        if args.max_train_datums_per_step > 0 and len(data) > args.max_train_datums_per_step:
            n_before_cap = len(data)
            cap = args.max_train_datums_per_step
            rl_data = data[:rl_data_len]
            sft_data = data[rl_data_len:]
            rng.shuffle(rl_data)
            rng.shuffle(sft_data)
            if len(sft_data) >= cap:
                data = sft_data[:cap]
            else:
                data = rl_data[: cap - len(sft_data)] + sft_data
            kept = len(data)
            print(
                f"  [train] capped kept datums "
                f"{n_before_cap}->{len(data)} for memory "
                f"(sft_kept={min(len(sft_data), len(data))})"
            )

        if data:
            try:
                heartbeat(runs_dir, phase="optim", round=round_idx,
                          step=step, n_kept=kept, n_data=len(data),
                          n_reference_sft=n_reference_sft,
                          n_sample_sft=n_sample_sft)
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
                    "n_reference_sft": n_reference_sft,
                    "n_sample_sft": n_sample_sft,
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
                if (
                    args.rollout_backend == "worldlines_sampling"
                    and args.refresh_sampler_every
                    and step % args.refresh_sampler_every == 0
                ):
                    name = f"{args.run_name}-rollout-step{step}"
                    print(f"  [sampler] exporting adapter -> {name}")
                    heartbeat(runs_dir, phase="sampler_export",
                              round=round_idx, step=step, name=name)
                    rollout_sampling_client = (
                        train.save_weights_and_get_sampling_client(
                            name=name,
                        )
                    )
                    append_jsonl(runs_dir / "history.jsonl", {
                        "ts": time.time(), "kind": "sampler_export",
                        "step": step, "round": round_idx,
                        "name": name,
                        "algo": "grpo-multi-turn",
                        "base_model": args.base_model,
                    })
                    print("  [sampler] adapter sampling client ready")
            except Exception as e:  # noqa: BLE001
                append_jsonl(runs_dir / "history.jsonl", {
                    "ts": time.time(), "kind": "train_error",
                    "step": step, "round": round_idx,
                    "error": f"{type(e).__name__}: {e}"[:1000],
                    "traceback": traceback.format_exc()[-2000:],
                    "n_kept": kept,
                    "n_reference_sft": n_reference_sft,
                    "n_sample_sft": n_sample_sft,
                    "algo": "grpo-multi-turn",
                    "base_model": args.base_model,
                })
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
    if args.save_final_sampler_name and step > 0:
        print(f"[sampler] saving final weights -> {args.save_final_sampler_name}")
        result = train.save_weights_for_sampler(
            name=args.save_final_sampler_name,
        ).result()
        path = getattr(result, "path", None)
        manifest = {
            "ts": time.time(),
            "kind": "final_sampler",
            "name": args.save_final_sampler_name,
            "path": str(path),
            "step": step,
            "base_model": args.base_model,
            "lora_rank": args.lora_rank,
            "rollout_backend": args.rollout_backend,
        }
        (runs_dir / "sampler_manifest.json").write_text(
            json.dumps(manifest, indent=2, default=str)
        )
        append_jsonl(runs_dir / "history.jsonl", manifest)
        print(f"[sampler] final path={path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
