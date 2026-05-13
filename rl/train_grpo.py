"""GRPO-style RL loop on mech_bench tasks via worldlines.

Adapted from rl-spark/worldlines-engdesign/src/engdesign_train.py.
One round =

    1. Pick a sub-batch of tasks (uniform — curriculum is a future
       knob, see rl-spark for the EMA-weighted version).
    2. For each task, sample K rollouts from the policy.
    3. ``mech_env.score(task, raw_text)`` →
       (parsed_ok, passed, score in [0, 100], failure_codes).
    4. Per-task reward = score + parse_bonus(if parsed) -
       length_alpha * log1p(completion_tokens).
       Truncated completions (stop_reason="length") get advantage 0.
    5. Advantage_i = (reward_i - mean_K) / (std_K + ε) with optional
       clip; "drop zero variance" skips entire groups.
    6. Build worldlines ``Datum`` objects with per-token weights
       (advantage on completion tokens, 0 on prompt tokens) and call
       ``train.forward_backward(data, loss_fn="cross_entropy")`` then
       ``train.optim_step(AdamParams(lr))``. Same multiplicative-CE
       advantage trick as engdesign_train.

Outputs run logs to ``runs/<run_name>/`` so the rl-spark dashboard
shape stays compatible — ``history.jsonl``, ``task_scores.jsonl``,
``heartbeat.json``.

Auth: relies on the user's Worldlines backend at
``--backend-url`` (default 127.0.0.1:18100) with API key
``wld-local``. The backend is whatever ``rl/launch_worldlines.sh``
spun up — Qwen3-1.7B PEFT LoRA trainer here.
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
from dataclasses import asdict
from pathlib import Path
from typing import Any

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))
sys.path.insert(0, str(THIS.parent))

import mech_env as env  # noqa: E402  (after sys.path)


SYSTEM_PROMPT_PATH = THIS.parent / "scripts" / "agent_system_prompt.md"
USER_PROMPT_TEMPLATE = """Solve mech_bench task **{task_id}**.

## prompt.md
{prompt_md}

## task.toml
```toml
{task_toml}
```

Emit ONE Python file named `design.py` that defines
`build_design(out_dir: Path) -> dict`. Wrap the full file in a
single fenced ```python ... ``` block. No prose outside the block.
"""


# --------------------------------------------------------------------- #
# tiny heartbeat + log files (mirror rl-spark layout)                    #
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
# Advantage math (uniform with rl-spark)                                 #
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
    """(r - mean) / (std + eps), with mask + optional clip + drop-zero-var."""
    if not rewards:
        return []
    if drop_zero_var and len(set(rewards)) <= 1:
        return [0.0] * len(rewards)
    valid = [r for r, m in zip(rewards, mask) if m]
    if not valid:
        return [0.0] * len(rewards)
    mu = sum(valid) / len(valid)
    if len(valid) > 1:
        sigma = statistics.pstdev(valid)
    else:
        sigma = 0.0
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
# main                                                                   #
# --------------------------------------------------------------------- #


def main() -> int:
    p = argparse.ArgumentParser(prog="train_grpo")
    p.add_argument("--backend-url", default="http://127.0.0.1:18100")
    p.add_argument("--api-key", default="wld-local")
    p.add_argument("--base-model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--tokenizer", default=None,
                   help="defaults to --base-model")
    p.add_argument("--run-name", default="mech-grpo")
    p.add_argument("--runs-root", default="runs",
                   help="root dir for per-run logs (relative to repo)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--tasks-per-round", type=int, default=4)
    p.add_argument("--samples-per-task", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--max-prompt-tokens", type=int, default=8192)
    p.add_argument("--rollout-temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--length-alpha", type=float, default=0.0)
    p.add_argument("--parse-bonus", type=float, default=5.0)
    p.add_argument("--adv-clip", type=float, default=5.0)
    p.add_argument("--drop-zero-var", action="store_true", default=True)
    p.add_argument("--mask-truncated", action="store_true", default=True)
    p.add_argument("--mask-prompt", action="store_true", default=True)
    p.add_argument("--checkpoint-every", type=int, default=10)
    p.add_argument("--families", default=None,
                   help="comma-separated family allowlist (e.g. "
                        "'mounting_plate_hole_pitch,spur_gear_ratio_analytic')")
    p.add_argument("--tiers", default=None,
                   help="comma-separated tier allowlist "
                        "(artifact_static / planar_kinematics / "
                        "transmission_analytic / contact_dynamics)")
    p.add_argument("--score-timeout", type=float, default=60.0,
                   help="seconds for one mech_bench evaluate")
    args = p.parse_args()

    rng = random.Random(args.seed)
    repo_root = Path(__file__).resolve().parent.parent
    runs_dir = repo_root / args.runs_root / args.run_name
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Load tasks.
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

    # Worldlines clients (lazy-imported so the script still parses on
    # machines without the SDK).
    import worldlines as wl  # type: ignore[import-not-found]
    from transformers import AutoTokenizer  # type: ignore[import-not-found]

    svc = wl.ServiceClient(base_url=args.backend_url, api_key=args.api_key)
    base_sampler = svc.create_sampling_client(base_model=args.base_model)
    rollout_sampler = base_sampler  # until we publish a trained adapter
    print(f"connected to {args.backend_url}, base={args.base_model}")

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

    def _chat_tokens(task: env.TaskInfo) -> list[int]:
        user = USER_PROMPT_TEMPLATE.format(
            task_id=task.task_id,
            prompt_md=task.prompt,
            task_toml=task.task_toml,
        )
        msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user},
        ]
        prompt_text = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
        )
        ids = tok.encode(prompt_text, add_special_tokens=False)
        return [int(x) for x in ids]

    step = 0
    t0 = time.time()
    heartbeat(runs_dir, phase="starting", round=0, step=0,
              base_model=args.base_model, algo="grpo",
              tokenizer=tokenizer_name,
              n_tasks=len(tasks))

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
            heartbeat(runs_dir, phase="rollout", round=round_idx,
                      step=step, current_task=ti.task_id)
            prompt_ids = _chat_tokens(ti)
            if len(prompt_ids) + args.max_tokens > args.max_prompt_tokens:
                print(f"  [skip] {ti.task_id} prompt too long "
                      f"({len(prompt_ids)} + {args.max_tokens})")
                continue
            mi = wl.ModelInput.from_ints(prompt_ids)
            params = wl.SamplingParams(
                max_tokens=args.max_tokens,
                temperature=args.rollout_temperature,
                top_p=args.top_p,
            )
            try:
                fut = rollout_sampler.sample(
                    prompt=mi,
                    num_samples=args.samples_per_task,
                    sampling_params=params,
                )
                resp = fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"  [warn] sample {ti.task_id}: {e}")
                continue

            rollouts: list[dict] = []
            for seq in resp.sequences:
                raw = tok.decode(list(seq.tokens), skip_special_tokens=True)
                ep = env.score(
                    ti, raw, parse_bonus=args.parse_bonus,
                    timeout_s=args.score_timeout,
                )
                ep.completion_tokens = len(seq.tokens)
                stop_reason = getattr(seq, "stop_reason", "stop")
                rollouts.append({
                    "raw": raw, "ep": ep,
                    "stop_reason": stop_reason,
                    "tokens": list(seq.tokens),
                    "prompt_ids": prompt_ids,
                })
            best = max((r["ep"].score for r in rollouts), default=0.0)
            n_parsed = sum(1 for r in rollouts if r["ep"].parsed_ok)
            n_passed = sum(1 for r in rollouts if r["ep"].passed)
            print(
                f"  {ti.task_id:48}  best={best:5.1f}  "
                f"parsed={n_parsed}/{len(rollouts)}  "
                f"passed={n_passed}/{len(rollouts)}"
            )
            for r in rollouts:
                ep = r["ep"]
                append_jsonl(runs_dir / "task_scores.jsonl", {
                    "ts": time.time(),
                    "task_id": ti.task_id, "tier": ti.tier,
                    "family": ti.family,
                    "score": ep.score, "passed": ep.passed,
                    "parsed_ok": ep.parsed_ok,
                    "failure_codes": ep.failure_codes,
                    "completion_tokens": ep.completion_tokens,
                    "stop_reason": r["stop_reason"],
                    "round": round_idx, "step": step,
                })
            groups.append({"task": ti, "rollouts": rollouts})

        heartbeat(runs_dir, phase="scoring", round=round_idx, step=step,
                  n_groups=len(groups))

        # Build advantage-weighted training data.
        data: list = []
        kept = 0
        all_rewards: list[float] = []
        for g in groups:
            rewards: list[float] = []
            for r in g["rollouts"]:
                rewards.append(_adjust_reward(
                    r["ep"].score, r["ep"].parse_bonus,
                    r["ep"].completion_tokens, args.length_alpha,
                ))
            mask = [True] * len(rewards)
            if args.mask_truncated:
                mask = [r["stop_reason"] != "length"
                        for r in g["rollouts"]]
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
                completion_ids = list(r["tokens"])
                if not completion_ids:
                    continue
                pl = len(r["prompt_ids"])
                cl = len(completion_ids)
                full_ids = list(r["prompt_ids"]) + completion_ids
                target_ids = full_ids[1:] + [0]
                if args.mask_prompt:
                    if pl >= 1:
                        weights = [0.0] * (pl - 1) + [adv] * cl + [0.0]
                    else:
                        weights = [adv] * cl + [0.0]
                    if len(weights) != len(full_ids):
                        weights = (
                            weights + [0.0] * len(full_ids)
                        )[: len(full_ids)]
                else:
                    weights = [adv] * len(full_ids)
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

        # Round-level history row.
        if groups:
            top_by_task = {
                g["task"].task_id: max((r["ep"].score for r in g["rollouts"]),
                                       default=0.0)
                for g in groups
            }
            n_passed_round = sum(
                1 for g in groups for r in g["rollouts"]
                if r["ep"].passed
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
                ),
                "base_model": args.base_model,
                "algo": "grpo",
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
                loss = float(getattr(fb, "loss", 0.0))
                heartbeat(runs_dir, phase="optim_done", round=round_idx,
                          step=step, n_kept=kept, last_loss=loss)
                append_jsonl(runs_dir / "history.jsonl", {
                    "ts": time.time(), "kind": "optim",
                    "step": step, "round": round_idx,
                    "loss": loss, "lr": args.lr, "n_kept": kept,
                    "algo": "grpo", "base_model": args.base_model,
                })
                print(f"  [train] loss={loss:.4f} kept={kept} step={step}")
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
                print(f"  [warn] train step: {e}\n{traceback.format_exc()}")
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
