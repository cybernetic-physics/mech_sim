#!/usr/bin/env python3
"""Exact TRL GRPO training for mech_bench verifier rewards.

This is the canonical GRPO path. It uses Hugging Face TRL's ``GRPOTrainer``
instead of the legacy Worldlines group-relative weighted-CE loop in
``rl/train_grpo.py``. The reward function executes the existing deterministic
``mech_bench`` verifier for each sampled completion and returns verified reward
only.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "rl"))

import mech_env as env  # noqa: E402
from rl.mech_bench_reward import score_completion  # noqa: E402
from rl.train_grpo import _build_user_prompt  # noqa: E402


SYSTEM_PROMPT_PATH = REPO_ROOT / "rl" / "agent_prompt_rl.md"
SCHEMA = "mech_bench.true_grpo_trl.v1"


def _completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts: list[str] = []
        for item in completion:
            if isinstance(item, dict):
                content = item.get("content")
                if content is not None:
                    parts.append(str(content))
            elif item is not None:
                parts.append(str(item))
        return "\n".join(parts)
    return "" if completion is None else str(completion)


def _read_split_file(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    return {
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    }


def _format_prompt(task: env.TaskInfo, system_prompt: str) -> str:
    return "\n\n".join([
        "### System",
        system_prompt.strip(),
        "### User",
        _build_user_prompt(task).strip(),
        "### Assistant",
    ])


def _build_rows(
    *,
    tasks_root: Path,
    split_file: Path | None,
    families: set[str] | None,
    tiers: set[str] | None,
    system_prompt: str,
    limit: int,
) -> list[dict[str, Any]]:
    tasks = env.list_tasks(
        root=tasks_root,
        split_file=split_file,
        families=families,
        tiers=tiers,
    )
    if limit > 0:
        tasks = tasks[:limit]
    rows: list[dict[str, Any]] = []
    for task in tasks:
        rows.append({
            "prompt": _format_prompt(task, system_prompt),
            "task_id": task.task_id,
            "task_dir": str(task.task_dir.resolve()),
            "family": task.family,
            "tier": task.tier,
        })
    return rows


def _filtered_config(cls: type, values: dict[str, Any]) -> Any:
    params = inspect.signature(cls.__init__).parameters
    accepted = {
        key: value for key, value in values.items()
        if key in params and value is not None
    }
    return cls(**accepted)


def _make_reward_func(
    *,
    log_path: Path,
    scratch_root: Path,
    timeout_s: float,
    reward_scale: float,
):
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def reward_func(
        prompts: list[Any],
        completions: list[Any],
        task_dir: list[str],
        task_id: list[str] | None = None,
        **_: Any,
    ) -> list[float]:
        rewards: list[float] = []
        rows: list[dict[str, Any]] = []
        ids = task_id or [""] * len(completions)
        for idx, completion in enumerate(completions):
            text = _completion_text(completion)
            task_path = Path(task_dir[idx]).resolve()
            candidate_scratch = scratch_root / f"reward_{time.time_ns()}_{idx}"
            candidate_scratch.mkdir(parents=True, exist_ok=True)
            result = score_completion(
                text,
                task_path,
                scratch_root=candidate_scratch,
                timeout_s=timeout_s,
            )
            reward = float(result.verified_score) * float(reward_scale)
            rewards.append(reward)
            rows.append({
                "ts": time.time(),
                "task_id": ids[idx],
                "task_dir": str(task_path),
                "reward": reward,
                **result.to_dict(),
            })
        with log_path.open("a") as f:
            for row in rows:
                f.write(json.dumps(row, sort_keys=True, default=str) + "\n")
        return rewards

    return reward_func


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tasks-root", default="tasks")
    parser.add_argument("--split-file", default=None)
    parser.add_argument("--families", default=None)
    parser.add_argument("--tiers", default=None)
    parser.add_argument("--limit-tasks", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=5.0e-6)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-prompt-length", type=int, default=4096)
    parser.add_argument("--max-completion-length", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--reward-scale", type=float, default=100.0)
    parser.add_argument("--reward-timeout-s", type=float, default=60.0)
    parser.add_argument("--save-steps", type=int, default=25)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="write dataset/config metadata without training")
    args = parser.parse_args()

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks_root = Path(args.tasks_root)
    if not tasks_root.is_absolute():
        tasks_root = REPO_ROOT / tasks_root
    split_file = Path(args.split_file).resolve() if args.split_file else None
    families = (
        {item.strip() for item in args.families.split(",") if item.strip()}
        if args.families else None
    )
    tiers = (
        {item.strip() for item in args.tiers.split(",") if item.strip()}
        if args.tiers else None
    )
    system_prompt = SYSTEM_PROMPT_PATH.read_text()
    rows = _build_rows(
        tasks_root=tasks_root,
        split_file=split_file,
        families=families,
        tiers=tiers,
        system_prompt=system_prompt,
        limit=max(0, int(args.limit_tasks)),
    )
    if not rows:
        print("error: no tasks matched", file=sys.stderr)
        return 2

    dataset_jsonl = out_dir / "train_prompts.jsonl"
    with dataset_jsonl.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    manifest = {
        "schema": SCHEMA,
        "argv": sys.argv,
        "model": args.model,
        "task_count": len(rows),
        "dataset_jsonl": str(dataset_jsonl),
        "split_file": str(split_file) if split_file else None,
        "families": sorted(families) if families else None,
        "tiers": sorted(tiers) if tiers else None,
        "algorithm": "trl.GRPOTrainer",
        "uses_policy_ratio_clipping": True,
        "uses_value_head": False,
        "reward": "mech_bench verified_score * reward_scale",
    }
    (out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    if args.dry_run:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0

    try:
        from datasets import Dataset  # type: ignore[import-not-found]
        from peft import LoraConfig  # type: ignore[import-not-found]
        from trl import GRPOConfig, GRPOTrainer  # type: ignore[import-not-found]
    except ImportError as exc:
        print(
            "error: exact GRPO requires the training-grpo extra. Run "
            "`uv sync --extra training-grpo` or invoke with "
            "`uv run --extra training-grpo ...`.\n"
            f"missing import: {exc}",
            file=sys.stderr,
        )
        return 2

    train_dataset = Dataset.from_list(rows)
    peft_config = LoraConfig(
        r=max(1, int(args.lora_rank)),
        lora_alpha=max(1, int(args.lora_alpha)),
        lora_dropout=max(0.0, float(args.lora_dropout)),
        bias="none",
        task_type="CAUSAL_LM",
    )
    config = _filtered_config(GRPOConfig, {
        "output_dir": str(out_dir),
        "learning_rate": float(args.learning_rate),
        "per_device_train_batch_size": int(args.per_device_train_batch_size),
        "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
        "num_generations": int(args.num_generations),
        "max_prompt_length": int(args.max_prompt_length),
        "max_completion_length": int(args.max_completion_length),
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "beta": float(args.beta),
        "epsilon": float(args.epsilon),
        "max_steps": int(args.max_steps),
        "save_steps": int(args.save_steps),
        "logging_steps": int(args.logging_steps),
        "seed": int(args.seed),
        "bf16": bool(args.bf16),
        "fp16": bool(args.fp16),
        "gradient_checkpointing": bool(args.gradient_checkpointing),
        "remove_unused_columns": False,
        "report_to": [],
    })
    scratch_root = Path(tempfile.mkdtemp(prefix="mech_true_grpo_"))
    reward_func = _make_reward_func(
        log_path=out_dir / "reward_log.jsonl",
        scratch_root=scratch_root,
        timeout_s=float(args.reward_timeout_s),
        reward_scale=float(args.reward_scale),
    )
    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=reward_func,
        args=config,
        train_dataset=train_dataset,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(out_dir / "final_adapter"))
    manifest["completed_ts"] = time.time()
    manifest["final_adapter"] = str(out_dir / "final_adapter")
    (out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
