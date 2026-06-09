#!/usr/bin/env python3
"""Run the MechanismRepair-TTRL online repair experiment.

The experiment tests a single causal hypothesis: starting from the same
seen-family policy, online verifier-derived GRPO updates during a held-out
repair episode improve verified repair success over the same verifier-feedback
loop with no weight updates, under the same actual verifier budget.

This script is intentionally cell-oriented. It writes one evidence row per
``split/task/seed/method`` cell so the paired analyzer can reject missing
coverage or unmatched budgets.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import shlex
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts.analyze_mechanism_repair_results import (
    build_expected_coverage,
)
from scripts.prepare_mechanism_repair_benchmark import (
    EVAL_SEEDS as LEGACY_EVAL_SEEDS,
    PRIMARY_BASELINE as LEGACY_PRIMARY_BASELINE,
    PRIMARY_BUDGET as LEGACY_PRIMARY_BUDGET,
    PRIMARY_METHOD as LEGACY_PRIMARY_METHOD,
    REQUIRED_METHODS as LEGACY_REQUIRED_METHODS,
)
from scripts.prepare_mechanism_repair_physics_benchmark import (
    EVAL_SEEDS as PHYSICS_EVAL_SEEDS,
    PRIMARY_BASELINE as PHYSICS_PRIMARY_BASELINE,
    PRIMARY_BUDGET as PHYSICS_PRIMARY_BUDGET,
    PRIMARY_METHOD as PHYSICS_PRIMARY_METHOD,
    REQUIRED_METHODS as PHYSICS_REQUIRED_METHODS,
)


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
REQUIRED_METHODS = tuple(
    dict.fromkeys([*LEGACY_REQUIRED_METHODS, *PHYSICS_REQUIRED_METHODS])
)
PRIMARY_METHOD = LEGACY_PRIMARY_METHOD
PRIMARY_BUDGET = LEGACY_PRIMARY_BUDGET
TTRL_METHODS = {
    LEGACY_PRIMARY_METHOD,
    "mechanical_evolve_ttrl",
    "mechanical_evolve_ttrl_tool_verified",
    "mechanical_evolve_ttrl_confidence",
}
SFT_METHODS = {"sft_model", "sft_seen_family"}
BASELINE_FEEDBACK_METHODS = {"llm_evolve_no_update"}


@dataclass(frozen=True)
class EvalMethod:
    name: str
    samples_per_task: int
    max_turns: int
    temperature: float
    top_p: float
    adapter_kind: str = "none"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark-dir",
        default="runs/mechanism_repair_ttrl_final",
        help="prepared benchmark directory from prepare_mechanism_repair_benchmark.py",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="experiment output directory; defaults to --benchmark-dir",
    )
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--runner-python", default=sys.executable)
    parser.add_argument("--sglang-base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--api-key", default="dummy")
    parser.add_argument(
        "--rollout-backend",
        default="sglang_chat",
        choices=["sglang_chat", "worldlines_sampling"],
    )
    parser.add_argument(
        "--methods",
        default=None,
        help="comma-separated methods; defaults to method_manifest.required_methods",
    )
    parser.add_argument(
        "--splits",
        default=None,
        help="comma-separated splits; defaults to A,B or shard contents",
    )
    parser.add_argument(
        "--eval-seeds",
        default=None,
        help="comma-separated evaluation seeds; defaults to method_manifest.eval_seeds",
    )
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument(
        "--cell-shard-file",
        default=None,
        help=(
            "optional experiment_shards/shard_XXXX.json from "
            "run_mechanism_repair_physics_experiment.py; restricts execution "
            "to the exact cells listed there"
        ),
    )
    parser.add_argument("--feedback-turns", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=1536)
    parser.add_argument("--max-context-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--eval-timeout-s", type=float, default=21600.0)
    parser.add_argument("--train-timeout-s", type=float, default=21600.0)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument(
        "--audit-retries",
        type=int,
        default=0,
        help="replacement audit retries for sample_and_score; final matched "
             "budget runs should keep this at 0",
    )
    parser.add_argument("--limit-tasks", type=int, default=0)
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-analysis", action="store_true")
    parser.add_argument(
        "--init-online-from-sft",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="initialize both llm_evolve_no_update and TTRL from the same "
             "seen-family SFT adapter",
    )
    parser.add_argument(
        "--sft-runner",
        default="uv run --extra training-grpo python",
    )
    parser.add_argument("--sft-max-steps", type=int, default=64)
    parser.add_argument("--sft-learning-rate", type=float, default=5.0e-6)
    parser.add_argument("--sft-max-grad-norm", type=float, default=0.0)
    parser.add_argument("--sft-max-seq-length", type=int, default=4096)
    parser.add_argument("--sft-lora-rank", type=int, default=16)
    parser.add_argument("--sft-load-in-4bit", action="store_true")
    parser.add_argument("--sft-load-in-8bit", action="store_true")
    parser.add_argument("--sft-prepare-kbit-training", action="store_true")
    parser.add_argument(
        "--sft-prepare-kbit-training-mode",
        default="peft",
        choices=("peft", "lightweight"),
    )
    parser.add_argument("--sft-torch-dtype", default=None)
    parser.add_argument("--sft-attn-implementation", default=None)
    parser.add_argument("--sft-device-map", default=None)
    parser.add_argument("--sft-gradient-checkpointing", action="store_true")
    parser.add_argument("--sft-trust-remote-code", action="store_true")
    parser.add_argument(
        "--ttrl-runner",
        default="uv run --extra training-grpo python",
    )
    parser.add_argument("--ttrl-model", default=None)
    parser.add_argument("--ttrl-max-steps", type=int, default=None)
    parser.add_argument("--ttrl-num-generations", type=int, default=4)
    parser.add_argument("--ttrl-learning-rate", type=float, default=5.0e-6)
    parser.add_argument("--ttrl-max-grad-norm", type=float, default=0.0)
    parser.add_argument("--ttrl-lora-rank", type=int, default=16)
    parser.add_argument(
        "--ttrl-reward-channel",
        default="artifact_progress",
        choices=("verified_score", "score", "artifact_progress"),
        help=(
            "reward channel used for online GRPO updates. Reported repair "
            "success remains strict verifier success regardless of this value"
        ),
    )
    parser.add_argument("--ttrl-load-in-4bit", action="store_true")
    parser.add_argument("--ttrl-load-in-8bit", action="store_true")
    parser.add_argument(
        "--ttrl-kbit-prepare-mode",
        default="peft",
        choices=("peft", "lightweight", "none"),
    )
    parser.add_argument("--ttrl-torch-dtype", default=None)
    parser.add_argument("--ttrl-attn-implementation", default=None)
    parser.add_argument("--ttrl-bf16", action="store_true")
    parser.add_argument("--ttrl-fp16", action="store_true")
    parser.add_argument("--ttrl-device-map", default=None)
    parser.add_argument("--ttrl-max-memory", default=None)
    parser.add_argument("--ttrl-gradient-checkpointing", action="store_true")
    parser.add_argument("--ttrl-trust-remote-code", action="store_true")
    parser.add_argument(
        "--ttrl-rollout-openai",
        action="store_true",
        help="debug only unless the serving stack refreshes the policy LoRA "
             "during GRPO; default local rollouts are the causal-clean path",
    )
    args = parser.parse_args()

    benchmark_dir = Path(args.benchmark_dir).expanduser().resolve()
    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else benchmark_dir
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    run_root = out_dir / "online_runs"
    run_root.mkdir(parents=True, exist_ok=True)
    ensure_required_artifact_dirs(out_dir)

    method_contract = load_method_contract(benchmark_dir)
    shard_cells = load_shard_cells(args.cell_shard_file)
    requested_methods = (
        sorted({str(cell["method"]) for cell in shard_cells})
        if shard_cells and args.methods is None
        else parse_csv(args.methods)
        if args.methods
        else list(method_contract["required_methods"])
    )
    allowed_methods = set(REQUIRED_METHODS) | set(method_contract["required_methods"])
    unknown = sorted(set(requested_methods) - allowed_methods)
    if unknown:
        raise SystemExit(f"unknown methods requested: {unknown}")
    splits = (
        sorted({str(cell["split"]) for cell in shard_cells})
        if shard_cells and args.splits is None
        else parse_csv(args.splits)
        if args.splits
        else ["A", "B"]
    )
    seeds = (
        sorted({int(cell["seed"]) for cell in shard_cells})
        if shard_cells and args.eval_seeds is None
        else [int(item) for item in parse_csv(args.eval_seeds)]
        if args.eval_seeds
        else list(method_contract["eval_seeds"])
    )
    validate_benchmark(benchmark_dir)
    family_by_task = canonical_family_by_task(benchmark_dir)
    budget = int(args.budget or method_contract["primary_budget"])
    args.budget = budget
    shard_filter = cell_filter_from_shard(shard_cells, budget=budget)
    feedback_turns = max(1, int(args.feedback_turns))
    if budget % feedback_turns:
        raise SystemExit(
            f"--budget={budget} must divide --feedback-turns={feedback_turns}"
        )
    ttrl_generations = max(1, int(args.ttrl_num_generations))
    ttrl_steps = (
        int(args.ttrl_max_steps)
        if args.ttrl_max_steps is not None
        else budget
    )
    if ttrl_steps % ttrl_generations:
        raise SystemExit(
            "TTRL budget mismatch: "
            f"max_steps={ttrl_steps} must be divisible by "
            f"num_generations={ttrl_generations}"
        )
    expected_ttrl_verifier_calls = (
        (ttrl_steps // ttrl_generations) * ttrl_generations
    )
    if expected_ttrl_verifier_calls != budget:
        raise SystemExit(
            "TTRL budget mismatch: "
            f"max_steps={ttrl_steps} with num_generations={ttrl_generations} "
            f"expects {expected_ttrl_verifier_calls} verifier calls, "
            f"not budget={budget}"
        )

    plan = build_plan(
        benchmark_dir=benchmark_dir,
        out_dir=out_dir,
        splits=splits,
        seeds=seeds,
        methods=requested_methods,
        budget=budget,
        feedback_turns=feedback_turns,
        audit_retries=int(args.audit_retries),
        limit_tasks=max(0, int(args.limit_tasks)),
        init_online_from_sft=bool(args.init_online_from_sft),
        ttrl_steps=ttrl_steps,
        ttrl_generations=ttrl_generations,
        ttrl_reward_channel=str(args.ttrl_reward_channel),
        shard_cells=shard_cells,
    )
    write_json(out_dir / "online_experiment_plan.json", plan)
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    rows_path = out_dir / "cell_results.jsonl"
    if args.resume_existing:
        rows = load_existing_rows(rows_path)
    else:
        reset_non_resume_outputs(out_dir=out_dir, run_root=run_root)
        rows = []
    seen_keys = {row_key(row) for row in rows}

    for split in splits:
        split_dir = benchmark_dir / f"splits_{split}"
        train_file = split_dir / "train.txt"
        test_file = make_eval_split_file(
            split_dir=split_dir,
            run_root=run_root,
            split=split,
            limit=max(0, int(args.limit_tasks)),
        )
        task_ids = read_ids(test_file)
        for seed in seeds:
            sft_adapter = None
            needs_sft = (
                bool(set(requested_methods) & SFT_METHODS)
                or (
                    bool(args.init_online_from_sft)
                    and bool(
                        set(requested_methods)
                        & (BASELINE_FEEDBACK_METHODS | TTRL_METHODS)
                    )
                )
            )
            if needs_sft:
                sft_dir = run_root / split / str(seed) / "sft_train"
                sft_adapter = run_or_load_sft(
                    args=args,
                    run_dir=sft_dir,
                    tasks_root=benchmark_dir / "tasks",
                    train_file=train_file,
                    seed=seed,
                    resume_existing=bool(args.resume_existing),
                )
            for method in requested_methods:
                if method in TTRL_METHODS:
                    for task_id in task_ids:
                        if not cell_wanted(
                            shard_filter,
                            split=split,
                            task_id=task_id,
                            seed=seed,
                            method=method,
                            budget=budget,
                        ):
                            continue
                        key = (split, task_id, seed, method)
                        if key in seen_keys:
                            continue
                        task_split = write_one_task_split(
                            run_root, split, seed, task_id
                        )
                        run_dir = (
                            run_root / split / str(seed) / method / task_id
                        )
                        row = run_or_load_ttrl_cell(
                            args=args,
                            run_dir=run_dir,
                            benchmark_dir=benchmark_dir,
                            split_file=task_split,
                            split=split,
                            task_id=task_id,
                            seed=seed,
                            budget=budget,
                            ttrl_steps=ttrl_steps,
                            ttrl_generations=ttrl_generations,
                            method=method,
                            reward_channel=reward_channel_for_method(
                                method,
                                default=str(args.ttrl_reward_channel),
                            ),
                            family_by_task=family_by_task,
                            evidence_root=out_dir,
                            init_adapter=sft_adapter
                            if bool(args.init_online_from_sft)
                            else None,
                            resume_existing=bool(args.resume_existing),
                        )
                        append_row(rows_path, row)
                        rows.append(row)
                        seen_keys.add(key)
                        write_results_bundle(out_dir, rows)
                    continue
                report_dir = run_root / split / str(seed) / f"eval_{method}"
                missing = [
                    task_id for task_id in task_ids
                    if (split, task_id, seed, method) not in seen_keys
                    and cell_wanted(
                        shard_filter,
                        split=split,
                        task_id=task_id,
                        seed=seed,
                        method=method,
                        budget=budget,
                    )
                ]
                if not missing:
                    continue
                spec = method_spec(
                    method,
                    budget=budget,
                    feedback_turns=feedback_turns,
                    sft_adapter=sft_adapter,
                    init_online_from_sft=bool(args.init_online_from_sft),
                )
                method_test_file = (
                    write_task_subset_split(
                        run_root=run_root,
                        split=split,
                        seed=seed,
                        method=method,
                        task_ids=missing,
                    )
                    if shard_filter is not None
                    else test_file
                )
                summary = run_or_load_eval_summary(
                    args=args,
                    method=spec,
                    report_dir=report_dir,
                    tasks_root=benchmark_dir / "tasks",
                    test_file=method_test_file,
                    seed=seed,
                    resume_existing=bool(args.resume_existing),
                )
                new_rows = rows_from_sample_summary(
                    summary=summary,
                    method=method,
                    split=split,
                    seed=seed,
                    budget=budget,
                    trace_root=report_dir,
                    family_by_task=family_by_task,
                    evidence_root=out_dir,
                )
                for row in new_rows:
                    key = row_key(row)
                    if key in seen_keys:
                        continue
                    append_row(rows_path, row)
                    rows.append(row)
                    seen_keys.add(key)
                write_results_bundle(out_dir, rows)
            write_results_bundle(out_dir, rows)

    write_results_bundle(out_dir, rows)
    if not args.skip_analysis:
        run_analysis(out_dir=out_dir, benchmark_dir=benchmark_dir)
    print(json.dumps({
        "rows": len(rows),
        "jsonl": str(rows_path),
        "json": str(out_dir / "cell_results.json"),
        "csv": str(out_dir / "cell_results.csv"),
        "claim_audit": str(out_dir / "claim_audit.json"),
    }, indent=2, sort_keys=True))
    return 0


def validate_benchmark(benchmark_dir: Path) -> None:
    required = [
        benchmark_dir / "benchmark_manifest.json",
        benchmark_dir / "method_manifest.json",
        benchmark_dir / "verifier_manifest.json",
        benchmark_dir / "split_manifest_A.json",
        benchmark_dir / "split_manifest_B.json",
        benchmark_dir / "tasks",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"prepared benchmark is incomplete: {missing}")
    manifest = json.loads((benchmark_dir / "benchmark_manifest.json").read_text())
    if manifest.get("experiment_ready") is not True:
        raise SystemExit("benchmark_manifest.json does not mark experiment_ready=true")


def load_method_contract(benchmark_dir: Path) -> dict[str, Any]:
    """Read method/seeds/budget from benchmark manifests.

    Older MechanismRepair-TTRL runs and the current MechanismRepair-Physics
    contract use different method names. The execution layer follows the
    benchmark's manifest instead of hard-coding one contract.
    """
    path = benchmark_dir / "method_manifest.json"
    if path.is_file():
        manifest = json.loads(path.read_text())
    else:
        manifest = {}
    required = [
        str(method)
        for method in manifest.get("required_methods", LEGACY_REQUIRED_METHODS)
    ]
    seeds = [int(seed) for seed in manifest.get("eval_seeds", LEGACY_EVAL_SEEDS)]
    return {
        "required_methods": required,
        "primary_method": str(
            manifest.get("primary_method")
            or (
                PHYSICS_PRIMARY_METHOD
                if required == list(PHYSICS_REQUIRED_METHODS)
                else LEGACY_PRIMARY_METHOD
            )
        ),
        "primary_baseline": str(
            manifest.get("primary_baseline")
            or (
                PHYSICS_PRIMARY_BASELINE
                if required == list(PHYSICS_REQUIRED_METHODS)
                else LEGACY_PRIMARY_BASELINE
            )
        ),
        "primary_budget": int(
            manifest.get("primary_budget_expensive_verifier_calls")
            or manifest.get("primary_budget_verifier_calls")
            or (
                PHYSICS_PRIMARY_BUDGET
                if required == list(PHYSICS_REQUIRED_METHODS)
                else LEGACY_PRIMARY_BUDGET
            )
        ),
        "eval_seeds": seeds,
    }


def load_shard_cells(path_value: str | None) -> list[dict[str, Any]]:
    if not path_value:
        return []
    path = Path(path_value).expanduser().resolve()
    payload = json.loads(path.read_text())
    cells = payload.get("cells") or []
    if not isinstance(cells, list):
        raise SystemExit(f"shard file has no cells list: {path}")
    out: list[dict[str, Any]] = []
    for cell in cells:
        item = dict(cell)
        item["_shard_file"] = str(path)
        out.append(item)
    return out


def cell_filter_from_shard(
    cells: list[dict[str, Any]],
    *,
    budget: int,
) -> set[tuple[str, str, int, str, int]] | None:
    if not cells:
        return None
    return {
        (
            str(cell["split"]),
            str(cell["task_id"]),
            int(cell["seed"]),
            str(cell["method"]),
            int(cell.get("budget", budget)),
        )
        for cell in cells
    }


def cell_wanted(
    shard_filter: set[tuple[str, str, int, str, int]] | None,
    *,
    split: str,
    task_id: str,
    seed: int,
    method: str,
    budget: int,
) -> bool:
    if shard_filter is None:
        return True
    return (split, task_id, int(seed), method, int(budget)) in shard_filter


def build_plan(
    *,
    benchmark_dir: Path,
    out_dir: Path,
    splits: list[str],
    seeds: list[int],
    methods: list[str],
    budget: int,
    feedback_turns: int,
    audit_retries: int,
    limit_tasks: int,
    init_online_from_sft: bool,
    ttrl_steps: int,
    ttrl_generations: int,
    ttrl_reward_channel: str,
    shard_cells: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    shard_cells = shard_cells or []
    shard_task_ids_by_split: dict[str, set[str]] = defaultdict(set)
    for cell in shard_cells:
        shard_task_ids_by_split[str(cell["split"])].add(str(cell["task_id"]))
    split_tasks = {}
    for split in splits:
        task_ids = read_ids(benchmark_dir / f"splits_{split}" / "test.txt")
        if shard_task_ids_by_split:
            wanted = shard_task_ids_by_split.get(split, set())
            task_ids = [task_id for task_id in task_ids if task_id in wanted]
        if limit_tasks:
            task_ids = task_ids[:limit_tasks]
        split_tasks[split] = task_ids
    total_cells = (
        len(shard_cells)
        if shard_cells
        else sum(len(task_ids) for task_ids in split_tasks.values())
        * len(seeds)
        * len(methods)
    )
    expected = build_expected_coverage(benchmark_dir)
    return {
        "schema": "mechanism_repair_ttrl.online_plan.v1",
        "hypothesis": (
            "online verifier-derived GRPO updates improve held-out verified "
            "mechanism repair over the same no-update verifier-feedback loop"
        ),
        "benchmark_dir": str(benchmark_dir),
        "out_dir": str(out_dir),
        "splits": splits,
        "split_tasks": split_tasks,
        "seeds": seeds,
        "methods": methods,
        "budget_verifier_calls_per_cell": budget,
        "audit_retries": audit_retries,
        "feedback_turns": feedback_turns,
        "llm_evolve_samples_per_task": budget,
        "llm_evolve_feedback_turns": feedback_turns,
        "llm_evolve_max_verifier_calls_per_task": budget,
        "ttrl_max_steps": ttrl_steps,
        "ttrl_num_generations": ttrl_generations,
        "ttrl_reward_channel": str(ttrl_reward_channel),
        "ttrl_rollout_evaluations_per_cell": ttrl_steps,
        "init_online_from_sft": init_online_from_sft,
        "planned_cells": total_cells,
        "cell_shard_file": (
            str(Path(shard_cells[0]["_shard_file"]).resolve())
            if shard_cells and shard_cells[0].get("_shard_file")
            else None
        ),
        "sharded_execution": bool(shard_cells),
        "full_expected_cells_from_manifest": len(expected["expected_cells"]),
        "limit_tasks": limit_tasks,
    }


def method_spec(
    method: str,
    *,
    budget: int,
    feedback_turns: int,
    sft_adapter: str | None,
    init_online_from_sft: bool,
) -> EvalMethod:
    if method == "frozen_model":
        return EvalMethod(method, budget, 1, 0.2, 0.95)
    if method in {"verifier_gated", "verifier_gated_search"}:
        return EvalMethod(method, budget, 1, 0.0, 1.0)
    if method == "no_update_search":
        return EvalMethod(method, budget, 1, 0.9, 0.95)
    if method == "adaptive_evolution":
        return EvalMethod(method, budget, feedback_turns, 0.9, 0.95)
    if method == "llm_evolve_no_update":
        return EvalMethod(
            method,
            budget,
            feedback_turns,
            0.7,
            0.95,
            adapter_kind="sft" if init_online_from_sft and sft_adapter else "none",
        )
    if method in SFT_METHODS:
        if not sft_adapter:
            raise SystemExit(f"{method} requested but SFT adapter is unavailable")
        return EvalMethod(method, budget, 1, 0.2, 0.95, adapter_kind="sft")
    raise SystemExit(f"sample evaluator cannot run method {method}")


def reward_channel_for_method(method: str, *, default: str) -> str:
    if method == "mechanical_evolve_ttrl_confidence":
        return "verified_score"
    if method == "mechanical_evolve_ttrl_tool_verified":
        return "artifact_progress"
    return default


def reset_non_resume_outputs(*, out_dir: Path, run_root: Path) -> None:
    for directory in (
        run_root,
        out_dir / "raw_completions",
        out_dir / "verifier_outputs",
        out_dir / "cad_artifacts",
        out_dir / "chrono_outputs",
        out_dir / "training_logs",
        out_dir / "adapter_checkpoints",
    ):
        if directory.exists():
            shutil.rmtree(directory)
    for file_name in (
        "cell_results.jsonl",
        "cell_results.json",
        "cell_results.csv",
        "results.json",
        "results.csv",
        "stats.json",
        "failure_analysis.json",
        "trace_pairs.json",
        "claim_audit.json",
    ):
        path = out_dir / file_name
        if path.exists():
            path.unlink()
    run_root.mkdir(parents=True, exist_ok=True)
    ensure_required_artifact_dirs(out_dir)


def run_or_load_sft(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    tasks_root: Path,
    train_file: Path,
    seed: int,
    resume_existing: bool,
) -> str:
    manifest = run_dir / "run_manifest.json"
    if resume_existing and manifest.is_file():
        payload = json.loads(manifest.read_text())
        require_learning_manifest(
            payload,
            manifest_path=manifest,
            label="SFT",
            expected_adapter_updates=int(args.sft_max_steps),
        )
        adapter = payload.get("final_adapter")
        if adapter and Path(adapter).exists():
            return str(adapter)
    if not resume_existing and run_dir.exists():
        shutil.rmtree(run_dir)
    cmd = [
        *shlex.split(args.sft_runner),
        str(REPO_ROOT / "rl" / "train_sft_peft.py"),
        "--model",
        str(args.base_model),
        "--output-dir",
        str(run_dir),
        "--tasks-root",
        str(tasks_root),
        "--split-file",
        str(train_file),
        "--max-steps",
        str(args.sft_max_steps),
        "--learning-rate",
        str(args.sft_learning_rate),
        "--max-grad-norm",
        str(args.sft_max_grad_norm),
        "--max-seq-length",
        str(args.sft_max_seq_length),
        "--lora-rank",
        str(args.sft_lora_rank),
        "--seed",
        str(seed),
    ]
    add_flag(cmd, args.sft_load_in_4bit, "--load-in-4bit")
    add_flag(cmd, args.sft_load_in_8bit, "--load-in-8bit")
    add_flag(
        cmd,
        args.sft_prepare_kbit_training,
        "--prepare-kbit-training",
    )
    add_option(
        cmd,
        "--prepare-kbit-training-mode",
        args.sft_prepare_kbit_training_mode,
    )
    add_flag(cmd, args.sft_gradient_checkpointing, "--gradient-checkpointing")
    add_flag(cmd, args.sft_trust_remote_code, "--trust-remote-code")
    add_option(cmd, "--torch-dtype", args.sft_torch_dtype)
    add_option(cmd, "--attn-implementation", args.sft_attn_implementation)
    add_option(cmd, "--device-map", args.sft_device_map)
    run(cmd, timeout=float(args.train_timeout_s))
    if not manifest.is_file():
        raise SystemExit(f"SFT did not write manifest: {manifest}")
    payload = json.loads(manifest.read_text())
    require_learning_manifest(
        payload,
        manifest_path=manifest,
        label="SFT",
        expected_adapter_updates=int(args.sft_max_steps),
    )
    adapter = payload.get("final_adapter")
    if not adapter or not Path(adapter).exists():
        raise SystemExit(f"SFT final_adapter missing in {manifest}: {adapter}")
    return str(adapter)


def run_or_load_eval_summary(
    *,
    args: argparse.Namespace,
    method: EvalMethod,
    report_dir: Path,
    tasks_root: Path,
    test_file: Path,
    seed: int,
    resume_existing: bool,
) -> dict[str, Any]:
    summary_path = report_dir / "smoke_summary.json"
    if resume_existing and summary_path.is_file():
        return json.loads(summary_path.read_text())
    if not resume_existing and report_dir.exists():
        shutil.rmtree(report_dir)
    max_verifier_calls = (
        method.samples_per_task
        if method.max_turns <= 1
        else getattr(args, "budget", method.samples_per_task)
    )
    if max_verifier_calls is None:
        max_verifier_calls = method.samples_per_task
    cmd = [
        str(args.runner_python),
        str(REPO_ROOT / "rl" / "sample_and_score.py"),
        "--base-url",
        str(args.sglang_base_url),
        "--api-key",
        str(args.api_key),
        "--base-model",
        str(args.base_model),
        "--rollout-backend",
        str(args.rollout_backend),
        "--tasks",
        str(tasks_root),
        "--report-dir",
        str(report_dir),
        "--samples-per-task",
        str(method.samples_per_task),
        "--max-turns",
        str(method.max_turns),
        "--max-tokens",
        str(args.max_tokens),
        "--temperature",
        str(method.temperature),
        "--top-p",
        str(method.top_p),
        "--seed",
        str(seed),
        "--timeout",
        str(args.timeout),
        "--concurrency",
        str(args.concurrency),
        "--audit-retries",
        str(args.audit_retries),
        "--max-verifier-calls-per-task",
        str(int(max_verifier_calls)),
        "--split-file",
        str(test_file),
    ]
    if method.adapter_kind == "sft":
        sft_manifest = report_dir.parent / "sft_train" / "run_manifest.json"
        payload = json.loads(sft_manifest.read_text())
        cmd.extend(["--sglang-lora-path", str(payload["final_adapter"])])
    run(cmd, timeout=float(args.eval_timeout_s))
    if not summary_path.is_file():
        raise SystemExit(f"sample_and_score did not write {summary_path}")
    return json.loads(summary_path.read_text())


def run_or_load_ttrl_cell(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    benchmark_dir: Path,
    split_file: Path,
    split: str,
    task_id: str,
    seed: int,
    budget: int,
    ttrl_steps: int,
    ttrl_generations: int,
    family_by_task: dict[str, str],
    evidence_root: Path,
    init_adapter: str | None,
    resume_existing: bool,
    method: str = PRIMARY_METHOD,
    reward_channel: str | None = None,
) -> dict[str, Any]:
    reward_log = run_dir / "reward_log.jsonl"
    if not resume_existing and run_dir.exists():
        shutil.rmtree(run_dir)
        reward_log = run_dir / "reward_log.jsonl"
    for attempt in range(2):
        if not (resume_existing and reward_log.is_file()):
            cmd = [
                *shlex.split(args.ttrl_runner),
                str(REPO_ROOT / "rl" / "train_true_grpo_trl.py"),
                "--model",
                str(args.ttrl_model or args.base_model),
                "--output-dir",
                str(run_dir),
                "--tasks-root",
                str(benchmark_dir / "tasks"),
                "--split-file",
                str(split_file),
                "--max-steps",
                str(ttrl_steps),
                "--learning-rate",
                str(args.ttrl_learning_rate),
                "--max-grad-norm",
                str(args.ttrl_max_grad_norm),
                "--per-device-train-batch-size",
                "1",
                "--gradient-accumulation-steps",
                "1",
                "--num-generations",
                str(ttrl_generations),
                "--max-prompt-length",
                str(args.max_context_tokens),
                "--max-completion-length",
                str(args.max_tokens),
                "--temperature",
                "0.7",
                "--top-p",
                "0.95",
                "--reward-timeout-s",
                str(args.timeout),
                "--reward-channel",
                str(
                    reward_channel
                    or getattr(args, "ttrl_reward_channel", "artifact_progress")
                ),
                "--lora-rank",
                str(args.ttrl_lora_rank),
                "--seed",
                str(seed),
            ]
            add_option(cmd, "--init-adapter", init_adapter)
            add_flag(cmd, args.ttrl_load_in_4bit, "--load-in-4bit")
            add_flag(cmd, args.ttrl_load_in_8bit, "--load-in-8bit")
            add_option(cmd, "--kbit-prepare-mode", args.ttrl_kbit_prepare_mode)
            add_flag(
                cmd,
                args.ttrl_gradient_checkpointing,
                "--gradient-checkpointing",
            )
            add_flag(cmd, args.ttrl_trust_remote_code, "--trust-remote-code")
            add_flag(cmd, getattr(args, "ttrl_bf16", False), "--bf16")
            add_flag(cmd, getattr(args, "ttrl_fp16", False), "--fp16")
            add_option(cmd, "--torch-dtype", args.ttrl_torch_dtype)
            add_option(cmd, "--attn-implementation", args.ttrl_attn_implementation)
            add_option(cmd, "--device-map", args.ttrl_device_map)
            add_option(cmd, "--max-memory", args.ttrl_max_memory)
            if args.ttrl_rollout_openai:
                cmd.extend([
                    "--rollout-openai-base-url",
                    str(args.sglang_base_url),
                    "--rollout-openai-model",
                    str(args.base_model),
                    "--rollout-openai-api-key",
                    str(args.api_key),
                ])
                add_option(cmd, "--rollout-openai-lora-path", init_adapter)
            run(cmd, timeout=float(args.train_timeout_s))
        manifest = run_dir / "run_manifest.json"
        if not manifest.is_file():
            if resume_existing and attempt == 0:
                print(
                    f"[resume] discarding incomplete TTRL run without manifest "
                    f"for {task_id}: {run_dir}",
                    file=sys.stderr,
                )
                shutil.rmtree(run_dir, ignore_errors=True)
                reward_log = run_dir / "reward_log.jsonl"
                continue
            raise SystemExit(f"TTRL did not write manifest: {manifest}")
        try:
            require_learning_manifest(
                json.loads(manifest.read_text()),
                manifest_path=manifest,
                label=f"TTRL {task_id}",
                expected_adapter_updates=int(ttrl_steps),
                min_rl_datums=int(budget),
            )
        except SystemExit as exc:
            if resume_existing and attempt == 0:
                print(
                    f"[resume] discarding incomplete TTRL run for {task_id}: "
                    f"{exc}",
                    file=sys.stderr,
                )
                shutil.rmtree(run_dir, ignore_errors=True)
                reward_log = run_dir / "reward_log.jsonl"
                continue
            raise
        break
    return row_from_ttrl_reward_log(
        reward_log=reward_log,
        split=split,
        task_id=task_id,
        seed=seed,
        budget=budget,
        method=method,
        run_dir=run_dir,
        family_by_task=family_by_task,
        evidence_root=evidence_root,
    )


def require_learning_manifest(
    payload: dict[str, Any],
    *,
    manifest_path: Path,
    label: str,
    expected_adapter_updates: int,
    min_rl_datums: int = 0,
) -> None:
    updates = int(payload.get("adapter_updates", 0) or 0)
    expected = int(expected_adapter_updates)
    if updates != expected:
        raise SystemExit(
            f"{label} wrote unusable adapter_updates={updates}; "
            f"expected {expected}: {manifest_path}"
        )
    guard = dict(payload.get("optimizer_guard") or {})
    bad_guard = {
        key: int(guard.get(key, 0) or 0)
        for key in [
            "skipped_nonfinite_gradient_steps",
            "skipped_all_nonfinite_gradient_steps",
            "sanitized_nonfinite_gradient_steps",
            "rolled_back_nonfinite_update_steps",
            "nonfinite_gradient_values",
            "nonfinite_parameter_values_after_update",
        ]
    }
    bad_guard = {key: value for key, value in bad_guard.items() if value}
    if bad_guard:
        raise SystemExit(
            f"{label} optimizer guard recorded nonfinite training events "
            f"{bad_guard}: {manifest_path}"
        )
    if min_rl_datums:
        n_rl_datums = int(payload.get("n_rl_datums", 0) or 0)
        if n_rl_datums < int(min_rl_datums):
            raise SystemExit(
                f"{label} wrote n_rl_datums={n_rl_datums}; "
                f"expected at least {int(min_rl_datums)}: {manifest_path}"
            )
        trained_tokens = int(payload.get("trained_tokens", 0) or 0)
        rl_trained_tokens = int(payload.get("rl_trained_tokens", 0) or 0)
        if trained_tokens <= 0:
            raise SystemExit(
                f"{label} wrote trained_tokens={trained_tokens}; "
                f"expected a positive token count: {manifest_path}"
            )
        if rl_trained_tokens <= 0:
            raise SystemExit(
                f"{label} wrote rl_trained_tokens={rl_trained_tokens}; "
                f"expected a positive RL token count: {manifest_path}"
            )


def rows_from_sample_summary(
    *,
    summary: dict[str, Any],
    method: str,
    split: str,
    seed: int,
    budget: int,
    trace_root: Path,
    family_by_task: dict[str, str],
    evidence_root: Path | None = None,
) -> list[dict[str, Any]]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in summary.get("all_samples", []) or []:
        by_task[str(row.get("task_id") or "")].append(dict(row))
    out: list[dict[str, Any]] = []
    for task_id, samples in sorted(by_task.items()):
        if not task_id:
            continue
        best = max(samples, key=sample_rank)
        verifier_calls = sum(int(item.get("verifier_calls", 0) or 0) for item in samples)
        cad_audits = sum(int(item.get("cad_audits", 0) or 0) for item in samples)
        chrono_audits = sum(int(item.get("chrono_audits", 0) or 0) for item in samples)
        sample_idx = int(best.get("sample_idx", 0) or 0)
        raw_paths, verifier_paths = materialize_sample_evidence(
            evidence_root=evidence_root,
            trace_root=trace_root,
            split=split,
            seed=seed,
            method=method,
            task_id=task_id,
            samples=samples,
        )
        cad_paths, chrono_paths = materialize_audit_evidence(
            evidence_root=evidence_root,
            split=split,
            seed=seed,
            method=method,
            task_id=task_id,
            rows=samples,
        )
        first_valid = first_valid_call(samples)
        best_metrics = dict(best.get("physical_metrics") or {})
        training = training_metadata_for_method(method, trace_root)
        out.append({
            "method": method,
            "method_variant": method,
            "split": split,
            "task_id": task_id,
            "family": family_by_task.get(task_id, str(best.get("family") or "")),
            "seed": int(seed),
            "verified_repair_success_at_32": bool(
                best.get("verifier_valid_passed")
                or (
                    best.get("evaluation_valid")
                    and best.get("hard_gate_passed")
                    and not (best.get("failure_codes") or [])
                )
            ),
            "best_verified_reward_at_32": float(
                best.get("verified_score", 0.0) or 0.0
            ),
            "budget": int(budget),
            "verifier_calls": verifier_calls,
            "cad_audits": cad_audits,
            "chrono_audits": chrono_audits,
            "actual_verifier_calls": verifier_calls,
            "actual_cad_calls": cad_audits,
            "actual_chrono_calls": chrono_audits,
            "failure_codes": best.get("failure_codes", []),
            "trace_path": str(trace_root / f"sample_{sample_idx}" / task_id),
            "summary_path": str(trace_root / "smoke_summary.json"),
            "raw_completion_paths": raw_paths,
            "verifier_output_paths": verifier_paths,
            "cad_artifact_paths": cad_paths,
            "chrono_output_paths": chrono_paths,
            "first_valid_verifier_call": first_valid,
            "strict_score_pass_rate": strict_pass_rate(samples),
            "wrong_mobility_rate": failure_rate(samples, "wrong_mobility"),
            "missing_port_rate": failure_rate(samples, "missing_port"),
            "ungrounded_port_rate": failure_rate(samples, "ungrounded_port"),
            "invalid_topology_rate": failure_rate(samples, "invalid_topology"),
            "invalid_artifact_rate": invalid_artifact_rate(samples),
            "cad_pass_rate": audit_pass_rate(samples, "cad_audits"),
            "chrono_real_geometry_rate": audit_pass_rate(samples, "chrono_audits"),
            "no_procedural_fallback_rate": no_procedural_fallback_rate(samples),
            "lockup_rate": failure_rate(samples, "lockup"),
            "contact_lockup_rate": failure_rate(samples, "contact_lockup"),
            "best_ratio_error_pct": best_metrics.get("ratio_error_pct"),
            "best_path_trace_error": best_metrics.get("path_trace_error"),
            "best_max_penetration_mm": best_metrics.get("max_penetration_mm"),
            "best_contact_force_rms_N": best_metrics.get("contact_force_rms_N"),
            "sampler_error_count": failure_count(samples, "sampler_error"),
            "invalid_artifact_count": invalid_artifact_count(samples),
            "timeout_count": failure_count(samples, "timeout"),
            "audit_retry_count": sum(
                int(item.get("audit_retry_count", 0) or 0) for item in samples
            ),
            "planned_max_verifier_calls": int(budget),
            "actual_budget_match_group": (
                f"{split}:{task_id}:{seed}:{int(budget)}"
            ),
            "adapter_updates": training["adapter_updates"],
            "trained_tokens": training["trained_tokens"],
            "rl_trained_tokens": training["rl_trained_tokens"],
            "n_rl_datums": training["n_rl_datums"],
            "adapter_path": training["adapter_path"],
            "actual_budget_matches_primary": verifier_calls == int(budget),
            "n_candidates": len(samples),
            "candidate_count": len(samples),
            "max_turns": int(summary.get("max_turns", 1) or 1),
        })
    return out


def row_from_ttrl_reward_log(
    *,
    reward_log: Path,
    split: str,
    task_id: str,
    seed: int,
    budget: int,
    run_dir: Path,
    family_by_task: dict[str, str],
    evidence_root: Path | None = None,
    method: str = PRIMARY_METHOD,
) -> dict[str, Any]:
    if not reward_log.is_file():
        raise SystemExit(f"TTRL reward log missing: {reward_log}")
    rows = [
        json.loads(line)
        for line in reward_log.read_text().splitlines()
        if line.strip()
    ]
    if not rows:
        raise SystemExit(f"TTRL reward log is empty: {reward_log}")
    matching = [row for row in rows if str(row.get("task_id")) == task_id]
    if not matching:
        matching = rows
    best = max(matching, key=sample_rank)
    verifier_calls = len(matching)
    cad_audits = sum(int(item.get("cad_audits", 0) or 0) for item in matching)
    chrono_audits = sum(int(item.get("chrono_audits", 0) or 0) for item in matching)
    family = family_by_task.get(task_id) or family_from_task_dir(best.get("task_dir"))
    raw_paths, verifier_paths = materialize_ttrl_evidence(
        evidence_root=evidence_root,
        split=split,
        seed=seed,
        method=method,
        task_id=task_id,
        rows=matching,
    )
    cad_paths, chrono_paths = materialize_audit_evidence(
        evidence_root=evidence_root,
        split=split,
        seed=seed,
        method=method,
        task_id=task_id,
        rows=matching,
    )
    best_metrics = dict(best.get("physical_metrics") or {})
    training = training_metadata_from_manifest(run_dir / "run_manifest.json")
    return {
        "method": method,
        "method_variant": method,
        "split": split,
        "task_id": task_id,
        "family": family,
        "seed": int(seed),
        "verified_repair_success_at_32": bool(
            best.get("evaluation_valid")
            and best.get("hard_gate_passed")
            and not (best.get("failure_codes") or [])
        ),
        "best_verified_reward_at_32": float(
            best.get("verified_score", 0.0) or 0.0
        ),
        "budget": int(budget),
        "verifier_calls": verifier_calls,
        "cad_audits": cad_audits,
        "chrono_audits": chrono_audits,
        "actual_verifier_calls": verifier_calls,
        "actual_cad_calls": cad_audits,
        "actual_chrono_calls": chrono_audits,
        "failure_codes": best.get("failure_codes", []),
        "trace_path": str(reward_log),
        "raw_completion_paths": raw_paths,
        "verifier_output_paths": verifier_paths,
        "cad_artifact_paths": cad_paths,
        "chrono_output_paths": chrono_paths,
        "run_dir": str(run_dir),
        "first_valid_verifier_call": first_valid_call(matching),
        "strict_score_pass_rate": strict_pass_rate(matching),
        "wrong_mobility_rate": failure_rate(matching, "wrong_mobility"),
        "missing_port_rate": failure_rate(matching, "missing_port"),
        "ungrounded_port_rate": failure_rate(matching, "ungrounded_port"),
        "invalid_topology_rate": failure_rate(matching, "invalid_topology"),
        "invalid_artifact_rate": invalid_artifact_rate(matching),
        "cad_pass_rate": audit_pass_rate(matching, "cad_audits"),
        "chrono_real_geometry_rate": audit_pass_rate(matching, "chrono_audits"),
        "no_procedural_fallback_rate": no_procedural_fallback_rate(matching),
        "lockup_rate": failure_rate(matching, "lockup"),
        "contact_lockup_rate": failure_rate(matching, "contact_lockup"),
        "best_ratio_error_pct": best_metrics.get("ratio_error_pct"),
        "best_path_trace_error": best_metrics.get("path_trace_error"),
        "best_max_penetration_mm": best_metrics.get("max_penetration_mm"),
        "best_contact_force_rms_N": best_metrics.get("contact_force_rms_N"),
        "sampler_error_count": failure_count(matching, "sampler_error"),
        "invalid_artifact_count": invalid_artifact_count(matching),
        "timeout_count": failure_count(matching, "timeout"),
        "audit_retry_count": sum(
            int(item.get("audit_retry_count", 0) or 0) for item in matching
        ),
        "planned_max_verifier_calls": int(budget),
        "actual_budget_match_group": f"{split}:{task_id}:{seed}:{int(budget)}",
        "adapter_updates": training["adapter_updates"],
        "trained_tokens": training["trained_tokens"],
        "rl_trained_tokens": training["rl_trained_tokens"],
        "n_rl_datums": training["n_rl_datums"],
        "actual_budget_matches_primary": verifier_calls == int(budget),
        "n_candidates": len(matching),
        "candidate_count": len(matching),
        "adapter_path": training["adapter_path"] or str(run_dir / "final_adapter"),
    }


def sample_rank(row: dict[str, Any]) -> tuple[bool, float, bool]:
    return (
        bool(
            row.get("evaluation_valid")
            and row.get("hard_gate_passed")
            and not (row.get("failure_codes") or [])
        ),
        float(row.get("verified_score", 0.0) or 0.0),
        bool(row.get("design_py_extracted", False)),
    )


def materialize_sample_evidence(
    *,
    evidence_root: Path | None,
    trace_root: Path,
    split: str,
    seed: int,
    method: str,
    task_id: str,
    samples: list[dict[str, Any]],
) -> tuple[list[str], list[str]]:
    raw_paths: list[str] = []
    verifier_paths: list[str] = []
    if evidence_root is None:
        return raw_paths, verifier_paths
    raw_dir = evidence_root / "raw_completions" / split / str(seed) / method / task_id
    verifier_dir = (
        evidence_root / "verifier_outputs" / split / str(seed) / method / task_id
    )
    raw_dir.mkdir(parents=True, exist_ok=True)
    verifier_dir.mkdir(parents=True, exist_ok=True)
    for item in sorted(samples, key=lambda row: int(row.get("sample_idx", 0) or 0)):
        idx = int(item.get("sample_idx", 0) or 0)
        turn_traces = item.get("turn_traces") or []
        if turn_traces:
            written_for_sample = 0
            for turn in turn_traces:
                turn_idx = int(turn.get("turn_idx", 0) or 0)
                call_idx = int(
                    turn.get("verifier_call_idx_within_sample", written_for_sample)
                    or 0
                )
                audit_attempt = int(turn.get("audit_attempt", 0) or 0)
                sampler_attempt = int(turn.get("sampler_attempt", 0) or 0)
                name_stem = (
                    f"sample_{idx:04d}_call_{call_idx:03d}_"
                    f"audit_{audit_attempt:02d}_attempt_{sampler_attempt:02d}_"
                    f"turn_{turn_idx:02d}"
                )
                raw_dest = raw_dir / f"{name_stem}.txt"
                raw_dest.write_text(str(turn.get("assistant_text") or ""))
                raw_paths.append(str(raw_dest))
                verifier_dest = verifier_dir / f"{name_stem}.json"
                verifier_row = {
                    "task_id": item.get("task_id"),
                    "family": item.get("family"),
                    "sample_idx": idx,
                    "turn_idx": turn_idx,
                    "audit_attempt": audit_attempt,
                    "sampler_attempt": sampler_attempt,
                    "verifier_call_idx_within_sample": call_idx,
                    "retry_trace_idx": turn.get("retry_trace_idx"),
                    "trace_kind": turn.get("trace_kind"),
                    "score": turn.get("dense_pct"),
                    "verified_score": turn.get("score"),
                    "hard_gate_passed": turn.get("passed"),
                    "evaluation_valid": turn.get("evaluation_valid"),
                    "design_py_extracted": turn.get("parsed_ok"),
                    "failure_codes": turn.get("failure_codes", []),
                    "feedback": turn.get("feedback", []),
                    "cad_audits": turn.get("cad_audits", 0),
                    "chrono_audits": turn.get("chrono_audits", 0),
                    "physical_metrics": turn.get("physical_metrics", {}),
                    "no_procedural_fallback": turn.get(
                        "no_procedural_fallback"
                    ),
                    "completion_tokens": turn.get("completion_tokens", 0),
                    "stop_reason": turn.get("stop_reason", ""),
                }
                verifier_dest.write_text(
                    json.dumps(
                        verifier_row,
                        indent=2,
                        sort_keys=True,
                        default=str,
                    ) + "\n"
                )
                verifier_paths.append(str(verifier_dest))
                written_for_sample += 1
            expected_calls = int(item.get("verifier_calls", 0) or 0)
            source = trace_root / f"sample_{idx}" / task_id / "completion.txt"
            terminal_text = (
                source.read_text()
                if source.is_file()
                else str(item.get("error") or "")
            )
            while written_for_sample < expected_calls:
                call_idx = written_for_sample
                turn_idx = written_for_sample
                name_stem = (
                    f"sample_{idx:04d}_call_{call_idx:03d}_"
                    "audit_00_attempt_00_"
                    f"turn_{turn_idx:02d}"
                )
                raw_dest = raw_dir / f"{name_stem}.txt"
                raw_dest.write_text(terminal_text)
                raw_paths.append(str(raw_dest))
                verifier_dest = verifier_dir / f"{name_stem}.json"
                verifier_row = {
                    "task_id": item.get("task_id"),
                    "family": item.get("family"),
                    "sample_idx": idx,
                    "turn_idx": turn_idx,
                    "audit_attempt": 0,
                    "sampler_attempt": 0,
                    "verifier_call_idx_within_sample": call_idx,
                    "retry_trace_idx": None,
                    "trace_kind": "terminal_sample_evidence",
                    "score": item.get("score"),
                    "verified_score": item.get("verified_score"),
                    "hard_gate_passed": item.get("hard_gate_passed"),
                    "evaluation_valid": item.get("evaluation_valid"),
                    "design_py_extracted": item.get("design_py_extracted"),
                    "failure_codes": item.get("failure_codes", []),
                    "feedback": item.get("feedback", []),
                    "cad_audits": item.get("cad_audits", 0),
                    "chrono_audits": item.get("chrono_audits", 0),
                    "physical_metrics": item.get("physical_metrics", {}),
                    "no_procedural_fallback": item.get(
                        "no_procedural_fallback"
                    ),
                    "completion_tokens": item.get("sample_tokens_out", 0),
                    "stop_reason": "terminal_sample_evidence",
                }
                verifier_dest.write_text(
                    json.dumps(
                        verifier_row,
                        indent=2,
                        sort_keys=True,
                        default=str,
                    ) + "\n"
                )
                verifier_paths.append(str(verifier_dest))
                written_for_sample += 1
            continue
        source = trace_root / f"sample_{idx}" / task_id / "completion.txt"
        raw_dest = raw_dir / f"sample_{idx:04d}.txt"
        if source.is_file():
            shutil.copyfile(source, raw_dest)
            raw_paths.append(str(raw_dest))
        verifier_dest = verifier_dir / f"sample_{idx:04d}.json"
        verifier_dest.write_text(
            json.dumps(item, indent=2, sort_keys=True, default=str) + "\n"
        )
        verifier_paths.append(str(verifier_dest))
    return raw_paths, verifier_paths


def materialize_ttrl_evidence(
    *,
    evidence_root: Path | None,
    split: str,
    seed: int,
    method: str,
    task_id: str,
    rows: list[dict[str, Any]],
) -> tuple[list[str], list[str]]:
    raw_paths: list[str] = []
    verifier_paths: list[str] = []
    if evidence_root is None:
        return raw_paths, verifier_paths
    raw_dir = evidence_root / "raw_completions" / split / str(seed) / method / task_id
    verifier_dir = (
        evidence_root / "verifier_outputs" / split / str(seed) / method / task_id
    )
    raw_dir.mkdir(parents=True, exist_ok=True)
    verifier_dir.mkdir(parents=True, exist_ok=True)
    for idx, row in enumerate(rows):
        text = str(row.get("completion_text") or row.get("completion_preview") or "")
        raw_dest = raw_dir / f"candidate_{idx:04d}.txt"
        raw_dest.write_text(text)
        raw_paths.append(str(raw_dest))
        verifier_dest = verifier_dir / f"candidate_{idx:04d}.json"
        verifier_row = dict(row)
        verifier_row.pop("completion_text", None)
        verifier_dest.write_text(
            json.dumps(verifier_row, indent=2, sort_keys=True, default=str) + "\n"
        )
        verifier_paths.append(str(verifier_dest))
    return raw_paths, verifier_paths


def materialize_audit_evidence(
    *,
    evidence_root: Path | None,
    split: str,
    seed: int,
    method: str,
    task_id: str,
    rows: list[dict[str, Any]],
) -> tuple[list[str], list[str]]:
    cad_paths: list[str] = []
    chrono_paths: list[str] = []
    if evidence_root is None:
        return cad_paths, chrono_paths
    cad_dir = evidence_root / "cad_artifacts" / split / str(seed) / method / task_id
    chrono_dir = (
        evidence_root / "chrono_outputs" / split / str(seed) / method / task_id
    )
    cad_dir.mkdir(parents=True, exist_ok=True)
    chrono_dir.mkdir(parents=True, exist_ok=True)
    for item_idx, item in enumerate(rows):
        traces = item.get("turn_traces") or []
        if traces:
            for trace_idx, trace in enumerate(traces):
                stem = evidence_stem(
                    item=item,
                    fallback_idx=item_idx,
                    suffix=f"turn_{trace_idx:04d}",
                )
                cad_paths.extend(
                    write_audit_json_if_present(
                        out_dir=cad_dir,
                        stem=stem,
                        kind="cad",
                        row=trace,
                    )
                )
                chrono_paths.extend(
                    write_audit_json_if_present(
                        out_dir=chrono_dir,
                        stem=stem,
                        kind="chrono",
                        row=trace,
                    )
                )
            continue
        stem = evidence_stem(item=item, fallback_idx=item_idx)
        cad_paths.extend(
            write_audit_json_if_present(
                out_dir=cad_dir,
                stem=stem,
                kind="cad",
                row=item,
            )
        )
        chrono_paths.extend(
            write_audit_json_if_present(
                out_dir=chrono_dir,
                stem=stem,
                kind="chrono",
                row=item,
            )
        )
    return cad_paths, chrono_paths


def evidence_stem(
    *,
    item: dict[str, Any],
    fallback_idx: int,
    suffix: str = "",
) -> str:
    sample_idx = int(item.get("sample_idx", fallback_idx) or 0)
    audit_attempt = int(item.get("audit_attempt", 0) or 0)
    sampler_attempt = int(item.get("sampler_attempt", 0) or 0)
    bits = [
        f"sample_{sample_idx:04d}",
        f"audit_{audit_attempt:02d}",
        f"attempt_{sampler_attempt:02d}",
    ]
    if suffix:
        bits.append(suffix)
    return "_".join(bits)


def write_audit_json_if_present(
    *,
    out_dir: Path,
    stem: str,
    kind: str,
    row: dict[str, Any],
) -> list[str]:
    field = "cad_audits" if kind == "cad" else "chrono_audits"
    if int(row.get(field, 0) or 0) <= 0:
        return []
    payload = {
        "schema": f"mechanism_repair_ttrl.{kind}_artifact_evidence.v1",
        "kind": kind,
        "audit_count": int(row.get(field, 0) or 0),
        "task_id": row.get("task_id"),
        "family": row.get("family"),
        "sample_idx": row.get("sample_idx"),
        "turn_idx": row.get("turn_idx"),
        "hard_gate_passed": row.get("hard_gate_passed", row.get("passed")),
        "evaluation_valid": row.get("evaluation_valid"),
        "failure_codes": row.get("failure_codes", []),
        "physical_metrics": row.get("physical_metrics", {}),
        "feedback": row.get("feedback", []),
    }
    dest = out_dir / f"{stem}.json"
    dest.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    return [str(dest)]


def first_valid_call(rows: list[dict[str, Any]]) -> int | None:
    for idx, row in enumerate(rows, start=1):
        if row.get("evaluation_valid") and row.get("design_py_extracted", True):
            return idx
    return None


def strict_pass_rate(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    passed = sum(
        1 for row in rows
        if row.get("evaluation_valid")
        and row.get("hard_gate_passed")
        and not (row.get("failure_codes") or [])
    )
    return passed / len(rows)


def failure_codes(row: dict[str, Any]) -> set[str]:
    raw = row.get("failure_codes") or []
    if isinstance(raw, str):
        raw = [raw]
    return {str(code).lower() for code in raw if str(code)}


def failure_count(rows: list[dict[str, Any]], code: str) -> int:
    wanted = code.lower()
    return sum(1 for row in rows if wanted in failure_codes(row))


def failure_rate(rows: list[dict[str, Any]], code: str) -> float:
    if not rows:
        return 0.0
    return failure_count(rows, code) / len(rows)


def invalid_artifact_count(rows: list[dict[str, Any]]) -> int:
    invalid_codes = {
        "invalid_artifact",
        "invalid_design",
        "runner_json_error",
        "design_build_error",
        "missing_cad",
        "untrusted_asset",
    }
    return sum(1 for row in rows if failure_codes(row) & invalid_codes)


def invalid_artifact_rate(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    return invalid_artifact_count(rows) / len(rows)


def audit_pass_rate(rows: list[dict[str, Any]], field: str) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if int(row.get(field, 0) or 0) > 0) / len(rows)


def no_procedural_fallback_rate(rows: list[dict[str, Any]]) -> float:
    observed = [
        bool(row.get("no_procedural_fallback"))
        for row in rows
        if row.get("no_procedural_fallback") is not None
    ]
    if not observed:
        return 0.0
    return sum(1 for item in observed if item) / len(observed)


def training_metadata_for_method(method: str, trace_root: Path) -> dict[str, Any]:
    if method in SFT_METHODS:
        return training_metadata_from_manifest(
            trace_root.parent / "sft_train" / "run_manifest.json"
        )
    return empty_training_metadata()


def training_metadata_from_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return empty_training_metadata()
    payload = json.loads(path.read_text())
    return {
        "adapter_updates": int(payload.get("adapter_updates", 0) or 0),
        "trained_tokens": int(payload.get("trained_tokens", 0) or 0),
        "rl_trained_tokens": int(payload.get("rl_trained_tokens", 0) or 0),
        "n_rl_datums": int(payload.get("n_rl_datums", 0) or 0),
        "adapter_path": str(payload.get("final_adapter") or ""),
    }


def empty_training_metadata() -> dict[str, Any]:
    return {
        "adapter_updates": 0,
        "trained_tokens": 0,
        "rl_trained_tokens": 0,
        "n_rl_datums": 0,
        "adapter_path": "",
    }


def family_from_task_dir(value: Any) -> str:
    if not value:
        return ""
    path = Path(str(value))
    meta_path = path / "metadata.json"
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text())
            return str(meta.get("family") or "")
        except json.JSONDecodeError:
            return ""
    return ""


def canonical_family_by_task(benchmark_dir: Path) -> dict[str, str]:
    manifest = json.loads((benchmark_dir / "benchmark_manifest.json").read_text())
    out: dict[str, str] = {}
    for row in manifest.get("tasks", []) or []:
        task_id = str(row.get("task_id") or "")
        family = str(
            row.get("canonical_family")
            or row.get("raw_family")
            or row.get("family")
            or ""
        )
        if task_id and family:
            out[task_id] = family
    return out


def run_analysis(*, out_dir: Path, benchmark_dir: Path) -> None:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "analyze_mechanism_repair_results.py"),
        "--results",
        str(out_dir / "cell_results.jsonl"),
        "--out-dir",
        str(out_dir),
        "--benchmark-dir",
        str(benchmark_dir),
    ]
    proc = run(cmd, timeout=3600.0, check=False)
    required_outputs = [
        out_dir / "stats.json",
        out_dir / "failure_analysis.json",
        out_dir / "trace_pairs.json",
        out_dir / "repair_taxonomy.json",
        out_dir / "claim_audit.json",
    ]
    missing = [str(path) for path in required_outputs if not path.is_file()]
    if proc.returncode not in {0, 2} or missing:
        raise SystemExit(
            "mechanism repair analysis failed to produce required artifacts: "
            f"rc={proc.returncode}, missing={missing}"
        )


def make_eval_split_file(
    *,
    split_dir: Path,
    run_root: Path,
    split: str,
    limit: int,
) -> Path:
    source = split_dir / "test.txt"
    if limit <= 0:
        return source
    ids = read_ids(source)[:limit]
    path = run_root / split / f"test_limit_{limit}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(ids) + "\n")
    return path


def write_one_task_split(
    run_root: Path,
    split: str,
    seed: int,
    task_id: str,
) -> Path:
    path = run_root / split / str(seed) / "one_task_splits" / f"{task_id}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(task_id + "\n")
    return path


def write_task_subset_split(
    *,
    run_root: Path,
    split: str,
    seed: int,
    method: str,
    task_ids: list[str],
) -> Path:
    path = (
        run_root
        / split
        / str(seed)
        / "method_task_splits"
        / f"{method}.txt"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(task_ids) + "\n")
    return path


def read_ids(path: Path) -> list[str]:
    return [
        Path(line.strip()).name
        for line in path.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def row_key(row: dict[str, Any]) -> tuple[str, str, int, str]:
    return (
        str(row["split"]),
        str(row["task_id"]),
        int(row["seed"]),
        str(row["method"]),
    )


def append_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def load_existing_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def write_results_bundle(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    ensure_required_artifact_dirs(out_dir)
    write_json(out_dir / "cell_results.json", {"rows": rows})
    write_json(out_dir / "results.json", {"rows": rows})
    csv_path = out_dir / "cell_results.csv"
    fields = sorted({key for row in rows for key in row})
    if not fields:
        write_artifact_indexes(out_dir, rows)
        return
    write_results_csv(csv_path, rows, fields)
    write_results_csv(out_dir / "results.csv", rows, fields)
    write_artifact_indexes(out_dir, rows)


def write_results_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fields: list[str],
) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(value) if isinstance(value, (list, dict)) else value
                for key, value in row.items()
            })


def ensure_required_artifact_dirs(out_dir: Path) -> None:
    for name in (
        "raw_completions",
        "verifier_outputs",
        "cad_artifacts",
        "chrono_outputs",
        "training_logs",
        "adapter_checkpoints",
    ):
        (out_dir / name).mkdir(parents=True, exist_ok=True)


def write_artifact_indexes(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    raw_paths: list[str] = []
    verifier_paths: list[str] = []
    cad_paths: list[str] = []
    chrono_paths: list[str] = []
    training_logs: set[str] = set()
    adapters: set[str] = set()
    for row in rows:
        raw_paths.extend(str(path) for path in row.get("raw_completion_paths", []) or [])
        verifier_paths.extend(
            str(path) for path in row.get("verifier_output_paths", []) or []
        )
        cad_paths.extend(str(path) for path in row.get("cad_artifact_paths", []) or [])
        chrono_paths.extend(str(path) for path in row.get("chrono_output_paths", []) or [])
        trace = str(row.get("trace_path") or "")
        if trace.endswith("reward_log.jsonl"):
            training_logs.add(trace)
        summary = str(row.get("summary_path") or "")
        if summary:
            verifier_paths.append(summary)
        adapter = str(row.get("adapter_path") or "")
        if adapter:
            adapters.add(adapter)
    write_json(
        out_dir / "raw_completions" / "index.json",
        {"paths": sorted(set(raw_paths)), "count": len(set(raw_paths))},
    )
    write_json(
        out_dir / "verifier_outputs" / "index.json",
        {"paths": sorted(set(verifier_paths)), "count": len(set(verifier_paths))},
    )
    write_json(
        out_dir / "cad_artifacts" / "index.json",
        {"paths": sorted(set(cad_paths)), "count": len(set(cad_paths))},
    )
    write_json(
        out_dir / "chrono_outputs" / "index.json",
        {"paths": sorted(set(chrono_paths)), "count": len(set(chrono_paths))},
    )
    write_json(
        out_dir / "training_logs" / "index.json",
        {"paths": sorted(training_logs), "count": len(training_logs)},
    )
    write_json(
        out_dir / "adapter_checkpoints" / "index.json",
        {"paths": sorted(adapters), "count": len(adapters)},
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def add_flag(cmd: list[str], enabled: bool, flag: str) -> None:
    if enabled:
        cmd.append(flag)


def add_option(cmd: list[str], flag: str, value: Any) -> None:
    if value is not None and value != "":
        cmd.extend([flag, str(value)])


def repo_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        str(REPO_ROOT)
        if not env.get("PYTHONPATH")
        else str(REPO_ROOT) + os.pathsep + env["PYTHONPATH"]
    )
    if extra:
        env.update(extra)
    return env


def run(
    cmd: list[str],
    *,
    timeout: float,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    print(f"[run] {shlex.join(cmd)}", file=sys.stderr, flush=True)
    started = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=repo_env(),
        text=True,
        timeout=timeout,
        check=False,
    )
    elapsed = time.perf_counter() - started
    print(
        f"[done] rc={proc.returncode} elapsed_s={elapsed:.1f}",
        file=sys.stderr,
        flush=True,
    )
    if check and proc.returncode != 0:
        raise SystemExit(proc.returncode)
    return proc


if __name__ == "__main__":
    raise SystemExit(main())
