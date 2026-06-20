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
import fcntl
import hashlib
import importlib.util
import json
import os
import shutil
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts.analyze_mechanism_repair_results import (
    build_expected_coverage,
)
from scripts.run_mechanism_repair_physics_experiment import (
    missing_evidence_for_row,
    missing_learning_evidence,
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
ADAPTER_WEIGHT_FILE_NAMES = {
    "adapter_model.safetensors",
    "adapter_model.bin",
    "pytorch_model.bin",
}
LEGACY_DEFAULT_SPLITS = ("A", "B")
PHYSICS_ANTI_SHORTCUT_SPLITS = (
    "hidden_perturbation",
    "external_style",
    "isomorphic",
)
PHYSICS_DEFAULT_SPLITS = ("A", "B", *PHYSICS_ANTI_SHORTCUT_SPLITS)


@dataclass(frozen=True)
class EvalMethod:
    name: str
    samples_per_task: int
    max_turns: int
    temperature: float
    top_p: float
    adapter_kind: str = "none"


class OptionalFileLock:
    def __init__(self, path: Path | None):
        self.path = path
        self._handle: Any | None = None

    def __enter__(self) -> None:
        if self.path is None:
            return None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w")
        fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX)
        return None

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._handle is None:
            return None
        try:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        finally:
            self._handle.close()
            self._handle = None
        return None


def optional_file_lock(path: Path | None) -> OptionalFileLock:
    return OptionalFileLock(path)


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
        choices=["sglang_chat", "worldlines_sampling", "transformers_local"],
    )
    parser.add_argument("--local-device", default="cpu")
    parser.add_argument(
        "--local-torch-dtype",
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
    )
    parser.add_argument("--local-trust-remote-code", action="store_true")
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
    parser.add_argument(
        "--require-runtime-preflight",
        action="store_true",
        help=(
            "check local sampler/training/physics runtime prerequisites before "
            "starting the shard and fail without writing result rows if they "
            "are unavailable"
        ),
    )
    parser.add_argument(
        "--runtime-preflight-only",
        action="store_true",
        help="print runtime preflight JSON and exit without running cells",
    )
    parser.add_argument("--preflight-timeout-s", type=float, default=3.0)
    parser.add_argument("--skip-analysis", action="store_true")
    parser.add_argument(
        "--evidence-layout",
        default="files",
        choices=("files", "bundled"),
        help=(
            "files writes one evidence file per call. bundled writes one "
            "JSONL bundle per split/seed/method/task and repeats that bundle "
            "path in row accounting where per-call cardinality is audited."
        ),
    )
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
    parser.add_argument(
        "--shared-sft-root",
        default=None,
        help=(
            "optional shared directory for split/seed SFT adapters. Use this "
            "for sharded runs so shards do not retrain and store duplicate "
            "SFT adapters."
        ),
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
    parser.add_argument(
        "--ttrl-steps-per-generation",
        type=int,
        default=None,
        help=(
            "TRL GRPO reuse window. Defaults to num_generations, giving "
            "matched-budget rollouts in memory-safe generation batches."
        ),
    )
    parser.add_argument("--ttrl-learning-rate", type=float, default=5.0e-6)
    parser.add_argument("--ttrl-max-grad-norm", type=float, default=0.0)
    parser.add_argument("--ttrl-lora-rank", type=int, default=16)
    parser.add_argument(
        "--ttrl-save-adapter-dtype",
        default="native",
        choices=("native", "bfloat16", "float16", "float32"),
        help="optional dtype cast for saved TTRL adapter checkpoints",
    )
    parser.add_argument(
        "--ttrl-adapter-retention",
        default="full",
        choices=("full", "metadata"),
        help=(
            "full keeps each TTRL final adapter weight directory. metadata "
            "keeps adapter config, manifest, file sizes, and hashes while "
            "removing per-cell weight payloads after the row is materialized."
        ),
    )
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
    shared_sft_root = (
        Path(args.shared_sft_root).expanduser().resolve()
        if args.shared_sft_root
        else None
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
    requested_methods = order_methods_for_budget_dependencies(requested_methods)
    allowed_methods = set(REQUIRED_METHODS) | set(method_contract["required_methods"])
    unknown = sorted(set(requested_methods) - allowed_methods)
    if unknown:
        raise SystemExit(f"unknown methods requested: {unknown}")
    splits = (
        sorted({str(cell["split"]) for cell in shard_cells})
        if shard_cells and args.splits is None
        else parse_csv(args.splits)
        if args.splits
        else default_splits_for_contract(method_contract)
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
    verifier_level_by_task = verifier_level_by_task_id(benchmark_dir)
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
    if int(args.ttrl_max_steps or 0) < 0:
        raise SystemExit("--ttrl-max-steps must be non-negative")
    if ttrl_steps <= 0:
        raise SystemExit("TTRL max steps must be positive")
    ttrl_steps_per_generation = (
        int(args.ttrl_steps_per_generation)
        if args.ttrl_steps_per_generation is not None
        else ttrl_steps_per_generation_for_budget(
            budget=budget,
            num_generations=ttrl_generations,
        )
    )
    if ttrl_steps_per_generation <= 0:
        raise SystemExit("TTRL steps_per_generation must be positive")
    if ttrl_steps_per_generation % ttrl_generations:
        raise SystemExit(
            "TTRL budget mismatch: steps_per_generation="
            f"{ttrl_steps_per_generation} must divide evenly by "
            f"num_generations={ttrl_generations}"
        )
    generation_batches = (
        (ttrl_steps + ttrl_steps_per_generation - 1)
        // ttrl_steps_per_generation
    )
    expected_ttrl_verifier_calls = generation_batches * ttrl_steps_per_generation
    if expected_ttrl_verifier_calls != budget:
        raise SystemExit(
            "TTRL budget mismatch: "
            f"max_steps={ttrl_steps}, "
            f"steps_per_generation={ttrl_steps_per_generation}, "
            f"num_generations={ttrl_generations} "
            f"expects {expected_ttrl_verifier_calls} verifier calls, "
            f"not budget={budget}"
        )
    needs_sft_for_run = methods_need_sft(
        requested_methods,
        init_online_from_sft=bool(args.init_online_from_sft),
    )
    sft_training_splits = (
        resolve_sft_training_splits(
            benchmark_dir=benchmark_dir,
            splits=splits,
            contract=method_contract,
        )
        if needs_sft_for_run
        else {}
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
        ttrl_steps_per_generation=ttrl_steps_per_generation,
        ttrl_reward_channel=str(args.ttrl_reward_channel),
        shard_cells=shard_cells,
        sft_training_splits=sft_training_splits,
    )
    if args.require_runtime_preflight or args.runtime_preflight_only:
        runtime_preflight = build_runtime_preflight(
            args=args,
            requested_methods=requested_methods,
            needs_sft=needs_sft_for_run,
            method_contract=method_contract,
        )
        if args.runtime_preflight_only:
            print(json.dumps(runtime_preflight, indent=2, sort_keys=True))
            return 0 if runtime_preflight["ready"] else 2
        if not runtime_preflight["ready"]:
            blockers = "; ".join(runtime_preflight["blockers"])
            raise SystemExit(f"runtime preflight failed: {blockers}")
        plan["runtime_preflight"] = runtime_preflight
    write_json(out_dir / "online_experiment_plan.json", plan)
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    rows_path = out_dir / "cell_results.jsonl"
    if args.resume_existing:
        rows = load_existing_rows(rows_path)
        if method_contract.get("is_physics"):
            rows, dropped_rows = prune_unusable_resume_rows(
                rows,
                out_dir=out_dir,
                budget=budget,
                verifier_level_by_task=verifier_level_by_task,
            )
            if dropped_rows:
                rewrite_rows(rows_path, rows)
                write_json(
                    out_dir / "resume_pruned_rows.json",
                    {
                        "schema": "mechanism_repair.resume_pruned_rows.v1",
                        "dropped_count": len(dropped_rows),
                        "dropped_rows": dropped_rows[:100],
                    },
                )
                write_results_bundle(out_dir, rows)
    else:
        reset_non_resume_outputs(out_dir=out_dir, run_root=run_root)
        rows = []
    seen_keys = {row_key(row) for row in rows}

    for split in splits:
        split_dir = benchmark_dir / f"splits_{split}"
        sft_train_split = sft_training_splits.get(split, split)
        train_file = benchmark_dir / f"splits_{sft_train_split}" / "train.txt"
        test_file = make_eval_split_file(
            split_dir=split_dir,
            run_root=run_root,
            split=split,
            limit=max(0, int(args.limit_tasks)),
        )
        task_entries = read_split_entries(test_file)
        task_ids = [task_id for task_id, _entry in task_entries]
        task_entry_by_id = {
            task_id: entry
            for task_id, entry in task_entries
        }
        for seed in seeds:
            sft_adapter = None
            sft_manifest_path: Path | None = None
            if needs_sft_for_run:
                sft_base = shared_sft_root or run_root
                sft_dir = sft_base / sft_train_split / str(seed) / "sft_train"
                lock_path = (
                    sft_base / sft_train_split / str(seed) / ".sft_train.lock"
                    if shared_sft_root
                    else None
                )
                with optional_file_lock(lock_path):
                    sft_adapter = run_or_load_sft(
                        args=args,
                        run_dir=sft_dir,
                        tasks_root=benchmark_dir / "tasks",
                        train_file=train_file,
                        seed=seed,
                        resume_existing=bool(args.resume_existing)
                        or shared_sft_root is not None,
                    )
                    sft_manifest_path = sft_dir / "run_manifest.json"
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
                            run_root,
                            split,
                            seed,
                            task_id,
                            task_entry=task_entry_by_id.get(task_id),
                        )
                        run_dir = (
                            run_root / split / str(seed) / method / task_id
                        )
                        cad_cap, chrono_cap = expensive_budget_caps_for_ttrl(
                            rows,
                            split=split,
                            task_id=task_id,
                            seed=seed,
                            budget=budget,
                            required=(
                                PHYSICS_PRIMARY_BASELINE
                                in set(requested_methods)
                            ),
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
                            ttrl_steps_per_generation=ttrl_steps_per_generation,
                            method=method,
                            reward_channel=reward_channel_for_method(
                                method,
                                default=str(args.ttrl_reward_channel),
                            ),
                            max_cad_audits=cad_cap,
                            max_chrono_audits=chrono_cap,
                            family_by_task=family_by_task,
                            verifier_level_by_task=verifier_level_by_task,
                            evidence_root=out_dir,
                            evidence_layout=str(args.evidence_layout),
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
                        task_entry_by_id=task_entry_by_id,
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
                    sft_manifest=sft_manifest_path,
                )
                new_rows = rows_from_sample_summary(
                    summary=summary,
                    method=method,
                    split=split,
                    seed=seed,
                    budget=budget,
                    trace_root=report_dir,
                    family_by_task=family_by_task,
                    verifier_level_by_task=verifier_level_by_task,
                    evidence_root=out_dir,
                    evidence_layout=str(args.evidence_layout),
                    sft_manifest=sft_manifest_path,
                )
                append_new_requested_rows(
                    rows_path=rows_path,
                    rows=rows,
                    seen_keys=seen_keys,
                    new_rows=new_rows,
                    shard_filter=shard_filter,
                    budget=budget,
                )
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
    method_contract = load_method_contract(benchmark_dir)
    required_splits = default_splits_for_contract(method_contract)
    required = [
        benchmark_dir / "benchmark_manifest.json",
        benchmark_dir / "method_manifest.json",
        benchmark_dir / "verifier_manifest.json",
        *[
            benchmark_dir / f"split_manifest_{split}.json"
            for split in required_splits
        ],
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
    schema = str(manifest.get("schema") or "")
    is_physics = (
        schema.startswith("mechanism_repair_physics.")
        or required == list(PHYSICS_REQUIRED_METHODS)
    )
    return {
        "schema": schema,
        "is_physics": is_physics,
        "required_methods": required,
        "primary_method": str(
            manifest.get("primary_method")
            or (PHYSICS_PRIMARY_METHOD if is_physics else LEGACY_PRIMARY_METHOD)
        ),
        "primary_baseline": str(
            manifest.get("primary_baseline")
            or (PHYSICS_PRIMARY_BASELINE if is_physics else LEGACY_PRIMARY_BASELINE)
        ),
        "primary_budget": int(
            manifest.get("primary_budget_expensive_verifier_calls")
            or manifest.get("primary_budget_verifier_calls")
            or (PHYSICS_PRIMARY_BUDGET if is_physics else LEGACY_PRIMARY_BUDGET)
        ),
        "eval_seeds": seeds,
    }


def default_splits_for_contract(contract: dict[str, Any]) -> list[str]:
    if bool(contract.get("is_physics")):
        return list(PHYSICS_DEFAULT_SPLITS)
    return list(LEGACY_DEFAULT_SPLITS)


def methods_need_sft(
    methods: list[str],
    *,
    init_online_from_sft: bool,
) -> bool:
    method_set = set(methods)
    return bool(method_set & SFT_METHODS) or (
        bool(init_online_from_sft)
        and bool(method_set & (BASELINE_FEEDBACK_METHODS | TTRL_METHODS))
    )


def build_runtime_preflight(
    *,
    args: argparse.Namespace,
    requested_methods: list[str],
    needs_sft: bool,
    method_contract: dict[str, Any],
) -> dict[str, Any]:
    """Return local runtime readiness for a real shard execution.

    This intentionally checks only local prerequisites. It never contacts a
    cluster or launches a model server.
    """
    checks: dict[str, Any] = {}
    blockers: list[str] = []
    method_set = set(requested_methods)
    sampler_methods = sorted(method_set - TTRL_METHODS)
    needs_chat_sampler = bool(sampler_methods) or bool(
        method_set & TTRL_METHODS and getattr(args, "ttrl_rollout_openai", False)
    )

    if needs_chat_sampler:
        if str(args.rollout_backend) == "sglang_chat":
            ok, detail = probe_openai_chat_server(
                str(args.sglang_base_url),
                timeout_s=float(args.preflight_timeout_s),
            )
            checks["sglang_chat"] = {
                "required": True,
                "base_url": str(args.sglang_base_url),
                "ok": ok,
                "detail": detail,
                "methods": sampler_methods,
            }
            if not ok:
                blockers.append(
                    f"sglang_chat server unavailable at {args.sglang_base_url}: "
                    f"{detail}"
                )
        elif str(args.rollout_backend) == "worldlines_sampling":
            ok = importlib.util.find_spec("worldlines") is not None
            checks["worldlines_sampling"] = {
                "required": True,
                "ok": ok,
                "methods": sampler_methods,
            }
            if not ok:
                blockers.append(
                    "worldlines_sampling requested but package 'worldlines' "
                    "is not importable"
                )
        elif str(args.rollout_backend) == "transformers_local":
            required = ["torch", "transformers"]
            if method_set & SFT_METHODS:
                required.append("peft")
            missing = [
                package
                for package in required
                if importlib.util.find_spec(package) is None
            ]
            checks["transformers_local"] = {
                "required": True,
                "ok": not missing,
                "missing": missing,
                "packages": required,
                "device": str(getattr(args, "local_device", "cpu")),
                "torch_dtype": str(getattr(args, "local_torch_dtype", "auto")),
                "methods": sampler_methods,
            }
            if missing:
                blockers.append(
                    "transformers_local packages missing: "
                    + ", ".join(sorted(missing))
                )
        else:
            blockers.append(f"unknown rollout backend: {args.rollout_backend}")
    else:
        checks["sampler"] = {"required": False, "ok": True}

    training_packages = ("torch", "transformers", "peft", "trl")
    needs_training = needs_sft or bool(method_set & TTRL_METHODS)
    if needs_training:
        missing = [
            package
            for package in training_packages
            if importlib.util.find_spec(package) is None
        ]
        checks["training_packages"] = {
            "required": True,
            "ok": not missing,
            "missing": missing,
            "packages": list(training_packages),
        }
        if missing:
            blockers.append(
                "training packages missing: " + ", ".join(sorted(missing))
            )
    else:
        checks["training_packages"] = {"required": False, "ok": True}

    if bool(method_contract.get("is_physics")):
        try:
            from mech_bench.adapters.chrono_contact import chrono_diagnostic

            chrono = chrono_diagnostic()
            chrono_ok = chrono.get("status") == "available"
            checks["chrono_contact"] = {
                "required": True,
                "ok": chrono_ok,
                "status": chrono.get("status"),
                "runner_status": chrono.get("runner_status"),
                "pychrono_importable": chrono.get("pychrono_importable"),
                "_chrono_impl_importable": chrono.get("_chrono_impl_importable"),
                "reason": chrono.get("reason"),
            }
            if not chrono_ok:
                blockers.append(
                    "chrono_contact unavailable: "
                    f"{chrono.get('reason') or chrono.get('status')}"
                )
        except Exception as exc:  # noqa: BLE001 - preflight boundary
            checks["chrono_contact"] = {
                "required": True,
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
            blockers.append(f"chrono diagnostic failed: {exc}")
    else:
        checks["chrono_contact"] = {"required": False, "ok": True}

    return {
        "schema": "mechanism_repair_ttrl.runtime_preflight.v1",
        "ready": not blockers,
        "blockers": blockers,
        "checks": checks,
    }


def probe_openai_chat_server(
    base_url: str,
    *,
    timeout_s: float,
) -> tuple[bool, str]:
    """Probe an OpenAI-compatible chat server without sampling tokens."""
    url = base_url.rstrip("/") + "/v1/models"
    request = urllib.request.Request(
        url,
        headers={"Authorization": "Bearer dummy"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=max(0.1, timeout_s)) as rsp:
            status = int(getattr(rsp, "status", 0) or 0)
            if 200 <= status < 400:
                return True, f"HTTP {status}"
            return False, f"HTTP {status}"
    except urllib.error.HTTPError as exc:
        if int(exc.code) in {401, 403}:
            return True, f"HTTP {exc.code}: auth required"
        return False, f"HTTP {exc.code}: {exc.reason}"
    except Exception as exc:  # noqa: BLE001 - diagnostic boundary
        return False, f"{type(exc).__name__}: {exc}"


def resolve_sft_training_splits(
    *,
    benchmark_dir: Path,
    splits: list[str],
    contract: dict[str, Any],
) -> dict[str, str]:
    return {
        split: resolve_sft_training_split(
            benchmark_dir=benchmark_dir,
            split=split,
            contract=contract,
        )
        for split in splits
    }


def resolve_sft_training_split(
    *,
    benchmark_dir: Path,
    split: str,
    contract: dict[str, Any],
) -> str:
    if split_has_train_rows(benchmark_dir, split):
        return split
    if (
        bool(contract.get("is_physics"))
        and split in PHYSICS_ANTI_SHORTCUT_SPLITS
        and split_has_train_rows(benchmark_dir, "A")
    ):
        return "A"
    raise SystemExit(
        f"split {split} has no SFT train rows; provide splits_{split}/train.txt "
        "or use a physics anti-shortcut split with splits_A/train.txt present"
    )


def split_has_train_rows(benchmark_dir: Path, split: str) -> bool:
    train_file = benchmark_dir / f"splits_{split}" / "train.txt"
    return train_file.is_file() and bool(read_ids(train_file))


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


def row_wanted(
    shard_filter: set[tuple[str, str, int, str, int]] | None,
    row: dict[str, Any],
    *,
    default_budget: int,
) -> bool:
    return cell_wanted(
        shard_filter,
        split=str(row["split"]),
        task_id=str(row["task_id"]),
        seed=int(row["seed"]),
        method=str(row["method"]),
        budget=int(row.get("budget", default_budget)),
    )


def append_new_requested_rows(
    *,
    rows_path: Path,
    rows: list[dict[str, Any]],
    seen_keys: set[tuple[str, str, int, str]],
    new_rows: list[dict[str, Any]],
    shard_filter: set[tuple[str, str, int, str, int]] | None,
    budget: int,
) -> dict[str, int]:
    counts = {"appended": 0, "duplicates": 0, "skipped_unrequested": 0}
    for row in new_rows:
        if not row_wanted(shard_filter, row, default_budget=budget):
            counts["skipped_unrequested"] += 1
            continue
        key = row_key(row)
        if key in seen_keys:
            counts["duplicates"] += 1
            continue
        append_row(rows_path, row)
        rows.append(row)
        seen_keys.add(key)
        counts["appended"] += 1
    return counts


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
    ttrl_steps_per_generation: int,
    ttrl_reward_channel: str,
    shard_cells: list[dict[str, Any]] | None = None,
    sft_training_splits: dict[str, str] | None = None,
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
        "ttrl_optimizer_steps": ttrl_steps,
        "ttrl_steps_per_generation": ttrl_steps_per_generation,
        "ttrl_num_generations": ttrl_generations,
        "ttrl_reward_channel": str(ttrl_reward_channel),
        "sft_training_splits": dict(sorted((sft_training_splits or {}).items())),
        "ttrl_rollout_evaluations_per_cell": (
            ((ttrl_steps + ttrl_steps_per_generation - 1)
             // ttrl_steps_per_generation)
            * ttrl_steps_per_generation
        ),
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


def order_methods_for_budget_dependencies(methods: list[str]) -> list[str]:
    """Run the no-update baseline before TTRL cells that consume its caps."""
    out = list(dict.fromkeys(methods))
    baseline = PHYSICS_PRIMARY_BASELINE
    if baseline not in out:
        return out

    def key(method: str) -> tuple[int, str]:
        if method == baseline:
            return (0, method)
        if method in TTRL_METHODS:
            return (2, method)
        return (1, method)

    return sorted(out, key=key)


def expensive_budget_caps_for_ttrl(
    rows: list[dict[str, Any]],
    *,
    split: str,
    task_id: str,
    seed: int,
    budget: int,
    required: bool,
) -> tuple[int | None, int | None]:
    for row in reversed(rows):
        if (
            str(row.get("split")) == str(split)
            and str(row.get("task_id")) == str(task_id)
            and int(row.get("seed", -1) or -1) == int(seed)
            and str(row.get("method")) == PHYSICS_PRIMARY_BASELINE
            and int(row.get("budget", budget) or budget) == int(budget)
        ):
            return (
                int(row.get("actual_cad_calls", row.get("cad_audits", 0)) or 0),
                int(
                    row.get(
                        "actual_chrono_calls",
                        row.get("chrono_audits", 0),
                    )
                    or 0
                ),
            )
    if required:
        raise SystemExit(
            "TTRL expensive-budget cap missing: "
            f"{split}/{task_id}/seed={seed}/budget={budget} has no "
            f"{PHYSICS_PRIMARY_BASELINE} row. Regenerate shards grouped by "
            "split/task/seed/budget so the baseline runs before TTRL."
        )
    return None, None


def prune_unusable_resume_rows(
    rows: list[dict[str, Any]],
    *,
    out_dir: Path,
    budget: int,
    verifier_level_by_task: dict[str, int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for row in rows:
        reasons = unusable_resume_row_reasons(
            row,
            out_dir=out_dir,
            budget=budget,
            verifier_level_by_task=verifier_level_by_task,
        )
        if reasons:
            dropped.append({
                "key": row_key_text(row),
                "reasons": reasons,
            })
            continue
        kept.append(row)
    return kept, dropped


def unusable_resume_row_reasons(
    row: dict[str, Any],
    *,
    out_dir: Path,
    budget: int,
    verifier_level_by_task: dict[str, int],
) -> list[str]:
    reasons: list[str] = []
    try:
        key = row_key(row)
    except (KeyError, TypeError, ValueError):
        return ["malformed_row_key"]

    split, task_id, seed, method = key
    row_budget = int(
        row.get("budget", row.get("budget_verifier_calls", budget)) or budget
    )
    verifier_level = int(
        row.get("verifier_level")
        or verifier_level_by_task.get(task_id, 0)
        or 0
    )
    cell = {
        "split": split,
        "task_id": task_id,
        "seed": int(seed),
        "method": method,
        "budget": row_budget,
        "verifier_level": verifier_level,
    }
    verifier_calls = int(
        row.get("actual_verifier_calls", row.get("verifier_calls", -1)) or -1
    )
    if verifier_calls != row_budget:
        reasons.append("verifier_budget_mismatch")
    if "actual_cad_calls" not in row and "cad_audits" not in row:
        reasons.append("missing_cad_accounting")
    if "actual_chrono_calls" not in row and "chrono_audits" not in row:
        reasons.append("missing_chrono_accounting")
    for item in missing_evidence_for_row(out_dir, cell, row):
        reasons.append(f"missing_{item['kind']}")
    if method in SFT_METHODS | TTRL_METHODS:
        learning_missing = missing_learning_evidence(
            out_dir,
            row,
            require_rl_evidence=method in TTRL_METHODS,
        )
        reasons.extend(f"missing_{item}" for item in learning_missing)
    return sorted(set(reasons))


def row_key_text(row: dict[str, Any]) -> str:
    try:
        split, task_id, seed, method = row_key(row)
    except (KeyError, TypeError, ValueError):
        return "<malformed>"
    budget = row.get("budget", row.get("budget_verifier_calls", ""))
    return f"{split}/{task_id}/seed{seed}/{method}/budget{budget}"


def reward_channel_for_method(method: str, *, default: str) -> str:
    if method == "mechanical_evolve_ttrl_confidence":
        return "verified_score"
    if method == "mechanical_evolve_ttrl_tool_verified":
        return "artifact_progress"
    return default


def ttrl_steps_per_generation_for_budget(
    *,
    budget: int,
    num_generations: int,
) -> int:
    budget = int(budget)
    num_generations = int(num_generations)
    if num_generations <= 0:
        raise SystemExit("--ttrl-num-generations must be positive")
    if budget % num_generations:
        raise SystemExit(
            f"TTRL budget mismatch: budget={budget} must divide evenly by "
            f"num_generations={num_generations}"
        )
    return num_generations


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
    sft_manifest: Path | None = None,
) -> dict[str, Any]:
    summary_path = report_dir / "smoke_summary.json"
    if resume_existing and summary_path.is_file():
        summary = json.loads(summary_path.read_text())
        requested_tasks = set(read_ids(test_file))
        if requested_tasks.issubset(sample_summary_task_ids(summary)):
            return summary
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
        "--local-device",
        str(getattr(args, "local_device", "cpu")),
        "--local-torch-dtype",
        str(getattr(args, "local_torch_dtype", "auto")),
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
    add_flag(
        cmd,
        bool(getattr(args, "local_trust_remote_code", False)),
        "--local-trust-remote-code",
    )
    if method.adapter_kind == "sft":
        manifest_path = sft_manifest or (
            report_dir.parent / "sft_train" / "run_manifest.json"
        )
        if not manifest_path.is_file():
            raise SystemExit(f"SFT manifest missing: {manifest_path}")
        payload = json.loads(manifest_path.read_text())
        cmd.extend(["--sglang-lora-path", str(payload["final_adapter"])])
    run(cmd, timeout=float(args.eval_timeout_s))
    if not summary_path.is_file():
        raise SystemExit(f"sample_and_score did not write {summary_path}")
    return json.loads(summary_path.read_text())


def sample_summary_task_ids(summary: dict[str, Any]) -> set[str]:
    return {
        str(row.get("task_id") or "")
        for row in summary.get("all_samples", []) or []
        if row.get("task_id")
    }


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
    ttrl_steps_per_generation: int | None,
    family_by_task: dict[str, str],
    evidence_root: Path,
    init_adapter: str | None,
    resume_existing: bool,
    verifier_level_by_task: dict[str, int] | None = None,
    method: str = PRIMARY_METHOD,
    reward_channel: str | None = None,
    max_cad_audits: int | None = None,
    max_chrono_audits: int | None = None,
    evidence_layout: str = "files",
) -> dict[str, Any]:
    reward_log = run_dir / "reward_log.jsonl"
    if not resume_existing and run_dir.exists():
        shutil.rmtree(run_dir)
        reward_log = run_dir / "reward_log.jsonl"
    if (
        resume_existing
        and reward_log.is_file()
        and reward_log_exceeds_expensive_caps(
            reward_log,
            max_cad_audits=max_cad_audits,
            max_chrono_audits=max_chrono_audits,
        )
    ):
        print(
            f"[resume] discarding uncapped/overspent TTRL run for {task_id}: "
            f"{reward_log}",
            file=sys.stderr,
        )
        shutil.rmtree(run_dir, ignore_errors=True)
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
                "--steps-per-generation",
                str(
                    ttrl_steps_per_generation
                    if ttrl_steps_per_generation is not None
                    else ttrl_steps_per_generation_for_budget(
                        budget=budget,
                        num_generations=ttrl_generations,
                    )
                ),
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
                "--save-adapter-dtype",
                str(args.ttrl_save_adapter_dtype),
                "--seed",
                str(seed),
            ]
            if max_cad_audits is not None:
                cmd.extend(["--max-cad-audits", str(int(max_cad_audits))])
            if max_chrono_audits is not None:
                cmd.extend([
                    "--max-chrono-audits",
                    str(int(max_chrono_audits)),
                ])
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
            payload = json.loads(manifest.read_text())
            require_learning_manifest(
                payload,
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
    retain_ttrl_adapter_checkpoint(
        manifest_path=run_dir / "run_manifest.json",
        split=split,
        seed=seed,
        method=method,
        task_id=task_id,
        evidence_root=evidence_root,
        retention=str(getattr(args, "ttrl_adapter_retention", "full") or "full"),
    )
    row = row_from_ttrl_reward_log(
        reward_log=reward_log,
        split=split,
        task_id=task_id,
        seed=seed,
        budget=budget,
        method=method,
        run_dir=run_dir,
        family_by_task=family_by_task,
        verifier_level_by_task=verifier_level_by_task,
        evidence_root=evidence_root,
        evidence_layout=evidence_layout,
    )
    if int(row.get("actual_verifier_calls", 0) or 0) != int(budget):
        raise SystemExit(
            "TTRL verifier budget mismatch: "
            f"{method} {split}/{task_id}/seed={seed} wrote "
            f"{row.get('actual_verifier_calls')} verifier calls; "
            f"expected {int(budget)}. Adjust --ttrl-max-steps and "
            "--ttrl-num-generations before running the full audit."
        )
    if (
        max_cad_audits is not None
        and int(row.get("actual_cad_calls", 0) or 0) > int(max_cad_audits)
    ):
        raise SystemExit(
            "TTRL CAD budget mismatch: "
            f"{method} {split}/{task_id}/seed={seed} wrote "
            f"{row.get('actual_cad_calls')} CAD calls; cap is {max_cad_audits}."
        )
    if (
        max_chrono_audits is not None
        and int(row.get("actual_chrono_calls", 0) or 0)
        > int(max_chrono_audits)
    ):
        raise SystemExit(
            "TTRL Chrono budget mismatch: "
            f"{method} {split}/{task_id}/seed={seed} wrote "
            f"{row.get('actual_chrono_calls')} Chrono calls; "
            f"cap is {max_chrono_audits}."
        )
    return row


def reward_log_exceeds_expensive_caps(
    reward_log: Path,
    *,
    max_cad_audits: int | None,
    max_chrono_audits: int | None,
) -> bool:
    if max_cad_audits is None and max_chrono_audits is None:
        return False
    cad = 0
    chrono = 0
    try:
        lines = reward_log.read_text().splitlines()
    except OSError:
        return False
    for line in lines:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            return True
        cad += int(row.get("cad_audits", 0) or 0)
        chrono += int(row.get("chrono_audits", 0) or 0)
        if max_cad_audits is not None and cad > int(max_cad_audits):
            return True
        if max_chrono_audits is not None and chrono > int(max_chrono_audits):
            return True
    return False


def retain_ttrl_adapter_checkpoint(
    *,
    manifest_path: Path,
    split: str,
    seed: int,
    method: str,
    task_id: str,
    evidence_root: Path | None,
    retention: str,
) -> str:
    payload = json.loads(manifest_path.read_text())
    adapter_path = Path(
        str(payload.get("final_adapter") or manifest_path.parent / "final_adapter")
    )
    if retention == "full":
        payload["adapter_retention"] = {
            "mode": "full",
            "path": str(adapter_path),
            "weights_retained": True,
        }
        payload["adapter_checkpoint_paths"] = [str(adapter_path)]
        manifest_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
        )
        return str(adapter_path)
    if retention != "metadata":
        raise SystemExit(f"unknown TTRL adapter retention mode: {retention}")

    root = evidence_root if evidence_root is not None else manifest_path.parent
    checkpoint_dir = (
        root
        / "adapter_checkpoints"
        / split
        / str(seed)
        / method
        / task_id
    )
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    files = adapter_file_manifest(adapter_path)
    copied_files: list[str] = []
    omitted_redundant_files: list[str] = []
    if adapter_path.is_dir():
        for source in sorted(path for path in adapter_path.rglob("*") if path.is_file()):
            if is_adapter_weight_file(source):
                continue
            rel = source.relative_to(adapter_path)
            if rel.name == "tokenizer.json":
                omitted_redundant_files.append(str(rel))
                continue
            dest = checkpoint_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
            copied_files.append(str(rel))
    manifest_copy = checkpoint_dir / "training_run_manifest.json"
    manifest_copy.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    )
    checkpoint_manifest = {
        "schema": "mechanism_repair.ttrl_adapter_checkpoint_metadata.v1",
        "mode": "metadata",
        "split": split,
        "seed": int(seed),
        "method": method,
        "task_id": task_id,
        "source_adapter_path": str(adapter_path),
        "source_run_manifest": str(manifest_path),
        "copied_non_weight_files": copied_files,
        "omitted_redundant_non_weight_files": omitted_redundant_files,
        "source_files": files,
        "weights_retained": False,
        "rationale": (
            "Per-cell TTRL adapter weights are omitted to keep the full "
            "paper-scale shard run within shared-storage limits; the reward "
            "log, trainer manifest, adapter config, file sizes, and hashes "
            "are retained for audit. Redundant tokenizer files are recorded "
            "in source_files but not copied into every per-cell checkpoint."
        ),
    }
    (checkpoint_dir / "checkpoint_manifest.json").write_text(
        json.dumps(checkpoint_manifest, indent=2, sort_keys=True, default=str) + "\n"
    )
    if adapter_path.exists() and adapter_path.resolve() != checkpoint_dir.resolve():
        shutil.rmtree(adapter_path)

    payload["final_adapter"] = str(checkpoint_dir)
    payload["adapter_checkpoint_paths"] = [str(checkpoint_dir)]
    payload["adapter_retention"] = {
        "mode": "metadata",
        "path": str(checkpoint_dir),
        "weights_retained": False,
        "source_adapter_path": str(adapter_path),
        "source_files": files,
    }
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    )
    return str(checkpoint_dir)


def adapter_file_manifest(adapter_path: Path) -> list[dict[str, Any]]:
    if not adapter_path.exists():
        return []
    if adapter_path.is_file():
        paths = [adapter_path]
        base = adapter_path.parent
    else:
        paths = sorted(path for path in adapter_path.rglob("*") if path.is_file())
        base = adapter_path
    out: list[dict[str, Any]] = []
    for path in paths:
        rel = path.relative_to(base)
        out.append({
            "path": str(rel),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "weight_file": is_adapter_weight_file(path),
        })
    return out


def is_adapter_weight_file(path: Path) -> bool:
    return path.name in ADAPTER_WEIGHT_FILE_NAMES or path.suffix == ".safetensors"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    verifier_level_by_task: dict[str, int] | None = None,
    evidence_root: Path | None = None,
    evidence_layout: str = "files",
    sft_manifest: Path | None = None,
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
        verifier_level = int((verifier_level_by_task or {}).get(task_id, 0) or 0)
        required_cad_audits, required_chrono_audits = required_audits_for_level(
            verifier_level=verifier_level,
            verifier_calls=verifier_calls,
        )
        sample_idx = int(best.get("sample_idx", 0) or 0)
        raw_paths, verifier_paths = materialize_sample_evidence(
            evidence_root=evidence_root,
            trace_root=trace_root,
            split=split,
            seed=seed,
            method=method,
            task_id=task_id,
            samples=samples,
            layout=evidence_layout,
        )
        cad_paths, chrono_paths = materialize_audit_evidence(
            evidence_root=evidence_root,
            split=split,
            seed=seed,
            method=method,
            task_id=task_id,
            rows=samples,
            required_cad_audits=required_cad_audits,
            required_chrono_audits=required_chrono_audits,
            layout=evidence_layout,
        )
        sampler_error_attempts = sampler_error_attempt_count(samples)
        first_valid = first_valid_call(samples)
        best_metrics = dict(best.get("physical_metrics") or {})
        training = training_metadata_for_method(
            method,
            trace_root,
            sft_manifest=sft_manifest,
        )
        out.append({
            "method": method,
            "method_variant": method,
            "split": split,
            "task_id": task_id,
            "family": family_by_task.get(task_id, str(best.get("family") or "")),
            "verifier_level": verifier_level or None,
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
            "required_cad_audits": required_cad_audits,
            "required_chrono_audits": required_chrono_audits,
            "actual_verifier_calls": verifier_calls,
            "actual_cad_calls": cad_audits,
            "actual_chrono_calls": chrono_audits,
            "failure_codes": best.get("failure_codes", []),
            "trace_path": str(trace_root / f"sample_{sample_idx}" / task_id),
            "summary_path": str(trace_root / "smoke_summary.json"),
            "training_log_paths": training["training_log_paths"],
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
            "sampler_error_count": sampler_error_attempts,
            "sampler_http_400_count": sum(
                int(item.get("sampler_http_400_count", 0) or 0)
                for item in samples
            ),
            "sampler_retry_count": sum(
                int(item.get("sampler_retry_count", 0) or 0)
                for item in samples
            ) + sampler_error_attempts,
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
            "adapter_checkpoint_paths": training["adapter_checkpoint_paths"],
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
    verifier_level_by_task: dict[str, int] | None = None,
    evidence_root: Path | None = None,
    method: str = PRIMARY_METHOD,
    evidence_layout: str = "files",
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
    verifier_level = int((verifier_level_by_task or {}).get(task_id, 0) or 0)
    required_cad_audits, required_chrono_audits = required_audits_for_level(
        verifier_level=verifier_level,
        verifier_calls=verifier_calls,
    )
    raw_paths, verifier_paths = materialize_ttrl_evidence(
        evidence_root=evidence_root,
        split=split,
        seed=seed,
        method=method,
        task_id=task_id,
        rows=matching,
        layout=evidence_layout,
    )
    cad_paths, chrono_paths = materialize_audit_evidence(
        evidence_root=evidence_root,
        split=split,
        seed=seed,
        method=method,
        task_id=task_id,
        rows=matching,
        required_cad_audits=required_cad_audits,
        required_chrono_audits=required_chrono_audits,
        layout=evidence_layout,
    )
    best_metrics = dict(best.get("physical_metrics") or {})
    training = training_metadata_from_manifest(run_dir / "run_manifest.json")
    return {
        "method": method,
        "method_variant": method,
        "split": split,
        "task_id": task_id,
        "family": family,
        "verifier_level": verifier_level or None,
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
        "required_cad_audits": required_cad_audits,
        "required_chrono_audits": required_chrono_audits,
        "actual_verifier_calls": verifier_calls,
        "actual_cad_calls": cad_audits,
        "actual_chrono_calls": chrono_audits,
        "failure_codes": best.get("failure_codes", []),
        "trace_path": str(reward_log),
        "training_log_paths": [str(reward_log)],
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
        "sampler_http_400_count": sum(
            int(item.get("sampler_http_400_count", 0) or 0)
            for item in matching
        ) + int(training.get("sampler_http_400_count", 0) or 0),
        "sampler_retry_count": sum(
            int(item.get("sampler_retry_count", 0) or 0)
            for item in matching
        ) + int(training.get("sampler_retry_count", 0) or 0),
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
        "adapter_checkpoint_paths": training["adapter_checkpoint_paths"]
        or (
            [training["adapter_path"]]
            if training["adapter_path"]
            else [str(run_dir / "final_adapter")]
        ),
    }


def sample_rank(row: dict[str, Any]) -> tuple[
    bool, float, bool, bool, bool, float, bool
]:
    codes = failure_codes(row)
    invalid_codes = {
        "invalid_artifact",
        "invalid_design",
        "runner_json_error",
        "design_build_error",
        "missing_cad",
        "untrusted_asset",
    }
    return (
        bool(
            row.get("evaluation_valid")
            and row.get("hard_gate_passed")
            and not (row.get("failure_codes") or [])
        ),
        float(row.get("verified_score", 0.0) or 0.0),
        bool(row.get("evaluation_valid", False)),
        bool(row.get("hard_gate_passed", False)),
        not bool(codes & invalid_codes),
        float(row.get("score", 0.0) or 0.0),
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
    layout: str = "files",
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
    if layout == "bundled":
        return materialize_sample_evidence_bundle(
            raw_dir=raw_dir,
            verifier_dir=verifier_dir,
            trace_root=trace_root,
            task_id=task_id,
            samples=samples,
        )
    if layout != "files":
        raise SystemExit(f"unknown evidence layout: {layout}")
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
                if str(turn.get("trace_kind") or "") == "sampler_error_retry":
                    continue
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


def materialize_sample_evidence_bundle(
    *,
    raw_dir: Path,
    verifier_dir: Path,
    trace_root: Path,
    task_id: str,
    samples: list[dict[str, Any]],
) -> tuple[list[str], list[str]]:
    raw_bundle = raw_dir / "raw_completions.jsonl"
    verifier_bundle = verifier_dir / "verifier_outputs.jsonl"
    raw_paths: list[str] = []
    verifier_paths: list[str] = []
    with raw_bundle.open("w") as raw_f, verifier_bundle.open("w") as verifier_f:
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
                    raw_record = {
                        "sample_idx": idx,
                        "turn_idx": turn_idx,
                        "verifier_call_idx_within_sample": call_idx,
                        "audit_attempt": audit_attempt,
                        "sampler_attempt": sampler_attempt,
                        "trace_kind": turn.get("trace_kind"),
                        "assistant_text": str(turn.get("assistant_text") or ""),
                    }
                    raw_f.write(json.dumps(raw_record, sort_keys=True, default=str) + "\n")
                    raw_paths.append(str(raw_bundle))
                    if str(turn.get("trace_kind") or "") == "sampler_error_retry":
                        continue
                    verifier_record = verifier_record_from_turn(
                        item=item,
                        turn=turn,
                        idx=idx,
                        turn_idx=turn_idx,
                        call_idx=call_idx,
                        audit_attempt=audit_attempt,
                        sampler_attempt=sampler_attempt,
                    )
                    verifier_f.write(
                        json.dumps(verifier_record, sort_keys=True, default=str) + "\n"
                    )
                    verifier_paths.append(str(verifier_bundle))
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
                    raw_record = {
                        "sample_idx": idx,
                        "turn_idx": turn_idx,
                        "verifier_call_idx_within_sample": call_idx,
                        "audit_attempt": 0,
                        "sampler_attempt": 0,
                        "trace_kind": "terminal_sample_evidence",
                        "assistant_text": terminal_text,
                    }
                    raw_f.write(json.dumps(raw_record, sort_keys=True, default=str) + "\n")
                    raw_paths.append(str(raw_bundle))
                    verifier_record = verifier_record_from_sample(
                        item=item,
                        idx=idx,
                        turn_idx=turn_idx,
                        call_idx=call_idx,
                    )
                    verifier_f.write(
                        json.dumps(verifier_record, sort_keys=True, default=str) + "\n"
                    )
                    verifier_paths.append(str(verifier_bundle))
                    written_for_sample += 1
                continue
            source = trace_root / f"sample_{idx}" / task_id / "completion.txt"
            text = source.read_text() if source.is_file() else ""
            raw_record = {"sample_idx": idx, "assistant_text": text}
            raw_f.write(json.dumps(raw_record, sort_keys=True, default=str) + "\n")
            raw_paths.append(str(raw_bundle))
            verifier_record = dict(item)
            verifier_f.write(
                json.dumps(verifier_record, sort_keys=True, default=str) + "\n"
            )
            verifier_paths.append(str(verifier_bundle))
    return raw_paths, verifier_paths


def verifier_record_from_turn(
    *,
    item: dict[str, Any],
    turn: dict[str, Any],
    idx: int,
    turn_idx: int,
    call_idx: int,
    audit_attempt: int,
    sampler_attempt: int,
) -> dict[str, Any]:
    return {
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
        "no_procedural_fallback": turn.get("no_procedural_fallback"),
        "completion_tokens": turn.get("completion_tokens", 0),
        "stop_reason": turn.get("stop_reason", ""),
    }


def verifier_record_from_sample(
    *,
    item: dict[str, Any],
    idx: int,
    turn_idx: int,
    call_idx: int,
) -> dict[str, Any]:
    return {
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
        "no_procedural_fallback": item.get("no_procedural_fallback"),
        "completion_tokens": item.get("sample_tokens_out", 0),
        "stop_reason": "terminal_sample_evidence",
    }


def materialize_ttrl_evidence(
    *,
    evidence_root: Path | None,
    split: str,
    seed: int,
    method: str,
    task_id: str,
    rows: list[dict[str, Any]],
    layout: str = "files",
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
    if layout == "bundled":
        raw_bundle = raw_dir / "raw_completions.jsonl"
        verifier_bundle = verifier_dir / "verifier_outputs.jsonl"
        with raw_bundle.open("w") as raw_f, verifier_bundle.open("w") as verifier_f:
            for idx, row in enumerate(rows):
                text = str(row.get("completion_text") or row.get("completion_preview") or "")
                raw_f.write(
                    json.dumps(
                        {"candidate_idx": idx, "completion_text": text},
                        sort_keys=True,
                        default=str,
                    ) + "\n"
                )
                raw_paths.append(str(raw_bundle))
                verifier_row = dict(row)
                verifier_row.pop("completion_text", None)
                verifier_f.write(
                    json.dumps(verifier_row, sort_keys=True, default=str) + "\n"
                )
                verifier_paths.append(str(verifier_bundle))
        return raw_paths, verifier_paths
    if layout != "files":
        raise SystemExit(f"unknown evidence layout: {layout}")
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
    required_cad_audits: int = 0,
    required_chrono_audits: int = 0,
    layout: str = "files",
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
    if layout == "bundled":
        return materialize_audit_evidence_bundle(
            cad_dir=cad_dir,
            chrono_dir=chrono_dir,
            rows=rows,
            required_cad_audits=required_cad_audits,
            required_chrono_audits=required_chrono_audits,
        )
    if layout != "files":
        raise SystemExit(f"unknown evidence layout: {layout}")
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
    if required_cad_audits > 0 and not cad_paths:
        cad_paths.append(
            str(write_audit_obligation_record(
                out_dir=cad_dir,
                kind="cad",
                rows=rows,
                required_audits=required_cad_audits,
            ))
        )
    if required_chrono_audits > 0 and not chrono_paths:
        chrono_paths.append(
            str(write_audit_obligation_record(
                out_dir=chrono_dir,
                kind="chrono",
                rows=rows,
                required_audits=required_chrono_audits,
            ))
        )
    return cad_paths, chrono_paths


def materialize_audit_evidence_bundle(
    *,
    cad_dir: Path,
    chrono_dir: Path,
    rows: list[dict[str, Any]],
    required_cad_audits: int = 0,
    required_chrono_audits: int = 0,
) -> tuple[list[str], list[str]]:
    cad_bundle = cad_dir / "cad_audits.jsonl"
    chrono_bundle = chrono_dir / "chrono_audits.jsonl"
    cad_paths: list[str] = []
    chrono_paths: list[str] = []
    with cad_bundle.open("w") as cad_f, chrono_bundle.open("w") as chrono_f:
        for item_idx, item in enumerate(rows):
            traces = item.get("turn_traces") or []
            candidates = traces if traces else [item]
            for trace_idx, row in enumerate(candidates):
                suffix = f"turn_{trace_idx:04d}" if traces else ""
                stem = evidence_stem(item=row, fallback_idx=item_idx, suffix=suffix)
                cad_record = audit_record_if_present(stem=stem, kind="cad", row=row)
                if cad_record:
                    cad_f.write(json.dumps(cad_record, sort_keys=True, default=str) + "\n")
                    cad_paths.append(str(cad_bundle))
                chrono_record = audit_record_if_present(stem=stem, kind="chrono", row=row)
                if chrono_record:
                    chrono_f.write(
                        json.dumps(chrono_record, sort_keys=True, default=str) + "\n"
                    )
                    chrono_paths.append(str(chrono_bundle))
        if required_cad_audits > 0 and not cad_paths:
            cad_f.write(
                json.dumps(
                    audit_obligation_record(
                        kind="cad",
                        rows=rows,
                        required_audits=required_cad_audits,
                    ),
                    sort_keys=True,
                    default=str,
                ) + "\n"
            )
            cad_paths.append(str(cad_bundle))
        if required_chrono_audits > 0 and not chrono_paths:
            chrono_f.write(
                json.dumps(
                    audit_obligation_record(
                        kind="chrono",
                        rows=rows,
                        required_audits=required_chrono_audits,
                    ),
                    sort_keys=True,
                    default=str,
                ) + "\n"
            )
            chrono_paths.append(str(chrono_bundle))
    return sorted(set(cad_paths)), sorted(set(chrono_paths))


def required_audits_for_level(
    *,
    verifier_level: int,
    verifier_calls: int,
) -> tuple[int, int]:
    calls = max(0, int(verifier_calls or 0))
    level = int(verifier_level or 0)
    return (calls if level >= 2 else 0, calls if level >= 3 else 0)


def write_audit_obligation_record(
    *,
    out_dir: Path,
    kind: str,
    rows: list[dict[str, Any]],
    required_audits: int,
) -> Path:
    dest = out_dir / f"{kind}_audit_obligation.json"
    dest.write_text(
        json.dumps(
            audit_obligation_record(
                kind=kind,
                rows=rows,
                required_audits=required_audits,
            ),
            indent=2,
            sort_keys=True,
            default=str,
        ) + "\n"
    )
    return dest


def audit_obligation_record(
    *,
    kind: str,
    rows: list[dict[str, Any]],
    required_audits: int,
) -> dict[str, Any]:
    failure_counts: dict[str, int] = defaultdict(int)
    task_id = None
    family = None
    for row in rows:
        task_id = task_id or row.get("task_id")
        family = family or row.get("family")
        for code in row.get("failure_codes") or []:
            failure_counts[str(code)] += 1
    return {
        "schema": "mechanism_repair_ttrl.audit_obligation.v1",
        "kind": kind,
        "status": "precondition_failed_no_actual_audit",
        "required_audits": int(required_audits),
        "actual_audits": 0,
        "task_id": task_id,
        "family": family,
        "candidate_count": len(rows),
        "failure_code_counts": dict(sorted(failure_counts.items())),
        "rationale": (
            "The benchmark cell requires this Level-2/Level-3 verifier "
            "evidence, but no candidate reached the tool-specific audit "
            "because earlier structural gates failed. This record preserves "
            "the failed evidence obligation without claiming that the CAD or "
            "Chrono tool executed."
        ),
    }


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
    payload = audit_record_if_present(stem=stem, kind=kind, row=row)
    if not payload:
        return []
    dest = out_dir / f"{stem}.json"
    dest.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    return [str(dest)]


def audit_record_if_present(
    *,
    stem: str,
    kind: str,
    row: dict[str, Any],
) -> dict[str, Any] | None:
    field = "cad_audits" if kind == "cad" else "chrono_audits"
    if int(row.get(field, 0) or 0) <= 0:
        return None
    return {
        "schema": f"mechanism_repair_ttrl.{kind}_artifact_evidence.v1",
        "kind": kind,
        "stem": stem,
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


def sampler_error_attempt_count(rows: list[dict[str, Any]]) -> int:
    total = failure_count(rows, "sampler_error")
    for row in rows:
        for turn in row.get("turn_traces") or []:
            if not isinstance(turn, dict):
                continue
            if str(turn.get("trace_kind") or "") == "sampler_error_retry":
                total += 1
                continue
            codes = turn.get("failure_codes") or []
            if isinstance(codes, str):
                codes = [codes]
            if any(str(code).lower() == "sampler_error" for code in codes):
                total += 1
    return total


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


def training_metadata_for_method(
    method: str,
    trace_root: Path,
    *,
    sft_manifest: Path | None = None,
) -> dict[str, Any]:
    if method in SFT_METHODS:
        return training_metadata_from_manifest(
            sft_manifest
            or (trace_root.parent / "sft_train" / "run_manifest.json")
        )
    return empty_training_metadata()


def training_metadata_from_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return empty_training_metadata()
    payload = json.loads(path.read_text())
    adapter_path = str(payload.get("final_adapter") or "")
    adapter_checkpoint_paths = [
        str(item)
        for item in (
            payload.get("adapter_checkpoint_paths")
            or ([adapter_path] if adapter_path else [])
        )
            if item
    ]
    training_log_paths = [
        str(item)
        for item in (
            payload.get("training_log_paths")
            or payload.get("training_logs")
            or [str(path)]
        )
        if item
    ]
    return {
        "adapter_updates": int(payload.get("adapter_updates", 0) or 0),
        "trained_tokens": int(payload.get("trained_tokens", 0) or 0),
        "rl_trained_tokens": int(payload.get("rl_trained_tokens", 0) or 0),
        "n_rl_datums": int(payload.get("n_rl_datums", 0) or 0),
        "adapter_path": adapter_path,
        "adapter_checkpoint_paths": adapter_checkpoint_paths,
        "training_log_paths": training_log_paths,
        "sampler_http_400_count": int(
            (payload.get("rollout_retry_stats") or {}).get(
                "sampler_http_400_count", 0
            )
            or 0
        ),
        "sampler_retry_count": int(
            (payload.get("rollout_retry_stats") or {}).get(
                "sampler_retry_count", 0
            )
            or 0
        ),
    }


def empty_training_metadata() -> dict[str, Any]:
    return {
        "adapter_updates": 0,
        "trained_tokens": 0,
        "rl_trained_tokens": 0,
        "n_rl_datums": 0,
        "adapter_path": "",
        "adapter_checkpoint_paths": [],
        "training_log_paths": [],
        "sampler_http_400_count": 0,
        "sampler_retry_count": 0,
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


def verifier_level_by_task_id(benchmark_dir: Path) -> dict[str, int]:
    manifest = json.loads((benchmark_dir / "benchmark_manifest.json").read_text())
    out: dict[str, int] = {}
    for row in manifest.get("tasks", []) or []:
        task_id = str(row.get("task_id") or "")
        if not task_id:
            continue
        try:
            out[task_id] = int(row.get("verifier_level", 0) or 0)
        except (TypeError, ValueError):
            out[task_id] = 0
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
    entries = [entry for _task_id, entry in read_split_entries(source)[:limit]]
    path = run_root / split / f"test_limit_{limit}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(entries) + "\n")
    return path


def write_one_task_split(
    run_root: Path,
    split: str,
    seed: int,
    task_id: str,
    *,
    task_entry: str | None = None,
) -> Path:
    path = run_root / split / str(seed) / "one_task_splits" / f"{task_id}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(task_entry or task_id) + "\n")
    return path


def write_task_subset_split(
    *,
    run_root: Path,
    split: str,
    seed: int,
    method: str,
    task_ids: list[str],
    task_entry_by_id: dict[str, str] | None = None,
) -> Path:
    path = (
        run_root
        / split
        / str(seed)
        / "method_task_splits"
        / f"{method}.txt"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    task_entry_by_id = task_entry_by_id or {}
    path.write_text(
        "\n".join(str(task_entry_by_id.get(task_id) or task_id) for task_id in task_ids)
        + "\n"
    )
    return path


def read_split_entries(path: Path) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    for line in path.read_text().splitlines():
        entry = line.strip()
        if not entry or entry.startswith("#"):
            continue
        entries.append((Path(entry).name, entry))
    return entries


def read_ids(path: Path) -> list[str]:
    return [task_id for task_id, _entry in read_split_entries(path)]


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


def rewrite_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, default=str) + "\n" for row in rows)
    )


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
        csv_path.write_text("")
        (out_dir / "results.csv").write_text("")
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
        for adapter_checkpoint in row.get("adapter_checkpoint_paths", []) or []:
            if adapter_checkpoint:
                adapters.add(str(adapter_checkpoint))
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
