#!/usr/bin/env python3
"""Plan and audit the MechanismRepair-Physics experiment.

This script is the physics-facing execution gate for ``goals.md``. It does not
pretend to produce paper results. It materializes the exact experiment plan and
then audits the output directory for method/seed/task coverage, matched actual
verifier budget, raw verifier evidence, CAD/Chrono evidence, anti-shortcut
coverage, and analysis artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from scripts.prepare_mechanism_repair_physics_benchmark import (
    EVAL_SEEDS,
    PRIMARY_BASELINE,
    PRIMARY_BUDGET,
    PRIMARY_METHOD,
    REQUIRED_FAMILIES,
    REQUIRED_METHODS,
    SUCCESS_DELTA_PCT,
    ensure_run_scaffold,
)


DEFAULT_BENCHMARK_DIR = "runs/mechanism_repair_physics_final"
DEFAULT_HEADLINE_SPLITS = ("A", "B")
DEFAULT_ANTI_SHORTCUT_SPLITS = (
    "hidden_perturbation",
    "external_style",
    "isomorphic",
)
MIN_TASKS_PER_FAMILY = 10
MIN_FINAL_TASKS = 120
MIN_LEVEL2PLUS_FRACTION = 0.40
MIN_LEVEL3_FRACTION = 0.25
MIN_POSITIVE_FAMILIES = 8
TTRL_METHODS = {
    "mechanical_evolve_ttrl",
    "mechanical_evolve_ttrl_tool_verified",
    "mechanical_evolve_ttrl_confidence",
}
SFT_METHODS = {"sft_seen_family"}
LEARNING_METHODS = TTRL_METHODS | SFT_METHODS
ANALYSIS_ARTIFACTS = (
    "stats.json",
    "failure_analysis.json",
    "trace_pairs.json",
    "repair_taxonomy.json",
)
RESULT_ARTIFACTS = (
    "cell_results.jsonl",
    "results.json",
    "results.csv",
)
REQUIRED_RUN_DIRS = (
    "raw_completions",
    "verifier_outputs",
    "cad_artifacts",
    "chrono_outputs",
    "training_logs",
    "adapter_checkpoints",
)
REQUIRED_SECONDARY_METRICS = (
    "cad_valid_rate",
    "chrono_valid_rate",
    "first_valid_verifier_call",
    "mobility_repair_success",
    "port_grounding_repair_success",
    "artifact_validity_repair_success",
    "contact_repair_success",
    "max_penetration_mm",
    "contact_force_rms_N",
    "ratio_error_pct",
    "stroke_error_mm",
    "path_chamfer_error",
    "lockup_rate",
    "invalid_topology_rate",
    "invalid_artifact_rate",
    "missing_port_rate",
    "ungrounded_port_rate",
    "wrong_mobility_rate",
    "adapter_updates",
    "rl_datums",
    "trained_tokens",
    "rl_trained_tokens",
)
REQUIRED_REPAIR_TAXONOMY_DIMENSIONS = (
    "topology_repair",
    "port_repair",
    "mobility_repair",
    "ratio_stroke_path_repair",
    "cad_artifact_repair",
    "material_mass_property_repair",
    "collision_clearance_repair",
    "contact_lockup_repair",
    "manufacturability_assembly_repair",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", default=DEFAULT_BENCHMARK_DIR)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--methods", default=",".join(REQUIRED_METHODS))
    parser.add_argument(
        "--splits",
        default=",".join(DEFAULT_HEADLINE_SPLITS),
        help="headline family-held-out splits to execute",
    )
    parser.add_argument(
        "--anti-shortcut-splits",
        default=",".join(DEFAULT_ANTI_SHORTCUT_SPLITS),
        help="hidden/isomorphic audit splits that must also be executed",
    )
    parser.add_argument(
        "--eval-seeds",
        default=",".join(str(seed) for seed in EVAL_SEEDS),
    )
    parser.add_argument("--budgets", default=str(PRIMARY_BUDGET))
    parser.add_argument("--limit-tasks", type=int, default=0)
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="number of deterministic cell shards; default audits the full grid",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="0-based deterministic shard index to plan/audit",
    )
    parser.add_argument(
        "--write-shard-files",
        type=int,
        default=0,
        metavar="N",
        help="write N deterministic shard JSON files and exit after auditing",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="exit non-zero unless the audit is complete",
    )
    args = parser.parse_args()

    benchmark_dir = Path(args.benchmark_dir).expanduser().resolve()
    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else benchmark_dir
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    ensure_run_scaffold(out_dir)

    benchmark_readiness_blockers = validate_benchmark(benchmark_dir)
    methods = parse_csv(args.methods)
    unknown_methods = sorted(set(methods) - set(REQUIRED_METHODS))
    if unknown_methods:
        raise SystemExit(f"unknown methods requested: {unknown_methods}")
    splits = parse_csv(args.splits)
    anti_splits = parse_csv(args.anti_shortcut_splits)
    seeds = [int(item) for item in parse_csv(args.eval_seeds)]
    budgets = [int(item) for item in parse_csv(args.budgets)]
    if PRIMARY_BUDGET not in budgets:
        raise SystemExit(f"--budgets must include primary budget {PRIMARY_BUDGET}")
    if args.num_shards < 1:
        raise SystemExit("--num-shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise SystemExit("--shard-index must be in [0, --num-shards)")

    task_index = load_task_index(benchmark_dir)
    plan = build_plan(
        benchmark_dir=benchmark_dir,
        out_dir=out_dir,
        methods=methods,
        splits=splits,
        anti_shortcut_splits=anti_splits,
        seeds=seeds,
        budgets=budgets,
        limit_tasks=max(0, int(args.limit_tasks)),
        task_index=task_index,
        shard_index=int(args.shard_index),
        num_shards=int(args.num_shards),
    )
    if int(args.write_shard_files) > 0:
        shard_plan = build_plan(
            benchmark_dir=benchmark_dir,
            out_dir=out_dir,
            methods=methods,
            splits=splits,
            anti_shortcut_splits=anti_splits,
            seeds=seeds,
            budgets=budgets,
            limit_tasks=max(0, int(args.limit_tasks)),
            task_index=task_index,
            shard_index=0,
            num_shards=1,
        )
        write_shard_files(
            out_dir=out_dir,
            plan=shard_plan,
            num_shards=int(args.write_shard_files),
        )
    write_json(out_dir / "physics_experiment_plan.json", public_plan(plan))
    audit = audit_existing_experiment(
        out_dir=out_dir,
        plan=plan,
        benchmark_readiness_blockers=benchmark_readiness_blockers,
    )
    write_json(out_dir / "budget_audit.json", audit["budget_audit"])
    write_json(out_dir / "anti_shortcut_audit.json", audit["anti_shortcut_audit"])
    write_json(out_dir / "claim_audit.json", audit["claim_audit"])

    print(json.dumps({
        "plan": str(out_dir / "physics_experiment_plan.json"),
        "budget_audit": str(out_dir / "budget_audit.json"),
        "anti_shortcut_audit": str(out_dir / "anti_shortcut_audit.json"),
        "claim_audit": str(out_dir / "claim_audit.json"),
        "goal_complete": audit["claim_audit"]["goal_complete"],
        "blockers": audit["claim_audit"]["blockers"],
    }, indent=2, sort_keys=True))
    if args.require_complete and not audit["claim_audit"]["goal_complete"]:
        return 2
    return 0


def validate_benchmark(benchmark_dir: Path) -> list[str]:
    required = [
        "benchmark_manifest.json",
        "method_manifest.json",
        "level_manifest.json",
        "verifier_manifest.json",
        "hidden_variant_manifest.json",
        *[
            f"split_manifest_{split}.json"
            for split in (*DEFAULT_HEADLINE_SPLITS, *DEFAULT_ANTI_SHORTCUT_SPLITS)
        ],
        "tasks",
    ]
    missing = [name for name in required if not (benchmark_dir / name).exists()]
    if missing:
        raise SystemExit(f"prepared physics benchmark is incomplete: {missing}")
    manifest = read_json(benchmark_dir / "benchmark_manifest.json")
    blockers: list[str] = []
    if manifest.get("experiment_ready") is not True:
        blockers.append("benchmark_manifest.json does not mark experiment_ready=true")
    methods = read_json(benchmark_dir / "method_manifest.json")
    if methods.get("required_methods") != list(REQUIRED_METHODS):
        blockers.append("method_manifest.json does not match goals.md methods")
    if methods.get("eval_seeds") and methods.get("eval_seeds") != list(EVAL_SEEDS):
        blockers.append("method_manifest.json does not match required eval seeds")
    if methods.get("primary_budget_expensive_verifier_calls") not in (
        None,
        PRIMARY_BUDGET,
    ):
        blockers.append("method_manifest.json does not match primary budget")

    level_manifest = read_json(benchmark_dir / "level_manifest.json")
    hidden_manifest = read_json(benchmark_dir / "hidden_variant_manifest.json")
    verifier_manifest = read_json(benchmark_dir / "verifier_manifest.json")

    tasks = manifest.get("tasks")
    if not isinstance(tasks, list):
        tasks = []
        blockers.append("benchmark_manifest.json does not list tasks")
    level_tasks = level_manifest.get("tasks")
    if not isinstance(level_tasks, list):
        level_tasks = []
        blockers.append("level_manifest.json does not list tasks")
    hidden_tasks = hidden_manifest.get("tasks")
    if not isinstance(hidden_tasks, list):
        hidden_tasks = []
        blockers.append("hidden_variant_manifest.json does not list tasks")

    task_ids = [str(task.get("task_id", "")) for task in tasks if task.get("task_id")]
    task_id_set = set(task_ids)
    if len(task_ids) != len(task_id_set):
        blockers.append("benchmark_manifest.json contains duplicate task ids")
    task_count = len(task_id_set)
    if task_count < MIN_FINAL_TASKS:
        blockers.append(f"benchmark has {task_count} tasks; need at least {MIN_FINAL_TASKS}")
    if int_value(manifest.get("task_count", task_count)) != task_count:
        blockers.append("benchmark_manifest.json task_count does not match listed tasks")

    headline_tasks = [
        task for task in tasks if bool(task.get("headline_eligible", True))
    ]
    headline_count = len(headline_tasks)
    if headline_count < MIN_FINAL_TASKS:
        blockers.append(
            f"benchmark has {headline_count} headline tasks; need at least {MIN_FINAL_TASKS}"
        )
    family_counts: dict[str, int] = {}
    for task in headline_tasks:
        family = str(task.get("family", ""))
        family_counts[family] = family_counts.get(family, 0) + 1
    for family in REQUIRED_FAMILIES:
        count = int(family_counts.get(family, 0))
        if count < MIN_TASKS_PER_FAMILY:
            blockers.append(
                f"{family}: only {count} headline tasks; need {MIN_TASKS_PER_FAMILY}"
            )
    unknown_families = sorted(set(family_counts) - set(REQUIRED_FAMILIES))
    if unknown_families:
        blockers.append(f"unexpected benchmark families: {unknown_families}")

    def verifier_level(task: dict[str, Any]) -> int:
        return int_value(task.get("verifier_level", task.get("level", 0)))

    level2plus = [task for task in headline_tasks if verifier_level(task) >= 2]
    level3 = [task for task in headline_tasks if verifier_level(task) >= 3]
    if headline_tasks and len(level2plus) / len(headline_tasks) < MIN_LEVEL2PLUS_FRACTION:
        blockers.append("Level-2/3 headline task share below 40 percent")
    if headline_tasks and len(level3) / len(headline_tasks) < MIN_LEVEL3_FRACTION:
        blockers.append("Level-3 headline task share below 25 percent")
    if level_manifest.get("headline_levels") != [2, 3]:
        blockers.append("level_manifest.json headline_levels must be [2, 3]")
    level_ids = {str(task.get("task_id", "")) for task in level_tasks if task.get("task_id")}
    hidden_ids = {str(task.get("task_id", "")) for task in hidden_tasks if task.get("task_id")}
    if level_ids != task_id_set:
        blockers.append("level_manifest.json task ids do not match benchmark tasks")
    if hidden_ids != task_id_set:
        blockers.append("hidden_variant_manifest.json task ids do not match benchmark tasks")

    tasks_root = benchmark_dir / "tasks"
    audit_tasks = {}
    audit = manifest.get("audit") if isinstance(manifest.get("audit"), dict) else {}
    for task in audit.get("tasks", []) if isinstance(audit.get("tasks"), list) else []:
        task_id = str(task.get("task_id", ""))
        if task_id:
            audit_tasks[task_id] = task
    if set(audit_tasks) != task_id_set:
        blockers.append("benchmark audit task ids do not match benchmark tasks")
    if audit.get("experiment_ready") is not True:
        blockers.append("benchmark audit does not mark experiment_ready=true")
    if audit.get("paper_blockers"):
        blockers.append("benchmark audit still has paper_blockers")
    if audit.get("blockers"):
        blockers.append("benchmark audit still has blockers")
    if audit.get("structural_blockers"):
        blockers.append("benchmark audit still has structural_blockers")
    if audit.get("validation_blockers"):
        blockers.append("benchmark audit still has validation_blockers")

    for task_id in sorted(task_id_set):
        if not (tasks_root / task_id).is_dir():
            blockers.append(f"{task_id}: task directory missing under tasks/")
        audit_task = audit_tasks.get(task_id, {})
        constraint_classes = audit_task.get("constraint_classes", [])
        if not isinstance(constraint_classes, list) or len(constraint_classes) < 3:
            blockers.append(f"{task_id}: fewer than 3 constraint classes")
        if int_value(audit_task.get("negative_control_count", 0)) < 2:
            blockers.append(f"{task_id}: fewer than 2 negative controls")
        if int_value(audit_task.get("effective_negative_control_count", 0)) < 2:
            blockers.append(f"{task_id}: fewer than 2 effective negative controls")
        if audit_task.get("has_hidden_variant") is not True:
            blockers.append(f"{task_id}: hidden variant missing")
        if audit_task.get("uses_fake_contact_oracle"):
            blockers.append(f"{task_id}: fake_contact_oracle still present")
        validation = audit_task.get("validation", {})
        if not isinstance(validation, dict):
            blockers.append(f"{task_id}: validation record missing")
            continue
        if validation.get("reference_passed") is not True:
            blockers.append(f"{task_id}: reference solution did not pass")
        if validation.get("reference_evaluation_valid") is not True:
            blockers.append(f"{task_id}: reference evaluation invalid")
        if validation.get("reference_hard_gate_passed") is not True:
            blockers.append(f"{task_id}: reference hard gate failed")
        if validation.get("reference_oracle_is_synthetic"):
            blockers.append(f"{task_id}: reference used synthetic oracle")
        if validation.get("negative_failures"):
            blockers.append(f"{task_id}: negative controls failed audit")

    for task in hidden_tasks:
        task_id = str(task.get("task_id", ""))
        if task.get("hidden_variant_present") is not True:
            blockers.append(f"{task_id}: hidden manifest marks variant missing")
        perturbations = task.get("perturbations", [])
        if not isinstance(perturbations, list) or len(perturbations) < 3:
            blockers.append(f"{task_id}: hidden manifest has fewer than 3 perturbations")

    for split in (*DEFAULT_HEADLINE_SPLITS, *DEFAULT_ANTI_SHORTCUT_SPLITS):
        split_manifest = read_json(benchmark_dir / f"split_manifest_{split}.json")
        split_test = split_manifest.get("splits", {}).get("test", [])
        if not isinstance(split_test, list) or not split_test:
            blockers.append(f"split_manifest_{split}.json has no test tasks")
        split_file = benchmark_dir / f"splits_{split}" / "test.txt"
        if not split_file.is_file():
            blockers.append(f"splits_{split}/test.txt is missing")
        elif not [line for line in split_file.read_text().splitlines() if line.strip()]:
            blockers.append(f"splits_{split}/test.txt is empty")
        if split in DEFAULT_HEADLINE_SPLITS:
            seen = set(split_manifest.get("seen_families") or [])
            unseen = set(split_manifest.get("unseen_families") or [])
            if not seen or not unseen or seen & unseen:
                blockers.append(f"split_manifest_{split}.json is not a valid family holdout")

    level3_contract = (
        verifier_manifest.get("level_3_contract")
        if isinstance(verifier_manifest.get("level_3_contract"), dict)
        else {}
    )
    if verifier_manifest.get("main_claim_allows_fake_oracle") is not False:
        blockers.append("verifier_manifest.json must forbid fake-oracle main claims")
    if (
        verifier_manifest.get("requires_real_pychrono") is not True
        and level3_contract.get("requires_real_pychrono") is not True
    ):
        blockers.append("verifier_manifest.json must require real PyChrono")

    return blockers


def build_plan(
    *,
    benchmark_dir: Path,
    out_dir: Path,
    methods: list[str],
    splits: list[str],
    anti_shortcut_splits: list[str],
    seeds: list[int],
    budgets: list[int],
    limit_tasks: int,
    task_index: dict[str, dict[str, Any]],
    shard_index: int = 0,
    num_shards: int = 1,
) -> dict[str, Any]:
    split_tasks: dict[str, list[str]] = {}
    all_splits = unique_list([*splits, *anti_shortcut_splits])
    for split in all_splits:
        task_ids = read_split_tasks(benchmark_dir, split)
        if limit_tasks:
            task_ids = task_ids[:limit_tasks]
        split_tasks[split] = task_ids

    expected_cells = []
    for split in all_splits:
        split_kind = "anti_shortcut" if split in anti_shortcut_splits else "headline"
        for task_id in split_tasks[split]:
            task_meta = task_index.get(task_id, {})
            for seed in seeds:
                for method in methods:
                    for budget in budgets:
                        expected_cells.append({
                            "split": split,
                            "split_kind": split_kind,
                            "task_id": task_id,
                            "family": task_meta.get("family", ""),
                            "verifier_level": int(task_meta.get("verifier_level", 0)),
                            "seed": int(seed),
                            "method": method,
                            "budget": int(budget),
                        })
    full_cell_count = len(expected_cells)
    if num_shards > 1:
        expected_cells = [
            cell for cell in expected_cells
            if cell_shard(cell, num_shards=num_shards) == shard_index
        ]
    return {
        "schema": "mechanism_repair_physics.experiment_plan.v1",
        "hypothesis": (
            "under equal CAD/Chrono verifier budget, online LoRA/GRPO updates "
            "from mechanical verifier feedback improve held-out Level-2/3 "
            "repair success over no-update and search/evolution baselines"
        ),
        "benchmark_dir": str(benchmark_dir),
        "out_dir": str(out_dir),
        "primary_method": PRIMARY_METHOD,
        "primary_baseline": PRIMARY_BASELINE,
        "required_methods": list(REQUIRED_METHODS),
        "methods": methods,
        "missing_required_methods": [
            method for method in REQUIRED_METHODS if method not in set(methods)
        ],
        "headline_splits": splits,
        "anti_shortcut_splits": anti_shortcut_splits,
        "seeds": seeds,
        "missing_required_seeds": [
            seed for seed in EVAL_SEEDS if seed not in set(seeds)
        ],
        "budgets": budgets,
        "primary_budget": PRIMARY_BUDGET,
        "success_delta_threshold_pct": SUCCESS_DELTA_PCT,
        "split_tasks": split_tasks,
        "expected_cells": expected_cells,
        "planned_cells": len(expected_cells),
        "full_planned_cells": full_cell_count,
        "shard_index": int(shard_index),
        "num_shards": int(num_shards),
        "sharded": bool(num_shards > 1),
        "limit_tasks": int(limit_tasks),
        "artifact_contract": {
            "raw_completions": "one or more raw model outputs per cell",
            "verifier_outputs": "one or more verifier reports per cell",
            "cad_artifacts": "required for Level-2 and Level-3 cells",
            "chrono_outputs": "required for Level-3 cells",
            "training_logs": "required for online-update methods",
            "adapter_checkpoints": "required for online-update methods",
        },
    }


def audit_existing_experiment(
    *,
    out_dir: Path,
    plan: dict[str, Any],
    benchmark_readiness_blockers: list[str] | None = None,
) -> dict[str, Any]:
    benchmark_readiness_blockers = list(benchmark_readiness_blockers or [])
    rows = load_rows(out_dir / "cell_results.jsonl")
    row_map: dict[tuple[str, str, int, str, int], dict[str, Any]] = {}
    duplicate_keys: list[str] = []
    row_keys: list[tuple[str, str, int, str, int]] = []
    for row in rows:
        key = row_key(row, default_budget=int(plan["primary_budget"]))
        row_keys.append(key)
        if key in row_map:
            duplicate_keys.append(cell_key_text(key))
        row_map[key] = row

    expected_cell_keys = {
        (
            str(cell["split"]),
            str(cell["task_id"]),
            int(cell["seed"]),
            str(cell["method"]),
            int(cell["budget"]),
        )
        for cell in plan["expected_cells"]
    }
    extra_keys = sorted({
        cell_key_text(key)
        for key in row_keys
        if key not in expected_cell_keys
    })

    missing_cells: list[dict[str, Any]] = []
    budget_mismatches: list[dict[str, Any]] = []
    primary_expensive_budget_excesses: list[dict[str, Any]] = []
    missing_evidence: list[dict[str, Any]] = []
    missing_learning: list[dict[str, Any]] = []
    missing_accounting: list[dict[str, Any]] = []
    for cell in plan["expected_cells"]:
        key = (
            str(cell["split"]),
            str(cell["task_id"]),
            int(cell["seed"]),
            str(cell["method"]),
            int(cell["budget"]),
        )
        row = row_map.get(key)
        if row is None:
            missing_cells.append(cell)
            continue
        verifier_calls = int_value(
            row.get("actual_verifier_calls", row.get("verifier_calls", -1))
        )
        if verifier_calls != int(cell["budget"]):
            budget_mismatches.append({
                "cell": cell,
                "actual_verifier_calls": verifier_calls,
            })
        cad_calls_present = has_any_key(row, ("actual_cad_calls", "cad_audits"))
        chrono_calls_present = has_any_key(row, ("actual_chrono_calls", "chrono_audits"))
        if not cad_calls_present or not chrono_calls_present:
            missing_accounting.append({
                "cell": cell,
                "missing": [
                    name for name, present in (
                        ("actual_cad_calls", cad_calls_present),
                        ("actual_chrono_calls", chrono_calls_present),
                    )
                    if not present
                ],
            })
        # CAD/Chrono calls are factual tool executions, not planned obligations.
        # A failed candidate may be rejected by earlier structural gates before
        # there is a valid CAD model or Chrono scene to run. Those cells are
        # still valid experimental failures as long as the row preserves
        # explicit Level-2/Level-3 evidence paths documenting the unreached
        # audit obligation; missing_evidence_for_row enforces those paths.
        missing_evidence.extend(missing_evidence_for_row(out_dir, cell, row))
        method = str(cell["method"])
        if method in LEARNING_METHODS:
            learning_missing = missing_learning_evidence(
                out_dir,
                row,
                require_rl_evidence=method in TTRL_METHODS,
            )
            if learning_missing:
                missing_learning.append({
                    "cell": cell,
                    "missing": learning_missing,
                })

    expected_groups: set[tuple[str, str, int, int]] = {
        (
            str(cell["split"]),
            str(cell["task_id"]),
            int(cell["seed"]),
            int(cell["budget"]),
        )
        for cell in plan["expected_cells"]
    }
    for split, task_id, seed, budget in sorted(expected_groups):
        primary_key = (split, task_id, seed, str(plan["primary_method"]), budget)
        baseline_key = (split, task_id, seed, str(plan["primary_baseline"]), budget)
        primary_row = row_map.get(primary_key)
        baseline_row = row_map.get(baseline_key)
        if primary_row is None or baseline_row is None:
            continue
        for kind, key_names in (
            ("cad", ("actual_cad_calls", "cad_audits")),
            ("chrono", ("actual_chrono_calls", "chrono_audits")),
        ):
            primary_calls = int_value(first_present(primary_row, key_names, -1))
            baseline_calls = int_value(first_present(baseline_row, key_names, -1))
            if primary_calls > baseline_calls:
                primary_expensive_budget_excesses.append({
                    "cell": {
                        "split": split,
                        "task_id": task_id,
                        "seed": int(seed),
                        "budget": int(budget),
                    },
                    "kind": kind,
                    "primary_method": str(plan["primary_method"]),
                    "primary_calls": primary_calls,
                    "baseline_method": str(plan["primary_baseline"]),
                    "baseline_calls": baseline_calls,
                })

    missing_result_artifacts = [
        name for name in RESULT_ARTIFACTS if not (out_dir / name).is_file()
    ]
    result_bundle_audit = audit_result_bundle(
        out_dir=out_dir,
        rows=rows,
        default_budget=int(plan["primary_budget"]),
    )
    missing_run_dirs = [
        name for name in REQUIRED_RUN_DIRS if not (out_dir / name).is_dir()
    ]
    missing_analysis = [
        name for name in ANALYSIS_ARTIFACTS if not (out_dir / name).is_file()
    ]
    missing_analysis_requirements = analysis_requirement_blockers(
        out_dir,
        anti_shortcut_splits=plan.get("anti_shortcut_splits") or [],
    )
    anti_cells = [
        cell for cell in plan["expected_cells"]
        if cell["split"] in set(plan["anti_shortcut_splits"])
    ]
    missing_anti = [
        cell for cell in anti_cells
        if (
            str(cell["split"]),
            str(cell["task_id"]),
            int(cell["seed"]),
            str(cell["method"]),
            int(cell["budget"]),
        ) not in row_map
    ]

    budget_audit = {
        "schema": "mechanism_repair_physics.budget_audit.v1",
        "primary_budget": int(plan["primary_budget"]),
        "planned_cells": int(plan["planned_cells"]),
        "full_planned_cells": int(plan.get("full_planned_cells", plan["planned_cells"])),
        "shard_index": int(plan.get("shard_index", 0)),
        "num_shards": int(plan.get("num_shards", 1)),
        "observed_rows": len(rows),
        "missing_cell_count": len(missing_cells),
        "extra_cell_count": len(extra_keys),
        "duplicate_cell_count": len(duplicate_keys),
        "budget_mismatch_count": len(budget_mismatches),
        "primary_expensive_budget_excess_count": (
            len(primary_expensive_budget_excesses)
        ),
        "missing_accounting_count": len(missing_accounting),
        "budget_matched": (
            not missing_cells
            and not extra_keys
            and not duplicate_keys
            and not budget_mismatches
            and not primary_expensive_budget_excesses
            and not missing_accounting
        ),
        "sample_missing_cells": missing_cells[:25],
        "sample_extra_cells": extra_keys[:25],
        "sample_duplicate_cells": duplicate_keys[:25],
        "sample_budget_mismatches": budget_mismatches[:25],
        "sample_primary_expensive_budget_excesses": (
            primary_expensive_budget_excesses[:25]
        ),
        "sample_missing_accounting": missing_accounting[:25],
    }
    anti_shortcut_audit = {
        "schema": "mechanism_repair_physics.anti_shortcut_audit.v1",
        "anti_shortcut_splits": list(plan["anti_shortcut_splits"]),
        "expected_anti_shortcut_cells": len(anti_cells),
        "missing_anti_shortcut_cells": len(missing_anti),
        "anti_shortcut_executed": bool(anti_cells) and not missing_anti,
        "sample_missing_anti_shortcut_cells": missing_anti[:25],
    }
    blockers = build_blockers(
        budget_audit=budget_audit,
        anti_shortcut_audit=anti_shortcut_audit,
        missing_evidence=missing_evidence,
        missing_learning=missing_learning,
        missing_result_artifacts=missing_result_artifacts,
        result_bundle_audit=result_bundle_audit,
        missing_run_dirs=missing_run_dirs,
        missing_analysis=missing_analysis,
        missing_analysis_requirements=missing_analysis_requirements,
    )
    if plan.get("missing_required_methods"):
        blockers.insert(0, "execute all required methods, not a smoke subset")
    if plan.get("missing_required_seeds"):
        blockers.insert(0, "execute all required evaluation seeds")
    if benchmark_readiness_blockers:
        blockers.insert(0, "fix benchmark readiness before claiming paper result")
    claim_audit = {
        "schema": "mechanism_repair_physics.experiment_claim_audit.v1",
        "goal_complete": not blockers,
        "benchmark_dir": plan["benchmark_dir"],
        "benchmark_readiness_blockers": benchmark_readiness_blockers,
        "planned_cells": int(plan["planned_cells"]),
        "full_planned_cells": int(plan.get("full_planned_cells", plan["planned_cells"])),
        "shard_index": int(plan.get("shard_index", 0)),
        "num_shards": int(plan.get("num_shards", 1)),
        "observed_rows": len(rows),
        "blockers": blockers,
        "missing_before_paper_claim": blockers,
        "missing_evidence_count": len(missing_evidence),
        "missing_learning_count": len(missing_learning),
        "missing_result_artifacts": missing_result_artifacts,
        "result_bundle_audit": result_bundle_audit,
        "missing_run_dirs": missing_run_dirs,
        "missing_analysis_artifacts": missing_analysis,
        "missing_analysis_requirements": missing_analysis_requirements,
        "sample_missing_evidence": missing_evidence[:25],
        "sample_missing_learning": missing_learning[:25],
    }
    if not blockers:
        claim_audit["claim_status"] = infer_claim_status(out_dir)
    return {
        "budget_audit": budget_audit,
        "anti_shortcut_audit": anti_shortcut_audit,
        "claim_audit": claim_audit,
    }


def public_plan(plan: dict[str, Any]) -> dict[str, Any]:
    """Return a compact, reconstructable plan for checked-in artifacts."""
    out = dict(plan)
    out.pop("expected_cells", None)
    out["expected_cells_materialized_in_audit"] = False
    return out


def write_shard_files(
    *,
    out_dir: Path,
    plan: dict[str, Any],
    num_shards: int,
) -> None:
    if num_shards < 1:
        raise SystemExit("--write-shard-files must be >= 1")
    shard_dir = out_dir / "experiment_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    cells = list(plan.get("expected_cells", []) or [])
    shard_payloads = []
    for shard_index in range(num_shards):
        shard_cells = [
            cell for cell in cells
            if cell_shard(cell, num_shards=num_shards) == shard_index
        ]
        payload = {
            "schema": "mechanism_repair_physics.experiment_shard.v1",
            "benchmark_dir": plan["benchmark_dir"],
            "out_dir": plan["out_dir"],
            "num_shards": int(num_shards),
            "shard_index": int(shard_index),
            "planned_cells": len(shard_cells),
            "full_planned_cells": len(cells),
            "cells": shard_cells,
            "replay_args": [
                "--benchmark-dir",
                plan["benchmark_dir"],
                "--out-dir",
                plan["out_dir"],
                "--methods",
                ",".join(plan["methods"]),
                "--splits",
                ",".join(plan["headline_splits"]),
                "--anti-shortcut-splits",
                ",".join(plan["anti_shortcut_splits"]),
                "--eval-seeds",
                ",".join(str(seed) for seed in plan["seeds"]),
                "--budgets",
                ",".join(str(budget) for budget in plan["budgets"]),
                "--num-shards",
                str(num_shards),
                "--shard-index",
                str(shard_index),
            ],
        }
        shard_payloads.append(payload)
    validate_shard_payloads(shard_payloads)
    temp_targets: list[tuple[Path, Path]] = []
    target_names: set[str] = set()
    temp_token = f"{os.getpid()}-{id(shard_payloads)}"
    try:
        for payload in shard_payloads:
            target = shard_dir / f"shard_{int(payload['shard_index']):04d}.json"
            temp = shard_dir / f".{target.name}.{temp_token}.tmp"
            temp_targets.append((temp, target))
            target_names.add(target.name)
            write_json(temp, payload)
        for temp, target in temp_targets:
            temp.replace(target)
        for stale in shard_dir.glob("shard_*.json"):
            if stale.name not in target_names:
                stale.unlink()
    finally:
        for temp, _target in temp_targets:
            if temp.exists():
                temp.unlink()


def validate_shard_payloads(payloads: list[dict[str, Any]]) -> None:
    bad_assignments: list[dict[str, Any]] = []
    group_shards: dict[tuple[str, str, int, int], set[int]] = {}
    for payload in payloads:
        num_shards = int(payload["num_shards"])
        shard_index = int(payload["shard_index"])
        for cell in payload.get("cells", []) or []:
            expected = cell_shard(cell, num_shards=num_shards)
            if expected != shard_index:
                bad_assignments.append({
                    "task_id": cell.get("task_id"),
                    "split": cell.get("split"),
                    "seed": cell.get("seed"),
                    "method": cell.get("method"),
                    "budget": cell.get("budget"),
                    "actual_shard": shard_index,
                    "expected_shard": expected,
                })
            group_shards.setdefault(cell_group_key(cell), set()).add(shard_index)
    split_groups = [
        {
            "split": split,
            "task_id": task_id,
            "seed": seed,
            "budget": budget,
            "shards": sorted(shards),
        }
        for (split, task_id, seed, budget), shards in group_shards.items()
        if len(shards) > 1
    ]
    if bad_assignments or split_groups:
        raise SystemExit(
            "invalid experiment shard partition: "
            + json.dumps(
                {
                    "bad_assignments": bad_assignments[:10],
                    "split_groups": split_groups[:10],
                },
                sort_keys=True,
            )
        )


def cell_group_key(cell: dict[str, Any]) -> tuple[str, str, int, int]:
    return (
        str(cell["split"]),
        str(cell["task_id"]),
        int(cell["seed"]),
        int(cell["budget"]),
    )


def cell_shard(cell: dict[str, Any], *, num_shards: int) -> int:
    split, task_id, seed, budget = cell_group_key(cell)
    key = json.dumps(
        {
            "split": split,
            "task_id": task_id,
            "seed": seed,
            "budget": budget,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % int(num_shards)


def build_blockers(
    *,
    budget_audit: dict[str, Any],
    anti_shortcut_audit: dict[str, Any],
    missing_evidence: list[dict[str, Any]],
    missing_learning: list[dict[str, Any]],
    missing_result_artifacts: list[str],
    result_bundle_audit: dict[str, Any],
    missing_run_dirs: list[str],
    missing_analysis: list[str],
    missing_analysis_requirements: list[str],
) -> list[str]:
    blockers: list[str] = []
    if budget_audit["missing_cell_count"]:
        blockers.append(
            "execute all required methods for all planned seeds/splits/tasks"
        )
    if budget_audit.get("extra_cell_count", 0):
        blockers.append("remove unplanned method/seed/task/budget cells")
    if budget_audit["duplicate_cell_count"]:
        blockers.append("deduplicate repeated method/seed/task/budget cells")
    if (
        budget_audit["budget_mismatch_count"]
        or budget_audit.get("primary_expensive_budget_excess_count", 0)
        or budget_audit["missing_accounting_count"]
    ):
        blockers.append("prove matched actual verifier/CAD/Chrono budget")
    if missing_evidence:
        blockers.append("record raw completions and verifier/CAD/Chrono outputs")
    if missing_learning:
        blockers.append("preserve training logs and adapter checkpoints for TTRL cells")
    if (
        missing_result_artifacts
        or missing_run_dirs
        or not result_bundle_audit.get("consistent", False)
    ):
        blockers.append("write final result bundle files and artifact directories")
    if not anti_shortcut_audit["anti_shortcut_executed"]:
        blockers.append("run hidden/isomorphic anti-shortcut variants")
    if missing_analysis:
        blockers.append(
            "write statistical, failure, trace-pair, repair-taxonomy, "
            "anti-shortcut, and budget analyses"
        )
    if missing_analysis_requirements:
        blockers.append(
            "compute required primary, hidden, anti-shortcut, family, and "
            "baseline statistical fields"
        )
    return blockers


def analysis_requirement_blockers(
    out_dir: Path,
    *,
    anti_shortcut_splits: list[str] | tuple[str, ...] | None = None,
) -> list[str]:
    stats_path = out_dir / "stats.json"
    if not stats_path.is_file():
        return []
    stats = read_json(stats_path)
    blockers: list[str] = []
    if stats.get("headline_metric_filter") != "verifier_level>=2":
        blockers.append("headline_metric_filter.verifier_level>=2")
    if int_value(stats.get("headline_metric_rows", 0)) <= 0:
        blockers.append("headline_metric_rows")
    if stats.get("primary_method") != PRIMARY_METHOD:
        blockers.append("primary_method")
    if stats.get("primary_baseline") != PRIMARY_BASELINE:
        blockers.append("primary_baseline")
    if int_value(stats.get("primary_budget_verifier_calls", 0)) != PRIMARY_BUDGET:
        blockers.append("primary_budget_verifier_calls")
    if int_value(stats.get("n_paired_cells", 0)) <= 0:
        blockers.append("n_paired_cells")
    contract = stats.get("analysis_contract") or {}
    if not isinstance(contract, dict):
        contract = {}
        blockers.append("analysis_contract")
    if contract.get("primary_method") != PRIMARY_METHOD:
        blockers.append("analysis_contract.primary_method")
    if contract.get("primary_baseline") != PRIMARY_BASELINE:
        blockers.append("analysis_contract.primary_baseline")
    if int_value(contract.get("primary_budget", 0)) != PRIMARY_BUDGET:
        blockers.append("analysis_contract.primary_budget")
    comparison = stats.get("primary_comparison") or {}
    for key in (
        "success_delta_pct",
        "success_delta_ci95",
        "success_sign_test_p_one_sided",
        "reward_delta_mean",
    ):
        if key not in comparison:
            blockers.append(f"primary_comparison.{key}")
    primary_table = stats.get("primary_result_table")
    required_table_fields = (
        "method",
        "level23_verified_repair_success_at_32",
        "hidden_variant_success_at_32",
        "anti_shortcut_pass_rate_at_32",
        "best_verified_reward_at_32",
        "actual_verifier_calls",
        "actual_cad_calls",
        "actual_chrono_calls",
    )
    if not isinstance(primary_table, list) or not primary_table:
        blockers.append("primary_result_table")
    else:
        missing_table_fields = sorted({
            field
            for row in primary_table
            if isinstance(row, dict)
            for field in required_table_fields
            if field not in row
        })
        if missing_table_fields:
            blockers.append(
                "primary_result_table fields: " + ", ".join(missing_table_fields)
            )
        missing_table_methods = sorted(
            set(REQUIRED_METHODS)
            - {
                str(row.get("method", ""))
                for row in primary_table
                if isinstance(row, dict)
            }
        )
        if missing_table_methods:
            blockers.append(
                "primary_result_table methods: "
                + ", ".join(missing_table_methods)
            )
    method_summary = stats.get("method_summary") or {}
    if not isinstance(method_summary, dict) or not method_summary:
        blockers.append("method_summary")
    else:
        missing_summary_methods = sorted(set(REQUIRED_METHODS) - set(method_summary))
        if missing_summary_methods:
            blockers.append(
                "method_summary methods: " + ", ".join(missing_summary_methods)
            )
        missing_secondary = sorted({
            metric
            for summary in method_summary.values()
            if isinstance(summary, dict)
            for metrics in [summary.get("secondary_metrics") or {}]
            for metric in REQUIRED_SECONDARY_METRICS
            if (
                metric not in metrics
                or not isinstance(metrics.get(metric), dict)
                or "n_present" not in metrics[metric]
                or "mean" not in metrics[metric]
            )
        })
        if missing_secondary:
            blockers.append(
                "method_summary.secondary_metrics: "
                + ", ".join(missing_secondary)
            )
    split_deltas = stats.get("split_deltas")
    if not isinstance(split_deltas, list):
        split_deltas = []
        blockers.append("split_deltas")
    reported_split_deltas = {
        str(row.get("split", ""))
        for row in split_deltas
        if isinstance(row, dict) and "success_delta" in row
    }
    required_anti_shortcut_splits = tuple(
        str(split)
        for split in (anti_shortcut_splits or DEFAULT_ANTI_SHORTCUT_SPLITS)
    )
    for split in required_anti_shortcut_splits:
        if split not in reported_split_deltas:
            blockers.append(f"split_deltas.{split}.success_delta")
    anti = stats.get("anti_shortcut_comparison") or {}
    if "anti_shortcut_pass_rate_delta" not in anti:
        blockers.append("anti_shortcut_comparison.anti_shortcut_pass_rate_delta")
    anti_splits = {str(split) for split in anti.get("splits", []) or []}
    missing_anti_splits = sorted(set(required_anti_shortcut_splits) - anti_splits)
    if missing_anti_splits:
        blockers.append(
            "anti_shortcut_comparison.splits: "
            + ", ".join(missing_anti_splits)
        )
    if int_value(anti.get("n_paired_cells", 0)) <= 0:
        blockers.append("anti_shortcut_comparison.n_paired_cells")
    paired = stats.get("paired_method_comparisons") or {}
    for method in ("adaptive_evolution", "verifier_gated_search"):
        method_row = paired.get(method) or {}
        if "primary_beats_on_success" not in method_row:
            blockers.append(
                f"paired_method_comparisons.{method}.primary_beats_on_success"
            )
    family_deltas = stats.get("family_deltas")
    if not isinstance(family_deltas, list) or len(family_deltas) < 12:
        blockers.append("family_deltas at least 12 families")
    elif any(
        not isinstance(row, dict) or "success_delta" not in row
        for row in family_deltas
    ):
        blockers.append("family_deltas.success_delta")
    leave_one = stats.get("leave_one_family_out")
    if not isinstance(leave_one, list) or len(leave_one) < 12:
        blockers.append("leave_one_family_out at least 12 families")
    elif any(
        not isinstance(row, dict) or "keeps_positive_success_delta" not in row
        for row in leave_one
    ):
        blockers.append("leave_one_family_out.keeps_positive_success_delta")
    required_trace_pairs = int_value(
        (stats.get("evidence_audit") or {}).get("required_min_trace_pairs", 24)
    )
    trace_pairs_path = out_dir / "trace_pairs.json"
    if trace_pairs_path.is_file():
        trace_pairs = read_json(trace_pairs_path).get("pairs")
        if not isinstance(trace_pairs, list):
            blockers.append("trace_pairs.pairs")
        elif len(trace_pairs) < required_trace_pairs:
            blockers.append(f"trace_pairs at least {required_trace_pairs} pairs")
    failure_path = out_dir / "failure_analysis.json"
    if failure_path.is_file():
        failure = read_json(failure_path)
        required_failure_list_fields = (
            "failure_code_transition_matrix",
            "first_to_final_attempt_changes",
            "ttrl_vs_no_update_failure_deltas",
            "repair_dimension_deltas",
            "adapter_update_timeline",
        )
        for field in required_failure_list_fields:
            if field not in failure:
                blockers.append(f"failure_analysis.{field}")
                continue
            value = failure.get(field)
            if not isinstance(value, list):
                blockers.append(f"failure_analysis.{field} list")
            elif not value:
                blockers.append(f"failure_analysis.{field} nonempty")
        if "hidden_perturbation_failure_analysis" not in failure:
            blockers.append("failure_analysis.hidden_perturbation_failure_analysis")
        hidden_failure = failure.get("hidden_perturbation_failure_analysis")
        if not isinstance(hidden_failure, dict) or hidden_failure.get("split") != "hidden_perturbation":
            blockers.append("failure_analysis.hidden_perturbation_failure_analysis")
        else:
            if int_value(hidden_failure.get("rows", 0)) <= 0:
                blockers.append(
                    "failure_analysis.hidden_perturbation_failure_analysis.rows"
                )
            if not isinstance(hidden_failure.get("failure_counts"), list):
                blockers.append(
                    "failure_analysis.hidden_perturbation_failure_analysis.failure_counts"
                )
    taxonomy_path = out_dir / "repair_taxonomy.json"
    if taxonomy_path.is_file():
        taxonomy = read_json(taxonomy_path)
        required_dims = set(REQUIRED_REPAIR_TAXONOMY_DIMENSIONS)
        observed_dims = set(taxonomy.get("required_goal_dimensions") or [])
        missing_dims = sorted(required_dims - observed_dims)
        if missing_dims:
            blockers.append(
                "repair_taxonomy.required_goal_dimensions: "
                + ", ".join(missing_dims)
            )
        goal_counts = taxonomy.get("goal_dimension_counts")
        if not isinstance(goal_counts, list):
            blockers.append("repair_taxonomy.goal_dimension_counts")
        else:
            count_dims = {
                str(row.get("dimension", ""))
                for row in goal_counts
                if isinstance(row, dict)
            }
            missing_count_dims = sorted(required_dims - count_dims)
            if missing_count_dims:
                blockers.append(
                    "repair_taxonomy.goal_dimension_counts: "
                    + ", ".join(missing_count_dims)
                )
            if any(
                not isinstance(row, dict)
                or not row.get("dimension")
                or "n" not in row
                for row in goal_counts
            ):
                blockers.append("repair_taxonomy.goal_dimension_counts rows")
        if not isinstance(taxonomy.get("dimension_map"), dict):
            blockers.append("repair_taxonomy.dimension_map")
    claim = stats.get("analysis_claim_audit") or {}
    claim_status = claim.get("claim_status")
    if claim_status not in {
        "supports_primary_hypothesis",
        "does_not_support_primary_hypothesis",
    }:
        blockers.append("analysis_claim_audit.claim_status")
    else:
        claim_blockers = claim.get("blockers")
        if not isinstance(claim_blockers, list):
            blockers.append("analysis_claim_audit.blockers")
            claim_blockers = []
        explicit_status = stats.get("claim_status")
        if explicit_status and explicit_status != claim_status:
            blockers.append("claim_status disagrees with analysis_claim_audit")
        if claim_status == "supports_primary_hypothesis":
            if claim_blockers:
                blockers.append("analysis_claim_audit.supported_with_blockers")
            positive_blockers = positive_claim_support_blockers(stats)
            if positive_blockers:
                blockers.append(
                    "analysis_claim_audit.unsupported_positive_claim: "
                    + ", ".join(positive_blockers)
                )
        elif not claim_blockers:
            blockers.append("analysis_claim_audit.unsupported_without_blockers")
    return blockers


def positive_claim_support_blockers(stats: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    comparison = stats.get("primary_comparison") or {}
    delta = float_value(comparison.get("success_delta_pct", 0.0))
    if delta < SUCCESS_DELTA_PCT:
        blockers.append("primary_comparison.success_delta_pct")
    ci_low = ci_low_value(comparison.get("success_delta_ci95"))
    p_value = float_value(comparison.get("success_sign_test_p_one_sided", 1.0))
    if ci_low <= 0.0 or p_value > 0.05:
        blockers.append("primary_comparison.statistical_support")
    if float_value(comparison.get("reward_delta_mean", 0.0)) <= 0.0:
        blockers.append("primary_comparison.reward_delta_mean")
    hidden_delta = split_success_delta(stats, "hidden_perturbation")
    if hidden_delta is None or hidden_delta <= 0.0:
        blockers.append("split_deltas.hidden_perturbation.success_delta")
    anti = stats.get("anti_shortcut_comparison") or {}
    if float_value(anti.get("anti_shortcut_pass_rate_delta", 0.0)) <= 0.0:
        blockers.append("anti_shortcut_comparison.anti_shortcut_pass_rate_delta")
    paired = stats.get("paired_method_comparisons") or {}
    for method in ("adaptive_evolution", "verifier_gated_search"):
        method_row = paired.get(method) or {}
        if method_row.get("primary_beats_on_success") is not True:
            blockers.append(
                f"paired_method_comparisons.{method}.primary_beats_on_success"
            )
    family_deltas = stats.get("family_deltas") or []
    positive_families = sum(
        1
        for row in family_deltas
        if isinstance(row, dict) and float_value(row.get("success_delta", 0.0)) > 0.0
    )
    if positive_families < MIN_POSITIVE_FAMILIES:
        blockers.append("family_deltas.positive_success_delta_count")
    leave_one = stats.get("leave_one_family_out") or []
    if not leave_one or any(
        not isinstance(row, dict)
        or row.get("keeps_positive_success_delta") is not True
        for row in leave_one
    ):
        blockers.append("leave_one_family_out.keeps_positive_success_delta")
    return blockers


def ci_low_value(raw_ci: Any) -> float:
    if isinstance(raw_ci, dict):
        return float_value(raw_ci.get("low", 0.0))
    if isinstance(raw_ci, list) and raw_ci:
        return float_value(raw_ci[0])
    return 0.0


def split_success_delta(stats: dict[str, Any], split: str) -> float | None:
    split_deltas = stats.get("split_deltas") or []
    for row in split_deltas:
        if isinstance(row, dict) and row.get("split") == split:
            return float_value(row.get("success_delta", 0.0))
    return None


def missing_evidence_for_row(
    out_dir: Path,
    cell: dict[str, Any],
    row: dict[str, Any],
) -> list[dict[str, Any]]:
    checks = [
        ("raw_completions", collect_paths(row, "raw_completion_paths", "raw_completion_path")),
        ("verifier_outputs", collect_paths(row, "verifier_output_paths", "verifier_output_path")),
    ]
    if int(cell["verifier_level"]) >= 2:
        checks.append(("cad_artifacts", collect_paths(row, "cad_artifact_paths", "cad_artifact_path")))
    if int(cell["verifier_level"]) >= 3:
        checks.append(("chrono_outputs", collect_paths(row, "chrono_output_paths", "chrono_output_path")))
    missing: list[dict[str, Any]] = []
    for label, paths in checks:
        if not paths:
            missing.append({"cell": cell, "kind": label, "reason": "no_paths"})
            continue
        absent = [path for path in paths if not resolve_path(out_dir, path).is_file()]
        if absent:
            missing.append({"cell": cell, "kind": label, "missing_paths": absent[:8]})
    return missing


def missing_learning_evidence(
    out_dir: Path,
    row: dict[str, Any],
    *,
    require_rl_evidence: bool,
) -> list[str]:
    missing: list[str] = []
    log_paths = collect_paths(row, "training_log_paths", "training_log_path")
    if not log_paths and not require_rl_evidence:
        log_paths = infer_sft_training_log_paths(out_dir, row)
    ckpt_paths = collect_paths(row, "adapter_checkpoint_paths", "adapter_checkpoint_path")
    if not log_paths or any(not resolve_path(out_dir, path).exists() for path in log_paths):
        missing.append("training_logs")
    if not ckpt_paths or any(not resolve_path(out_dir, path).exists() for path in ckpt_paths):
        missing.append("adapter_checkpoints")
    elif any(not adapter_checkpoint_has_weights(resolve_path(out_dir, path)) for path in ckpt_paths):
        missing.append("adapter_checkpoint_weights")
    updates = int_value(row.get("adapter_updates", row.get("online_update_steps", 0)))
    if updates <= 0:
        missing.append("adapter_updates")
    trained_tokens = int_value(row.get("trained_tokens", 0))
    if trained_tokens <= 0:
        missing.append("trained_tokens")
    if require_rl_evidence:
        rl_datums = int_value(row.get("n_rl_datums", row.get("rl_datums", 0)))
        if rl_datums <= 0:
            missing.append("rl_datums")
        rl_trained_tokens = int_value(row.get("rl_trained_tokens", 0))
        if rl_trained_tokens <= 0:
            missing.append("rl_trained_tokens")
    return missing


def infer_sft_training_log_paths(out_dir: Path, row: dict[str, Any]) -> list[str]:
    inferred: list[str] = []
    raw_paths = [
        *collect_paths(row, "adapter_checkpoint_paths", "adapter_checkpoint_path"),
        *collect_paths(row, "adapter_paths", "adapter_path"),
    ]
    for raw_path in raw_paths:
        adapter_path = resolve_path(out_dir, raw_path)
        base = adapter_path.parent if adapter_path.name == "final_adapter" else adapter_path
        for candidate in (base / "run_manifest.json", base / "train_sft.jsonl"):
            if candidate.is_file():
                inferred.append(str(candidate))
    return unique_list(inferred)


def adapter_checkpoint_has_weights(path: Path) -> bool:
    if path.is_file():
        return is_adapter_weight_file(path)
    if not path.is_dir():
        return False
    return any(is_adapter_weight_file(item) for item in path.rglob("*") if item.is_file())


def is_adapter_weight_file(path: Path) -> bool:
    name = path.name
    return (
        name in {"adapter_model.safetensors", "adapter_model.bin"}
        or (path.suffix == ".safetensors" and "adapter" in name)
    )


def infer_claim_status(out_dir: Path) -> str:
    stats = read_json(out_dir / "stats.json")
    analysis_claim = stats.get("analysis_claim_audit") or {}
    strict = analysis_claim.get("claim_status")
    if strict in {
        "supports_primary_hypothesis",
        "does_not_support_primary_hypothesis",
    }:
        return str(strict)
    explicit = stats.get("claim_status")
    if explicit in {
        "supports_primary_hypothesis",
        "does_not_support_primary_hypothesis",
    }:
        return str(explicit)
    comparison = stats.get("primary_comparison") or {}
    delta = float_value(comparison.get("success_delta_pct", 0.0))
    ci = comparison.get("success_delta_ci95") or [0.0, 0.0]
    ci_low = float_value(
        ci.get("low", 0.0)
        if isinstance(ci, dict)
        else ci[0] if isinstance(ci, list) and ci else 0.0
    )
    p_value = float_value(comparison.get("success_sign_test_p_one_sided", 1.0))
    if delta >= SUCCESS_DELTA_PCT and ci_low > 0.0 and p_value <= 0.05:
        return "supports_primary_hypothesis"
    return "does_not_support_primary_hypothesis"


def load_task_index(benchmark_dir: Path) -> dict[str, dict[str, Any]]:
    manifest = read_json(benchmark_dir / "level_manifest.json")
    out: dict[str, dict[str, Any]] = {}
    for row in manifest.get("tasks", []) or []:
        task_id = str(row.get("task_id", ""))
        if task_id:
            out[task_id] = dict(row)
    return out


def read_split_tasks(benchmark_dir: Path, split: str) -> list[str]:
    split_file = benchmark_dir / f"splits_{split}" / "test.txt"
    if not split_file.is_file():
        raise SystemExit(f"missing split test file: {split_file}")
    tasks = []
    for line in split_file.read_text().splitlines():
        item = line.strip()
        if not item:
            continue
        tasks.append(Path(item).name)
    return tasks


def load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def audit_result_bundle(
    *,
    out_dir: Path,
    rows: list[dict[str, Any]],
    default_budget: int,
) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    expected_counts = row_key_counter(rows, default_budget=default_budget)
    expected_json_signatures = row_signature_counter(
        rows,
        default_budget=default_budget,
        mode="json",
    )
    expected_csv_fields = sorted({key for row in rows for key in row})
    expected_csv_signatures = row_signature_counter(
        rows,
        default_budget=default_budget,
        mode="csv",
        fields=expected_csv_fields,
    )
    for artifact, path, loader in (
        ("results.json", out_dir / "results.json", load_result_json_rows),
        ("results.csv", out_dir / "results.csv", load_result_csv_rows),
    ):
        if not path.is_file():
            continue
        try:
            artifact_rows = loader(path)
        except (OSError, ValueError, csv.Error) as exc:
            errors.append({
                "artifact": artifact,
                "reason": "parse_error",
                "error": str(exc),
            })
            continue
        if not isinstance(artifact_rows, list):
            errors.append({"artifact": artifact, "reason": "rows_not_list"})
            continue
        artifact_counts = row_key_counter(
            artifact_rows,
            default_budget=default_budget,
        )
        row_keys_match = (
            len(artifact_rows) == len(rows) and artifact_counts == expected_counts
        )
        if not row_keys_match:
            errors.append({
                "artifact": artifact,
                "reason": "row_key_mismatch",
                "jsonl_rows": len(rows),
                "artifact_rows": len(artifact_rows),
                "missing_from_artifact": counter_sample(
                    expected_counts - artifact_counts
                ),
                "extra_in_artifact": counter_sample(
                    artifact_counts - expected_counts
                ),
            })
            continue
        if artifact == "results.csv":
            artifact_fields = sorted({
                key
                for row in artifact_rows
                for key in row
                if key is not None
            })
            if artifact_fields != expected_csv_fields:
                errors.append({
                    "artifact": artifact,
                    "reason": "csv_fields_mismatch",
                    "missing_fields": sorted(
                        set(expected_csv_fields) - set(artifact_fields)
                    ),
                    "extra_fields": sorted(
                        set(artifact_fields) - set(expected_csv_fields)
                    ),
                })
                continue
            artifact_signatures = row_signature_counter(
                artifact_rows,
                default_budget=default_budget,
                mode="csv",
                fields=expected_csv_fields,
            )
            expected_signatures = expected_csv_signatures
        else:
            artifact_signatures = row_signature_counter(
                artifact_rows,
                default_budget=default_budget,
                mode="json",
            )
            expected_signatures = expected_json_signatures
        if artifact_signatures != expected_signatures:
            errors.append({
                "artifact": artifact,
                "reason": "row_payload_mismatch",
                "mismatched_jsonl_rows": signature_counter_sample(
                    expected_signatures - artifact_signatures
                ),
                "mismatched_artifact_rows": signature_counter_sample(
                    artifact_signatures - expected_signatures
                ),
            })
    artifacts_present = all((out_dir / name).is_file() for name in RESULT_ARTIFACTS)
    return {
        "schema": "mechanism_repair_physics.result_bundle_audit.v1",
        "artifacts_present": artifacts_present,
        "jsonl_rows": len(rows),
        "consistent": artifacts_present and not errors,
        "error_count": len(errors),
        "errors": errors[:25],
    }


def load_result_json_rows(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return [dict(row) for row in data]
    if isinstance(data, dict) and isinstance(data.get("rows"), list):
        return [dict(row) for row in data["rows"]]
    raise ValueError("results JSON does not contain rows")


def load_result_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def row_key_counter(
    rows: list[dict[str, Any]],
    *,
    default_budget: int,
) -> Counter[str]:
    return Counter(
        cell_key_text(row_key(row, default_budget=default_budget))
        for row in rows
    )


def row_signature_counter(
    rows: list[dict[str, Any]],
    *,
    default_budget: int,
    mode: str,
    fields: list[str] | None = None,
) -> Counter[str]:
    return Counter(
        row_signature(
            row,
            default_budget=default_budget,
            mode=mode,
            fields=fields,
        )
        for row in rows
    )


def row_signature(
    row: dict[str, Any],
    *,
    default_budget: int,
    mode: str,
    fields: list[str] | None = None,
) -> str:
    key = cell_key_text(row_key(row, default_budget=default_budget))
    if mode == "json":
        payload = json.dumps(row, sort_keys=True, separators=(",", ":"))
    elif mode == "csv":
        if fields is None:
            raise ValueError("CSV row signatures require fields")
        payload = json.dumps(
            {field: csv_cell_text(row.get(field, "")) for field in fields},
            sort_keys=True,
            separators=(",", ":"),
        )
    else:
        raise ValueError(f"unknown row signature mode: {mode}")
    return f"{key}\t{payload}"


def csv_cell_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value)
    return str(value)


def counter_sample(counts: Counter[str], limit: int = 25) -> list[str]:
    sample: list[str] = []
    for key, count in sorted(counts.items()):
        sample.extend([key] * int(count))
        if len(sample) >= limit:
            return sample[:limit]
    return sample


def signature_counter_sample(counts: Counter[str], limit: int = 25) -> list[str]:
    sample: list[str] = []
    for signature, count in sorted(counts.items()):
        key = signature.split("\t", 1)[0]
        sample.extend([key] * int(count))
        if len(sample) >= limit:
            return sample[:limit]
    return sample


def row_key(
    row: dict[str, Any],
    *,
    default_budget: int,
) -> tuple[str, str, int, str, int]:
    return (
        str(row.get("split") or row.get("split_name") or ""),
        str(row.get("task_id") or ""),
        int_value(row.get("seed", 0)),
        str(row.get("method") or ""),
        int_value(row.get("budget", row.get("budget_verifier_calls", default_budget))),
    )


def cell_key_text(key: tuple[str, str, int, str, int]) -> str:
    split, task_id, seed, method, budget = key
    return f"{split}/{task_id}/seed{seed}/{method}/budget{budget}"


def collect_paths(row: dict[str, Any], list_key: str, scalar_key: str) -> list[str]:
    raw = row.get(list_key)
    if raw is None:
        raw = row.get(scalar_key)
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw] if raw else []
    if isinstance(raw, list):
        return [str(item) for item in raw if str(item)]
    return []


def resolve_path(out_dir: Path, raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        return out_dir / path
    if path.exists():
        return path
    out_dir = out_dir.resolve()
    parts = path.parts
    for index in range(len(parts) - 1, -1, -1):
        if parts[index] == out_dir.name:
            return out_dir.joinpath(*parts[index + 1 :])
    return path


def parse_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def unique_list(items: list[str]) -> list[str]:
    out: list[str] = []
    for item in items:
        if item not in out:
            out.append(item)
    return out


def has_any_key(row: dict[str, Any], keys: tuple[str, ...]) -> bool:
    return any(key in row for key in keys)


def first_present(
    row: dict[str, Any],
    keys: tuple[str, ...],
    default: Any = None,
) -> Any:
    for key in keys:
        if key in row:
            return row[key]
    return default


def int_value(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def float_value(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
