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
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.prepare_mechanism_repair_physics_benchmark import (
    EVAL_SEEDS,
    PRIMARY_BASELINE,
    PRIMARY_BUDGET,
    PRIMARY_METHOD,
    REQUIRED_METHODS,
    SUCCESS_DELTA_PCT,
    ensure_run_scaffold,
)


DEFAULT_BENCHMARK_DIR = "runs/mechanism_repair_physics_final"
DEFAULT_HEADLINE_SPLITS = ("A", "B")
DEFAULT_ANTI_SHORTCUT_SPLITS = ("hidden_perturbation", "external_style")
TTRL_METHODS = {
    "mechanical_evolve_ttrl",
    "mechanical_evolve_ttrl_tool_verified",
    "mechanical_evolve_ttrl_confidence",
}
ANALYSIS_ARTIFACTS = (
    "stats.json",
    "failure_analysis.json",
    "trace_pairs.json",
    "repair_taxonomy.json",
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

    validate_benchmark(benchmark_dir)
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
    audit = audit_existing_experiment(out_dir=out_dir, plan=plan)
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


def validate_benchmark(benchmark_dir: Path) -> None:
    required = [
        "benchmark_manifest.json",
        "method_manifest.json",
        "level_manifest.json",
        "verifier_manifest.json",
        "hidden_variant_manifest.json",
        "tasks",
    ]
    missing = [name for name in required if not (benchmark_dir / name).exists()]
    if missing:
        raise SystemExit(f"prepared physics benchmark is incomplete: {missing}")
    manifest = read_json(benchmark_dir / "benchmark_manifest.json")
    if manifest.get("experiment_ready") is not True:
        raise SystemExit("benchmark_manifest.json does not mark experiment_ready=true")
    methods = read_json(benchmark_dir / "method_manifest.json")
    if methods.get("required_methods") != list(REQUIRED_METHODS):
        raise SystemExit("method_manifest.json does not match goals.md methods")


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


def audit_existing_experiment(*, out_dir: Path, plan: dict[str, Any]) -> dict[str, Any]:
    rows = load_rows(out_dir / "cell_results.jsonl")
    row_map: dict[tuple[str, str, int, str, int], dict[str, Any]] = {}
    duplicate_keys: list[str] = []
    for row in rows:
        key = row_key(row, default_budget=int(plan["primary_budget"]))
        if key in row_map:
            duplicate_keys.append(cell_key_text(key))
        row_map[key] = row

    missing_cells: list[dict[str, Any]] = []
    budget_mismatches: list[dict[str, Any]] = []
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
        if int(cell["verifier_level"]) >= 2 and int_value(
            row.get("actual_cad_calls", row.get("cad_audits", 0))
        ) <= 0:
            budget_mismatches.append({"cell": cell, "actual_cad_calls": 0})
        if int(cell["verifier_level"]) >= 3 and int_value(
            row.get("actual_chrono_calls", row.get("chrono_audits", 0))
        ) <= 0:
            budget_mismatches.append({"cell": cell, "actual_chrono_calls": 0})
        missing_evidence.extend(missing_evidence_for_row(out_dir, cell, row))
        if str(cell["method"]) in TTRL_METHODS:
            learning_missing = missing_learning_evidence(out_dir, row)
            if learning_missing:
                missing_learning.append({
                    "cell": cell,
                    "missing": learning_missing,
                })

    missing_analysis = [
        name for name in ANALYSIS_ARTIFACTS if not (out_dir / name).is_file()
    ]
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
        "duplicate_cell_count": len(duplicate_keys),
        "budget_mismatch_count": len(budget_mismatches),
        "missing_accounting_count": len(missing_accounting),
        "budget_matched": (
            not missing_cells
            and not duplicate_keys
            and not budget_mismatches
            and not missing_accounting
        ),
        "sample_missing_cells": missing_cells[:25],
        "sample_duplicate_cells": duplicate_keys[:25],
        "sample_budget_mismatches": budget_mismatches[:25],
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
        missing_analysis=missing_analysis,
    )
    if plan.get("missing_required_methods"):
        blockers.insert(0, "execute all required methods, not a smoke subset")
    if plan.get("missing_required_seeds"):
        blockers.insert(0, "execute all required evaluation seeds")
    claim_audit = {
        "schema": "mechanism_repair_physics.experiment_claim_audit.v1",
        "goal_complete": not blockers,
        "benchmark_dir": plan["benchmark_dir"],
        "planned_cells": int(plan["planned_cells"]),
        "full_planned_cells": int(plan.get("full_planned_cells", plan["planned_cells"])),
        "shard_index": int(plan.get("shard_index", 0)),
        "num_shards": int(plan.get("num_shards", 1)),
        "observed_rows": len(rows),
        "blockers": blockers,
        "missing_before_paper_claim": blockers,
        "missing_evidence_count": len(missing_evidence),
        "missing_learning_count": len(missing_learning),
        "missing_analysis_artifacts": missing_analysis,
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
        write_json(shard_dir / f"shard_{shard_index:04d}.json", payload)


def cell_shard(cell: dict[str, Any], *, num_shards: int) -> int:
    key = json.dumps(
        {
            "split": cell["split"],
            "task_id": cell["task_id"],
            "seed": int(cell["seed"]),
            "method": cell["method"],
            "budget": int(cell["budget"]),
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
    missing_analysis: list[str],
) -> list[str]:
    blockers: list[str] = []
    if budget_audit["missing_cell_count"]:
        blockers.append(
            "execute all required methods for all planned seeds/splits/tasks"
        )
    if budget_audit["duplicate_cell_count"]:
        blockers.append("deduplicate repeated method/seed/task/budget cells")
    if budget_audit["budget_mismatch_count"] or budget_audit["missing_accounting_count"]:
        blockers.append("prove matched actual verifier/CAD/Chrono budget")
    if missing_evidence:
        blockers.append("record raw completions and verifier/CAD/Chrono outputs")
    if missing_learning:
        blockers.append("preserve training logs and adapter checkpoints for TTRL cells")
    if not anti_shortcut_audit["anti_shortcut_executed"]:
        blockers.append("run hidden/isomorphic anti-shortcut variants")
    if missing_analysis:
        blockers.append(
            "write statistical, failure, trace-pair, repair-taxonomy, "
            "anti-shortcut, and budget analyses"
        )
    return blockers


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


def missing_learning_evidence(out_dir: Path, row: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    log_paths = collect_paths(row, "training_log_paths", "training_log_path")
    ckpt_paths = collect_paths(row, "adapter_checkpoint_paths", "adapter_checkpoint_path")
    if not log_paths or any(not resolve_path(out_dir, path).exists() for path in log_paths):
        missing.append("training_logs")
    if not ckpt_paths or any(not resolve_path(out_dir, path).exists() for path in ckpt_paths):
        missing.append("adapter_checkpoints")
    updates = int_value(row.get("adapter_updates", row.get("online_update_steps", 0)))
    if updates <= 0:
        missing.append("adapter_updates")
    return missing


def infer_claim_status(out_dir: Path) -> str:
    stats = read_json(out_dir / "stats.json")
    explicit = stats.get("claim_status")
    if explicit in {
        "supports_primary_hypothesis",
        "does_not_support_primary_hypothesis",
    }:
        return str(explicit)
    comparison = stats.get("primary_comparison") or {}
    delta = float_value(comparison.get("success_delta_pct", 0.0))
    ci = comparison.get("success_delta_ci95") or [0.0, 0.0]
    ci_low = float_value(ci[0] if isinstance(ci, list) and ci else 0.0)
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
    return path if path.is_absolute() else out_dir / path


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
