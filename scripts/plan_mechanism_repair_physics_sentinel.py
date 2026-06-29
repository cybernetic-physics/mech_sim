#!/usr/bin/env python3
"""Plan and audit a high-fidelity MechanismRepair-Physics sentinel run.

The sentinel is not a paper claim. It is a low-N, same-fidelity futility gate
for the expensive goals.md experiment: exact same tasks, verifier budget, CAD /
Chrono evidence contract, online runner, and primary TTRL-vs-no-update paired
contrast, but only a balanced subset of cells.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from scripts.prepare_mechanism_repair_physics_benchmark import (
    EVAL_SEEDS,
    PRIMARY_BASELINE,
    PRIMARY_BUDGET,
    PRIMARY_METHOD,
    REQUIRED_FAMILIES,
)


DEFAULT_BENCHMARK_DIR = "runs/mechanism_repair_physics_final"
DEFAULT_OUT_DIR = "runs/mechanism_repair_physics_sentinel"
DEFAULT_SPLITS = ("hidden_perturbation", "isomorphic")
DEFAULT_CALIBRATOR_METHODS = ("frozen_model", "sft_seen_family")
DEFAULT_MIN_EFFECT_PP = 5.0
BENCHMARK_FILES = (
    "benchmark_manifest.json",
    "method_manifest.json",
    "level_manifest.json",
    "verifier_manifest.json",
    "hidden_variant_manifest.json",
    "split_manifest_A.json",
    "split_manifest_B.json",
    "split_manifest_external_style.json",
    "split_manifest_hidden_perturbation.json",
    "split_manifest_isomorphic.json",
)
BENCHMARK_DIR_PREFIXES = ("splits_",)
RUN_ARTIFACT_DIRS = (
    "raw_completions",
    "verifier_outputs",
    "cad_artifacts",
    "chrono_outputs",
    "training_logs",
    "adapter_checkpoints",
    "shared_sft",
    "shard_runs",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", default=DEFAULT_BENCHMARK_DIR)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--splits", default=",".join(DEFAULT_SPLITS))
    parser.add_argument(
        "--eval-seeds",
        default=",".join(str(seed) for seed in EVAL_SEEDS[:2]),
        help="sentinel seeds; first seed is the first-stage signal shard",
    )
    parser.add_argument("--primary-method", default=PRIMARY_METHOD)
    parser.add_argument("--baseline-method", default=PRIMARY_BASELINE)
    parser.add_argument(
        "--calibrator-methods",
        default=",".join(DEFAULT_CALIBRATOR_METHODS),
        help="optional non-updating calibrators run after the primary pair",
    )
    parser.add_argument("--budget", type=int, default=PRIMARY_BUDGET)
    parser.add_argument("--tasks-per-family-per-split", type=int, default=1)
    parser.add_argument("--min-effect-pp", type=float, default=DEFAULT_MIN_EFFECT_PP)
    parser.add_argument(
        "--source-run-dir",
        action="append",
        default=[],
        help="optional existing run directory to include in the audit",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the plan/audit without writing the sentinel run tree",
    )
    args = parser.parse_args()

    benchmark_dir = Path(args.benchmark_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    plan = build_sentinel_plan(
        benchmark_dir=benchmark_dir,
        out_dir=out_dir,
        splits=parse_csv(args.splits),
        seeds=[int(seed) for seed in parse_csv(args.eval_seeds)],
        primary_method=str(args.primary_method),
        baseline_method=str(args.baseline_method),
        calibrator_methods=parse_csv(args.calibrator_methods),
        budget=int(args.budget),
        tasks_per_family_per_split=int(args.tasks_per_family_per_split),
        min_effect_pp=float(args.min_effect_pp),
    )

    if not args.dry_run:
        materialize_benchmark_scaffold(source_dir=benchmark_dir, out_dir=out_dir)
        write_sentinel_artifacts(out_dir=out_dir, plan=plan)

    audit = audit_sentinel_run(
        out_dir=out_dir,
        planned_cells=plan["expected_cells"],
        primary_method=str(args.primary_method),
        baseline_method=str(args.baseline_method),
        budget=int(args.budget),
        min_effect_pp=float(args.min_effect_pp),
        source_run_dirs=[Path(item).expanduser().resolve() for item in args.source_run_dir],
    )
    if not args.dry_run:
        write_json(out_dir / "sentinel_audit.json", audit)

    print(json.dumps({
        "schema": "mechanism_repair_physics.sentinel_cli.v1",
        "out_dir": str(out_dir),
        "plan": None if args.dry_run else str(out_dir / "sentinel_plan.json"),
        "audit": None if args.dry_run else str(out_dir / "sentinel_audit.json"),
        "planned_cells": int(plan["planned_cells"]),
        "shards": [
            {
                "shard": f"shard_{int(stage['shard_index']):04d}",
                "stage": stage["stage"],
                "planned_cells": len(stage["cells"]),
                "methods": stage["methods"],
                "seeds": stage["seeds"],
            }
            for stage in plan["stages"]
        ],
        "observed_cells": audit["observed_cell_count"],
        "missing_cells": audit["missing_cell_count"],
        "primary_delta": audit["primary_pair_summary"]["delta_success_rate"],
        "decision": audit["decision"],
        "first_stage_submit_command": plan["first_stage_submit_command"],
    }, indent=2, sort_keys=True))
    return 0


def build_sentinel_plan(
    *,
    benchmark_dir: Path,
    out_dir: Path,
    splits: list[str],
    seeds: list[int],
    primary_method: str,
    baseline_method: str,
    calibrator_methods: list[str],
    budget: int,
    tasks_per_family_per_split: int,
    min_effect_pp: float,
) -> dict[str, Any]:
    if not splits:
        raise SystemExit("at least one split is required")
    if not seeds:
        raise SystemExit("at least one eval seed is required")
    if tasks_per_family_per_split < 1:
        raise SystemExit("--tasks-per-family-per-split must be >= 1")

    tasks = load_task_index(benchmark_dir)
    selected = select_balanced_tasks(
        benchmark_dir=benchmark_dir,
        task_index=tasks,
        splits=splits,
        tasks_per_family_per_split=tasks_per_family_per_split,
    )
    stage_specs: list[dict[str, Any]] = [
        {
            "stage": "primary_pair_seed0_signal",
            "rationale": (
                "Earliest futility gate: paired no-update baseline and "
                "tool-verified TTRL on every selected family/split at seed 0."
            ),
            "seeds": [int(seeds[0])],
            "methods": [baseline_method, primary_method],
        }
    ]
    if len(seeds) > 1:
        stage_specs.append({
            "stage": "primary_pair_replicate_seeds",
            "rationale": (
                "Replication gate for the primary paired contrast before "
                "spending calibrator compute."
            ),
            "seeds": [int(seed) for seed in seeds[1:]],
            "methods": [baseline_method, primary_method],
        })
    if calibrator_methods:
        stage_specs.append({
            "stage": "non_updating_calibrators",
            "rationale": (
                "Sanity calibrators on the exact same selected cells; run only "
                "after the primary pair is not futile."
            ),
            "seeds": [int(seed) for seed in seeds],
            "methods": list(dict.fromkeys(calibrator_methods)),
        })

    stages = []
    all_cells: list[dict[str, Any]] = []
    for shard_index, spec in enumerate(stage_specs):
        cells = cells_for_stage(
            selected_tasks=selected,
            seeds=spec["seeds"],
            methods=spec["methods"],
            budget=budget,
        )
        stage = {
            **spec,
            "shard_index": int(shard_index),
            "planned_cells": len(cells),
            "cells": cells,
        }
        stages.append(stage)
        all_cells.extend(cells)

    validate_primary_pair_groups(
        all_cells,
        primary_method=primary_method,
        baseline_method=baseline_method,
        budget=budget,
    )
    selected_summary = summarize_selected_tasks(selected)
    return {
        "schema": "mechanism_repair_physics.sentinel_plan.v1",
        "hypothesis": (
            "For non-toy Level-2/3 mechanism repair tasks, online GRPO/LoRA "
            "updates from strict CAD/Chrono verifier feedback should improve "
            f"{primary_method} verified repair success over the matched "
            f"{baseline_method} no-update verifier-feedback loop under the "
            f"same {budget}-call verifier budget. The sentinel rejects this "
            "hypothesis early if the paired delta is non-positive on the "
            "balanced hidden/isomorphic subset."
        ),
        "benchmark_dir": str(benchmark_dir),
        "out_dir": str(out_dir),
        "primary_method": primary_method,
        "baseline_method": baseline_method,
        "calibrator_methods": list(dict.fromkeys(calibrator_methods)),
        "budget": int(budget),
        "splits": splits,
        "seeds": [int(seed) for seed in seeds],
        "tasks_per_family_per_split": int(tasks_per_family_per_split),
        "selected_task_count": len({
            (item["split"], item["task_id"])
            for item in selected
        }),
        "selected_task_summary": selected_summary,
        "planned_cells": len(all_cells),
        "num_shards": len(stages),
        "stages": stages,
        "expected_cells": all_cells,
        "per_cell_fidelity_contract": {
            "same_benchmark_tasks": True,
            "same_online_runner": "scripts/run_mechanism_repair_online_experiment.py",
            "same_primary_budget": int(budget),
            "same_cad_chrono_evidence_contract": True,
            "same_primary_baseline_pairing": True,
        },
        "stopping_rule": {
            "audit_after_each_primary_pair_stage": True,
            "minimum_effect_pp_to_continue": float(min_effect_pp),
            "futility_stop": (
                "If the completed paired primary delta is <= 0 percentage "
                "points, stop the full-grid run and revise the method."
            ),
            "weak_signal": (
                "If the delta is positive but below the minimum effect, run at "
                "most one replicate stage before spending calibrator/full-grid "
                "compute."
            ),
        },
        "first_stage_local_command": local_stage_command(out_dir, 0),
        "first_stage_submit_command": first_stage_submit_command(out_dir, len(stages)),
    }


def select_balanced_tasks(
    *,
    benchmark_dir: Path,
    task_index: dict[str, dict[str, Any]],
    splits: list[str],
    tasks_per_family_per_split: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    family_order = list(REQUIRED_FAMILIES)
    for split in splits:
        split_task_ids = read_split_task_ids(benchmark_dir / f"splits_{split}" / "test.txt")
        by_family: dict[str, list[str]] = defaultdict(list)
        for task_id in split_task_ids:
            task = task_index.get(task_id)
            if not task:
                raise SystemExit(f"{split} references unknown task id: {task_id}")
            by_family[str(task["family"])].append(task_id)
        for family in family_order:
            candidates = by_family.get(family, [])
            if not candidates:
                continue
            for task_id in candidates[:tasks_per_family_per_split]:
                task = task_index[task_id]
                selected.append({
                    "split": split,
                    "split_kind": (
                        "anti_shortcut"
                        if split in {"hidden_perturbation", "external_style", "isomorphic"}
                        else "headline"
                    ),
                    "task_id": task_id,
                    "family": str(task["family"]),
                    "verifier_level": int(task.get("verifier_level", 0) or 0),
                })
    if not selected:
        raise SystemExit("balanced sentinel task selection is empty")
    return selected


def cells_for_stage(
    *,
    selected_tasks: list[dict[str, Any]],
    seeds: list[int],
    methods: list[str],
    budget: int,
) -> list[dict[str, Any]]:
    cells = []
    for task in selected_tasks:
        for seed in seeds:
            for method in methods:
                cells.append({
                    "split": task["split"],
                    "split_kind": task["split_kind"],
                    "task_id": task["task_id"],
                    "family": task["family"],
                    "verifier_level": int(task["verifier_level"]),
                    "seed": int(seed),
                    "method": method,
                    "budget": int(budget),
                })
    return cells


def materialize_benchmark_scaffold(*, source_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in BENCHMARK_FILES:
        source = source_dir / name
        if source.is_file():
            shutil.copy2(source, out_dir / name)
    for source in source_dir.iterdir():
        if not source.is_dir():
            continue
        if source.name == "tasks" or any(
            source.name.startswith(prefix) for prefix in BENCHMARK_DIR_PREFIXES
        ):
            shutil.copytree(source, out_dir / source.name, dirs_exist_ok=True)
    for name in RUN_ARTIFACT_DIRS:
        (out_dir / name).mkdir(parents=True, exist_ok=True)


def write_sentinel_artifacts(*, out_dir: Path, plan: dict[str, Any]) -> None:
    shard_dir = out_dir / "experiment_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    target_names = set()
    for stage in plan["stages"]:
        shard_index = int(stage["shard_index"])
        shard_name = f"shard_{shard_index:04d}.json"
        target_names.add(shard_name)
        payload = {
            "schema": "mechanism_repair_physics.experiment_shard.v1",
            "sentinel_schema": "mechanism_repair_physics.sentinel_shard.v1",
            "benchmark_dir": str(Path(plan["out_dir"]).resolve()),
            "out_dir": str(Path(plan["out_dir"]).resolve()),
            "num_shards": int(plan["num_shards"]),
            "shard_index": shard_index,
            "stage": stage["stage"],
            "planned_cells": len(stage["cells"]),
            "full_planned_cells": int(plan["planned_cells"]),
            "cells": stage["cells"],
            "replay_args": [
                "--benchmark-dir",
                str(Path(plan["out_dir"]).resolve()),
                "--out-dir",
                str(Path(plan["out_dir"]).resolve()),
                "--cell-shard-file",
                str((shard_dir / shard_name).resolve()),
            ],
        }
        write_json(shard_dir / shard_name, payload)
    for stale in shard_dir.glob("shard_*.json"):
        if stale.name not in target_names:
            stale.unlink()
    write_json(out_dir / "sentinel_plan.json", public_plan(plan))


def audit_sentinel_run(
    *,
    out_dir: Path,
    planned_cells: list[dict[str, Any]],
    primary_method: str,
    baseline_method: str,
    budget: int,
    min_effect_pp: float,
    source_run_dirs: list[Path] | None = None,
) -> dict[str, Any]:
    source_run_dirs = source_run_dirs or []
    planned_by_key = {
        cell_key(cell, default_budget=budget): cell
        for cell in planned_cells
    }
    rows = load_rows_for_audit(out_dir, source_run_dirs=source_run_dirs)
    row_map: dict[tuple[str, str, int, str, int], dict[str, Any]] = {}
    duplicate_keys = []
    for row in rows:
        key = cell_key(row, default_budget=budget)
        if key not in planned_by_key:
            continue
        if key in row_map:
            duplicate_keys.append(key_text(key))
            continue
        row_map[key] = row

    missing = [
        cell for key, cell in sorted(planned_by_key.items())
        if key not in row_map
    ]
    pairs = paired_primary_rows(
        planned_cells=planned_cells,
        row_map=row_map,
        primary_method=primary_method,
        baseline_method=baseline_method,
        budget=budget,
    )
    pair_summary = summarize_pairs(pairs)
    decision = sentinel_decision(
        paired_count=int(pair_summary["paired_count"]),
        planned_pair_count=planned_primary_pair_count(
            planned_cells,
            primary_method=primary_method,
            baseline_method=baseline_method,
        ),
        delta_success_rate=pair_summary["delta_success_rate"],
        min_effect_pp=min_effect_pp,
    )
    return {
        "schema": "mechanism_repair_physics.sentinel_audit.v1",
        "out_dir": str(out_dir),
        "source_run_dirs": [str(path) for path in source_run_dirs],
        "planned_cell_count": len(planned_cells),
        "observed_cell_count": len(row_map),
        "missing_cell_count": len(missing),
        "duplicate_cell_count": len(duplicate_keys),
        "primary_method": primary_method,
        "baseline_method": baseline_method,
        "budget": int(budget),
        "primary_pair_summary": pair_summary,
        "by_split": summarize_pairs_by(pairs, "split"),
        "by_family": summarize_pairs_by(pairs, "family"),
        "by_verifier_level": summarize_pairs_by(pairs, "verifier_level"),
        "decision": decision,
        "sample_missing_cells": missing[:25],
        "sample_duplicate_keys": duplicate_keys[:25],
    }


def paired_primary_rows(
    *,
    planned_cells: list[dict[str, Any]],
    row_map: dict[tuple[str, str, int, str, int], dict[str, Any]],
    primary_method: str,
    baseline_method: str,
    budget: int,
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    for cell in planned_cells:
        if str(cell["method"]) not in {primary_method, baseline_method}:
            continue
        group = (
            str(cell["split"]),
            str(cell["task_id"]),
            int(cell["seed"]),
            int(cell.get("budget", budget)),
        )
        groups[group] = cell

    pairs = []
    for split, task_id, seed, group_budget in sorted(groups):
        primary_key = (split, task_id, seed, primary_method, group_budget)
        baseline_key = (split, task_id, seed, baseline_method, group_budget)
        primary_row = row_map.get(primary_key)
        baseline_row = row_map.get(baseline_key)
        if primary_row is None or baseline_row is None:
            continue
        cell = groups[(split, task_id, seed, group_budget)]
        primary_success = success_value(primary_row, budget=group_budget)
        baseline_success = success_value(baseline_row, budget=group_budget)
        pairs.append({
            "split": split,
            "task_id": task_id,
            "seed": int(seed),
            "budget": int(group_budget),
            "family": str(cell.get("family") or ""),
            "verifier_level": int(cell.get("verifier_level", 0) or 0),
            "primary_success": primary_success,
            "baseline_success": baseline_success,
            "delta": int(primary_success) - int(baseline_success),
        })
    return pairs


def summarize_pairs(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(pairs)
    if not pairs:
        return {
            "paired_count": 0,
            "primary_success_rate": None,
            "baseline_success_rate": None,
            "delta_success_rate": None,
            "primary_wins": 0,
            "baseline_wins": 0,
            "ties": 0,
        }
    primary = sum(1 for item in pairs if item["primary_success"])
    baseline = sum(1 for item in pairs if item["baseline_success"])
    primary_wins = sum(1 for item in pairs if item["delta"] > 0)
    baseline_wins = sum(1 for item in pairs if item["delta"] < 0)
    ties = n - primary_wins - baseline_wins
    return {
        "paired_count": n,
        "primary_success_rate": primary / n,
        "baseline_success_rate": baseline / n,
        "delta_success_rate": (primary - baseline) / n,
        "primary_wins": primary_wins,
        "baseline_wins": baseline_wins,
        "ties": ties,
    }


def summarize_pairs_by(pairs: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        grouped[str(pair.get(field, ""))].append(pair)
    return [
        {"group": group, **summarize_pairs(items)}
        for group, items in sorted(grouped.items())
    ]


def sentinel_decision(
    *,
    paired_count: int,
    planned_pair_count: int,
    delta_success_rate: float | None,
    min_effect_pp: float,
) -> dict[str, Any]:
    if paired_count <= 0 or delta_success_rate is None:
        return {
            "status": "incomplete",
            "action": "run_primary_pair_stage",
            "reason": "no completed primary-baseline pairs yet",
        }
    delta_pp = 100.0 * float(delta_success_rate)
    if delta_pp <= 0.0:
        return {
            "status": "futile",
            "action": "stop_full_grid_and_revise_method",
            "reason": f"paired primary delta is non-positive ({delta_pp:.2f} pp)",
        }
    if delta_pp < float(min_effect_pp):
        return {
            "status": "weak_signal",
            "action": "run_at_most_one_replicate_before_calibrators",
            "reason": (
                f"paired primary delta {delta_pp:.2f} pp is below "
                f"{float(min_effect_pp):.2f} pp"
            ),
        }
    if paired_count < planned_pair_count:
        return {
            "status": "promising_incomplete",
            "action": "finish_primary_pair_stages_before_calibrators",
            "reason": (
                f"paired primary delta {delta_pp:.2f} pp is positive but "
                f"only {paired_count}/{planned_pair_count} pairs are complete"
            ),
        }
    return {
        "status": "promising",
        "action": "run_calibrators_then_consider_full_grid",
        "reason": f"paired primary delta is {delta_pp:.2f} pp",
    }


def planned_primary_pair_count(
    planned_cells: list[dict[str, Any]],
    *,
    primary_method: str,
    baseline_method: str,
) -> int:
    groups: dict[tuple[str, str, int, int], set[str]] = defaultdict(set)
    for cell in planned_cells:
        method = str(cell["method"])
        if method not in {primary_method, baseline_method}:
            continue
        group = (
            str(cell["split"]),
            str(cell["task_id"]),
            int(cell["seed"]),
            int(cell["budget"]),
        )
        groups[group].add(method)
    return sum(
        1 for methods in groups.values()
        if {primary_method, baseline_method}.issubset(methods)
    )


def validate_primary_pair_groups(
    cells: list[dict[str, Any]],
    *,
    primary_method: str,
    baseline_method: str,
    budget: int,
) -> None:
    methods_by_group: dict[tuple[str, str, int, int], set[str]] = defaultdict(set)
    for cell in cells:
        split, task_id, seed, method, cell_budget = cell_key(
            cell,
            default_budget=budget,
        )
        methods_by_group[(split, task_id, seed, cell_budget)].add(method)
    missing = []
    for (split, task_id, seed, cell_budget), methods in sorted(methods_by_group.items()):
        if primary_method in methods and baseline_method not in methods:
            missing.append({
                "split": split,
                "task_id": task_id,
                "seed": seed,
                "budget": cell_budget,
                "missing": baseline_method,
            })
    if missing:
        raise SystemExit(
            "primary TTRL cells must be paired with same-shard baseline: "
            + json.dumps(missing[:10], sort_keys=True)
        )


def load_rows_for_audit(
    out_dir: Path,
    *,
    source_run_dirs: list[Path],
) -> list[dict[str, Any]]:
    roots = [out_dir, *source_run_dirs]
    rows = []
    seen_paths = set()
    for root in roots:
        candidates = []
        if (root / "cell_results.jsonl").is_file():
            candidates.append(root / "cell_results.jsonl")
        candidates.extend(sorted((root / "shard_runs").glob("shard_*/cell_results.jsonl")))
        for path in candidates:
            if path in seen_paths:
                continue
            seen_paths.add(path)
            rows.extend(read_jsonl(path))
    return rows


def summarize_selected_tasks(selected: list[dict[str, Any]]) -> dict[str, Any]:
    by_split = Counter(str(item["split"]) for item in selected)
    by_family = Counter(str(item["family"]) for item in selected)
    by_level = Counter(str(item["verifier_level"]) for item in selected)
    return {
        "by_split": dict(sorted(by_split.items())),
        "by_family": dict(sorted(by_family.items())),
        "by_verifier_level": dict(sorted(by_level.items())),
    }


def local_stage_command(out_dir: Path, shard_index: int) -> str:
    shard = f"shard_{int(shard_index):04d}"
    return " ".join([
        ".venv/bin/python",
        "scripts/run_mechanism_repair_online_experiment.py",
        "--benchmark-dir",
        shell_token(str(out_dir)),
        "--out-dir",
        shell_token(str(out_dir / "shard_runs" / shard)),
        "--cell-shard-file",
        shell_token(str(out_dir / "experiment_shards" / f"{shard}.json")),
        "--shared-sft-root",
        shell_token(str(out_dir / "shared_sft")),
        "--resume-existing",
        "--skip-analysis",
        "--audit-retries",
        "0",
        "--evidence-layout",
        "bundled",
        "--require-runtime-preflight",
    ])


def first_stage_submit_command(out_dir: Path, num_shards: int) -> str:
    submit_out_dir = repo_relative_path(out_dir)
    return " ".join([
        "OUT_DIR=" + shell_token(submit_out_dir),
        "REMOTE_ROOT=/matx/u/knatalia/corl_mechanism_repair_physics_sentinel",
        "JOB_NAME=corl_mech_sent",
        f"NUM_SHARDS={int(num_shards)}",
        "SHARD_INDICES=0",
        "MAX_ARRAY_TASKS=1",
        "ARRAY_CONCURRENCY=1",
        "USE_PREPLANNED_SHARDS=1",
        "RESTAGE_REMOTE_REPO=1",
        "ALLOW_DESTRUCTIVE_RESTAGE=1",
        "SYNC_LOCAL_BENCHMARK=1",
        "SUBMIT_DEPENDENTS=0",
        "scripts/submit_mechanism_repair_physics_matx.sh",
        "--submit",
    ])


def repo_relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def public_plan(plan: dict[str, Any]) -> dict[str, Any]:
    out = dict(plan)
    out.pop("expected_cells", None)
    public_stages = []
    for stage in out["stages"]:
        item = dict(stage)
        item.pop("cells", None)
        public_stages.append(item)
    out["stages"] = public_stages
    out["expected_cells_materialized_in_shards"] = True
    return out


def load_task_index(benchmark_dir: Path) -> dict[str, dict[str, Any]]:
    manifest = read_json(benchmark_dir / "benchmark_manifest.json")
    tasks = {}
    for row in manifest.get("tasks", []) or []:
        task_id = str(row.get("task_id") or "")
        if not task_id:
            continue
        family = str(row.get("family") or "")
        if not family:
            raise SystemExit(f"task {task_id} has no family")
        tasks[task_id] = row
    if not tasks:
        raise SystemExit(f"no tasks found in {benchmark_dir / 'benchmark_manifest.json'}")
    return tasks


def read_split_task_ids(path: Path) -> list[str]:
    if not path.is_file():
        raise SystemExit(f"missing split file: {path}")
    task_ids = []
    for line in path.read_text(encoding="utf-8").splitlines():
        entry = line.strip()
        if not entry or entry.startswith("#"):
            continue
        task_ids.append(Path(entry).name)
    return task_ids


def success_value(row: dict[str, Any], *, budget: int) -> bool:
    for name in (
        f"verified_repair_success_at_{int(budget)}",
        "verified_repair_success_at_32",
        "verified_repair_success",
        "repair_success",
        "success",
    ):
        if name in row:
            return bool(row.get(name))
    return False


def cell_key(
    item: dict[str, Any],
    *,
    default_budget: int,
) -> tuple[str, str, int, str, int]:
    return (
        str(item.get("split") or item.get("split_name") or ""),
        str(item.get("task_id") or ""),
        int(item.get("seed", 0) or 0),
        str(item.get("method") or ""),
        int(item.get("budget", item.get("budget_verifier_calls", default_budget))),
    )


def key_text(key: tuple[str, str, int, str, int]) -> str:
    split, task_id, seed, method, budget = key
    return f"{split}/{task_id}/seed{seed}/{method}/budget{budget}"


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def shell_token(value: str) -> str:
    if not value:
        return "''"
    safe = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_./:-")
    if all(char in safe for char in value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
