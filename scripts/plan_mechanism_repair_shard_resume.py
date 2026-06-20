#!/usr/bin/env python3
"""Summarize MechanismRepair-Physics shard completion for safe resume.

This is a local artifact-inspection helper. It does not contact MATX or submit
jobs. Its purpose is to turn a partial run directory into a single safe next
``SHARD_INDICES=<n>`` target for the conservative submit path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_RUN_DIR = "runs/mechanism_repair_physics_final"
DEFAULT_BUDGET = 32


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    parser.add_argument(
        "--default-budget",
        type=int,
        default=DEFAULT_BUDGET,
        help="budget to use only for legacy rows missing budget fields",
    )
    parser.add_argument(
        "--out-json",
        default=None,
        help="optional path to also write the JSON report",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    report = build_report(run_dir=run_dir, default_budget=int(args.default_budget))
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(text, end="")
    if args.out_json:
        out_json = Path(args.out_json).expanduser()
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(text, encoding="utf-8")
    return 0


def build_report(
    *,
    run_dir: Path,
    default_budget: int = DEFAULT_BUDGET,
) -> dict[str, Any]:
    shard_dir = run_dir / "experiment_shards"
    shard_files = sorted(shard_dir.glob("shard_*.json"))
    if not shard_files:
        raise SystemExit(f"no shard files found under {shard_dir}")

    shard_reports = [
        summarize_shard(
            run_dir=run_dir,
            shard_file=path,
            default_budget=default_budget,
        )
        for path in shard_files
    ]
    incomplete = [item for item in shard_reports if item["status"] != "complete"]
    next_item = incomplete[0] if incomplete else None
    next_shard_index = (
        int(next_item["shard_index"])
        if next_item is not None
        else None
    )
    resume_env = (
        (
            f"SHARD_INDICES={next_shard_index} "
            "RESTAGE_REMOTE_REPO=0 SUBMIT_DEPENDENTS=0"
        )
        if next_shard_index is not None
        else None
    )
    return {
        "schema": "mechanism_repair_physics.shard_resume_plan.v1",
        "run_dir": str(run_dir),
        "shard_count": len(shard_reports),
        "complete_shard_count": sum(
            1 for item in shard_reports if item["status"] == "complete"
        ),
        "incomplete_shard_count": len(incomplete),
        "expected_rows": sum(int(item["expected_rows"]) for item in shard_reports),
        "observed_rows": sum(int(item["observed_rows"]) for item in shard_reports),
        "missing_rows": sum(int(item["missing_rows"]) for item in shard_reports),
        "duplicate_rows": sum(int(item["duplicate_rows"]) for item in shard_reports),
        "unexpected_rows": sum(int(item["unexpected_rows"]) for item in shard_reports),
        "merge_ready": not incomplete,
        "next_shard_index": next_shard_index,
        "resume_env": resume_env,
        "shards": shard_reports,
    }


def summarize_shard(
    *,
    run_dir: Path,
    shard_file: Path,
    default_budget: int,
) -> dict[str, Any]:
    payload = read_json(shard_file)
    shard_name = shard_file.stem
    shard_index = int(payload.get("shard_index", shard_name.rsplit("_", 1)[-1]))
    expected_cells = payload.get("cells") or []
    expected_keys = {
        cell_key(cell, default_budget=default_budget)
        for cell in expected_cells
    }

    rows_path = run_dir / "shard_runs" / shard_name / "cell_results.jsonl"
    rows = read_jsonl(rows_path) if rows_path.is_file() else []
    observed_keys: set[tuple[str, str, int, str, int]] = set()
    duplicate_keys: list[tuple[str, str, int, str, int]] = []
    for row in rows:
        key = cell_key(row, default_budget=default_budget)
        if key in observed_keys:
            duplicate_keys.append(key)
        observed_keys.add(key)

    missing_keys = sorted(expected_keys - observed_keys)
    unexpected_keys = sorted(observed_keys - expected_keys)
    blockers: list[str] = []
    if not rows_path.is_file():
        blockers.append("missing shard_runs cell_results.jsonl")
    if missing_keys:
        blockers.append("missing expected cells")
    if unexpected_keys:
        blockers.append("unexpected cells not in shard file")
    if duplicate_keys:
        blockers.append("duplicate cell rows")

    status = "complete" if not blockers else "partial"
    if not rows_path.is_file():
        status = "missing_output"
    if unexpected_keys or duplicate_keys:
        status = "invalid"

    return {
        "shard": shard_name,
        "shard_index": shard_index,
        "status": status,
        "expected_rows": len(expected_keys),
        "observed_rows": len(rows),
        "unique_observed_rows": len(observed_keys),
        "missing_rows": len(missing_keys),
        "unexpected_rows": len(unexpected_keys),
        "duplicate_rows": len(duplicate_keys),
        "cell_results": str(rows_path),
        "blockers": blockers,
        "sample_missing": [key_text(key) for key in missing_keys[:10]],
        "sample_unexpected": [key_text(key) for key in unexpected_keys[:10]],
        "sample_duplicates": [key_text(key) for key in duplicate_keys[:10]],
    }


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
        int(first_present(item, ("budget", "budget_verifier_calls"), default_budget)),
    )


def first_present(
    item: dict[str, Any],
    names: tuple[str, ...],
    default: int,
) -> Any:
    for name in names:
        if item.get(name) is not None:
            return item[name]
    return default


def key_text(key: tuple[str, str, int, str, int]) -> str:
    split, task_id, seed, method, budget = key
    return f"{split}/{task_id}/seed{seed}/{method}/budget{budget}"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


if __name__ == "__main__":
    raise SystemExit(main())
