#!/usr/bin/env python3
"""Summarize MechanismRepair-Physics shard completion for safe resume.

This is a local artifact-inspection helper. It does not contact MATX or submit
jobs. Its purpose is to turn a partial run directory into a single safe next
``SHARD_INDICES=<n>`` target for the conservative submit path.
"""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from typing import Any

from scripts.run_mechanism_repair_physics_experiment import (
    LEARNING_METHODS,
    TTRL_METHODS,
    missing_evidence_for_row,
    missing_learning_evidence,
)


DEFAULT_RUN_DIR = "runs/mechanism_repair_physics_final"
DEFAULT_BUDGET = 32
DEFAULT_PYTHON = ".venv/bin/python"
PRIMARY_BASELINE = "llm_evolve_no_update"


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
    parser.add_argument(
        "--python",
        default=DEFAULT_PYTHON,
        help="Python executable to put in generated local commands",
    )
    parser.add_argument(
        "--rollout-backend",
        default="sglang_chat",
        choices=("sglang_chat", "worldlines_sampling", "transformers_local"),
        help="rollout backend to put in generated local shard commands",
    )
    parser.add_argument("--local-device", default="cpu")
    parser.add_argument(
        "--local-torch-dtype",
        default="auto",
        choices=("auto", "float32", "float16", "bfloat16"),
    )
    parser.add_argument("--local-trust-remote-code", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    report = build_report(
        run_dir=run_dir,
        default_budget=int(args.default_budget),
        python_executable=str(args.python),
        rollout_backend=str(args.rollout_backend),
        local_device=str(args.local_device),
        local_torch_dtype=str(args.local_torch_dtype),
        local_trust_remote_code=bool(args.local_trust_remote_code),
    )
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
    python_executable: str = DEFAULT_PYTHON,
    rollout_backend: str = "sglang_chat",
    local_device: str = "cpu",
    local_torch_dtype: str = "auto",
    local_trust_remote_code: bool = False,
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
    local_command = (
        local_shard_command(
            run_dir=run_dir,
            shard_index=next_shard_index,
            python_executable=python_executable,
            rollout_backend=rollout_backend,
            local_device=local_device,
            local_torch_dtype=local_torch_dtype,
            local_trust_remote_code=local_trust_remote_code,
        )
        if next_shard_index is not None
        else None
    )
    merge_command = merge_shards_command(
        run_dir=run_dir,
        shard_count=len(shard_reports),
        python_executable=python_executable,
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
        "missing_evidence_count": sum(
            int(item["missing_evidence_count"]) for item in shard_reports
        ),
        "missing_learning_count": sum(
            int(item["missing_learning_count"]) for item in shard_reports
        ),
        "merge_ready": not incomplete,
        "next_shard_index": next_shard_index,
        "resume_env": resume_env,
        "next_shard_file": (
            str(run_dir / "experiment_shards" / f"shard_{next_shard_index:04d}.json")
            if next_shard_index is not None
            else None
        ),
        "next_shard_output_dir": (
            str(run_dir / "shard_runs" / f"shard_{next_shard_index:04d}")
            if next_shard_index is not None
            else None
        ),
        "local_shard_command": local_command,
        "local_shard_command_text": (
            shlex.join(local_command) if local_command else None
        ),
        "merge_command": merge_command if not incomplete else None,
        "merge_command_text": (
            shlex.join(merge_command) if not incomplete else None
        ),
        "finalize_when_merge_ready_command": merge_command,
        "finalize_when_merge_ready_command_text": shlex.join(merge_command),
        "ttrl_baseline_group_error_count": sum(
            len(item["ttrl_baseline_group_errors"]) for item in shard_reports
        ),
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
    ttrl_baseline_errors = ttrl_baseline_group_errors(
        expected_cells,
        default_budget=default_budget,
    )

    rows_path = run_dir / "shard_runs" / shard_name / "cell_results.jsonl"
    rows = read_jsonl(rows_path) if rows_path.is_file() else []
    observed_keys: set[tuple[str, str, int, str, int]] = set()
    duplicate_keys: list[tuple[str, str, int, str, int]] = []
    expected_cell_by_key = {
        cell_key(cell, default_budget=default_budget): cell
        for cell in expected_cells
    }
    missing_evidence: list[dict[str, Any]] = []
    missing_learning: list[dict[str, Any]] = []
    for row in rows:
        key = cell_key(row, default_budget=default_budget)
        if key in observed_keys:
            duplicate_keys.append(key)
        observed_keys.add(key)
        cell = expected_cell_by_key.get(key)
        if cell is None:
            continue
        audit_cell = {
            **cell,
            "verifier_level": int(cell.get("verifier_level", 1) or 1),
        }
        missing_evidence.extend(
            missing_evidence_for_row(rows_path.parent, audit_cell, row)
        )
        method = str(row.get("method") or cell.get("method") or "")
        if method in LEARNING_METHODS:
            learning_missing = missing_learning_evidence(
                rows_path.parent,
                row,
                require_rl_evidence=method in TTRL_METHODS,
            )
            if learning_missing:
                missing_learning.append({
                    "cell": audit_cell,
                    "missing": learning_missing,
                })

    missing_keys = sorted(expected_keys - observed_keys)
    unexpected_keys = sorted(observed_keys - expected_keys)
    blockers: list[str] = []
    if not rows_path.is_file():
        blockers.append("missing shard_runs cell_results.jsonl")
    if ttrl_baseline_errors:
        blockers.append("TTRL cells missing same-shard no-update baseline")
    if missing_keys:
        blockers.append("missing expected cells")
    if unexpected_keys:
        blockers.append("unexpected cells not in shard file")
    if duplicate_keys:
        blockers.append("duplicate cell rows")
    if missing_evidence:
        blockers.append("missing evidence files")
    if missing_learning:
        blockers.append("missing learning evidence")

    status = "complete" if not blockers else "partial"
    if not rows_path.is_file():
        status = "missing_output"
    if unexpected_keys or duplicate_keys:
        status = "invalid"
    if ttrl_baseline_errors:
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
        "missing_evidence_count": len(missing_evidence),
        "missing_learning_count": len(missing_learning),
        "cell_results": str(rows_path),
        "blockers": blockers,
        "sample_missing": [key_text(key) for key in missing_keys[:10]],
        "sample_unexpected": [key_text(key) for key in unexpected_keys[:10]],
        "sample_duplicates": [key_text(key) for key in duplicate_keys[:10]],
        "sample_missing_evidence": missing_evidence[:10],
        "sample_missing_learning": missing_learning[:10],
        "ttrl_baseline_group_errors": ttrl_baseline_errors[:10],
    }


def ttrl_baseline_group_errors(
    cells: list[dict[str, Any]],
    *,
    default_budget: int,
) -> list[dict[str, Any]]:
    methods_by_group: dict[tuple[str, str, int, int], set[str]] = {}
    for cell in cells:
        split, task_id, seed, method, budget = cell_key(
            cell,
            default_budget=default_budget,
        )
        group = (split, task_id, seed, budget)
        methods_by_group.setdefault(group, set()).add(method)

    errors: list[dict[str, Any]] = []
    for group, methods in sorted(methods_by_group.items()):
        ttrl_methods = sorted(methods & TTRL_METHODS)
        if ttrl_methods and PRIMARY_BASELINE not in methods:
            split, task_id, seed, budget = group
            errors.append({
                "split": split,
                "task_id": task_id,
                "seed": int(seed),
                "budget": int(budget),
                "ttrl_methods": ttrl_methods,
                "missing_method": PRIMARY_BASELINE,
            })
    return errors


def local_shard_command(
    *,
    run_dir: Path,
    shard_index: int,
    python_executable: str,
    rollout_backend: str = "sglang_chat",
    local_device: str = "cpu",
    local_torch_dtype: str = "auto",
    local_trust_remote_code: bool = False,
) -> list[str]:
    shard_name = f"shard_{int(shard_index):04d}"
    cmd = [
        python_executable,
        "scripts/run_mechanism_repair_online_experiment.py",
        "--benchmark-dir",
        str(run_dir),
        "--out-dir",
        str(run_dir / "shard_runs" / shard_name),
        "--cell-shard-file",
        str(run_dir / "experiment_shards" / f"{shard_name}.json"),
        "--shared-sft-root",
        str(run_dir / "shared_sft"),
        "--resume-existing",
        "--skip-analysis",
        "--audit-retries",
        "0",
        "--evidence-layout",
        "bundled",
        "--require-runtime-preflight",
    ]
    if rollout_backend != "sglang_chat":
        cmd.extend(["--rollout-backend", rollout_backend])
    if rollout_backend == "transformers_local":
        cmd.extend([
            "--local-device",
            local_device,
            "--local-torch-dtype",
            local_torch_dtype,
            "--concurrency",
            "1",
        ])
        if local_device == "cpu":
            cmd.extend(["--sft-use-cpu", "--ttrl-use-cpu"])
        if local_trust_remote_code:
            cmd.append("--local-trust-remote-code")
    return cmd


def merge_shards_command(
    *,
    run_dir: Path,
    shard_count: int,
    python_executable: str,
) -> list[str]:
    return [
        python_executable,
        "scripts/merge_mechanism_repair_shards.py",
        "--benchmark-dir",
        str(run_dir),
        "--out-dir",
        str(run_dir),
        "--shard-root",
        str(run_dir / "shard_runs"),
        "--require-all-shards",
        str(int(shard_count)),
        "--require-complete",
    ]


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
