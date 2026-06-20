from __future__ import annotations

import json
from pathlib import Path

from scripts.plan_mechanism_repair_shard_resume import build_report


def _cell(task_id: str, method: str) -> dict:
    return {
        "split": "A",
        "task_id": task_id,
        "seed": 20260610,
        "method": method,
        "budget": 32,
    }


def _write_shard(run_dir: Path, index: int, cells: list[dict]) -> None:
    path = run_dir / "experiment_shards" / f"shard_{index:04d}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema": "mechanism_repair_physics.experiment_shard.v1",
                "num_shards": 2,
                "shard_index": index,
                "planned_cells": len(cells),
                "cells": cells,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_rows(run_dir: Path, index: int, rows: list[dict]) -> None:
    path = run_dir / "shard_runs" / f"shard_{index:04d}" / "cell_results.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_resume_plan_selects_first_missing_output_shard(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    shard0 = [_cell("task_a", "frozen_model")]
    shard1 = [_cell("task_b", "frozen_model")]
    _write_shard(run_dir, 0, shard0)
    _write_shard(run_dir, 1, shard1)
    _write_rows(run_dir, 1, shard1)

    report = build_report(run_dir=run_dir)

    assert report["merge_ready"] is False
    assert report["next_shard_index"] == 0
    assert report["resume_env"] == (
        "SHARD_INDICES=0 RESTAGE_REMOTE_REPO=0 SUBMIT_DEPENDENTS=0"
    )
    assert report["shards"][0]["status"] == "missing_output"


def test_resume_plan_marks_partial_shard(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    cells = [_cell("task_a", "frozen_model"), _cell("task_a", "llm_evolve_no_update")]
    _write_shard(run_dir, 0, cells)
    _write_rows(run_dir, 0, [cells[0]])

    report = build_report(run_dir=run_dir)

    assert report["next_shard_index"] == 0
    assert report["missing_rows"] == 1
    assert report["shards"][0]["status"] == "partial"
    assert report["shards"][0]["sample_missing"] == [
        "A/task_a/seed20260610/llm_evolve_no_update/budget32"
    ]


def test_resume_plan_requires_clean_complete_shards_for_merge(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    cells0 = [_cell("task_a", "frozen_model")]
    cells1 = [_cell("task_b", "frozen_model")]
    _write_shard(run_dir, 0, cells0)
    _write_shard(run_dir, 1, cells1)
    _write_rows(run_dir, 0, cells0)
    _write_rows(run_dir, 1, cells1)

    report = build_report(run_dir=run_dir)

    assert report["merge_ready"] is True
    assert report["next_shard_index"] is None
    assert report["resume_env"] is None
    assert report["complete_shard_count"] == 2


def test_resume_plan_flags_duplicate_or_unexpected_rows(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    expected = [_cell("task_a", "frozen_model")]
    unexpected = _cell("task_x", "frozen_model")
    _write_shard(run_dir, 0, expected)
    _write_rows(run_dir, 0, [expected[0], expected[0], unexpected])

    report = build_report(run_dir=run_dir)

    assert report["merge_ready"] is False
    assert report["shards"][0]["status"] == "invalid"
    assert report["duplicate_rows"] == 1
    assert report["unexpected_rows"] == 1


def test_resume_plan_uses_budget_verifier_calls_when_budget_is_none(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    expected = [_cell("task_a", "frozen_model")]
    row = dict(expected[0])
    row["budget"] = None
    row["budget_verifier_calls"] = 32
    _write_shard(run_dir, 0, expected)
    _write_rows(run_dir, 0, [row])

    report = build_report(run_dir=run_dir)

    assert report["merge_ready"] is True
    assert report["missing_rows"] == 0
