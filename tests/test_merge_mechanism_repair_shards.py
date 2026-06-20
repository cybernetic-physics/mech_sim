from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.merge_mechanism_repair_shards import load_shard_rows


def _write_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")


def _write_expected_shard(path: Path, cells: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema": "mechanism_repair_physics.experiment_shard.v1",
                "cells": cells,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _cell(
    task_id: str,
    *,
    method: str = "frozen_model",
    budget: int = 32,
) -> dict:
    return {
        "split": "A",
        "task_id": task_id,
        "seed": 20260610,
        "method": method,
        "budget": budget,
    }


def test_merge_shard_rows_normalizes_relative_paths(tmp_path: Path) -> None:
    shard_root = tmp_path / "shard_runs"
    shard_dir = shard_root / "shard_0000"
    (shard_dir / "raw").mkdir(parents=True)
    (shard_dir / "raw" / "completion.txt").write_text("design\n")
    _write_row(
        shard_dir / "cell_results.jsonl",
        {
            "split": "A",
            "task_id": "task_a",
            "seed": 20260610,
            "method": "frozen_model",
            "raw_completion_paths": ["raw/completion.txt"],
            "verifier_output_paths": [],
        },
    )

    rows, summaries = load_shard_rows(shard_root, require_all_shards=1)

    assert len(rows) == 1
    assert len(summaries) == 1
    assert rows[0]["raw_completion_paths"] == [
        str((shard_dir / "raw" / "completion.txt").resolve())
    ]


def test_merge_shard_rows_accepts_complete_expected_manifest(
    tmp_path: Path,
) -> None:
    shard_root = tmp_path / "shard_runs"
    expected_dir = tmp_path / "experiment_shards"
    row = _cell("task_a")
    _write_row(shard_root / "shard_0000" / "cell_results.jsonl", row)
    _write_expected_shard(expected_dir / "shard_0000.json", [row])

    rows, summaries = load_shard_rows(
        shard_root,
        require_all_shards=1,
        expected_shard_dir=expected_dir,
    )

    assert len(rows) == 1
    assert summaries[0]["expected_rows"] == 1
    assert summaries[0]["missing_cell_count"] == 0
    assert summaries[0]["unexpected_cell_count"] == 0


def test_merge_shard_rows_rejects_incomplete_expected_manifest(
    tmp_path: Path,
) -> None:
    shard_root = tmp_path / "shard_runs"
    expected_dir = tmp_path / "experiment_shards"
    _write_row(shard_root / "shard_0000" / "cell_results.jsonl", _cell("task_a"))
    _write_expected_shard(
        expected_dir / "shard_0000.json",
        [_cell("task_a"), _cell("task_b")],
    )

    with pytest.raises(SystemExit, match="incomplete or mismatched shard coverage"):
        load_shard_rows(
            shard_root,
            require_all_shards=1,
            expected_shard_dir=expected_dir,
        )


def test_merge_shard_rows_distinguishes_budget_curve_cells(
    tmp_path: Path,
) -> None:
    shard_root = tmp_path / "shard_runs"
    expected_dir = tmp_path / "experiment_shards"
    low_budget = _cell("task_a", budget=16)
    high_budget = _cell("task_a", budget=32)
    _write_row(shard_root / "shard_0000" / "cell_results.jsonl", low_budget)
    _write_row(shard_root / "shard_0001" / "cell_results.jsonl", high_budget)
    _write_expected_shard(expected_dir / "shard_0000.json", [low_budget])
    _write_expected_shard(expected_dir / "shard_0001.json", [high_budget])

    rows, _summaries = load_shard_rows(
        shard_root,
        require_all_shards=2,
        expected_shard_dir=expected_dir,
    )

    assert [row["budget"] for row in rows] == [16, 32]


def test_merge_shard_rows_rejects_duplicate_cells(tmp_path: Path) -> None:
    shard_root = tmp_path / "shard_runs"
    row = {
        "split": "A",
        "task_id": "task_a",
        "seed": 20260610,
        "method": "frozen_model",
    }
    _write_row(shard_root / "shard_0000" / "cell_results.jsonl", row)
    _write_row(shard_root / "shard_0001" / "cell_results.jsonl", row)

    with pytest.raises(SystemExit, match="duplicate"):
        load_shard_rows(shard_root, require_all_shards=2)


def test_merge_reclassifies_missing_collision_geometry_as_design_failure(
    tmp_path: Path,
) -> None:
    shard_root = tmp_path / "shard_runs"
    shard_dir = shard_root / "shard_0000"
    verifier = shard_dir / "verifier_outputs.jsonl"
    verifier.parent.mkdir(parents=True)
    verifier.write_text(
        "required contact bodies lack Chrono collision geometry: body disc\n",
        encoding="utf-8",
    )
    _write_row(
        shard_dir / "cell_results.jsonl",
        {
            "split": "A",
            "task_id": "task_a",
            "seed": 20260610,
            "method": "llm_evolve_no_update",
            "failure_codes": ["capability_unavailable"],
            "verifier_output_paths": ["verifier_outputs.jsonl"],
        },
    )

    rows, _summaries = load_shard_rows(shard_root, require_all_shards=1)

    assert rows[0]["failure_codes"] == ["invalid_artifact"]
    assert rows[0]["capability_unavailable_reclassified"] is True


def test_merge_preserves_true_capability_unavailable_without_collision_evidence(
    tmp_path: Path,
) -> None:
    shard_root = tmp_path / "shard_runs"
    shard_dir = shard_root / "shard_0000"
    verifier = shard_dir / "verifier_outputs.jsonl"
    verifier.parent.mkdir(parents=True)
    verifier.write_text("pychrono not importable\n", encoding="utf-8")
    _write_row(
        shard_dir / "cell_results.jsonl",
        {
            "split": "A",
            "task_id": "task_a",
            "seed": 20260610,
            "method": "llm_evolve_no_update",
            "failure_codes": ["capability_unavailable"],
            "verifier_output_paths": ["verifier_outputs.jsonl"],
        },
    )

    rows, _summaries = load_shard_rows(shard_root, require_all_shards=1)

    assert rows[0]["failure_codes"] == ["capability_unavailable"]
    assert "capability_unavailable_reclassified" not in rows[0]
