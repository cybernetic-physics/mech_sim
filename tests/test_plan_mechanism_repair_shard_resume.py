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
    shard_dir = run_dir / "shard_runs" / f"shard_{index:04d}"
    materialized_rows = [
        _row_with_default_evidence(shard_dir, row, row_index)
        for row_index, row in enumerate(rows)
    ]
    path = run_dir / "shard_runs" / f"shard_{index:04d}" / "cell_results.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in materialized_rows),
        encoding="utf-8",
    )


def _row_with_default_evidence(
    shard_dir: Path,
    row: dict,
    row_index: int,
) -> dict:
    out = dict(row)

    def add_path(field: str, relative: str, text: str) -> None:
        path = shard_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        out[field] = [relative]

    if "raw_completion_paths" not in out and "raw_completion_path" not in out:
        add_path("raw_completion_paths", f"raw/{row_index}.txt", "design\n")
    if "verifier_output_paths" not in out and "verifier_output_path" not in out:
        add_path("verifier_output_paths", f"verifier/{row_index}.json", "{}\n")
    verifier_level = int(out.get("verifier_level", 1) or 1)
    if (
        verifier_level >= 2
        and "cad_artifact_paths" not in out
        and "cad_artifact_path" not in out
    ):
        add_path("cad_artifact_paths", f"cad/{row_index}.json", "{}\n")
    if (
        verifier_level >= 3
        and "chrono_output_paths" not in out
        and "chrono_output_path" not in out
    ):
        add_path("chrono_output_paths", f"chrono/{row_index}.json", "{}\n")
    return out


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
    assert report["next_shard_file"] == str(
        run_dir / "experiment_shards" / "shard_0000.json"
    )
    assert report["next_shard_output_dir"] == str(
        run_dir / "shard_runs" / "shard_0000"
    )
    assert report["local_shard_command"] == [
        ".venv/bin/python",
        "scripts/run_mechanism_repair_online_experiment.py",
        "--benchmark-dir",
        str(run_dir),
        "--out-dir",
        str(run_dir / "shard_runs" / "shard_0000"),
        "--cell-shard-file",
        str(run_dir / "experiment_shards" / "shard_0000.json"),
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
    assert "run_mechanism_repair_online_experiment.py" in (
        report["local_shard_command_text"]
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
    assert report["local_shard_command"] is None
    assert report["merge_command"] == [
        ".venv/bin/python",
        "scripts/merge_mechanism_repair_shards.py",
        "--benchmark-dir",
        str(run_dir),
        "--out-dir",
        str(run_dir),
        "--shard-root",
        str(run_dir / "shard_runs"),
        "--require-all-shards",
        "2",
        "--require-complete",
    ]
    assert "merge_mechanism_repair_shards.py" in report["merge_command_text"]
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


def test_resume_plan_rejects_ttrl_without_same_group_baseline(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    expected = [_cell("task_a", "mechanical_evolve_ttrl_tool_verified")]
    _write_shard(run_dir, 0, expected)

    report = build_report(run_dir=run_dir)

    assert report["merge_ready"] is False
    assert report["ttrl_baseline_group_error_count"] == 1
    assert report["shards"][0]["status"] == "invalid"
    assert report["shards"][0]["ttrl_baseline_group_errors"] == [
        {
            "split": "A",
            "task_id": "task_a",
            "seed": 20260610,
            "budget": 32,
            "ttrl_methods": ["mechanical_evolve_ttrl_tool_verified"],
            "missing_method": "llm_evolve_no_update",
        }
    ]


def test_resume_plan_rejects_missing_row_evidence(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    expected = [_cell("task_a", "frozen_model")]
    expected[0]["verifier_level"] = 3
    row = {
        **expected[0],
        "raw_completion_paths": [],
        "verifier_output_paths": [],
        "cad_artifact_paths": [],
        "chrono_output_paths": [],
    }
    _write_shard(run_dir, 0, expected)
    _write_rows(run_dir, 0, [row])

    report = build_report(run_dir=run_dir)

    assert report["merge_ready"] is False
    assert report["missing_evidence_count"] == 4
    assert report["shards"][0]["status"] == "partial"
    assert "missing evidence files" in report["shards"][0]["blockers"]
    assert {
        item["kind"] for item in report["shards"][0]["sample_missing_evidence"]
    } == {
        "raw_completions",
        "verifier_outputs",
        "cad_artifacts",
        "chrono_outputs",
    }


def test_resume_plan_rejects_missing_learning_evidence(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    expected = [
        _cell("task_a", "llm_evolve_no_update"),
        _cell("task_a", "mechanical_evolve_ttrl_tool_verified"),
    ]
    _write_shard(run_dir, 0, expected)
    _write_rows(run_dir, 0, expected)

    report = build_report(run_dir=run_dir)

    assert report["merge_ready"] is False
    assert report["missing_learning_count"] == 1
    assert report["shards"][0]["status"] == "partial"
    assert "missing learning evidence" in report["shards"][0]["blockers"]
    assert report["shards"][0]["sample_missing_learning"][0]["missing"] == [
        "training_logs",
        "adapter_checkpoints",
        "adapter_updates",
        "trained_tokens",
        "rl_datums",
        "rl_trained_tokens",
    ]
