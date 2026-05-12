"""Benchmark runner / aggregation tests."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from mech_bench.benchmark import (
    check_negative_controls,
    run_suite,
    run_task,
)
from mech_bench.dashboard_payload import build_benchmark_dashboard_payload
from mech_bench.generators.benchmark_suite import generate_suite


@pytest.fixture(scope="module")
def suite_dir(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("bench_suite")
    generate_suite(out, count_per_family=1, base_seed=7)
    return out


def test_run_suite_produces_benchmark_summary(suite_dir: Path, tmp_path):
    rd = tmp_path / "reports"
    summary = run_suite(suite_dir, report_dir=rd, eval_mode="public")
    assert (rd / "benchmark_summary.json").exists()
    on_disk = json.loads((rd / "benchmark_summary.json").read_text())
    assert on_disk["version"] == "mech_bench.benchmark_summary.v1"
    assert on_disk["n_tasks"] == summary["n_tasks"]
    # Aggregate fields documented in benchmark.py:
    for k in (
        "overall_score_mean", "overall_score_median",
        "pass_rate", "hard_gate_pass_rate",
        "pass_by_tier", "score_by_tier",
        "pass_by_family", "score_by_family",
        "failure_code_histogram", "runtime_by_tier",
        "capability_unavailable_n",
    ):
        assert k in on_disk, k


def test_run_suite_per_task_reports_written(suite_dir: Path, tmp_path):
    rd = tmp_path / "reports"
    run_suite(suite_dir, report_dir=rd, eval_mode="public")
    fourbars = [p for p in rd.iterdir()
                if p.is_dir() and p.name.startswith("fourbar_path_")]
    assert fourbars, "no per-task report for fourbar tasks"
    one = fourbars[0]
    assert (one / "scorecard.json").exists()
    assert (one / "metrics.json").exists()
    assert (one / "dashboard_payload.json").exists()


def test_run_suite_tier_and_family_metrics_present(suite_dir: Path, tmp_path):
    rd = tmp_path / "reports"
    summary = run_suite(suite_dir, report_dir=rd, eval_mode="public")
    tiers = summary["pass_by_tier"]
    assert {"artifact_static", "planar_kinematics",
            "transmission_analytic", "contact_dynamics"} <= set(tiers)
    families = summary["pass_by_family"]
    # At least 5 distinct families from the requirement.
    assert len(families) >= 5


def test_run_suite_capability_unavailable_counted(suite_dir: Path, tmp_path):
    summary = run_suite(suite_dir, report_dir=tmp_path / "reports",
                          eval_mode="public")
    assert summary["capability_unavailable_n"] >= 2  # Tier 3 stubs


def test_run_suite_both_emits_generalization_gap(suite_dir: Path, tmp_path):
    rd = tmp_path / "reports"
    summary = run_suite(suite_dir, report_dir=rd, eval_mode="both")
    assert summary["public_score_mean"] is not None
    assert summary["hidden_score_mean"] is not None
    assert summary["generalization_gap_mean"] is not None
    for task in summary["tasks"]:
        # Both variants ran for every task.
        assert task["public_score"] is not None
        assert task["hidden_score"] is not None
    # Per-task report dirs contain variant subdirs in "both" mode.
    sample = next(p for p in rd.iterdir()
                  if p.is_dir() and p.name.startswith("fourbar_path_"))
    assert (sample / "public").is_dir()
    assert (sample / "hidden").is_dir()


def test_check_negative_controls_pass_on_reference_suite(
    suite_dir: Path,
):
    _, summary = check_negative_controls(suite_dir, eval_mode="public")
    assert summary["all_passed"], summary["failures"]


def test_check_negative_controls_detects_wrong_expectation(
    suite_dir: Path, tmp_path,
):
    """If we deliberately mark a control as expecting a code that
    will not appear, the checker must report it as failed."""
    sabotaged = tmp_path / "sabotaged"
    sabotaged.mkdir()
    # Copy the suite and break one expected_failures.json.
    src_task = next(p for p in suite_dir.iterdir()
                     if p.is_dir() and p.name.startswith("fourbar_path_"))
    dst_task = sabotaged / src_task.name
    shutil.copytree(src_task, dst_task)
    spec = json.loads((dst_task / "expected_failures.json").read_text())
    # Set an impossible expected code for the first control.
    spec["controls"][0]["expected_failure_codes"] = [
        "excessive_torque_ripple"
    ]
    (dst_task / "expected_failures.json").write_text(
        json.dumps(spec, indent=2)
    )
    _, summary = check_negative_controls(sabotaged, eval_mode="public")
    assert summary["all_passed"] is False
    assert summary["failures"], summary


def test_run_task_named_negative(suite_dir: Path, tmp_path):
    task_dir = next(p for p in suite_dir.iterdir()
                     if p.is_dir()
                     and p.name.startswith("fourbar_path_"))
    res = run_task(task_dir, submission="negative",
                   negative="wrong_mobility_extra_fixed",
                   eval_mode="public",
                   scratch_root=tmp_path / "scr")
    assert res.public_passed is False
    assert "wrong_mobility" in (res.public_failure_codes or [])


def test_benchmark_dashboard_payload_shape(suite_dir: Path, tmp_path):
    rd = tmp_path / "reports"
    summary = run_suite(suite_dir, report_dir=rd, eval_mode="both")
    payload = build_benchmark_dashboard_payload(summary)
    assert payload["version"] == "mech_bench.benchmark_dashboard_payload.v1"
    assert payload["overview"]["n_tasks"] == summary["n_tasks"]
    assert isinstance(payload["task_table"], list)
    assert payload["task_table"], "task_table should not be empty"
    assert isinstance(payload["tier_heatmap"], list)
    assert isinstance(payload["family_heatmap"], list)
    assert isinstance(payload["score_distribution"], list)
