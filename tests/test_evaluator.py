"""End-to-end evaluator tests.

These prove the inversion holds: a generic evaluator + capability
dispatch + probe registry score the reference solution highly and
catch each negative control on a different failure code.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mech_bench.evaluator import evaluate
from mech_bench.feedback import FailureCode


TASK_DIR = Path(__file__).resolve().parent.parent / "tasks" / "fourbar_path_t001"


def _codes(report) -> set[str]:
    return {f.code.value if hasattr(f.code, "value") else str(f.code)
            for f in report.feedback}


def test_reference_solution_passes(tmp_path):
    """The reference solution should pass the hard gate and score
    very close to 1.0 (it generated the target path)."""
    report = evaluate(
        TASK_DIR,
        TASK_DIR / "reference_solution",
        scratch_dir=tmp_path,
    )
    assert report.hard_gate_passed, report.feedback
    # The reference solution generated the target itself, so the
    # Chamfer distance should be essentially zero.
    chamfer = report.metrics["coupler_path.chamfer"]
    assert chamfer < 1e-6, f"reference chamfer {chamfer} not near zero"
    assert report.score > 0.99


def test_negative_control_wrong_coupler_offset_fails_path(tmp_path):
    """Same topology, perturbed coupler-point pose — should pass
    the mobility gate but fail the path probe."""
    report = evaluate(
        TASK_DIR,
        TASK_DIR / "negative_solutions" / "wrong_coupler_offset",
        scratch_dir=tmp_path,
    )
    assert report.hard_gate_passed, (
        "Wrong-offset control should still satisfy mobility=1; "
        f"got feedback {[f.code.value for f in report.feedback]}"
    )
    codes = _codes(report)
    assert "path_error" in codes
    assert report.score < 0.5, f"score {report.score} too forgiving"


def test_negative_control_extra_fixed_fails_mobility(tmp_path):
    """Adds an extra fixed joint → mobility drops below 1 → hard
    gate fails → score is exactly 0."""
    report = evaluate(
        TASK_DIR,
        TASK_DIR / "negative_solutions" / "wrong_mobility_extra_fixed",
        scratch_dir=tmp_path,
    )
    assert not report.hard_gate_passed
    codes = _codes(report)
    assert "wrong_mobility" in codes
    assert report.score == pytest.approx(0.0)


def test_capability_dispatch_picks_planar_adapter(tmp_path):
    """Indirect check: if dispatch is broken, the path probe would
    fail with SIMULATOR_DIVERGENCE rather than path_error. Reference
    succeeds → dispatch is working."""
    report = evaluate(
        TASK_DIR,
        TASK_DIR / "reference_solution",
        scratch_dir=tmp_path,
    )
    codes = _codes(report)
    assert "simulator_divergence" not in codes
    assert "capability_unavailable" not in codes


def test_invalid_submission_caught_before_probes(tmp_path):
    """A submission without design.py is rejected with
    INVALID_ARTIFACT, not a probe failure."""
    bad = tmp_path / "empty_submission"
    bad.mkdir()
    report = evaluate(TASK_DIR, bad, scratch_dir=tmp_path / "scr")
    assert not report.hard_gate_passed
    codes = _codes(report)
    assert FailureCode.INVALID_ARTIFACT.value in codes
    assert report.score == 0.0
