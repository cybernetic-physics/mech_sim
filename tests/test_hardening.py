"""End-to-end tests for the hardened evaluator.

Covers:
  * capability_unavailable surfaces when no adapter is registered for
    a probe's requirements (not simulator_divergence).
  * Malicious geometry paths are rejected before any probe runs.
  * write_report_bundle produces the expected four files.
  * Public report redacts private_trace and hidden metrics.
"""

from __future__ import annotations

import importlib
import json
import textwrap
from pathlib import Path

import pytest

from mech_bench.evaluator import (
    build_execution_plan,
    evaluate,
    load_task,
    write_report_bundle,
)
from mech_bench.feedback import Failure, FailureCode, Severity
from mech_bench.probes import Capability, Probe, register_probe
from mech_bench.schema import (
    EvalConfig,
    EvalReport,
    FeedbackVisibility,
    ProbeResult,
    ProbeSpec,
)


TASK_DIR = Path(__file__).resolve().parent.parent / "tasks" / "fourbar_path_t001"


def _make_task(tmp_path: Path, design_src: str,
               eval_extra: str = "",
               required_ports: str = '["input_port"]') -> tuple[Path, Path]:
    """Materialize a minimal task + submission pair on disk."""
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / "fixtures").mkdir()
    (task_dir / "task.toml").write_text(textwrap.dedent(f"""
        [task]
        id = "t_test"
        family = "planar_test"
        difficulty = 1
        units = "mm"

        [requirements]
        required_ports = {required_ports}
        expected_mobility = 1
    """).strip())
    (task_dir / "eval_config.toml").write_text(textwrap.dedent(f"""
        [[probes]]
        id = "mobility"
        type = "dof_grubler"
        space = "planar"
        expected = 1
        hard_gate = true
        severity = "critical"
        {eval_extra}
    """).strip())

    sub = tmp_path / "submission"
    sub.mkdir()
    (sub / "design.py").write_text(design_src)
    return task_dir, sub


def test_capability_unavailable_when_no_adapter():
    """A probe whose capabilities no adapter provides should produce
    CAPABILITY_UNAVAILABLE, not SIMULATOR_DIVERGENCE."""

    @register_probe
    class _NeedsExoticCap(Probe):
        type_name = "_needs_exotic"
        capabilities_required = frozenset({Capability.FEA_STATIC})

        def run(self, ir, sim_outputs, config):  # pragma: no cover
            return ProbeResult(probe_id="", probe_type=self.type_name,
                                passed=True, score=1.0)

    cfg = EvalConfig(probes=[
        ProbeSpec(id="exotic", type="_needs_exotic", weight=1.0),
    ])
    plan = build_execution_plan(cfg)
    assert plan.probes[0].available is False
    assert plan.probes[0].adapter_type is None

    # Clean up registry so this doesn't leak into other tests.
    from mech_bench.probes import _REGISTRY
    _REGISTRY.pop("_needs_exotic", None)


def test_unknown_probe_type_marked_unavailable():
    cfg = EvalConfig(probes=[
        ProbeSpec(id="nope", type="not_a_real_probe", weight=1.0),
    ])
    plan = build_execution_plan(cfg)
    p0 = plan.probes[0]
    assert p0.available is False
    assert p0.probe_known is False


def test_malicious_geometry_path_fails_before_probes(tmp_path: Path):
    design_src = textwrap.dedent('''
        from pathlib import Path
        def build_design(out_dir: Path) -> dict:
            return {
                "schema_version": "design_ir.v2",
                "parts": [
                    {"id": "ground", "fixed": True, "mass_kg": 0.0,
                     "geometry": {"mesh": "../../etc/passwd"}},
                    {"id": "crank", "mass_kg": 0.02},
                ],
                "joints": [{
                    "id": "j1", "type": "revolute",
                    "parent": "ground", "child": "crank",
                    "axis_world": (0.0, 0.0, 1.0),
                    "anchor_world_mm": (0.0, 0.0, 0.0),
                }],
                "ports": {
                    "input_port": {
                        "id": "input_port", "part": "j1",
                        "kind": "revolute_joint",
                    },
                },
            }
    ''')
    task_dir, sub = _make_task(tmp_path, design_src)
    report = evaluate(task_dir, sub, scratch_dir=tmp_path / "scratch")
    assert report.hard_gate_passed is False
    assert report.score == 0.0
    codes = {f.code.value for f in report.feedback}
    assert FailureCode.INVALID_ARTIFACT.value in codes
    # Probes must NOT have run when validation hard-fails.
    assert report.probe_results == []


def test_missing_required_port_short_circuits_probes(tmp_path: Path):
    design_src = textwrap.dedent('''
        from pathlib import Path
        def build_design(out_dir: Path) -> dict:
            return {
                "schema_version": "design_ir.v2",
                "parts": [
                    {"id": "ground", "fixed": True, "mass_kg": 0.0},
                    {"id": "crank", "mass_kg": 0.02},
                ],
                "joints": [{
                    "id": "j1", "type": "revolute",
                    "parent": "ground", "child": "crank",
                    "axis_world": (0.0, 0.0, 1.0),
                    "anchor_world_mm": (0.0, 0.0, 0.0),
                }],
                "ports": {},
            }
    ''')
    task_dir, sub = _make_task(tmp_path, design_src)
    report = evaluate(task_dir, sub, scratch_dir=tmp_path / "scratch")
    assert report.hard_gate_passed is False
    codes = {f.code.value for f in report.feedback}
    assert FailureCode.MISSING_PORT.value in codes
    assert report.probe_results == []


def test_negative_mass_rejected_before_probes(tmp_path: Path):
    design_src = textwrap.dedent('''
        from pathlib import Path
        def build_design(out_dir: Path) -> dict:
            return {
                "schema_version": "design_ir.v2",
                "parts": [
                    {"id": "ground", "fixed": True, "mass_kg": 0.0},
                    {"id": "crank", "mass_kg": -0.5},
                ],
                "joints": [{
                    "id": "j1", "type": "revolute",
                    "parent": "ground", "child": "crank",
                    "axis_world": (0.0, 0.0, 1.0),
                    "anchor_world_mm": (0.0, 0.0, 0.0),
                }],
                "ports": {
                    "input_port": {
                        "id": "input_port", "part": "j1",
                        "kind": "revolute_joint",
                    },
                },
            }
    ''')
    task_dir, sub = _make_task(tmp_path, design_src)
    report = evaluate(task_dir, sub, scratch_dir=tmp_path / "scratch")
    assert report.hard_gate_passed is False
    codes = {f.code.value for f in report.feedback}
    assert FailureCode.INVALID_MASS_PROPERTIES.value in codes


def test_report_bundle_writes_expected_files(tmp_path: Path):
    report = evaluate(
        TASK_DIR,
        TASK_DIR / "reference_solution",
        scratch_dir=tmp_path / "scratch",
    )
    _, cfg = load_task(TASK_DIR)
    out_dir = tmp_path / "bundle"
    paths = write_report_bundle(report, out_dir, visibility=cfg.visibility)

    for key in ("scorecard", "scorecard_public", "metrics",
                "feedback_public"):
        assert key in paths
        assert paths[key].exists()

    scorecard = json.loads((out_dir / "scorecard.json").read_text())
    assert scorecard["version"] == "eval_report.v1"
    assert scorecard["task_id"] == "fourbar_path_t001"
    assert scorecard["task_family"] == "planar_4bar"
    assert "run_id" in scorecard and scorecard["run_id"]
    assert "tier_results" in scorecard
    assert "timings" in scorecard

    pub = json.loads((out_dir / "scorecard.public.json").read_text())
    # Public report must not leak full metrics outside the allowlist.
    allowed = set(cfg.visibility.public_metrics)
    for k in pub["metrics"]:
        assert k in allowed, f"metric {k!r} leaked into public report"

    metrics_blob = json.loads((out_dir / "metrics.json").read_text())
    assert "score" in metrics_blob
    assert "hard_gate_passed" in metrics_blob


def test_public_report_redacts_private_trace():
    failure = Failure(
        code=FailureCode.PATH_ERROR,
        severity=Severity.MAJOR,
        message="x",
        private_trace="/tmp/secret_trace.h5",
    )
    report = EvalReport(
        task_id="t", task_family="planar_test", difficulty=1,
        run_id="run_x",
        score=0.0, hard_gate_passed=False,
        probe_results=[],
        feedback=[failure],
    )
    pub = report.to_dict(public=True, visibility=FeedbackVisibility())
    full = report.to_dict(public=False, visibility=FeedbackVisibility())
    assert "private_trace" not in pub["feedback"][0]
    assert full["feedback"][0]["private_trace"] == "/tmp/secret_trace.h5"


def test_public_report_redacts_hidden_metrics():
    report = EvalReport(
        task_id="t", task_family="planar_test", difficulty=1,
        run_id="run_x",
        score=1.0, hard_gate_passed=True,
        probe_results=[],
        metrics={"public_one.x": 1.0, "hidden.x": 9.0,
                 "neither.x": 2.0},
    )
    vis = FeedbackVisibility(
        public_metrics=["public_one.x"],
        hidden_metrics=["hidden.x"],
    )
    pub = report.to_dict(public=True, visibility=vis)
    assert "public_one.x" in pub["metrics"]
    assert "hidden.x" not in pub["metrics"]
    assert "neither.x" not in pub["metrics"]


def test_reference_still_passes_after_hardening(tmp_path: Path):
    report = evaluate(
        TASK_DIR,
        TASK_DIR / "reference_solution",
        scratch_dir=tmp_path,
    )
    assert report.hard_gate_passed is True
    assert report.score > 0.99


def test_negative_extra_fixed_still_fails_mobility(tmp_path: Path):
    """Wrong-mobility control adds a fixed joint creating two fixed
    parts...wait, no: it adds a fixed *joint*, which is a topology
    change but no second fixed part. So mobility should still flag
    it. Make sure validation didn't swallow it."""
    report = evaluate(
        TASK_DIR,
        TASK_DIR / "negative_solutions" / "wrong_mobility_extra_fixed",
        scratch_dir=tmp_path,
    )
    assert report.hard_gate_passed is False
    codes = {f.code.value for f in report.feedback}
    assert FailureCode.WRONG_MOBILITY.value in codes


def test_negative_coupler_offset_still_fails_path(tmp_path: Path):
    report = evaluate(
        TASK_DIR,
        TASK_DIR / "negative_solutions" / "wrong_coupler_offset",
        scratch_dir=tmp_path,
    )
    assert report.hard_gate_passed is True
    codes = {f.code.value for f in report.feedback}
    assert FailureCode.PATH_ERROR.value in codes
