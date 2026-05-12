"""Adversarial tests for validation strictness, inertia eigvals,
redaction consistency, evaluation_valid semantics, and the
all-hard-gate scoring fix.

These tests exist because the v1 audit found that the surface
between "trusted evaluator" and "agent submission" was leaking in
several subtle ways. Each test pins one of those holes shut.
"""

from __future__ import annotations

import json
import math
import textwrap
from pathlib import Path

import pytest

from mech_bench.evaluator import (
    build_execution_plan,
    evaluate,
    load_task,
    sanitize_metric_value,
    sanitize_metrics_dict,
    sanitize_report_for_json,
    write_report_bundle,
)
from mech_bench.feedback import Failure, FailureCode, Severity
from mech_bench.probes import (
    Capability,
    Probe,
    _REGISTRY as PROBE_REGISTRY,
    register_probe,
)
from mech_bench.schema import (
    DesignIR,
    EvalConfig,
    EvalReport,
    FeedbackVisibility,
    ProbeResult,
    ProbeSpec,
    TaskSpec,
)
from mech_bench.validation import validate_design_ir


TASK_DIR = Path(__file__).resolve().parent.parent / "tasks" / "fourbar_path_t001"


def _good_ir_dict() -> dict:
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
        "ports": {"input_port": {
            "id": "input_port", "part": "j1",
            "kind": "revolute_joint",
        }},
    }


def _codes(failures) -> set[str]:
    return {f.code.value for f in failures}


# --------------------------------------------------------------------- #
# Totality: validation never raises                                     #
# --------------------------------------------------------------------- #


def test_validation_total_on_nondict_geometry():
    raw = _good_ir_dict()
    raw["parts"][1]["geometry"] = ["not", "a", "dict"]
    ir = DesignIR.from_dict(raw)
    failures = validate_design_ir(ir)
    assert FailureCode.INVALID_ARTIFACT.value in _codes(failures)


def test_validation_total_on_geometry_with_null_byte(tmp_path: Path):
    raw = _good_ir_dict()
    raw["parts"][1]["geometry"] = {"mesh": "mesh\x00.stl"}
    ir = DesignIR.from_dict(raw)
    failures = validate_design_ir(ir, build_root=tmp_path)
    assert FailureCode.INVALID_ARTIFACT.value in _codes(failures)


def test_validation_total_on_geometry_with_control_chars(tmp_path: Path):
    raw = _good_ir_dict()
    raw["parts"][1]["geometry"] = {"mesh": "abc\ndef.stl"}
    ir = DesignIR.from_dict(raw)
    failures = validate_design_ir(ir, build_root=tmp_path)
    assert FailureCode.INVALID_ARTIFACT.value in _codes(failures)


def test_validation_total_on_backslash_traversal(tmp_path: Path):
    raw = _good_ir_dict()
    raw["parts"][1]["geometry"] = {"mesh": "..\\..\\secret.stl"}
    ir = DesignIR.from_dict(raw)
    failures = validate_design_ir(ir, build_root=tmp_path)
    assert FailureCode.INVALID_ARTIFACT.value in _codes(failures)


def test_validation_total_on_raw_backslash(tmp_path: Path):
    raw = _good_ir_dict()
    raw["parts"][1]["geometry"] = {"mesh": "subdir\\mesh.stl"}
    ir = DesignIR.from_dict(raw)
    failures = validate_design_ir(ir, build_root=tmp_path)
    assert FailureCode.INVALID_ARTIFACT.value in _codes(failures)


def test_validation_total_on_symlink_escape(tmp_path: Path):
    # Put a symlink inside build_root that points outside.
    outside = tmp_path / "outside_secret"
    outside.write_text("nope")
    build_root = tmp_path / "build"
    build_root.mkdir()
    link = build_root / "lnk.stl"
    link.symlink_to(outside)
    raw = _good_ir_dict()
    raw["parts"][1]["geometry"] = {"mesh": "lnk.stl"}
    ir = DesignIR.from_dict(raw)
    failures = validate_design_ir(ir, build_root=build_root)
    assert FailureCode.INVALID_ARTIFACT.value in _codes(failures)


def test_validation_does_not_raise_on_garbage_inertia():
    """Even a deeply malformed inertia must come back as a Failure,
    not an exception."""
    raw = _good_ir_dict()
    raw["parts"][1]["inertia_kg_m2"] = "definitely not a 3x3 matrix"
    ir = DesignIR.from_dict(raw)
    # Should not raise:
    failures = validate_design_ir(ir)
    assert FailureCode.INVALID_MASS_PROPERTIES.value in _codes(failures)


# --------------------------------------------------------------------- #
# Strict type / enum / id checks                                        #
# --------------------------------------------------------------------- #


def test_part_fixed_must_be_bool():
    raw = _good_ir_dict()
    raw["parts"][0]["fixed"] = 1  # truthy int, not bool
    ir = DesignIR.from_dict(raw)
    failures = validate_design_ir(ir)
    assert FailureCode.SCHEMA_ERROR.value in _codes(failures)


def test_joint_type_must_be_known():
    raw = _good_ir_dict()
    raw["joints"][0]["type"] = "wormhole"
    ir = DesignIR.from_dict(raw)
    failures = validate_design_ir(ir)
    assert FailureCode.SCHEMA_ERROR.value in _codes(failures)


def test_port_kind_must_be_known():
    raw = _good_ir_dict()
    raw["ports"]["input_port"]["kind"] = "telepathic"
    ir = DesignIR.from_dict(raw)
    failures = validate_design_ir(ir)
    assert FailureCode.SCHEMA_ERROR.value in _codes(failures)


def test_part_id_with_slash_rejected():
    raw = _good_ir_dict()
    raw["parts"][1]["id"] = "crank/evil"
    raw["joints"][0]["child"] = "crank/evil"
    ir = DesignIR.from_dict(raw)
    failures = validate_design_ir(ir)
    assert FailureCode.SCHEMA_ERROR.value in _codes(failures)


def test_part_id_with_whitespace_rejected():
    raw = _good_ir_dict()
    raw["parts"][1]["id"] = "  whitespace  "
    raw["joints"][0]["child"] = "  whitespace  "
    ir = DesignIR.from_dict(raw)
    failures = validate_design_ir(ir)
    assert FailureCode.SCHEMA_ERROR.value in _codes(failures)


def test_part_id_with_control_char_rejected():
    raw = _good_ir_dict()
    raw["parts"][1]["id"] = "crank\x01"
    raw["joints"][0]["child"] = "crank\x01"
    ir = DesignIR.from_dict(raw)
    failures = validate_design_ir(ir)
    assert FailureCode.SCHEMA_ERROR.value in _codes(failures)


def test_part_id_starting_with_digit_rejected():
    raw = _good_ir_dict()
    raw["parts"][1]["id"] = "1crank"
    raw["joints"][0]["child"] = "1crank"
    ir = DesignIR.from_dict(raw)
    failures = validate_design_ir(ir)
    assert FailureCode.SCHEMA_ERROR.value in _codes(failures)


def test_part_params_must_be_dict():
    raw = _good_ir_dict()
    raw["parts"][1]["params"] = "not a dict"
    ir = DesignIR.from_dict(raw)
    failures = validate_design_ir(ir)
    assert FailureCode.SCHEMA_ERROR.value in _codes(failures)


def test_joint_params_must_be_dict():
    raw = _good_ir_dict()
    raw["joints"][0]["params"] = ["not", "dict"]
    ir = DesignIR.from_dict(raw)
    failures = validate_design_ir(ir)
    assert FailureCode.SCHEMA_ERROR.value in _codes(failures)


def test_designir_params_must_be_dict():
    raw = _good_ir_dict()
    raw["params"] = [1, 2, 3]
    ir = DesignIR.from_dict(raw)
    failures = validate_design_ir(ir)
    assert FailureCode.SCHEMA_ERROR.value in _codes(failures)


# --------------------------------------------------------------------- #
# Inertia eigenvalue-based validation                                   #
# --------------------------------------------------------------------- #


def test_off_diagonal_inertia_with_negative_eigenvalue_rejected():
    """[[1,2,0],[2,1,0],[0,0,1]] is symmetric but has eigenvalues
    {3, -1, 1}. Diagonal-only checks would accept it."""
    raw = _good_ir_dict()
    raw["parts"][1]["inertia_kg_m2"] = (
        (1.0, 2.0, 0.0),
        (2.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    ir = DesignIR.from_dict(raw)
    failures = validate_design_ir(ir)
    assert FailureCode.INVALID_MASS_PROPERTIES.value in _codes(failures)


def test_all_zero_inertia_on_positive_mass_part_rejected():
    raw = _good_ir_dict()
    raw["parts"][1]["inertia_kg_m2"] = (
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
    )
    ir = DesignIR.from_dict(raw)
    failures = validate_design_ir(ir)
    assert FailureCode.INVALID_MASS_PROPERTIES.value in _codes(failures)


def test_diagonal_triangle_inequality_violation_rejected():
    raw = _good_ir_dict()
    raw["parts"][1]["inertia_kg_m2"] = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 100.0),
    )
    ir = DesignIR.from_dict(raw)
    failures = validate_design_ir(ir)
    assert FailureCode.INVALID_MASS_PROPERTIES.value in _codes(failures)


def test_valid_diagonal_inertia_passes():
    raw = _good_ir_dict()
    raw["parts"][1]["inertia_kg_m2"] = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.5),
    )
    ir = DesignIR.from_dict(raw)
    failures = validate_design_ir(ir)
    assert FailureCode.INVALID_MASS_PROPERTIES.value not in _codes(failures)


# --------------------------------------------------------------------- #
# Sanitation utilities                                                  #
# --------------------------------------------------------------------- #


def test_sanitize_metric_value_handles_nonfinite():
    assert sanitize_metric_value(float("nan")) is None
    assert sanitize_metric_value(float("inf")) is None
    assert sanitize_metric_value(-float("inf")) is None
    assert sanitize_metric_value(1.5) == 1.5
    assert sanitize_metric_value(None) is None


def test_sanitize_metrics_dict_filters_inf():
    out = sanitize_metrics_dict({"good": 1.0, "bad": float("inf")})
    assert out == {"good": 1.0, "bad": None}


def test_sanitize_report_for_json_recurses():
    blob = {"a": [float("nan"), 1.0], "b": {"c": float("inf")}}
    out = sanitize_report_for_json(blob)
    assert out == {"a": [None, 1.0], "b": {"c": None}}
    # Round-trips through strict json:
    json.dumps(out, allow_nan=False)


def test_report_bundle_is_strict_json(tmp_path: Path):
    """The report bundle must serialize cleanly under allow_nan=False
    even if a probe injected NaN/Inf into its metrics."""
    report = evaluate(
        TASK_DIR,
        TASK_DIR / "reference_solution",
        scratch_dir=tmp_path / "scratch",
    )
    # Inject a NaN by hand to simulate a faulty probe.
    report.metrics["fake.bad"] = float("inf")
    _, cfg = load_task(TASK_DIR)
    out_dir = tmp_path / "bundle"
    write_report_bundle(report, out_dir, visibility=cfg.visibility)
    # Should parse:
    for fname in ("scorecard.json", "scorecard.public.json",
                  "metrics.json", "feedback.public.json"):
        text = (out_dir / fname).read_text()
        json.loads(text)  # must not raise
        # And the raw string must not contain JSON-extension tokens.
        assert "NaN" not in text
        assert "Infinity" not in text


# --------------------------------------------------------------------- #
# Public redaction consistency                                          #
# --------------------------------------------------------------------- #


def test_hidden_metric_wins_over_public_allowlist():
    rep = EvalReport(
        task_id="t", task_family="planar_test", difficulty=1,
        run_id="r", score=1.0, hard_gate_passed=True,
        probe_results=[],
        metrics={"probe.x": 1.0, "probe.y": 2.0},
    )
    vis = FeedbackVisibility(
        public_metrics=["probe.x", "probe.y"],
        hidden_metrics=["probe.y"],
    )
    pub = rep.to_dict(public=True, visibility=vis)
    assert "probe.x" in pub["metrics"]
    assert "probe.y" not in pub["metrics"]


def test_probe_level_hidden_metric_redacted():
    pr = ProbeResult(
        probe_id="coupler_path", probe_type="path_trace_chamfer",
        passed=True, score=1.0,
        metrics={"chamfer": 0.01, "secret": 999.0},
    )
    rep = EvalReport(
        task_id="t", task_family="planar_test", difficulty=1,
        run_id="r", score=1.0, hard_gate_passed=True,
        probe_results=[pr],
        metrics={"coupler_path.chamfer": 0.01,
                 "coupler_path.secret": 999.0},
    )
    vis = FeedbackVisibility(
        public_metrics=["coupler_path.chamfer"],
        hidden_metrics=["coupler_path.secret"],
    )
    pub = rep.to_dict(public=True, visibility=vis)
    probe_view = pub["probe_results"][0]
    assert "chamfer" in probe_view["metrics"]
    assert "secret" not in probe_view["metrics"]
    assert "coupler_path.secret" not in pub["metrics"]


def test_no_public_allowlist_still_hides_hidden_metrics():
    rep = EvalReport(
        task_id="t", task_family="planar_test", difficulty=1,
        run_id="r", score=1.0, hard_gate_passed=True,
        probe_results=[],
        metrics={"x": 1.0, "secret": 2.0},
    )
    vis = FeedbackVisibility(public_metrics=[], hidden_metrics=["secret"])
    pub = rep.to_dict(public=True, visibility=vis)
    assert "x" in pub["metrics"]
    assert "secret" not in pub["metrics"]


def test_failure_extra_private_does_not_leak():
    f = Failure(
        code=FailureCode.PATH_ERROR,
        severity=Severity.MAJOR,
        message="x",
        extra={"trusted_only": "/tmp/secret_log.txt"},
        extra_public={"hint_id": "h42"},
    )
    pub = f.public()
    assert "extra" not in pub
    assert pub.get("extra_public") == {"hint_id": "h42"}


def test_failure_private_trace_redacted():
    f = Failure(
        code=FailureCode.PATH_ERROR,
        severity=Severity.MAJOR,
        message="x",
        private_trace="/tmp/private.h5",
    )
    assert "private_trace" not in f.public()


# --------------------------------------------------------------------- #
# Capability-unavailable invalidates the evaluation                     #
# --------------------------------------------------------------------- #


def _make_task_files(tmp_path: Path, probe_type: str,
                     hard_gate: bool, weight: float = 0.0,
                     extra_probe: str = "") -> tuple[Path, Path]:
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / "fixtures").mkdir()
    (task_dir / "task.toml").write_text(textwrap.dedent('''
        [task]
        id = "t_capability"
        family = "planar_test"
        difficulty = 1
        units = "mm"

        [requirements]
        required_ports = ["input_port"]
        expected_mobility = 1
    ''').strip())
    (task_dir / "eval_config.toml").write_text(textwrap.dedent(f'''
        [[probes]]
        id = "mobility"
        type = "dof_grubler"
        space = "planar"
        expected = 1
        hard_gate = true
        severity = "critical"

        [[probes]]
        id = "exotic"
        type = "{probe_type}"
        hard_gate = {str(hard_gate).lower()}
        weight = {weight}
        severity = "major"
        {extra_probe}
    ''').strip())
    sub = tmp_path / "submission"
    sub.mkdir()
    (sub / "design.py").write_text(textwrap.dedent('''
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
                "ports": {"input_port": {
                    "id": "input_port", "part": "j1",
                    "kind": "revolute_joint",
                }},
            }
    '''))
    return task_dir, sub


def test_unknown_probe_type_invalidates_evaluation(tmp_path: Path):
    task_dir, sub = _make_task_files(
        tmp_path, "does_not_exist_probe", hard_gate=False, weight=1.0)
    report = evaluate(task_dir, sub, scratch_dir=tmp_path / "scratch")
    assert report.evaluation_valid is False
    assert report.hard_gate_passed is False
    assert report.score == 0.0
    codes = {f.code.value for f in report.feedback}
    assert FailureCode.CAPABILITY_UNAVAILABLE.value in codes


def test_capability_unavailable_blocks_reward(tmp_path: Path):
    """Register a probe that needs an exotic capability with no
    adapter. Even though the mobility hard gate passes, the report
    must come back invalid with score=0."""

    @register_probe
    class _NeedsExotic(Probe):
        type_name = "_needs_exotic_strict"
        capabilities_required = frozenset({Capability.FEA_STATIC})

        def run(self, ir, sim_outputs, config):  # pragma: no cover
            return ProbeResult(probe_id="", probe_type=self.type_name,
                                passed=True, score=1.0)

    try:
        task_dir, sub = _make_task_files(
            tmp_path, "_needs_exotic_strict", hard_gate=False,
            weight=1.0)
        report = evaluate(task_dir, sub,
                          scratch_dir=tmp_path / "scratch")
        assert report.evaluation_valid is False, (
            "A probe with no adapter must invalidate the eval, even if "
            "the hard gate passes."
        )
        assert report.score == 0.0
        assert report.hard_gate_passed is False
    finally:
        PROBE_REGISTRY.pop("_needs_exotic_strict", None)


# --------------------------------------------------------------------- #
# All-hard-gate scoring                                                 #
# --------------------------------------------------------------------- #


def test_all_hard_gate_task_scores_one_on_pass(tmp_path: Path):
    """If the only probe is a hard gate that passes, the task is
    fully satisfied and score should be 1.0 (not 0.0)."""
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / "fixtures").mkdir()
    (task_dir / "task.toml").write_text(textwrap.dedent('''
        [task]
        id = "t_all_gate"
        family = "planar_test"
        difficulty = 1
        units = "mm"

        [requirements]
        required_ports = ["input_port"]
        expected_mobility = 1
    ''').strip())
    (task_dir / "eval_config.toml").write_text(textwrap.dedent('''
        [[probes]]
        id = "mobility"
        type = "dof_grubler"
        space = "planar"
        expected = 1
        hard_gate = true
        severity = "critical"
    ''').strip())
    sub = tmp_path / "submission"
    sub.mkdir()
    (sub / "design.py").write_text(textwrap.dedent('''
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
                "ports": {"input_port": {
                    "id": "input_port", "part": "j1",
                    "kind": "revolute_joint",
                }},
            }
    '''))
    report = evaluate(task_dir, sub, scratch_dir=tmp_path / "scratch")
    assert report.evaluation_valid is True
    assert report.hard_gate_passed is True
    assert report.score == pytest.approx(1.0)


# --------------------------------------------------------------------- #
# Non-finite probe score handled                                        #
# --------------------------------------------------------------------- #


def test_probe_returning_nan_invalidates_evaluation(tmp_path: Path):
    """If a probe somehow returns NaN, the evaluator must not silently
    pass it through to the agent."""

    @register_probe
    class _NanScore(Probe):
        type_name = "_nan_score"
        capabilities_required = frozenset({Capability.NONE})

        def run(self, ir, sim_outputs, config):
            return ProbeResult(probe_id="", probe_type=self.type_name,
                                passed=True, score=float("nan"))

    try:
        task_dir, sub = _make_task_files(
            tmp_path, "_nan_score", hard_gate=False, weight=1.0)
        report = evaluate(task_dir, sub,
                          scratch_dir=tmp_path / "scratch")
        assert report.evaluation_valid is False
        # Score is finite even though the probe returned NaN.
        assert math.isfinite(report.score)
        codes = {f.code.value for f in report.feedback}
        assert FailureCode.SIMULATOR_DIVERGENCE.value in codes
    finally:
        PROBE_REGISTRY.pop("_nan_score", None)
