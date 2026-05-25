"""Evaluator-dispatch tests for the reusable probe library.

These exercise the planner + adapter dispatch on synthetic eval
configs to prove:

  * required_ports trips before any adapter runs when input_port is
    absent (no path-trace simulator divergence; the topology probe
    catches it first).
  * A multi-probe config that mixes NONE probes with adapter-backed
    probes runs each needed adapter exactly once.
  * Probes that ask for capabilities no registered adapter provides
    surface CAPABILITY_UNAVAILABLE rather than throwing.
  * Adapter config from [adapters.X] in eval_config.toml is plumbed
    through to adapter.run().
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mech_bench.adapters import (
    SimAdapter,
    _REGISTRY as ADAPTER_REGISTRY,
    register_adapter,
)
from mech_bench.evaluator import build_execution_plan, evaluate
from mech_bench.feedback import FailureCode
from mech_bench.probes import Capability
from mech_bench.schema import (
    DesignIR,
    EvalConfig,
    FeedbackVisibility,
    ProbeSpec,
    TaskSpec,
)


TASK_DIR = (Path(__file__).resolve().parent.parent
            / "tasks" / "fourbar_path_t001")


def _codes(report) -> set[str]:
    return {f.code.value if hasattr(f.code, "value") else str(f.code)
            for f in report.feedback}


@pytest.fixture
def _no_contact_adapters(monkeypatch):
    for name, adapter in list(ADAPTER_REGISTRY.items()):
        if Capability.CONTACT_FORCES in adapter.capabilities_provided:
            monkeypatch.delitem(ADAPTER_REGISTRY, name, raising=False)


# --------------------------------------------------------------------- #
# Planner unit tests                                                    #
# --------------------------------------------------------------------- #


def test_planner_routes_none_probes_without_adapter():
    cfg = EvalConfig(
        probes=[
            ProbeSpec(id="ports", type="required_ports", config={
                "ports": ["input_port"],
            }),
            ProbeSpec(id="mob", type="dof_grubler", config={"space": "planar"}),
        ],
    )
    plan = build_execution_plan(cfg)
    assert plan.adapters_to_run() == []
    for p in plan.probes:
        assert p.available
        assert p.adapter_type is None


def test_planner_picks_planar_adapter_for_planar_probes():
    cfg = EvalConfig(
        probes=[
            ProbeSpec(id="ratio", type="port_velocity_ratio", config={
                "expected": -4.0, "tolerance_pct": 5.0,
            }),
            ProbeSpec(id="lock", type="lockup", config={
                "min_output_motion_rad": 0.05,
            }),
        ],
    )
    plan = build_execution_plan(cfg)
    adapters = plan.adapters_to_run()
    assert adapters == ["planar_kinematics"]


def test_planner_marks_unavailable_for_unsatisfied_cap(_no_contact_adapters):
    cfg = EvalConfig(
        probes=[
            ProbeSpec(id="ce", type="contact_engagement", config={
                "required_pairs": ["a:b"],
            }),
            ProbeSpec(id="sf", type="safety_factor", config={"min_fos": 1.5}),
        ],
    )
    plan = build_execution_plan(cfg)
    for p in plan.probes:
        assert not p.available
        assert p.adapter_type is None
        assert "No registered adapter" in p.reason


# --------------------------------------------------------------------- #
# End-to-end dispatch on the four-bar reference                          #
# --------------------------------------------------------------------- #


def test_required_ports_catches_missing_input_before_path_trace(tmp_path):
    """A submission that omits ``input_port`` is rejected by the
    validation layer's MISSING_PORT before the planar adapter is
    invoked (so we never see path_error or simulator_divergence).

    The reference task already enumerates ``required_ports`` in
    task.toml; validation runs before any probe. We confirm that
    behavior is intact: a forged submission missing input_port short-
    circuits at validation.
    """
    forged = tmp_path / "submission"
    forged.mkdir()
    (forged / "design.py").write_text(
        "from pathlib import Path\n"
        "def build_design(out_dir: Path) -> dict:\n"
        "    return {\n"
        "        'schema_version': 'design_ir.v2',\n"
        "        'parts': [{'id': 'ground', 'fixed': True,\n"
        "                   'mass_kg': 0.0,\n"
        "                   'com_local_mm': (0.0, 0.0, 0.0)}],\n"
        "        'joints': [],\n"
        "        'ports': {},\n"
        "        'params': {},\n"
        "    }\n"
    )
    report = evaluate(TASK_DIR, forged, scratch_dir=tmp_path / "scr")
    assert not report.hard_gate_passed
    assert FailureCode.MISSING_PORT.value in _codes(report)
    assert FailureCode.PATH_ERROR.value not in _codes(report)
    assert FailureCode.SIMULATOR_DIVERGENCE.value not in _codes(report)


# --------------------------------------------------------------------- #
# Adapter config plumbing                                               #
# --------------------------------------------------------------------- #


@pytest.fixture
def _spy_adapter(monkeypatch):
    """Capture adapter.run config calls without permanently mutating
    the global adapter registry. Yields a list of captured configs.
    """
    captured: list[dict] = []
    original = ADAPTER_REGISTRY["planar_kinematics"]

    class SpyPlanar(original):  # type: ignore[misc, valid-type]
        def run(self, ir, config):  # type: ignore[override]
            captured.append(dict(config))
            return original.run(self, ir, config)

    monkeypatch.setitem(ADAPTER_REGISTRY, "planar_kinematics", SpyPlanar)
    return captured


def test_adapter_config_from_eval_config_is_plumbed(_spy_adapter, tmp_path):
    """An ``[adapters.planar_kinematics]`` table in eval_config.toml
    overrides the adapter's defaults at run time."""
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    fixtures = task_dir / "fixtures"
    fixtures.mkdir()
    # Copy reference fixture so the chamfer probe has a target.
    (fixtures / "target_path.csv").write_text(
        (TASK_DIR / "fixtures" / "target_path.csv").read_text()
    )
    (task_dir / "task.toml").write_text(
        (TASK_DIR / "task.toml").read_text()
    )
    (task_dir / "prompt.md").write_text("test prompt")
    eval_toml = (TASK_DIR / "eval_config.toml").read_text()
    eval_toml += (
        "\n[adapters.planar_kinematics]\n"
        "samples = 144\n"
        "strict_geometry = false\n"
    )
    (task_dir / "eval_config.toml").write_text(eval_toml)

    report = evaluate(task_dir, TASK_DIR / "reference_solution",
                      scratch_dir=tmp_path / "scr")
    assert report.hard_gate_passed
    assert _spy_adapter, "planar adapter should have run at least once"
    cfg = _spy_adapter[0]
    assert cfg["samples"] == 144
    assert cfg["strict_geometry"] is False


# --------------------------------------------------------------------- #
# Multi-adapter dispatch (NONE + planar)                                #
# --------------------------------------------------------------------- #


def test_multi_adapter_run_mixes_none_and_planar(tmp_path):
    """A config that lists a NONE probe + a planar probe should run
    the planar adapter exactly once for the planar probe, and skip it
    for the NONE one. We confirm via report timings + outcome.
    """
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    fixtures = task_dir / "fixtures"
    fixtures.mkdir()
    (fixtures / "target_path.csv").write_text(
        (TASK_DIR / "fixtures" / "target_path.csv").read_text()
    )
    (task_dir / "prompt.md").write_text("p")
    (task_dir / "task.toml").write_text(
        (TASK_DIR / "task.toml").read_text()
    )
    (task_dir / "eval_config.toml").write_text(
        '[[probes]]\n'
        'id = "ports"\n'
        'type = "required_ports"\n'
        'ports = ["input_port", "output_port", "coupler_point"]\n'
        'hard_gate = true\n\n'
        '[[probes]]\n'
        'id = "mobility"\n'
        'type = "dof_grubler"\n'
        'space = "planar"\n'
        'expected = 1\n'
        'hard_gate = true\n\n'
        '[[probes]]\n'
        'id = "coupler_path"\n'
        'type = "path_trace_chamfer"\n'
        'moving_frame = "coupler_point"\n'
        'target_csv = "target_path.csv"\n'
        'normalize = true\n'
        'max_chamfer = 0.05\n'
        'weight = 1.0\n\n'
        '[hard_gate]\n'
        'require = ["ports", "mobility"]\n'
    )

    report = evaluate(task_dir, TASK_DIR / "reference_solution",
                      scratch_dir=tmp_path / "scr")
    assert report.evaluation_valid, report.feedback
    assert report.hard_gate_passed, report.feedback
    assert "adapter.planar_kinematics" in report.timings
    # Adapter ran exactly once.
    runs = [k for k in report.timings
            if k.startswith("adapter.planar_kinematics")]
    assert len(runs) == 1
    # All three probe types appear.
    types = {r.probe_type for r in report.probe_results}
    assert {"required_ports", "dof_grubler", "path_trace_chamfer"} <= types
    # No probe surfaced CAPABILITY_UNAVAILABLE / SIMULATOR_DIVERGENCE.
    codes = _codes(report)
    assert FailureCode.CAPABILITY_UNAVAILABLE.value not in codes
    assert FailureCode.SIMULATOR_DIVERGENCE.value not in codes


def test_contact_probe_without_adapter_is_capability_unavailable(
    _no_contact_adapters, tmp_path,
):
    """A task asking for contact_engagement without any contact-force-
    capable adapter registered must yield CAPABILITY_UNAVAILABLE on
    that probe — not a silent pass or an opaque exception."""
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    fixtures = task_dir / "fixtures"
    fixtures.mkdir()
    (task_dir / "prompt.md").write_text("p")
    (task_dir / "task.toml").write_text(
        (TASK_DIR / "task.toml").read_text()
    )
    (task_dir / "eval_config.toml").write_text(
        '[[probes]]\n'
        'id = "mobility"\n'
        'type = "dof_grubler"\n'
        'space = "planar"\n'
        'expected = 1\n'
        'hard_gate = true\n\n'
        '[[probes]]\n'
        'id = "contact"\n'
        'type = "contact_engagement"\n'
        'required_pairs = ["a:b"]\n'
        'min_rms_force_N = 0.5\n'
        'min_engagement_fraction = 0.05\n'
        'weight = 1.0\n\n'
        '[hard_gate]\n'
        'require = ["mobility"]\n'
    )
    report = evaluate(task_dir, TASK_DIR / "reference_solution",
                      scratch_dir=tmp_path / "scr")
    codes = _codes(report)
    assert FailureCode.CAPABILITY_UNAVAILABLE.value in codes
    assert report.evaluation_valid is False
