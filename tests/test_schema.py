"""Schema sanity tests."""

from __future__ import annotations

from mech_bench.feedback import Failure, FailureCode, Severity
from mech_bench.schema import (
    DesignIR,
    EvalConfig,
    Joint,
    Part,
    Port,
    ProbeSpec,
    TaskSpec,
)


def test_design_ir_from_dict_roundtrip():
    raw = {
        "schema_version": "design_ir.v2",
        "parts": [
            {"id": "ground", "fixed": True, "mass_kg": 0.0},
            {"id": "crank", "mass_kg": 0.02},
        ],
        "joints": [
            {
                "id": "j1", "type": "revolute",
                "parent": "ground", "child": "crank",
                "axis_world": (0.0, 0.0, 1.0),
                "anchor_world_mm": (0.0, 0.0, 0.0),
            },
        ],
        "ports": {
            "input_port": {
                "id": "input_port", "part": "j1",
                "kind": "revolute_joint",
            },
        },
    }
    ir = DesignIR.from_dict(raw)
    assert ir.schema_version == "design_ir.v2"
    assert len(ir.parts) == 2
    assert ir.parts[0].fixed is True
    assert isinstance(ir.parts[0], Part)
    assert isinstance(ir.joints[0], Joint)
    assert isinstance(ir.ports["input_port"], Port)
    assert ir.parts[0].id == "ground"


def test_task_spec_from_dict():
    d = {
        "task": {"id": "t1", "family": "planar_4bar", "difficulty": 2,
                 "units": "mm"},
        "requirements": {"required_ports": ["input_port"],
                          "expected_mobility": 1,
                          "max_envelope_mm": [200, 200, 50]},
        "objective": {"description": "trace"},
    }
    task = TaskSpec.from_dict(d, prompt="hello")
    assert task.id == "t1"
    assert task.expected_mobility == 1
    assert task.envelope_mm == (200, 200, 50)
    assert task.prompt == "hello"


def test_eval_config_from_dict():
    d = {
        "probes": [
            {"id": "mobility", "type": "dof_grubler",
             "expected": 1, "hard_gate": True},
            {"id": "path", "type": "path_trace_chamfer",
             "weight": 1.0, "moving_frame": "coupler_point",
             "target_csv": "x.csv", "max_chamfer": 0.05},
        ],
        "hard_gate": {"require": ["mobility"]},
        "feedback": {"public_metrics": ["path.chamfer"]},
    }
    cfg = EvalConfig.from_dict(d)
    assert len(cfg.probes) == 2
    assert cfg.probes[0].id == "mobility"
    assert cfg.probes[0].hard_gate is True
    assert cfg.probes[1].weight == 1.0
    assert cfg.probes[1].config["target_csv"] == "x.csv"
    assert cfg.hard_gate_probes == ["mobility"]
    assert cfg.visibility.public_metrics == ["path.chamfer"]


def test_failure_grammar_is_closed_enum():
    # The codes are an Enum — agents can rely on membership.
    assert FailureCode("wrong_mobility") is FailureCode.WRONG_MOBILITY
    f = Failure(
        code=FailureCode.PATH_ERROR,
        severity=Severity.MAJOR,
        message="x",
        metric="chamfer",
        observed=0.1,
        target=0.05,
        private_trace="/tmp/run123/hidden.h5",
    )
    pub = f.public()
    assert "private_trace" not in pub
    assert pub["code"] == "path_error"
    assert pub["observed"] == 0.1


def test_probe_spec_default_weight_zero():
    s = ProbeSpec(id="x", type="dof_grubler")
    assert s.weight == 0.0
    assert s.hard_gate is False
