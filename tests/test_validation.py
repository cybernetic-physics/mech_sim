"""Validation-layer tests.

These exercise validate_design_ir directly. Evaluator integration is
covered in test_evaluator.py.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from mech_bench.feedback import FailureCode
from mech_bench.schema import DesignIR, TaskSpec
from mech_bench.validation import validate_design_ir


def _good_ir_dict() -> dict:
    return {
        "schema_version": "design_ir.v2",
        "parts": [
            {"id": "ground", "fixed": True, "mass_kg": 0.0,
             "com_local_mm": (0.0, 0.0, 0.0)},
            {"id": "crank", "mass_kg": 0.02,
             "com_local_mm": (15.0, 0.0, 0.0)},
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


def _good_task() -> TaskSpec:
    return TaskSpec(
        id="t1", family="planar_4bar", difficulty=1, units="mm",
        prompt="", required_ports=["input_port"],
        expected_mobility=1,
    )


def _codes(failures) -> set[str]:
    return {f.code.value for f in failures}


def test_clean_ir_yields_no_failures():
    ir = DesignIR.from_dict(_good_ir_dict())
    task = _good_task()
    assert validate_design_ir(ir, task=task) == []


def test_invalid_schema_version():
    raw = _good_ir_dict()
    raw["schema_version"] = "design_ir.v1"
    ir = DesignIR.from_dict(raw)
    assert FailureCode.SCHEMA_ERROR.value in _codes(validate_design_ir(ir))


def test_duplicate_part_ids():
    raw = _good_ir_dict()
    raw["parts"].append(dict(raw["parts"][0]))
    ir = DesignIR.from_dict(raw)
    codes = _codes(validate_design_ir(ir))
    assert FailureCode.SCHEMA_ERROR.value in codes


def test_joint_references_missing_part():
    raw = _good_ir_dict()
    raw["joints"][0]["parent"] = "nonexistent"
    ir = DesignIR.from_dict(raw)
    codes = _codes(validate_design_ir(ir))
    assert FailureCode.WRONG_TOPOLOGY.value in codes


def test_revolute_port_references_missing_joint():
    raw = _good_ir_dict()
    raw["ports"]["input_port"]["part"] = "does_not_exist"
    ir = DesignIR.from_dict(raw)
    codes = _codes(validate_design_ir(ir))
    assert FailureCode.MISSING_PORT.value in codes


def test_required_port_missing():
    raw = _good_ir_dict()
    raw["ports"].pop("input_port")
    ir = DesignIR.from_dict(raw)
    task = _good_task()
    codes = _codes(validate_design_ir(ir, task=task))
    assert FailureCode.MISSING_PORT.value in codes


def test_nan_mass_rejected():
    raw = _good_ir_dict()
    raw["parts"][1]["mass_kg"] = float("nan")
    ir = DesignIR.from_dict(raw)
    codes = _codes(validate_design_ir(ir))
    assert FailureCode.INVALID_MASS_PROPERTIES.value in codes


def test_nan_com_rejected():
    raw = _good_ir_dict()
    raw["parts"][1]["com_local_mm"] = (float("nan"), 0.0, 0.0)
    ir = DesignIR.from_dict(raw)
    codes = _codes(validate_design_ir(ir))
    assert FailureCode.INVALID_MASS_PROPERTIES.value in codes


def test_negative_moving_part_mass():
    raw = _good_ir_dict()
    raw["parts"][1]["mass_kg"] = -0.05
    ir = DesignIR.from_dict(raw)
    codes = _codes(validate_design_ir(ir))
    assert FailureCode.INVALID_MASS_PROPERTIES.value in codes


def test_nonphysical_inertia_triangle_inequality():
    raw = _good_ir_dict()
    raw["parts"][1]["inertia_kg_m2"] = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 100.0),
    )
    ir = DesignIR.from_dict(raw)
    codes = _codes(validate_design_ir(ir))
    assert FailureCode.INVALID_MASS_PROPERTIES.value in codes


def test_nonsymmetric_inertia_rejected():
    raw = _good_ir_dict()
    raw["parts"][1]["inertia_kg_m2"] = (
        (1.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    ir = DesignIR.from_dict(raw)
    codes = _codes(validate_design_ir(ir))
    assert FailureCode.INVALID_MASS_PROPERTIES.value in codes


def test_zero_joint_axis_rejected():
    raw = _good_ir_dict()
    raw["joints"][0]["axis_world"] = (0.0, 0.0, 0.0)
    ir = DesignIR.from_dict(raw)
    codes = _codes(validate_design_ir(ir))
    assert FailureCode.SCHEMA_ERROR.value in codes


def test_two_fixed_parts_violates_topology():
    raw = _good_ir_dict()
    raw["parts"].append({"id": "extra_ground", "fixed": True,
                          "mass_kg": 0.0})
    ir = DesignIR.from_dict(raw)
    codes = _codes(validate_design_ir(ir, task=_good_task()))
    assert FailureCode.WRONG_TOPOLOGY.value in codes


def test_geometry_path_traversal_rejected(tmp_path: Path):
    raw = _good_ir_dict()
    raw["parts"][1]["geometry"] = {"mesh": "../../etc/passwd"}
    ir = DesignIR.from_dict(raw)
    failures = validate_design_ir(ir, build_root=tmp_path)
    codes = _codes(failures)
    assert FailureCode.INVALID_ARTIFACT.value in codes


def test_geometry_absolute_path_outside_build_root_rejected(tmp_path: Path):
    raw = _good_ir_dict()
    raw["parts"][1]["geometry"] = {"mesh": "/etc/passwd"}
    ir = DesignIR.from_dict(raw)
    failures = validate_design_ir(ir, build_root=tmp_path)
    codes = _codes(failures)
    assert FailureCode.INVALID_ARTIFACT.value in codes


def test_geometry_path_inside_build_root_accepted(tmp_path: Path):
    raw = _good_ir_dict()
    (tmp_path / "mesh.stl").write_text("dummy")
    raw["parts"][1]["geometry"] = {"mesh": "mesh.stl"}
    ir = DesignIR.from_dict(raw)
    failures = validate_design_ir(ir, build_root=tmp_path)
    assert failures == []


def test_material_reference_must_exist():
    raw = _good_ir_dict()
    raw["parts"][1]["material"] = "missing_material"
    ir = DesignIR.from_dict(raw)
    codes = _codes(validate_design_ir(ir))
    assert FailureCode.SCHEMA_ERROR.value in codes


def test_material_properties_are_validated():
    raw = _good_ir_dict()
    raw["materials"] = {
        "bad": {
            "density_kg_m3": -1.0,
            "poisson_ratio": 0.75,
        },
    }
    raw["parts"][1]["material"] = "bad"
    ir = DesignIR.from_dict(raw)
    failures = validate_design_ir(ir)
    codes = _codes(failures)
    assert FailureCode.SCHEMA_ERROR.value in codes
    assert any("density_kg_m3" in f.message for f in failures)
    assert any("poisson_ratio" in f.message for f in failures)


def test_top_level_physics_fields_must_be_dicts():
    raw = _good_ir_dict()
    raw["load_cases"] = ["not", "a", "dict"]
    ir, errors = DesignIR.try_from_dict(raw)
    assert ir is None
    assert any("'load_cases' must be a dict" in e for e in errors)
