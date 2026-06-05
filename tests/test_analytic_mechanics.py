"""Tests for the geometry-grounded analytic-mechanics oracle.

The defining property versus the fake oracle: outputs depend on GEOMETRY, not
config constants. Gears placed far apart produce zero contact force; the
transmission ratio is computed from tooth counts, not from declared_ratio.
"""

from __future__ import annotations

import numpy as np

from mech_bench.adapters.analytic_mechanics import AnalyticMechanics
from mech_bench.schema import DesignIR


def _gear_pair_ir(
    *,
    teeth_in: int = 10,
    teeth_out: int = 30,
    center_distance_mm: float = 20.0,
    declared_ratio: float = 999.0,  # a LIE the oracle must ignore
) -> DesignIR:
    """Pinion at origin, gear at center_distance along x. With module 1.0,
    pitch radii are teeth/2, so they mesh when center_distance == r_in + r_out.
    """
    parts = [
        {"id": "frame", "role": "ground", "fixed": True},
        {"id": "pinion", "role": "input", "params": {"teeth": teeth_in}},
        {"id": "gear", "role": "output", "params": {"teeth": teeth_out}},
    ]
    joints = [
        {"id": "input_axis", "type": "revolute", "parent": "frame",
         "child": "pinion", "anchor_world_mm": (0.0, 0.0, 0.0)},
        {"id": "output_axis", "type": "revolute", "parent": "frame",
         "child": "gear", "anchor_world_mm": (center_distance_mm, 0.0, 0.0)},
        {"id": "mesh", "type": "contact_pair", "parent": "pinion",
         "child": "gear"},
    ]
    ports = {
        "input_port": {"id": "input_port", "part": "input_axis",
                       "kind": "revolute_joint"},
        "output_port": {"id": "output_port", "part": "output_axis",
                        "kind": "revolute_joint"},
    }
    return DesignIR.from_dict({
        "schema_version": "design_ir.v2",
        "parts": parts, "joints": joints, "ports": ports,
        "params": {"declared_ratio": declared_ratio},
    })


def _run(ir, **cfg):
    return AnalyticMechanics().run(ir, cfg)


def test_ratio_from_teeth_not_declared():
    # teeth_out/teeth_in = 30/10 = 3.0, regardless of the lying declared_ratio.
    out = _run(_gear_pair_ir(teeth_in=10, teeth_out=30, declared_ratio=999.0))
    assert out["scalar_metrics"]["ratio_observed"] == 3.0
    assert out["scalar_metrics"]["ratio_known"] == 1.0


def test_output_velocity_follows_geometry_ratio():
    out = _run(_gear_pair_ir(teeth_in=10, teeth_out=30), input_speed_rad_s=3.0)
    out_vel = out["joint_velocities"]["output_port"]
    # out_speed = in_speed / ratio = 3.0 / 3.0 = 1.0
    assert np.allclose(out_vel, 1.0)


def test_meshing_gears_engage():
    # r_in=5, r_out=15, sum=20 == center distance -> perfectly meshed.
    out = _run(_gear_pair_ir(teeth_in=10, teeth_out=30, center_distance_mm=20.0),
               output_load_Nm=1.0)
    force = out["contact_forces"]["gear:pinion"]
    assert float(np.max(force)) > 0.0
    assert out["scalar_metrics"]["n_contacts_max"] == 1.0


def test_far_apart_gears_do_not_engage():
    # The central anti-hack property: 100 m apart -> zero contact force.
    out = _run(_gear_pair_ir(center_distance_mm=100_000.0), output_load_Nm=1.0)
    force = out["contact_forces"]["gear:pinion"]
    assert float(np.max(force)) == 0.0
    assert out["scalar_metrics"]["n_contacts_max"] == 0.0


def test_engagement_is_monotonic_in_separation():
    loads = dict(output_load_Nm=1.0)
    perfect = _run(_gear_pair_ir(center_distance_mm=20.0), **loads)
    edge = _run(_gear_pair_ir(center_distance_mm=21.5), **loads)  # within 10% clearance
    gone = _run(_gear_pair_ir(center_distance_mm=30.0), **loads)
    fp = float(np.max(perfect["contact_forces"]["gear:pinion"]))
    fe = float(np.max(edge["contact_forces"]["gear:pinion"]))
    fg = float(np.max(gone["contact_forces"]["gear:pinion"]))
    assert fp > fe > fg
    assert fg == 0.0


def test_penetration_from_overlap():
    # center distance 15 < r_sum 20 -> 5 mm interpenetration.
    out = _run(_gear_pair_ir(center_distance_mm=15.0))
    pen = out["penetration"]["gear:pinion"]
    assert float(np.max(pen)) == 5.0


def test_unknown_geometry_yields_no_credit():
    # No teeth, no radii -> ratio undetermined, zero contact force (honest).
    parts = [
        {"id": "frame", "role": "ground", "fixed": True},
        {"id": "a", "role": "input"},
        {"id": "b", "role": "output"},
    ]
    joints = [
        {"id": "ja", "type": "revolute", "parent": "frame", "child": "a",
         "anchor_world_mm": (0.0, 0.0, 0.0)},
        {"id": "jb", "type": "revolute", "parent": "frame", "child": "b",
         "anchor_world_mm": (10.0, 0.0, 0.0)},
        {"id": "m", "type": "contact_pair", "parent": "a", "child": "b"},
    ]
    ports = {
        "input_port": {"id": "input_port", "part": "ja", "kind": "revolute_joint"},
        "output_port": {"id": "output_port", "part": "jb", "kind": "revolute_joint"},
    }
    ir = DesignIR.from_dict({"schema_version": "design_ir.v2", "parts": parts,
                             "joints": joints, "ports": ports,
                             "params": {"declared_ratio": 5.0}})
    out = _run(ir, output_load_Nm=1.0)
    # No teeth / radii -> the adapter declines (capability_unavailable) rather
    # than fabricate a verdict. No silent pass.
    assert out["__capability_unavailable__"] is True
    assert out["contact_forces"] == {}
    assert out["metadata"]["oracle_is_synthetic"] is False


def test_declines_when_no_contact_pair_declared():
    # Real teeth/positions but only revolute joints (no contact_pair) -> the
    # conservative oracle declines so placeholder stubs stay capability_unavailable.
    parts = [
        {"id": "frame", "role": "ground", "fixed": True},
        {"id": "pinion", "role": "input", "params": {"teeth": 16}},
        {"id": "gear", "role": "output", "params": {"teeth": 64}},
    ]
    joints = [
        {"id": "ia", "type": "revolute", "parent": "frame", "child": "pinion",
         "anchor_world_mm": (0.0, 0.0, 0.0)},
        {"id": "oa", "type": "revolute", "parent": "frame", "child": "gear",
         "anchor_world_mm": (40.0, 0.0, 0.0)},
    ]
    ports = {
        "input_port": {"id": "input_port", "part": "ia", "kind": "revolute_joint"},
        "output_port": {"id": "output_port", "part": "oa", "kind": "revolute_joint"},
    }
    ir = DesignIR.from_dict({"schema_version": "design_ir.v2", "parts": parts,
                             "joints": joints, "ports": ports, "params": {}})
    out = _run(ir, output_load_Nm=1.0)
    assert out["__capability_unavailable__"] is True


def test_power_balance_ideal_is_zero():
    out = _run(_gear_pair_ir(), input_speed_rad_s=10.0, output_load_Nm=0.75)
    assert out["scalar_metrics"]["power_balance_error_pct"] < 1e-6


def test_oracle_is_not_synthetic():
    out = _run(_gear_pair_ir())
    assert out["metadata"]["oracle_is_synthetic"] is False
    assert out["metadata"]["is_physical_oracle"] is True


def test_deterministic():
    a = _run(_gear_pair_ir(), output_load_Nm=2.0)
    b = _run(_gear_pair_ir(), output_load_Nm=2.0)
    assert np.array_equal(a["contact_forces"]["gear:pinion"],
                          b["contact_forces"]["gear:pinion"])
    assert a["scalar_metrics"] == b["scalar_metrics"]


def test_capabilities_and_cost():
    from mech_bench.probes import Capability
    assert Capability.CONTACT_FORCES in AnalyticMechanics.capabilities_provided
    assert Capability.MESH_OVERLAP not in AnalyticMechanics.capabilities_provided
    # cheaper than the fake oracle so the dispatcher prefers the real one.
    assert AnalyticMechanics.cost_tier < 50


def test_registered_in_registry():
    from mech_bench.adapters import _REGISTRY
    assert "analytic_mechanics" in _REGISTRY
