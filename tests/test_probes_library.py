"""Unit tests for the reusable probe library.

Each probe gets a focused test with synthetic ``sim_outputs`` so we
verify behavior independently of any simulator adapter. Evaluator
integration tests live in test_evaluator_dispatch.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from mech_bench.feedback import FailureCode
from mech_bench.probes import get_probe
from mech_bench.schema import DesignIR


def _ir_minimal(*, with_input_port: bool = True) -> DesignIR:
    parts = [
        {"id": "ground", "fixed": True, "mass_kg": 0.0,
         "com_local_mm": (0.0, 0.0, 0.0)},
        {"id": "crank", "mass_kg": 0.02,
         "com_local_mm": (15.0, 0.0, 0.0)},
        {"id": "coupler", "mass_kg": 0.05,
         "com_local_mm": (45.0, 0.0, 0.0)},
        {"id": "rocker", "mass_kg": 0.04,
         "com_local_mm": (40.0, 0.0, 0.0)},
    ]
    joints = [
        {"id": "j_in", "type": "revolute",
         "parent": "ground", "child": "crank",
         "axis_world": (0.0, 0.0, 1.0),
         "anchor_world_mm": (0.0, 0.0, 0.0)},
        {"id": "j_bc", "type": "revolute",
         "parent": "crank", "child": "coupler",
         "axis_world": (0.0, 0.0, 1.0),
         "anchor_world_mm": (30.0, 0.0, 0.0)},
        {"id": "j_cd", "type": "revolute",
         "parent": "coupler", "child": "rocker",
         "axis_world": (0.0, 0.0, 1.0),
         "anchor_world_mm": (50.0, 50.0, 0.0)},
        {"id": "j_out", "type": "revolute",
         "parent": "ground", "child": "rocker",
         "axis_world": (0.0, 0.0, 1.0),
         "anchor_world_mm": (100.0, 0.0, 0.0)},
    ]
    ports: dict = {
        "output_port": {
            "id": "output_port", "part": "j_out",
            "kind": "revolute_joint",
            "pose_local_mm": (0.0, 0.0, 0.0),
        },
        "coupler_point": {
            "id": "coupler_point", "part": "coupler",
            "kind": "frame",
            "pose_local_mm": (35.0, 18.0, 0.0),
        },
    }
    if with_input_port:
        ports["input_port"] = {
            "id": "input_port", "part": "j_in",
            "kind": "revolute_joint",
            "pose_local_mm": (0.0, 0.0, 0.0),
        }
    return DesignIR.from_dict({
        "schema_version": "design_ir.v2",
        "parts": parts,
        "joints": joints,
        "ports": ports,
    })


# --------------------------------------------------------------------- #
# required_ports                                                        #
# --------------------------------------------------------------------- #


def test_required_ports_pass():
    probe = get_probe("required_ports")
    ir = _ir_minimal()
    result = probe.run(ir, {}, {
        "ports": ["input_port", "output_port", "coupler_point"],
        "require_kinds": {"input_port": "revolute_joint"},
    })
    assert result.passed
    assert result.score == pytest.approx(1.0)
    assert result.failures == []


def test_required_ports_missing_port():
    probe = get_probe("required_ports")
    ir = _ir_minimal(with_input_port=False)
    result = probe.run(ir, {}, {
        "ports": ["input_port", "output_port"],
    })
    assert not result.passed
    codes = [f.code for f in result.failures]
    assert FailureCode.MISSING_PORT in codes


def test_required_ports_wrong_kind():
    probe = get_probe("required_ports")
    ir = _ir_minimal()
    result = probe.run(ir, {}, {
        "ports": ["coupler_point"],
        "require_kinds": {"coupler_point": "revolute_joint"},
    })
    assert not result.passed
    codes = [f.code for f in result.failures]
    assert FailureCode.WRONG_TOPOLOGY in codes


def test_required_ports_grounded_check():
    probe = get_probe("required_ports")
    ir = _ir_minimal()
    # input_port references j_in which touches ground → grounded.
    ok = probe.run(ir, {}, {
        "ports": ["input_port"],
        "require_grounded": ["input_port"],
    })
    assert ok.passed
    # coupler_point references the coupler (not fixed) → NOT grounded.
    bad = probe.run(ir, {}, {
        "ports": ["coupler_point"],
        "require_grounded": ["coupler_point"],
    })
    assert not bad.passed
    assert any(f.code == FailureCode.WRONG_TOPOLOGY for f in bad.failures)


# --------------------------------------------------------------------- #
# trusted_asset_preflight                                               #
# --------------------------------------------------------------------- #


def test_trusted_asset_preflight_passes_metadata_gate():
    probe = get_probe("trusted_asset_preflight")
    raw = {
        "schema_version": "design_ir.v2",
        "units": "mm",
        "materials": {
            "al6061": {
                "density_kg_m3": 2700.0,
                "provenance": "datasheet",
            },
        },
        "parts": [
            {"id": "frame", "fixed": True, "mass_kg": 0.0,
             "geometry": {"cad": "frame.step"},
             "material": "al6061"},
            {"id": "link", "mass_kg": 0.1,
             "geometry": {"cad": "link.step"},
             "material": "al6061"},
        ],
        "joints": [
            {"id": "j1", "type": "revolute",
             "parent": "frame", "child": "link"},
        ],
        "ports": {},
    }
    result = probe.run(DesignIR.from_dict(raw), {}, {
        "require_geometry_roles": ["cad"],
        "require_materials": True,
        "require_material_properties": ["density_kg_m3"],
        "require_provenance": True,
    })
    assert result.passed
    assert result.metrics["trusted_mass_properties_recomputed"] == 0.0


def test_trusted_asset_preflight_missing_cad_and_material_fail():
    probe = get_probe("trusted_asset_preflight")
    ir = _ir_minimal()
    result = probe.run(ir, {}, {
        "require_geometry_roles": ["cad"],
        "require_materials": True,
    })
    assert not result.passed
    codes = {f.code for f in result.failures}
    assert FailureCode.INVALID_ARTIFACT in codes
    assert FailureCode.SCHEMA_ERROR in codes


def test_trusted_asset_preflight_refuses_unimplemented_trusted_mass():
    probe = get_probe("trusted_asset_preflight")
    ir = _ir_minimal()
    result = probe.run(ir, {}, {
        "require_trusted_mass_properties": True,
    })
    assert not result.passed
    assert any(
        f.code == FailureCode.INVALID_MASS_PROPERTIES
        for f in result.failures
    )


def test_trusted_asset_preflight_accepts_trusted_cad_mass_properties():
    probe = get_probe("trusted_asset_preflight")
    inertia = ((1.0e-5, 0.0, 0.0), (0.0, 2.0e-5, 0.0), (0.0, 0.0, 3.0e-5))
    raw = {
        "schema_version": "design_ir.v2",
        "units": "mm",
        "params": {"cad_source": {"kernel": "FreeCAD/OCCT"}},
        "parts": [
            {"id": "frame", "fixed": True, "mass_kg": 0.0},
            {
                "id": "link",
                "mass_kg": 0.1,
                "geometry": {"cad": "link.step"},
                "params": {
                    "cad_mass_properties": {
                        "mass_kg": 0.1,
                        "com_local_mm": (1.0, 2.0, 3.0),
                        "inertia_kg_m2": inertia,
                    },
                },
            },
        ],
        "joints": [
            {"id": "j1", "type": "revolute",
             "parent": "frame", "child": "link"},
        ],
        "ports": {},
    }
    result = probe.run(DesignIR.from_dict(raw), {}, {
        "require_trusted_mass_properties": True,
    })
    assert result.passed
    assert result.metrics["trusted_mass_properties_recomputed"] == 1.0
    assert result.metrics["parts_requiring_trusted_mass_properties"] == 1.0
    assert result.metrics["parts_with_trusted_mass_properties"] == 1.0


# --------------------------------------------------------------------- #
# port_velocity_ratio                                                   #
# --------------------------------------------------------------------- #


def test_port_velocity_ratio_pass_on_synthetic():
    probe = get_probe("port_velocity_ratio")
    t = np.linspace(0.0, 1.0, 200)
    v_in = np.ones_like(t)
    v_out = -4.0 * v_in
    sim = {
        "joint_velocities": {
            "input_port": v_in,
            "output_port": v_out,
        },
        "time_s": t,
    }
    result = probe.run(_ir_minimal(), sim, {
        "input_port": "input_port",
        "output_port": "output_port",
        "expected": -4.0,
        "tolerance_pct": 5.0,
    })
    assert result.passed
    assert result.metrics["ratio_observed"] == pytest.approx(-4.0, rel=1e-6)
    assert result.score == pytest.approx(1.0)


def test_port_velocity_ratio_fails_when_off():
    probe = get_probe("port_velocity_ratio")
    v_in = np.ones(200)
    v_out = -3.0 * v_in  # 25% off from -4
    sim = {
        "joint_velocities": {"input_port": v_in, "output_port": v_out},
    }
    result = probe.run(_ir_minimal(), sim, {
        "expected": -4.0, "tolerance_pct": 5.0,
    })
    assert not result.passed
    assert any(f.code == FailureCode.WRONG_RATIO for f in result.failures)
    assert result.score == pytest.approx(0.0)


def test_port_velocity_ratio_derives_from_positions():
    """When joint_velocities are absent the probe finite-differences
    joint_positions on the shared time axis."""
    probe = get_probe("port_velocity_ratio")
    t = np.linspace(0.0, 2 * np.pi, 200)
    p_in = t
    p_out = -2.5 * t
    sim = {
        "joint_positions": {"input_port": p_in, "output_port": p_out},
        "time_s": t,
    }
    result = probe.run(_ir_minimal(), sim, {
        "expected": -2.5, "tolerance_pct": 1.0,
    })
    assert result.passed, result.failures


def test_port_velocity_ratio_missing_traces_surfaces_simulator_divergence():
    probe = get_probe("port_velocity_ratio")
    result = probe.run(_ir_minimal(), {}, {
        "expected": 1.0, "tolerance_pct": 5.0,
    })
    assert not result.passed
    assert any(f.code == FailureCode.SIMULATOR_DIVERGENCE
               for f in result.failures)


# --------------------------------------------------------------------- #
# swept_collision                                                       #
# --------------------------------------------------------------------- #


def test_swept_collision_pass_under_threshold():
    probe = get_probe("swept_collision")
    pen = {
        "crank:coupler": np.array([0.0, 0.01, 0.02, 0.0]),
    }
    result = probe.run(_ir_minimal(), {"penetration": pen}, {
        "max_penetration_mm": 0.05,
    })
    assert result.passed
    assert result.metrics["max_penetration_mm"] == pytest.approx(0.02)


def test_swept_collision_fails_over_threshold():
    probe = get_probe("swept_collision")
    pen = {
        "crank:coupler": np.array([0.0, 0.2, 0.4, 0.1]),
    }
    result = probe.run(_ir_minimal(), {"penetration": pen,
                                       "time_s": np.array(
                                           [0, 0.1, 0.2, 0.3])}, {
        "max_penetration_mm": 0.05,
    })
    assert not result.passed
    codes = [f.code for f in result.failures]
    assert FailureCode.EXCESSIVE_PENETRATION in codes
    assert result.metrics["max_penetration_mm"] == pytest.approx(0.4)
    assert result.metrics["worst_time_s"] == pytest.approx(0.2)


def test_swept_collision_unavailable():
    probe = get_probe("swept_collision")
    result = probe.run(_ir_minimal(), {
        "__capability_unavailable__": True,
    }, {"max_penetration_mm": 0.05})
    codes = [f.code for f in result.failures]
    assert FailureCode.CAPABILITY_UNAVAILABLE in codes


def test_swept_collision_allowed_pair_skipped():
    probe = get_probe("swept_collision")
    pen = {"crank:coupler": np.array([0.5, 0.6])}
    result = probe.run(_ir_minimal(), {"penetration": pen}, {
        "max_penetration_mm": 0.05,
        "allowed_pairs": ["crank:coupler"],
    })
    assert result.passed, result.failures


# --------------------------------------------------------------------- #
# contact_engagement                                                    #
# --------------------------------------------------------------------- #


def test_contact_engagement_pass():
    probe = get_probe("contact_engagement")
    forces = {
        "disc:ring_pins": np.full(100, 2.0),
    }
    result = probe.run(_ir_minimal(), {"contact_forces": forces}, {
        "required_pairs": ["disc:ring_pins"],
        "min_rms_force_N": 0.5,
        "min_engagement_fraction": 0.05,
    })
    assert result.passed
    assert result.metrics["contact.disc:ring_pins.rms_N"] == pytest.approx(2.0)
    assert result.metrics[
        "contact.disc:ring_pins.engagement_fraction"] == pytest.approx(1.0)


def test_contact_engagement_missing_pair():
    probe = get_probe("contact_engagement")
    result = probe.run(_ir_minimal(),
                       {"contact_forces": {"unrelated": np.ones(10)}},
                       {"required_pairs": ["disc:ring_pins"],
                        "min_rms_force_N": 0.5,
                        "min_engagement_fraction": 0.05})
    assert not result.passed
    assert any(f.code == FailureCode.MISSING_CONTACT for f in result.failures)


def test_contact_engagement_under_engaged():
    probe = get_probe("contact_engagement")
    # Pair exists but force is too weak.
    forces = {"gear1:gear2": np.full(100, 0.01)}
    result = probe.run(_ir_minimal(), {"contact_forces": forces}, {
        "required_pairs": ["gear1:gear2"],
        "min_rms_force_N": 0.5,
        "min_engagement_fraction": 0.05,
    })
    assert not result.passed
    assert any(f.code == FailureCode.MISSING_CONTACT for f in result.failures)


def test_contact_engagement_capability_unavailable():
    probe = get_probe("contact_engagement")
    result = probe.run(_ir_minimal(),
                       {"__capability_unavailable__": True},
                       {"required_pairs": ["a:b"]})
    assert any(f.code == FailureCode.CAPABILITY_UNAVAILABLE
               for f in result.failures)


# --------------------------------------------------------------------- #
# lockup                                                                #
# --------------------------------------------------------------------- #


def test_lockup_pass_when_output_moves():
    probe = get_probe("lockup")
    t = np.linspace(0, 2 * np.pi, 100)
    sim = {
        "joint_positions": {"input_port": t, "output_port": 0.5 * t},
        "time_s": t,
    }
    result = probe.run(_ir_minimal(), sim, {"min_output_motion_rad": 0.1})
    assert result.passed
    assert result.metrics["lockup_detected"] == pytest.approx(0.0)


def test_lockup_detects_zero_output_motion():
    probe = get_probe("lockup")
    t = np.linspace(0, 2 * np.pi, 100)
    sim = {
        "joint_positions": {
            "input_port": t,
            "output_port": np.zeros_like(t),
        },
        "time_s": t,
    }
    result = probe.run(_ir_minimal(), sim, {"min_output_motion_rad": 0.05})
    assert not result.passed
    assert any(f.code == FailureCode.LOCKUP for f in result.failures)
    assert result.metrics["lockup_detected"] == pytest.approx(1.0)


def test_lockup_undriven_input_awards_no_credit():
    """Anti-hack gate: if the input itself never moved, the lockup test is
    vacuous — a static (dead) design must NOT collect a degenerate pass."""
    probe = get_probe("lockup")
    n = 100
    sim = {
        "joint_positions": {
            "input_port": np.zeros(n),
            "output_port": np.zeros(n),
        },
        "time_s": np.linspace(0, 1, n),
    }
    result = probe.run(_ir_minimal(), sim, {})
    assert not result.passed
    assert result.score == 0.0
    codes = {f.code.value if hasattr(f.code, "value") else str(f.code)
             for f in result.failures}
    assert "degenerate_test" in codes


# --------------------------------------------------------------------- #
# torque_load_trial                                                     #
# --------------------------------------------------------------------- #


def test_torque_load_trial_pass():
    probe = get_probe("torque_load_trial")
    sim = {
        "joint_velocities": {
            "input_port": np.full(100, 10.0),
            "output_port": np.full(100, 2.5),
        },
        "scalar_metrics": {
            "input_power_W_mean": 5.0,
            "output_power_W_mean": 5.0,
            "input_torque_ripple_pct": 3.0,
        },
    }
    result = probe.run(_ir_minimal(), sim, {
        "input_port": "input_port",
        "output_port": "output_port",
        "input_speed_rad_s": 10.0,
        "output_load_Nm": 2.0,
        "min_output_speed_rad_s": 0.5,
        "max_power_error_pct": 5.0,
        "max_torque_ripple_pct": 10.0,
    })
    assert result.passed
    # Ripple of 3% leaves 70% headroom on the 10% threshold; combined
    # with full credit on the other two checks, the mean lands at 0.9.
    assert result.score >= 0.85


def test_torque_load_trial_lockup_under_load():
    probe = get_probe("torque_load_trial")
    sim = {
        "joint_velocities": {
            "input_port": np.full(100, 10.0),
            "output_port": np.zeros(100),
        },
    }
    result = probe.run(_ir_minimal(), sim, {
        "min_output_speed_rad_s": 0.5,
    })
    assert not result.passed
    assert any(f.code == FailureCode.LOCKUP for f in result.failures)


def test_torque_load_trial_power_balance_error():
    probe = get_probe("torque_load_trial")
    sim = {
        "joint_velocities": {
            "input_port": np.full(100, 10.0),
            "output_port": np.full(100, 2.0),
        },
        "scalar_metrics": {
            "input_power_W_mean": 10.0,
            "output_power_W_mean": 2.0,
        },
    }
    result = probe.run(_ir_minimal(), sim, {
        "min_output_speed_rad_s": 0.1,
        "max_power_error_pct": 10.0,
    })
    assert not result.passed
    assert any(f.code == FailureCode.POWER_BALANCE_ERROR
               for f in result.failures)


def test_torque_load_trial_torque_ripple_high():
    probe = get_probe("torque_load_trial")
    sim = {
        "joint_velocities": {
            "input_port": np.full(100, 10.0),
            "output_port": np.full(100, 5.0),
        },
        "scalar_metrics": {
            "input_torque_ripple_pct": 40.0,
        },
    }
    result = probe.run(_ir_minimal(), sim, {
        "min_output_speed_rad_s": 0.1,
        "max_torque_ripple_pct": 10.0,
    })
    assert not result.passed
    assert any(f.code == FailureCode.EXCESSIVE_TORQUE_RIPPLE
               for f in result.failures)


# --------------------------------------------------------------------- #
# printability_dfam                                                     #
# --------------------------------------------------------------------- #


def test_printability_pass():
    probe = get_probe("printability_dfam")
    sim = {
        "mesh_metrics": {
            "crank": {"min_wall_mm": 2.0, "max_overhang_deg": 30.0},
        },
    }
    result = probe.run(_ir_minimal(), sim, {
        "min_wall_mm": 1.2, "max_overhang_deg": 50.0,
    })
    assert result.passed
    assert result.metrics["worst_min_wall_mm"] == pytest.approx(2.0)


def test_printability_fails_thin_wall():
    probe = get_probe("printability_dfam")
    sim = {
        "mesh_metrics": {
            "crank": {"min_wall_mm": 0.6, "max_overhang_deg": 20.0},
        },
    }
    result = probe.run(_ir_minimal(), sim, {
        "min_wall_mm": 1.2, "max_overhang_deg": 50.0,
    })
    assert not result.passed
    assert any(f.code == FailureCode.UNPRINTABLE for f in result.failures)


def test_printability_capability_unavailable_without_metrics():
    probe = get_probe("printability_dfam")
    result = probe.run(_ir_minimal(), {}, {"min_wall_mm": 1.2})
    assert any(f.code == FailureCode.CAPABILITY_UNAVAILABLE
               for f in result.failures)


# --------------------------------------------------------------------- #
# safety_factor                                                         #
# --------------------------------------------------------------------- #


def test_safety_factor_pass():
    probe = get_probe("safety_factor")
    sim = {
        "safety_factors": {
            "output_pin_bending": 2.5,
            "hertz": 3.1,
        },
    }
    result = probe.run(_ir_minimal(), sim, {"min_fos": 1.5})
    assert result.passed
    assert result.metrics["min_fos"] == pytest.approx(2.5)


def test_safety_factor_fails_below_min():
    probe = get_probe("safety_factor")
    sim = {
        "safety_factors": {
            "output_pin_bending": 0.8,
            "hertz": 3.1,
        },
    }
    result = probe.run(_ir_minimal(), sim, {"min_fos": 1.5})
    assert not result.passed
    assert any(f.code == FailureCode.INSUFFICIENT_SAFETY_FACTOR
               for f in result.failures)
    assert result.metrics["min_fos"] == pytest.approx(0.8)


def test_safety_factor_filters_by_check_name():
    probe = get_probe("safety_factor")
    sim = {
        "safety_factors": {
            "output_pin_bending": 0.8,
            "irrelevant_check": 0.1,
        },
    }
    result = probe.run(_ir_minimal(), sim, {
        "min_fos": 1.5,
        "checks": ["output_pin_bending"],
    })
    assert not result.passed
    assert result.metrics["min_fos"] == pytest.approx(0.8)


def test_safety_factor_capability_unavailable():
    probe = get_probe("safety_factor")
    result = probe.run(_ir_minimal(), {}, {"min_fos": 1.5})
    assert any(f.code == FailureCode.CAPABILITY_UNAVAILABLE
               for f in result.failures)


# --------------------------------------------------------------------- #
# Score range invariant                                                 #
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("probe_name,sim,cfg", [
    ("required_ports", {}, {"ports": ["input_port"]}),
    ("port_velocity_ratio",
     {"joint_velocities": {"input_port": np.ones(10),
                            "output_port": -4 * np.ones(10)}},
     {"expected": -4.0, "tolerance_pct": 5.0}),
    ("swept_collision",
     {"penetration": {"a:b": np.array([0.01, 0.02])}},
     {"max_penetration_mm": 0.05}),
    ("contact_engagement",
     {"contact_forces": {"a:b": np.ones(10)}},
     {"required_pairs": ["a:b"], "min_rms_force_N": 0.5,
      "min_engagement_fraction": 0.05}),
    ("lockup",
     {"joint_positions": {"input_port": np.linspace(0, 1, 50),
                          "output_port": np.linspace(0, 0.5, 50)},
      "time_s": np.linspace(0, 1, 50)},
     {"min_output_motion_rad": 0.1}),
    ("printability_dfam",
     {"mesh_metrics": {"x": {"min_wall_mm": 2.0, "max_overhang_deg": 20.0}}},
     {"min_wall_mm": 1.2, "max_overhang_deg": 50.0}),
    ("safety_factor",
     {"safety_factors": {"x": 2.0}},
     {"min_fos": 1.5}),
])
def test_probe_score_in_unit_interval(probe_name, sim, cfg):
    probe = get_probe(probe_name)
    result = probe.run(_ir_minimal(), sim, cfg)
    assert 0.0 <= result.score <= 1.0, (
        f"{probe_name} produced score {result.score} outside [0,1]"
    )
