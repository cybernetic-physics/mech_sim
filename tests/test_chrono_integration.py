"""Project Chrono integration plumbing that does not require PyChrono."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest


def _minimal_ir():
    from mech_bench.schema import DesignIR, Joint, Part, Port

    return DesignIR(
        schema_version="design_ir.v2",
        parts=[
            Part(id="frame", role="ground", fixed=True, mass_kg=0.0),
            Part(id="input", role="input", mass_kg=0.1),
            Part(id="output", role="output", mass_kg=0.2),
        ],
        joints=[
            Joint(id="j_in", type="revolute", parent="frame",
                  child="input", axis_world=(0, 0, 1),
                  anchor_world_mm=(0, 0, 0)),
            Joint(id="j_out", type="revolute", parent="frame",
                  child="output", axis_world=(0, 0, 1),
                  anchor_world_mm=(10, 0, 0)),
        ],
        ports={
            "input_port": Port(id="input_port", part="j_in",
                               kind="revolute_joint"),
            "output_port": Port(id="output_port", part="j_out",
                                kind="revolute_joint"),
        },
    )


def _cycloidal_ir():
    from mech_bench.schema import DesignIR, Joint, Part, Port

    return DesignIR(
        schema_version="design_ir.v2",
        parts=[
            Part(id="housing", role="ground", fixed=True, mass_kg=0.0),
            Part(id="eccentric", role="eccentric", mass_kg=0.05),
            Part(id="disc", role="cycloidal_disc", mass_kg=0.08,
                 params={"pins": 10}),
            Part(id="carrier", role="carrier", mass_kg=0.04),
        ],
        joints=[
            Joint(id="input_revolute", type="revolute", parent="housing",
                  child="eccentric", axis_world=(0, 0, 1),
                  anchor_world_mm=(0, 0, 0)),
            Joint(id="eccentric_disc", type="revolute", parent="eccentric",
                  child="disc", axis_world=(0, 0, 1),
                  anchor_world_mm=(1, 0, 0)),
            Joint(id="output_revolute", type="revolute", parent="housing",
                  child="carrier", axis_world=(0, 0, 1),
                  anchor_world_mm=(0, 0, 0)),
            Joint(id="ring_contact", type="contact_pair", parent="housing",
                  child="disc", axis_world=(0, 0, 1),
                  anchor_world_mm=(0, 0, 0)),
        ],
        ports={
            "input_port": Port(id="input_port", part="input_revolute",
                               kind="revolute_joint"),
            "output_port": Port(id="output_port", part="output_revolute",
                                kind="revolute_joint"),
        },
        params={"pins": 10, "declared_ratio": 9.0},
    )


def _cycloidal_cfg(contact_model: str) -> dict:
    return {
        "samples": 120,
        "duration_s": 0.6,
        "contact_model": contact_model,
        "friction": 0.35,
        "restitution": 0.02,
        "young_modulus": 2.1e9,
        "normal_stiffness": 5.0e6,
        "damping": 800.0,
        "contact_margin": 0.0005,
        "contact_envelope": 0.001,
        "timestep": 1.0e-3,
        "solver_iterations": 80,
        "_mech_bench": {
            "task": {
                "id": "cycloidal_lowN_stub_s0001",
                "family": "cycloidal_lowN_stub",
                "difficulty": 3,
                "units": "mm",
            },
            "probe_specs": [
                {
                    "id": "torque",
                    "type": "torque_load_trial",
                    "config": {
                        "input_port": "input_port",
                        "output_port": "output_port",
                        "input_speed_rad_s": 10.0,
                        "output_load_Nm": 0.05,
                        "min_output_speed_rad_s": 0.001,
                        "max_power_error_pct": 25.0,
                        "max_torque_ripple_pct": 30.0,
                    },
                },
            ],
        },
    }


def test_evaluator_runtime_context_is_serializable_and_adapter_scoped(tmp_path):
    from mech_bench.evaluator import (
        ExecutionPlan,
        ProbePlan,
        _adapter_runtime_context,
    )
    from mech_bench.probes import Capability
    from mech_bench.schema import EvalConfig, ProbeSpec, TaskSpec

    task = TaskSpec(id="t1", family="contact", difficulty=1, units="mm",
                    prompt="")
    cfg = EvalConfig(probes=[
        ProbeSpec(
            id="contact",
            type="contact_engagement",
            config={"required_pairs": ["input:output"]},
        ),
        ProbeSpec(id="ports", type="required_ports", config={}),
    ])
    plan = ExecutionPlan(probes=[
        ProbePlan(
            probe_id="contact",
            probe_type="contact_engagement",
            capabilities=frozenset({Capability.CONTACT_FORCES}),
            adapter_type="chrono_contact",
        ),
        ProbePlan(
            probe_id="ports",
            probe_type="required_ports",
            capabilities=frozenset({Capability.NONE}),
            adapter_type=None,
        ),
    ])

    ctx = _adapter_runtime_context(
        task=task,
        cfg=cfg,
        plan=plan,
        adapter_name="chrono_contact",
        build_root=tmp_path,
    )

    assert ctx["task"]["id"] == "t1"
    assert ctx["build_root"] == str(tmp_path.resolve())
    assert [p["id"] for p in ctx["probe_specs"]] == ["contact"]
    assert ctx["probe_specs"][0]["config"]["required_pairs"] == ["input:output"]


def test_chrono_runtime_spec_extracts_contacts_drive_and_load(tmp_path):
    from mech_bench.adapters import _chrono_impl

    ir = _minimal_ir()
    cfg = {
        "_mech_bench": {
            "build_root": str(tmp_path),
            "probe_specs": [
                {
                    "id": "contact",
                    "type": "contact_engagement",
                    "config": {"required_pairs": ["output:input"]},
                },
                {
                    "id": "torque",
                    "type": "torque_load_trial",
                    "config": {
                        "input_port": "input_port",
                        "output_port": "output_port",
                        "input_speed_rad_s": 12.5,
                        "output_load_Nm": 0.75,
                    },
                },
            ],
        },
        "contact_pairs": ["input:output"],
    }

    spec = _chrono_impl._runtime_spec(ir, cfg)

    assert spec.contact_pairs == ["input:output"]
    assert spec.build_root == tmp_path.resolve()
    assert spec.motors == [{
        "id": "drive_torque",
        "joint_id": "j_in",
        "port_id": "input_port",
        "mode": "speed",
        "value": 12.5,
    }]
    assert spec.loads == [{
        "id": "load_torque",
        "joint_id": "j_out",
        "port_id": "output_port",
        "mode": "torque",
        "value": 0.75,
    }]


def test_chrono_impl_direct_run_reports_missing_pychrono():
    if importlib.util.find_spec("pychrono") is not None:
        pytest.skip("host has PyChrono; direct missing-dependency path not active")

    from mech_bench.adapters import _chrono_impl

    out = _chrono_impl.run(_minimal_ir(), {})

    assert out["__capability_unavailable__"] is True
    assert out["metadata"]["simulator"] == "project_chrono"
    assert "pychrono not importable" in out["metadata"]["preflight_issues"][0]


def test_chrono_cycloidal_nsc_vs_smc_thresholds():
    if importlib.util.find_spec("pychrono") is None:
        pytest.skip("requires PyChrono")

    from mech_bench.adapters import _chrono_impl

    ir = _cycloidal_ir()
    nsc = _chrono_impl.run(ir, _cycloidal_cfg("nsc"))
    smc = _chrono_impl.run(ir, _cycloidal_cfg("smc"))

    nsc_m = nsc["scalar_metrics"]
    smc_m = smc["scalar_metrics"]
    assert nsc_m["lockup_detected"] == 1.0
    assert abs(nsc_m["out_omega_med"]) < 1e-6
    assert nsc_m["ratio_observed"] == float("inf")
    assert nsc["passed"] is False

    assert smc["metadata"]["contact_model"] == "smc"
    assert smc["metadata"]["config"]["friction"] == 0.35
    assert smc["metadata"]["config"]["solver_iterations"] == 80.0
    assert smc_m["lockup_detected"] == 0.0
    assert abs(smc_m["out_omega_med"]) > 0.5
    assert np.isfinite(smc_m["ratio_observed"])
    assert smc_m["max_penetration_mm"] < 1.0
    assert smc_m["n_contacts_max"] < nsc_m["n_contacts_max"]
    assert smc_m["power_balance_error_pct"] <= 25.0
    assert smc_m["torque_ripple_pct"] <= 30.0
    assert smc["passed"] is True
    for key in (
        "lockup_detected",
        "ratio_observed",
        "in_omega_med",
        "out_omega_med",
        "max_penetration_mm",
        "max_constraint_error_mm",
        "n_contacts_max",
        "top_contact_pairs",
        "contact_force_rms_N",
        "power_balance_error_pct",
        "torque_ripple_pct",
    ):
        assert key in smc_m
