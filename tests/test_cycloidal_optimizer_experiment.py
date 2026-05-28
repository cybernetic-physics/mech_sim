from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from types import SimpleNamespace


def _load_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "optimize_cycloidal_chrono_candidates.py"
    )
    spec = importlib.util.spec_from_file_location(
        "optimize_cycloidal_chrono_candidates", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _good_metrics() -> dict[str, float]:
    return {
        "lockup_detected": 0.0,
        "ratio_observed": 9.1,
        "ratio_error_pct": 1.1,
        "out_omega_med": -1.1,
        "max_penetration_mm": 0.22,
        "contact_force_rms_N": 950.0,
        "n_contacts_max": 18.0,
        "power_balance_error_pct": 15.0,
        "torque_ripple_pct": 20.0,
    }


def test_fast_reward_favors_plausible_cycloidal_actuator_params():
    mod = _load_module()

    plausible = mod.fast_cps_actuator_reward({
        "pins": 10,
        "clearance": 0.58,
        "driver_circle_diameter": 38.0,
        "driver_pin_collision_shrink_mm": 0.58,
        "line_segment_count": 44,
    })
    bad = mod.fast_cps_actuator_reward({
        "pins": 14,
        "clearance": 0.05,
        "driver_circle_diameter": 60.0,
        "driver_pin_collision_shrink_mm": 1.2,
        "line_segment_count": 12,
    })

    assert 0.0 <= plausible["score"] <= 100.0
    assert 0.0 <= bad["score"] <= 100.0
    assert plausible["score"] > bad["score"] + 25.0
    assert plausible["components"]["target_ratio"] == 1.0


def test_verified_reward_is_zero_until_all_hard_gates_pass():
    mod = _load_module()
    limits = mod.VerificationLimits()
    metrics = _good_metrics()

    assert mod.verified_gate_passed(
        cad_generated=True,
        cad_static_ok=True,
        chrono_real_geometry=True,
        metrics=metrics,
        limits=limits,
    )
    assert mod.verified_reward(
        fast_reward=80.0,
        cad_generated=True,
        cad_static_ok=True,
        chrono_real_geometry=True,
        metrics=metrics,
        limits=limits,
    ) > 0.0

    lockup_metrics = dict(metrics, lockup_detected=1.0, out_omega_med=0.01)
    assert mod.verified_reward(
        fast_reward=80.0,
        cad_generated=True,
        cad_static_ok=True,
        chrono_real_geometry=True,
        metrics=lockup_metrics,
        limits=limits,
    ) == 0.0

    nonfinite_ratio = dict(metrics, ratio_observed=math.inf)
    assert mod.verified_reward(
        fast_reward=80.0,
        cad_generated=True,
        cad_static_ok=True,
        chrono_real_geometry=True,
        metrics=nonfinite_ratio,
        limits=limits,
    ) == 0.0

    assert mod.verified_reward(
        fast_reward=80.0,
        cad_generated=True,
        cad_static_ok=False,
        chrono_real_geometry=True,
        metrics=metrics,
        limits=limits,
    ) == 0.0


def test_chrono_config_uses_target_and_contact_settings():
    mod = _load_module()
    cfg = mod._chrono_config(
        assets=SimpleNamespace(root=Path("/tmp/build")),
        samples=41,
        duration_s=0.15,
        limits=mod.VerificationLimits(min_output_speed_rad_s=0.75),
        trial=mod.ChronoTrialConfig(
            input_speed_rad_s=14.0,
            output_load_Nm=1.1,
            young_modulus=2.0e8,
            normal_stiffness=1.0e8,
            damping=350.0,
            friction=0.1,
        ),
    )

    probe_cfg = cfg["_mech_bench"]["probe_specs"][0]["config"]
    assert probe_cfg["input_speed_rad_s"] == 14.0
    assert probe_cfg["output_load_Nm"] == 1.1
    assert probe_cfg["min_output_speed_rad_s"] == 0.75
    assert cfg["young_modulus"] == 2.0e8
    assert cfg["normal_stiffness"] == 1.0e8
    assert cfg["damping"] == 350.0
    assert cfg["friction"] == 0.1


def test_method_table_reports_requested_baseline_columns():
    mod = _load_module()
    limits = mod.VerificationLimits()
    good = _good_metrics()
    lockup = dict(
        good,
        lockup_detected=1.0,
        out_omega_med=0.0,
        ratio_observed=math.inf,
    )
    rows = [
        {
            "method": "seed",
            "fast_reward": 55.0,
            "verified_reward": 0.0,
            "cad_generated": True,
            "cad_static_ok": True,
            "chrono_real_geometry": True,
            "metrics": lockup,
            "verified_gate_passed": False,
            "defect_count": 3,
        },
        {
            "method": "random",
            "fast_reward": 60.0,
            "verified_reward": 35.0,
            "cad_generated": True,
            "cad_static_ok": True,
            "chrono_real_geometry": True,
            "metrics": good,
            "verified_gate_passed": True,
            "defect_count": 0,
        },
        {
            "method": "cma_es_fast_only",
            "fast_reward": 78.0,
            "verified_reward": 0.0,
            "cad_generated": False,
            "cad_static_ok": False,
            "chrono_real_geometry": False,
            "metrics": {},
            "verified_gate_passed": False,
            "defect_count": 1,
        },
        {
            "method": "verifier_gated",
            "fast_reward": 82.0,
            "verified_reward": 70.0,
            "cad_generated": True,
            "cad_static_ok": True,
            "chrono_real_geometry": True,
            "metrics": good,
            "verified_gate_passed": True,
            "defect_count": 0,
        },
        {
            "method": "verifier_gated",
            "fast_reward": 84.0,
            "verified_reward": 0.0,
            "cad_generated": True,
            "cad_static_ok": False,
            "chrono_real_geometry": True,
            "metrics": lockup,
            "verified_gate_passed": False,
            "defect_count": 4,
        },
    ]

    table = mod._method_table(rows, limits=limits)
    by_method = {row["method"]: row for row in table}

    assert list(by_method) == list(mod.METHOD_ORDER)
    for column in mod.TABLE_COLUMNS:
        assert column in by_method["seed"]
    assert by_method["seed"]["best_verified_reward"] == 0.0
    assert by_method["seed"]["lockup rate"] == 1.0
    assert by_method["random"]["CAD pass rate"] == 1.0
    assert by_method["random"]["Chrono pass rate"] == 1.0
    assert by_method["cma_es_fast_only"]["CAD pass rate"] == 0.0
    assert by_method["verifier_gated"]["best_fast_reward"] == 84.0
    assert by_method["verifier_gated"]["best_verified_reward"] == 70.0
    assert by_method["verifier_gated"]["CAD pass rate"] == 0.5
    assert by_method["verifier_gated"]["Chrono pass rate"] == 0.5
    assert by_method["verifier_gated"]["mean defect count"] == 2.0


def test_method_table_can_report_verifier_only_subset():
    mod = _load_module()

    assert mod._selected_methods("verifier_gated") == ["verifier_gated"]
    table = mod._method_table([], methods=["verifier_gated"])

    assert [row["method"] for row in table] == ["verifier_gated"]


def test_verifier_gated_plan_includes_boundary_refinement_candidate():
    mod = _load_module()

    plans = mod._experiment_plans(SimpleNamespace(
        seed=20260525,
        random_candidates=8,
        cma_candidates=8,
        verifier_pool=32,
        verifier_audit_k=8,
    ))
    verifier_ids = {candidate.id for candidate in plans["verifier_gated"]}

    assert "vg_refine_driver_circle_0495" in verifier_ids
    assert "vg_refine_strict_anchor" in verifier_ids
    assert len(plans["verifier_gated"]) == 8


def test_verifier_selection_keeps_strict_physics_portfolio():
    mod = _load_module()
    pool = [
        mod.Candidate(
            id=f"fast_{idx}",
            method="verifier_gated",
            params={
                "pins": 10,
                "eccentricity": 2.95,
                "clearance": 0.60 + idx * 0.01,
                "driver_circle_diameter": 56.0,
                "driver_pin_collision_shrink_mm": 0.45,
                "line_segment_count": 42,
            },
            proposer="fast_reward_pool",
        )
        for idx in range(12)
    ]
    pool.extend(mod._verifier_refinement_candidates())

    selected = mod._select_verifier_candidates(pool, audit_k=4)

    assert selected[0].id == "vg_refine_strict_anchor"
    assert any(
        candidate.proposer == "strict_power_ripple_refinement"
        for candidate in selected
    )
