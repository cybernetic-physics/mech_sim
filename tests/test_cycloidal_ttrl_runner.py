from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_runner():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_cycloidal_mechanical_evolve_ttrl.py"
    )
    spec = importlib.util.spec_from_file_location(
        "run_cycloidal_mechanical_evolve_ttrl", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_ttrl_dataset_groups_verifier_rewards(tmp_path):
    mod = _load_runner()
    archive = mod.mech.MapElitesArchive()
    good = {
        "id": "good",
        "params": {
            "pins": 11,
            "eccentricity": 2.02,
            "clearance": 0.34,
            "driver_circle_diameter": 49.5,
            "driver_pin_collision_shrink_mm": 0.13,
        },
        "fast_reward": 75.0,
        "verified_reward": 65.0,
        "verified_gate_passed": True,
        "defects": [],
        "metrics": {"out_omega_med": 1.2},
        "proposal": {"parent_id": "root"},
    }
    bad = {
        "id": "bad",
        "params": dict(good["params"], eccentricity=1.9),
        "verified_reward": 0.0,
        "defects": ["lockup"],
        "proposal": {"parent_id": "root"},
    }
    archive.insert(good)
    path = tmp_path / "grpo.jsonl"

    count = mod.write_grpo_dataset(
        path,
        rows=[good, bad],
        archive=archive,
        limits=mod.cyclo.VerificationLimits(
            max_power_balance_error_pct=90,
            max_torque_ripple_pct=1000,
        ),
    )

    assert count == 1
    record = json.loads(path.read_text().strip())
    assert record["parent_id"] == "root"
    assert len(record["responses"]) == 2
    assert record["responses"][0]["reward"] == 65.0
    assert record["prompt"]["paper_gate"]["max_torque_ripple_pct"] == 1000


def test_win_conditions_require_ttrl_to_beat_baselines_and_stability():
    mod = _load_runner()
    table = [
        {"method": "verifier_gated", "best_verified_reward": 65.0},
        {"method": "llm_evolve_no_update", "best_verified_reward": 66.0},
        {
            "method": "mechanical_evolve_ttrl",
            "best_verified_reward": 70.0,
            "adapter_updates": 4,
        },
    ]

    wins = mod.win_conditions_from_table(
        method_table=table,
        stability={"repeat_count": 3, "pass_count": 3},
    )

    assert wins["ttrl_beats_no_update"] is True
    assert wins["ttrl_beats_all_baselines"] is True
    assert wins["ttrl_has_adapter_updates"] is True
    assert wins["best_survives_regenerated_cad"] is True
