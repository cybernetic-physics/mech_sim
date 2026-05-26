from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "mechanical_evolve.py"
    spec = importlib.util.spec_from_file_location("mechanical_evolve", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_map_elites_keeps_best_candidate_per_cell():
    mod = _load_module()
    archive = mod.MapElitesArchive()
    base = {
        "id": "bad",
        "params": {"pins": 11, "eccentricity": 1.98, "clearance": 0.34,
                   "driver_circle_diameter": 49.5,
                   "driver_pin_collision_shrink_mm": 0.13},
        "fast_reward": 80.0,
        "verified_reward": 0.0,
        "verified_gate_passed": False,
        "defect_count": 3,
    }
    better = dict(base, id="better", verified_reward=60.0,
                  verified_gate_passed=True, defect_count=0)

    assert archive.insert(base) is True
    assert archive.insert(better) is True
    elites = archive.elites()

    assert len(elites) == 1
    assert elites[0]["id"] == "better"


def test_policy_update_uses_verifier_rewards():
    mod = _load_module()
    policy = mod.PolicyState()
    before = policy.weights["boundary_refine"]
    policy.update([
        {
            "verified_reward": 80.0,
            "defect_count": 0,
            "proposal": {"operator": "boundary_refine"},
        },
        {
            "verified_reward": 0.0,
            "defect_count": 4,
            "proposal": {"operator": "random_restart"},
        },
    ], lr=1.0)

    assert policy.weights["boundary_refine"] > before
    assert policy.weights["random_restart"] < 1.0
    assert policy.updates == 1


def test_model_payload_normalizes_candidate_params():
    mod = _load_module()
    proposals = mod.parse_model_payload(
        {
            "candidates": [
                {
                    "params": {
                        "pins": 50,
                        "eccentricity": 9.0,
                        "clearance": -1.0,
                        "driver_circle_diameter": 100.0,
                        "driver_pin_collision_shrink_mm": -3.0,
                    },
                    "notes": "clamped by runner",
                }
            ]
        },
        method="llm_zero_shot",
        id_prefix="test",
        proposer="unit",
    )

    assert len(proposals) == 1
    params = proposals[0].params
    assert params["pins"] == 14
    assert params["line_segment_count"] == 56
    assert params["eccentricity"] == 3.0
    assert params["clearance"] == 0.25
    assert params["driver_circle_diameter"] == 58.0
    assert params["driver_pin_collision_shrink_mm"] == 0.0


def test_dry_run_modes_write_archive_and_grpo_dataset(tmp_path):
    mod = _load_module()
    limits = mod.cyclo.VerificationLimits()
    runner = mod.MechanicalEvolveRunner(
        out_dir=tmp_path,
        seed=123,
        limits=limits,
        samples=5,
        duration_s=0.01,
        dry_run=True,
        proposal_jsonl=None,
        model_command=None,
        trainer_command=None,
    )

    evolve = runner.evolve_only(generations=1, population=8, audit_k=4)
    assert evolve["best"]["verified_reward"] > 0.0
    assert (tmp_path / "archive.json").is_file()
    assert (tmp_path / "lineage.jsonl").is_file()

    train = runner.rlvr_train(generations=1, population=6, audit_k=3)
    assert train["dataset_count"] > 0
    assert (tmp_path / "grpo_dataset.jsonl").is_file()

    adapt = runner.test_time_adapt(
        target_id="heldout",
        rounds=1,
        population=6,
        audit_k=3,
        policy_lr=1.0,
    )
    assert adapt["policy"]["updates"] >= 1
