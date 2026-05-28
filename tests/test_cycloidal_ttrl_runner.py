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
        trial=mod.cyclo.ChronoTrialConfig(output_load_Nm=1.0),
    )

    assert count == 1
    record = json.loads(path.read_text().strip())
    assert record["parent_id"] == "root"
    assert len(record["responses"]) == 2
    assert record["responses"][0]["reward"] == 65.0
    assert record["prompt"]["paper_gate"]["max_torque_ripple_pct"] == 1000
    assert record["prompt"]["chrono_trial"]["output_load_Nm"] == 1.0


def test_win_conditions_require_ttrl_to_beat_baselines_and_stability():
    mod = _load_runner()
    table = [
        {
            "method": "verifier_gated",
            "best_verified_reward": 65.0,
            "best_ratio_error_pct": 12.0,
        },
        {
            "method": "llm_evolve_no_update",
            "best_verified_reward": 66.0,
            "best_ratio_error_pct": 30.0,
        },
        {
            "method": "mechanical_evolve_ttrl",
            "best_verified_reward": 70.0,
            "best_ratio_error_pct": 5.0,
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
    assert wins["equal_budget_success"] is True


def test_equal_budget_markdown_states_verifier_invariants(tmp_path):
    mod = _load_runner()
    path = tmp_path / "summary.md"
    summary = {
        "budget": {"target_chrono_audits": 160},
        "verifier": {
            "contact_model": "smc",
            "procedural_cycloidal_fallback": False,
            "samples": 41,
            "duration_s": 0.15,
            "limits": {"max_contacts": 128},
        },
        "win_conditions": {"equal_budget_success": True},
        "method_table": [
            {
                "method": "verifier_gated",
                "candidate_count": 160,
                "chrono_audits": 160,
                "best_verified_reward": 65.0,
            },
            {
                "method": "llm_evolve_no_update",
                "candidate_count": 200,
                "chrono_audits": 160,
                "best_verified_reward": 0.0,
            },
            {
                "method": "mechanical_evolve_ttrl",
                "candidate_count": 220,
                "chrono_audits": 160,
                "best_verified_reward": 70.0,
                "adapter_updates": 5,
            },
        ],
    }

    mod.write_markdown_summary(path, summary)

    text = path.read_text()
    assert "equal Chrono audit budget: yes" in text
    assert "identical verifier settings: yes" in text
    assert "procedural_cycloidal_fallback=false: true" in text
    assert "TTRL wins under equal budget: yes" in text


def test_lora_training_reuses_previous_adapter_after_transient_oom(tmp_path, monkeypatch):
    mod = _load_runner()
    previous = tmp_path / "prev_adapter"
    previous.mkdir()
    (previous / "adapters.safetensors").write_bytes(b"weights")
    round_dir = tmp_path / "round"
    train_dir = round_dir / "mlx_lora"
    train_dir.mkdir(parents=True)
    (train_dir / "training_summary.json").write_text(
        json.dumps({
            "example_count": 3,
            "best_training_reward": 67.0,
            "trainer": {
                "train_loss": 0.1,
                "trained_tokens": 89,
                "peak_mem_gb": 25.9,
            },
        })
    )
    calls = []

    def fake_run_command(command, *, cwd, log_path, timeout_s):
        calls.append(log_path.name)
        return mod.CommandResult(
            status="failed",
            returncode=-6,
            command=command,
            log_path=str(log_path),
            stdout_tail="[METAL] Command buffer execution failed: Insufficient Memory",
            stderr_tail="",
        )

    monkeypatch.setattr(mod, "run_command", fake_run_command)
    monkeypatch.setattr(mod.time, "sleep", lambda _: None)
    args = type("Args", (), {
        "model": "local-model",
        "lora_iters": 4,
        "lora_batch_size": 1,
        "lora_grad_accumulation_steps": 1,
        "lora_learning_rate": 1e-5,
        "lora_num_layers": 4,
        "lora_rank": 4,
        "lora_scale": 16.0,
        "lora_dropout": 0.0,
        "lora_max_examples": 256,
        "lora_min_reward": 1.0,
        "seed": 123,
        "allow_zero_reward_lora": False,
        "train_timeout_s": 1.0,
        "lora_train_retries": 1,
        "lora_retry_sleep_s": 0.0,
    })()

    trainer = mod.train_lora_round(
        args=args,
        round_dir=round_dir,
        dataset_path=tmp_path / "dataset.jsonl",
        archive_path=tmp_path / "archive.json",
        previous_adapter=previous,
        round_idx=2,
    )

    assert trainer["status"] == "reused_previous_adapter_after_failure"
    assert trainer["fallback_adapter_reused"] is True
    assert len(trainer["attempts"]) == 2
    assert calls == ["train.log", "train_retry_01.log"]
