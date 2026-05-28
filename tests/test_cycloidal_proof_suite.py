from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


def _load_suite():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_cycloidal_mechanical_evolve_proof_suite.py"
    )
    spec = importlib.util.spec_from_file_location(
        "run_cycloidal_mechanical_evolve_proof_suite", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_proof_suite_aggregates_paired_deltas():
    mod = _load_suite()
    trials = [
        {
            "target": "nominal",
            "seed": 1,
            "method_table": [
                {"method": "verifier_gated", "best_verified_reward": 10.0, "candidate_count": 2, "chrono_audits": 2},
                {"method": "llm_evolve_no_update", "best_verified_reward": 12.0, "candidate_count": 2, "chrono_audits": 2},
                {"method": "mechanical_evolve_ttrl", "best_verified_reward": 15.0, "candidate_count": 2, "chrono_audits": 2},
            ],
        },
        {
            "target": "high_load",
            "seed": 2,
            "method_table": [
                {"method": "verifier_gated", "best_verified_reward": 9.0, "candidate_count": 2, "chrono_audits": 2},
                {"method": "llm_evolve_no_update", "best_verified_reward": 11.0, "candidate_count": 2, "chrono_audits": 2},
                {"method": "mechanical_evolve_ttrl", "best_verified_reward": 14.0, "candidate_count": 2, "chrono_audits": 2},
            ],
        },
    ]

    paired = mod.paired_deltas(trials)
    by_baseline = {row["baseline"]: row for row in paired}

    assert by_baseline["verifier_gated"]["mean_delta_ttrl_minus_baseline"] == 5.0
    assert by_baseline["llm_evolve_no_update"]["win_rate"] == 1.0


def test_proof_condition_requires_positive_ci():
    mod = _load_suite()
    method_table = [
        {"method": "verifier_gated", "best_verified_reward_mean": 10.0},
        {"method": "llm_evolve_no_update", "best_verified_reward_mean": 11.0},
        {"method": "mechanical_evolve_ttrl", "best_verified_reward_mean": 12.0},
    ]
    paired = [
        {"baseline": "verifier_gated", "mean_delta_ttrl_minus_baseline": 2.0, "ci95_low": 0.1},
        {"baseline": "llm_evolve_no_update", "mean_delta_ttrl_minus_baseline": 1.0, "ci95_low": -0.5},
    ]

    conditions = mod.proof_conditions(method_table, paired)

    assert conditions["ttrl_mean_beats_all_baselines"] is True
    assert conditions["paired_mean_delta_positive_vs_all"] is True
    assert conditions["paired_ci95_positive_vs_all"] is False
    assert conditions["paper_grade_statistical_claim_supported"] is False


def test_markdown_names_equal_budget_failure_regime(tmp_path):
    mod = _load_suite()
    summary = {
        "budget": {"chrono_audits_per_method_per_trial": 160},
        "method_table": [
            {
                "method": "verifier_gated",
                "trial_count": 1,
                "chrono_audits": 160,
                "best_verified_reward_mean": 74.0,
                "best_verified_reward_ci95_low": 74.0,
                "best_verified_reward_ci95_high": 74.0,
            },
            {
                "method": "mechanical_evolve_ttrl",
                "trial_count": 1,
                "chrono_audits": 160,
                "best_verified_reward_mean": 70.0,
                "best_verified_reward_ci95_low": 70.0,
                "best_verified_reward_ci95_high": 70.0,
                "adapter_updates": 5,
                "trained_tokens": 1800,
            },
        ],
        "paired_deltas": [
            {
                "baseline": "verifier_gated",
                "trial_count": 1,
                "mean_delta_ttrl_minus_baseline": -4.0,
                "ci95_low": -4.0,
                "ci95_high": -4.0,
                "win_rate": 0.0,
            }
        ],
        "target_table": [
            {
                "target": "high_speed",
                "target_kind": "target",
                "seed": 20260527,
                "ttrl_best_verified_reward": 70.0,
                "best_non_ttrl_verified_reward": 80.0,
                "ttrl_wins": False,
            }
        ],
        "proof_conditions": {
            "ttrl_mean_beats_all_baselines": False,
            "paired_mean_delta_positive_vs_all": False,
            "paired_ci95_positive_vs_all": False,
            "secondary_metric_support_vs_all": True,
            "paper_grade_statistical_claim_supported": False,
        },
    }
    path = tmp_path / "summary.md"

    mod.write_markdown(path, summary)

    text = path.read_text()
    assert "TTRL does not yet win under the equal Chrono audit budget" in text
    assert "high_speed/seed_20260527" in text
    assert "procedural_cycloidal_fallback=false" in text


def test_proof_suite_passes_trial_seed_to_runner(tmp_path, monkeypatch):
    mod = _load_suite()
    captured = {}

    def fake_run_command(command, *, cwd, log_path, timeout_s):
        captured["command"] = command
        result_path = Path(command[command.index("--results-json") + 1])
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(
            '{"method_table": [{"method": "mechanical_evolve_ttrl"}]}\n'
        )
        return {"status": "completed", "returncode": 0}

    monkeypatch.setattr(mod, "run_command", fake_run_command)
    args = SimpleNamespace(
        model="local-model",
        budget=1,
        baseline_audits=1,
        verifier_pool=1,
        rounds=1,
        proposals_per_round=1,
        audits_per_round=1,
        mutation_fill=0,
        samples=3,
        duration_s=0.01,
        power_balance_limit_pct=90.0,
        torque_ripple_limit_pct=1000.0,
        contact_force_limit_N=3000.0,
        max_contacts=128.0,
        lora_iters=1,
        lora_max_examples=1,
        lora_num_layers=1,
        lora_rank=1,
        lora_scale=16.0,
        ttrl_base_exploration_frac=0.5,
        stability_repeats=0,
        trial_timeout_s=1.0,
        rerun=True,
    )

    mod.run_or_load_trial(
        args=args,
        out_dir=tmp_path,
        target=mod.TARGETS["nominal"],
        seed=12345,
    )

    command = captured["command"]
    assert command[command.index("--seed") + 1] == "12345"
    assert command[command.index("--stability-repeats") + 1] == "0"


def test_proof_suite_resumes_partial_trial_artifacts(tmp_path, monkeypatch):
    mod = _load_suite()
    captured = {}

    def fake_run_command(command, *, cwd, log_path, timeout_s):
        captured["command"] = command
        result_path = Path(command[command.index("--results-json") + 1])
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(
            '{"method_table": [{"method": "mechanical_evolve_ttrl"}]}\n'
        )
        return {"status": "completed", "returncode": 0}

    monkeypatch.setattr(mod, "run_command", fake_run_command)
    trial_dir = tmp_path / "nominal" / "seed_7"
    baseline_json = (
        trial_dir / "run" / "baselines" / "cycloidal_optimizer_strict_matched.json"
    )
    baseline_json.parent.mkdir(parents=True)
    baseline_json.write_text("{}\n")
    no_update_json = trial_dir / "run" / "llm_evolve_no_update" / "summary.json"
    no_update_json.parent.mkdir(parents=True)
    no_update_json.write_text("{}\n")
    args = SimpleNamespace(
        model="local-model",
        budget=1,
        baseline_audits=1,
        verifier_pool=1,
        rounds=1,
        proposals_per_round=1,
        audits_per_round=1,
        mutation_fill=0,
        samples=3,
        duration_s=0.01,
        power_balance_limit_pct=90.0,
        torque_ripple_limit_pct=1000.0,
        contact_force_limit_N=3000.0,
        max_contacts=128.0,
        lora_iters=1,
        lora_max_examples=1,
        lora_num_layers=1,
        lora_rank=1,
        lora_scale=16.0,
        ttrl_base_exploration_frac=0.5,
        stability_repeats=0,
        trial_timeout_s=1.0,
        rerun=False,
    )

    mod.run_or_load_trial(
        args=args,
        out_dir=tmp_path,
        target=mod.TARGETS["nominal"],
        seed=7,
    )

    command = captured["command"]
    assert "--skip-baselines" in command
    assert "--skip-no-update" in command
