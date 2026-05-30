from __future__ import annotations

import json
import shlex
from pathlib import Path
from types import SimpleNamespace

from scripts.run_family_generalization_benchmark import (
    MethodSpec,
    PAPER_TTRL_BASE_MODEL,
    REQUIRED_METHODS,
    audit_eval_coverage,
    audit_no_procedural_fallback,
    audit_physical_metric_coverage,
    audit_split,
    audit_split_task_verifiers,
    audit_methods,
    build_preflight_full_run_command,
    build_peft_sft_cmd,
    build_true_grpo_cmd,
    can_skip_worldlines_init_for_resume,
    enforce_paper_verifier_ready_split,
    family_run_uses_worldlines,
    flatten_eval,
    load_or_freeze_family_split,
    load_or_run_eval_summary,
    matched_eval_samples_per_task,
    resolve_family_tasks_root,
    summarize_split_role_metrics,
    train_peft_sft_model,
    ttrl_beats_required_baselines_on_unseen,
    write_preflight_report,
    write_results,
)


_PHYSICAL_ROW_METRICS = {
    "best_out_omega_med": 1.0,
    "best_ratio_error_pct": 2.0,
    "best_power_balance_error_pct": 3.0,
    "best_torque_ripple_pct": 4.0,
    "best_max_penetration_mm": 0.2,
    "best_contact_force_rms_N": 20.0,
}


def _method(name: str, samples_per_task: int = 1) -> MethodSpec:
    return MethodSpec(
        name=name,
        model_path=None,
        rollout_backend="worldlines_sampling",
        sglang_lora_path=None,
        samples_per_task=samples_per_task,
        max_turns=1,
        temperature=0.2,
        top_p=0.95,
        seed_offset=0,
        baseline_kind="test_method",
    )


def test_flatten_eval_uses_verifier_valid_pass_rate_for_verified_pass() -> None:
    row = flatten_eval(
        _method("frozen_model", samples_per_task=1),
        {
            "n_samples": 1,
            "samples_per_task": 1,
            "n_tasks": 1,
            "pass_rate_best_of_k": 0.0,
            "verifier_valid_pass_rate_best_of_k": 1.0,
            "pass_rate_raw": 0.0,
            "verifier_valid_pass_rate_raw": 1.0,
            "tasks": [
                {
                    "task_id": "cam_follower_contact_stub_s0001",
                    "verified_score": 0.8,
                    "strict_passed": False,
                    "verifier_valid_passed": True,
                }
            ],
        },
    )

    assert row["verified_pass_rate"] == 1.0
    assert row["strict_score_pass_rate"] == 0.0
    assert row["best_verified_reward"] == 0.8
    assert row["baseline_kind"] == "test_method"
    assert row["max_turns"] == 1


def test_flatten_eval_emits_family_metrics() -> None:
    row = flatten_eval(
        _method("frozen_model", samples_per_task=2),
        {
            "n_samples": 2,
            "samples_per_task": 2,
            "n_tasks": 2,
            "pass_rate_best_of_k": 0.5,
            "verifier_valid_pass_rate_best_of_k": 1.0,
            "tasks": [
                {
                    "task_id": "cam_follower_contact_stub_s0001",
                    "family": "cam_follower_contact_stub",
                    "verified_score": 0.8,
                    "strict_passed": False,
                    "verifier_valid_passed": True,
                    "verifier_calls": 2,
                    "cad_audits": 1,
                    "chrono_audits": 1,
                    "no_procedural_fallback": True,
                    "repair_attempted": True,
                    "repair_succeeded": True,
                    "physical_metrics": {
                        "out_omega_med": 0.8,
                        "ratio_error_pct": 4.0,
                        "power_balance_error_pct": 6.0,
                        "torque_ripple_pct": 8.0,
                        "max_penetration_mm": 0.2,
                        "contact_force_rms_N": 20.0,
                    },
                    "failure_codes": [],
                },
                {
                    "task_id": "lead_screw_linear_travel_s0001",
                    "family": "lead_screw_linear_travel",
                    "verified_score": 1.0,
                    "strict_passed": True,
                    "verifier_valid_passed": True,
                    "cad_audits": 0,
                    "chrono_audits": 0,
                    "no_procedural_fallback": False,
                    "physical_metrics": {
                        "out_omega_med": 1.1,
                        "ratio_error_pct": 2.0,
                        "power_balance_error_pct": 10.0,
                        "torque_ripple_pct": 4.0,
                        "max_penetration_mm": 0.4,
                        "contact_force_rms_N": 10.0,
                    },
                    "failure_codes": ["lockup"],
                },
            ],
        },
    )

    families = {item["family"]: item for item in row["families"]}
    assert families["cam_follower_contact_stub"]["verified_pass_rate"] == 1.0
    assert families["cam_follower_contact_stub"]["strict_score_pass_rate"] == 0.0
    assert families["cam_follower_contact_stub"]["mean_verified_reward"] == 0.8
    assert families["cam_follower_contact_stub"]["cad_pass_rate"] == 1.0
    assert families["cam_follower_contact_stub"]["chrono_real_geometry_rate"] == 1.0
    assert families["cam_follower_contact_stub"]["no_procedural_fallback_rate"] == 1.0
    assert families["cam_follower_contact_stub"]["repair_attempt_count"] == 1
    assert families["cam_follower_contact_stub"]["repair_success_rate"] == 1.0
    assert families["lead_screw_linear_travel"]["strict_score_pass_rate"] == 1.0
    assert families["lead_screw_linear_travel"]["no_procedural_fallback_rate"] == 0.0
    assert families["lead_screw_linear_travel"]["lockup_rate"] == 1.0
    assert row["cad_pass_rate"] == 0.5
    assert row["chrono_real_geometry_rate"] == 0.5
    assert row["no_procedural_fallback_rate"] == 0.5
    assert row["lockup_rate"] == 0.5
    assert row["repair_success_rate"] == 1.0
    assert row["best_out_omega_med"] == 1.1
    assert row["best_ratio_error_pct"] == 2.0
    assert row["best_power_balance_error_pct"] == 6.0
    assert row["best_torque_ripple_pct"] == 4.0
    assert row["best_max_penetration_mm"] == 0.2
    assert row["best_contact_force_rms_N"] == 10.0


def test_write_results_backfills_metric_defaults_for_old_rows(tmp_path: Path) -> None:
    rows = [
        {
            "method": method,
            "candidate_count": 1,
            "verifier_calls": 1,
            "n_tasks": 1,
            "verified_pass_rate": 0.5,
            "strict_score_pass_rate": 0.5,
            "best_verified_reward": 1.0,
            "adapter_updates": 0,
            "trained_tokens": 0,
            "rl_trained_tokens": 0,
            "n_rl_datums": 0,
        }
        for method in REQUIRED_METHODS
    ]

    write_results(tmp_path, _split(), rows)
    payload = json.loads(
        (tmp_path / "family_generalization_results.json").read_text()
    )

    assert payload["rows"][0]["cad_audits"] == 0
    assert payload["rows"][0]["chrono_audits"] == 0
    assert payload["rows"][0]["cad_pass_rate"] == 0.0
    assert payload["rows"][0]["chrono_real_geometry_rate"] == 0.0
    assert payload["rows"][0]["no_procedural_fallback_rate"] == 0.0
    assert payload["rows"][0]["lockup_rate"] == 0.0
    assert payload["rows"][0]["repair_success_rate"] == 0.0
    assert payload["rows"][0]["best_ratio_error_pct"] is None


def test_split_role_metrics_expose_unseen_family_result() -> None:
    split = _split()
    rows = [
        {
            "method": "frozen_model",
            "family": "slider_crank",
            "n_tasks": 2,
            "verified_pass_rate": 0.5,
            "mean_verified_reward": 0.4,
            "best_verified_reward": 0.8,
            "lockup_rate": 0.5,
            "no_procedural_fallback_rate": 0.5,
            "repair_attempt_count": 0,
            "repair_success_count": 0,
        },
        {
            "method": "mechanical_evolve_ttrl",
            "family": "slider_crank",
            "n_tasks": 2,
            "verified_pass_rate": 1.0,
            "mean_verified_reward": 0.9,
            "best_verified_reward": 1.0,
            "lockup_rate": 0.0,
            "no_procedural_fallback_rate": 1.0,
            "repair_attempt_count": 2,
            "repair_success_count": 1,
        },
    ]

    out = summarize_split_role_metrics(split, rows)
    by_method = {row["method"]: row for row in out}

    assert by_method["mechanical_evolve_ttrl"]["split_role"] == "unseen"
    assert by_method["mechanical_evolve_ttrl"]["best_verified_reward_mean"] == 0.9
    assert by_method["mechanical_evolve_ttrl"]["no_procedural_fallback_rate"] == 1.0
    assert by_method["mechanical_evolve_ttrl"]["repair_success_rate"] == 0.5


def test_unseen_baseline_gate_uses_split_role_best_reward_mean() -> None:
    split_role_rows = [
        {
            "method": method,
            "split_role": "unseen",
            "best_verified_reward_mean": (
                0.9 if method == "mechanical_evolve_ttrl" else 0.5
            ),
        }
        for method in REQUIRED_METHODS
    ]

    assert ttrl_beats_required_baselines_on_unseen(split_role_rows) is True

    split_role_rows[0]["best_verified_reward_mean"] = 1.0
    assert ttrl_beats_required_baselines_on_unseen(split_role_rows) is False


def test_flatten_eval_uses_actual_verifier_call_count() -> None:
    method = _method("llm_evolve_no_update", samples_per_task=2)
    method = MethodSpec(
        name=method.name,
        model_path=method.model_path,
        rollout_backend=method.rollout_backend,
        sglang_lora_path=method.sglang_lora_path,
        samples_per_task=method.samples_per_task,
        max_turns=3,
        temperature=method.temperature,
        top_p=method.top_p,
        seed_offset=method.seed_offset,
        baseline_kind=method.baseline_kind,
    )
    row = flatten_eval(
        method,
        {
            "n_samples": 2,
            "n_verifier_calls": 5,
            "samples_per_task": 2,
            "n_tasks": 1,
            "pass_rate_best_of_k": 0.0,
            "verifier_valid_pass_rate_best_of_k": 0.0,
            "tasks": [],
        },
    )

    assert row["candidate_count"] == 2
    assert row["verifier_calls"] == 5
    assert row["planned_max_verifier_calls"] == 6
    assert row["verifier_calls_per_candidate"] == 2.5


def test_flatten_eval_uses_actual_chrono_audit_count() -> None:
    row = flatten_eval(
        _method("frozen_model", samples_per_task=2),
        {
            "n_samples": 2,
            "n_verifier_calls": 2,
            "n_cad_audits": 1,
            "n_chrono_audits": 1,
            "samples_per_task": 2,
            "n_tasks": 1,
            "pass_rate_best_of_k": 0.0,
            "verifier_valid_pass_rate_best_of_k": 0.0,
            "tasks": [],
        },
    )

    assert row["cad_audits"] == 1
    assert row["chrono_audits"] == 1


def _split(*, verifier_ready: bool = True) -> dict:
    test_meta = {
        "canonical_family": "slider_crank",
    }
    if verifier_ready:
        test_meta.update({
            "has_chrono_contact_config": True,
            "chrono_procedural_fallback_disabled": True,
            "has_trusted_asset_preflight": True,
            "requires_trusted_mass_properties": True,
        })
    return {
        "seen_families": ["fourbar"],
        "unseen_families": ["slider_crank"],
        "splits": {
            "train": ["fourbar_path_t001"],
            "val": [],
            "test": ["slider_crank_stroke_s0001"],
        },
        "task_index": {
            "fourbar_path_t001": {"canonical_family": "fourbar"},
            "slider_crank_stroke_s0001": test_meta,
        },
    }


def test_audit_split_requires_no_train_eval_family_overlap() -> None:
    split = _split()
    assert audit_split(split)["family_heldout"] is True

    split["task_index"]["slider_crank_stroke_s0001"] = {
        "canonical_family": "fourbar"
    }
    audit = audit_split(split)

    assert audit["family_heldout"] is False
    assert audit["train_test_family_overlap"] == ["fourbar"]


def test_audit_methods_requires_all_methods_and_equal_budget() -> None:
    rows = [
        {
            "method": method,
            "verifier_calls": 4,
            "cad_audits": 0,
            "chrono_audits": 0,
        }
        for method in REQUIRED_METHODS
    ]

    audit = audit_methods(rows)
    assert audit["required_methods_present"] is True
    assert audit["equal_verifier_budget"] is True
    assert audit["equal_cad_budget"] is True
    assert audit["positive_cad_budget"] is False
    assert audit["equal_chrono_budget"] is True
    assert audit["positive_chrono_budget"] is False

    positive_rows = [dict(row, cad_audits=2, chrono_audits=2) for row in rows]
    audit = audit_methods(positive_rows)
    assert audit["equal_cad_budget"] is True
    assert audit["positive_cad_budget"] is True
    assert audit["equal_chrono_budget"] is True
    assert audit["positive_chrono_budget"] is True

    audit = audit_methods(rows[:-1])
    assert audit["required_methods_present"] is False
    assert audit["missing_required_methods"] == [REQUIRED_METHODS[-1]]
    assert audit["equal_verifier_budget"] is False

    uneven_rows = [dict(row) for row in rows]
    uneven_rows[0]["verifier_calls"] = 5
    audit = audit_methods(uneven_rows)
    assert audit["required_methods_present"] is True
    assert audit["equal_verifier_budget"] is False


def test_audit_eval_coverage_requires_full_test_split_per_method() -> None:
    split = _split()
    rows = [
        {"method": method, "n_tasks": 1}
        for method in REQUIRED_METHODS
    ]

    audit = audit_eval_coverage(split, rows)
    assert audit["complete_required_eval_coverage"] is True

    rows[0]["n_tasks"] = 0
    audit = audit_eval_coverage(split, rows)
    assert audit["complete_required_eval_coverage"] is False
    assert audit["incomplete_required_methods"] == [REQUIRED_METHODS[0]]


def test_audit_split_task_verifiers_requires_paper_verifier_specs() -> None:
    split = _split(verifier_ready=False)
    audit = audit_split_task_verifiers(split)

    assert audit["paper_verifier_ready_test_tasks"] is False
    assert audit["missing_chrono_contact_config"] == [
        "slider_crank_stroke_s0001"
    ]
    assert audit["missing_trusted_asset_preflight"] == [
        "slider_crank_stroke_s0001"
    ]

    split["task_index"]["slider_crank_stroke_s0001"].update({
        "has_chrono_contact_config": True,
        "chrono_procedural_fallback_disabled": True,
        "has_trusted_asset_preflight": True,
        "requires_trusted_mass_properties": True,
    })
    audit = audit_split_task_verifiers(split)
    assert audit["paper_verifier_ready_test_tasks"] is True


def test_enforce_paper_verifier_ready_split_fails_fast() -> None:
    try:
        enforce_paper_verifier_ready_split(
            _split(verifier_ready=False),
            allow_non_paper_tasks=False,
        )
    except SystemExit as exc:
        assert "not paper-verifier-ready" in str(exc)
    else:
        raise AssertionError("expected SystemExit")

    audit = enforce_paper_verifier_ready_split(
        _split(verifier_ready=False),
        allow_non_paper_tasks=True,
    )
    assert audit["paper_verifier_ready_test_tasks"] is False


def test_write_preflight_report_emits_split_and_verifier_audits(
    tmp_path: Path,
) -> None:
    split = _split()
    command = "uv run python scripts/run_family_generalization_benchmark.py"
    payload = write_preflight_report(
        docs_dir=tmp_path,
        split=split,
        split_json=tmp_path / "split_manifest.json",
        tasks_root=tmp_path / "tasks",
        recommended_full_run_command=command,
    )

    written = json.loads(
        (tmp_path / "family_generalization_preflight.json").read_text()
    )
    assert payload == written
    assert written["split_audit"]["family_heldout"] is True
    assert (
        written["split_task_verifier_audit"]["paper_verifier_ready_test_tasks"]
        is True
    )
    assert written["recommended_full_run_command"] == command


def test_build_preflight_full_run_command_resumes_frozen_paper_root(
    tmp_path: Path,
) -> None:
    args = SimpleNamespace(
        base_model=PAPER_TTRL_BASE_MODEL,
        sft_trainer="peft_sft",
        ttrl_trainer="trl_grpo",
        eval_rollout_backend="sglang_chat",
        seen_families="cycloidal,belt",
        unseen_families="planetary",
        split_seed=20260528,
        train_rounds=6,
        tasks_per_round=4,
        samples_per_task=4,
        max_turns=2,
        max_tokens=1536,
        max_context_tokens=8192,
        temperature=0.7,
        top_p=0.95,
        timeout=180.0,
        train_timeout_s=21600.0,
        eval_timeout_s=21600.0,
        concurrency=2,
        lora_rank=16,
        lr=1.0e-4,
        sft_runner="uv run --extra training-grpo python",
        ttrl_grpo_runner="uv run --extra training-grpo python",
        max_train_datums_per_step=2,
        rlvr_refresh_sampler_every=1,
        rlvr_verifier_pass_fallback_weight=1.0,
        worldlines_base_url="http://127.0.0.1:18100",
        sglang_base_url="http://127.0.0.1:30000",
        api_key="wld-local",
        runner_python="/tmp/worldlines/.venv/bin/python",
        worldlines_root="/tmp/worldlines",
        worldlines_venv="/tmp/worldlines/.venv",
        worldlines_artifact_root="/tmp/wld-family-artifacts",
        worldlines_launch_timeout_s=600.0,
        match_planned_verifier_budget=True,
        eval_limit=None,
        sft_max_steps=None,
        sft_load_in_4bit=False,
        sft_torch_dtype=None,
        sft_device_map=None,
        sft_trust_remote_code=False,
        ttrl_grpo_max_steps=None,
        ttrl_grpo_num_generations=None,
        ttrl_grpo_load_in_4bit=False,
        ttrl_grpo_torch_dtype=None,
        ttrl_grpo_device_map=None,
        ttrl_grpo_trust_remote_code=False,
        manage_worldlines=False,
        allow_single_sample_rlvr=False,
        allow_zero_update_models=False,
    )

    command = build_preflight_full_run_command(
        args=args,
        tasks_root=tmp_path / "paper_tasks",
        out_dir=tmp_path / "run",
        docs_dir=tmp_path / "docs",
    )
    argv = shlex.split(command)

    assert argv[:4] == [
        "uv",
        "run",
        "python",
        "scripts/run_family_generalization_benchmark.py",
    ]
    assert argv[argv.index("--tasks-root") + 1] == str(tmp_path / "paper_tasks")
    assert "--keep-out-dir" in argv
    assert "--resume-existing" in argv
    assert "--materialize-paper-tasks" not in argv
    assert "--paper-task-overwrite" not in argv
    assert "--preflight-only" not in argv


def test_audit_physical_metric_coverage_requires_all_metric_fields() -> None:
    rows = [
        {
            "method": method,
            **_PHYSICAL_ROW_METRICS,
        }
        for method in REQUIRED_METHODS
    ]

    audit = audit_physical_metric_coverage(rows)
    assert audit["complete_required_physical_metrics"] is True

    rows[0]["best_ratio_error_pct"] = None
    audit = audit_physical_metric_coverage(rows)
    assert audit["complete_required_physical_metrics"] is False
    assert audit["missing_required_physical_metrics_by_method"] == {
        REQUIRED_METHODS[0]: ["best_ratio_error_pct"]
    }


def test_audit_no_procedural_fallback_requires_all_methods() -> None:
    rows = [
        {
            "method": method,
            "no_procedural_fallback_rate": 1.0,
        }
        for method in REQUIRED_METHODS
    ]

    audit = audit_no_procedural_fallback(rows)
    assert audit["complete_no_procedural_fallback_evidence"] is True

    rows[0]["no_procedural_fallback_rate"] = 0.0
    audit = audit_no_procedural_fallback(rows)
    assert audit["complete_no_procedural_fallback_evidence"] is False
    assert audit["methods_missing_no_procedural_fallback_evidence"] == [
        REQUIRED_METHODS[0]
    ]


def test_write_results_claim_requires_ttrl_trained_tokens(tmp_path: Path) -> None:
    rows = [
        {
            "method": "frozen_model",
            "candidate_count": 2,
            "verified_pass_rate": 0.5,
            "strict_score_pass_rate": 0.5,
            "best_verified_reward": 1.0,
            "n_tasks": 1,
            "adapter_updates": 0,
            "trained_tokens": 0,
            "rl_trained_tokens": 0,
            "n_rl_datums": 0,
            "no_procedural_fallback_rate": 1.0,
            **_PHYSICAL_ROW_METRICS,
            "families": [
                {
                    "family": "slider_crank",
                    "n_tasks": 1,
                    "verified_pass_rate": 0.5,
                    "strict_score_pass_rate": 0.5,
                    "mean_verified_reward": 1.0,
                    "best_verified_reward": 1.0,
                    "failure_count": 0,
                }
            ],
        },
        {
            "method": "mechanical_evolve_ttrl",
            "candidate_count": 2,
            "verified_pass_rate": 1.0,
            "strict_score_pass_rate": 1.0,
            "best_verified_reward": 1.0,
            "adapter_updates": 2,
            "trained_tokens": 0,
            "rl_trained_tokens": 0,
            "n_rl_datums": 0,
            "no_procedural_fallback_rate": 1.0,
            **_PHYSICAL_ROW_METRICS,
            "families": [
                {
                    "family": "slider_crank",
                    "n_tasks": 1,
                    "verified_pass_rate": 1.0,
                    "strict_score_pass_rate": 1.0,
                    "mean_verified_reward": 1.0,
                    "best_verified_reward": 1.0,
                    "failure_count": 0,
                }
            ],
        },
    ]

    write_results(tmp_path, _split(), rows)
    payload = json.loads(
        (tmp_path / "family_generalization_results.json").read_text()
    )

    assert payload["mechanical_evolve_ttrl_beats_all"] is True
    assert payload["mechanical_evolve_ttrl_adaptation_valid"] is False
    assert payload["claim_status"] == "does_not_yet_support_family_heldout_transfer"
    assert (tmp_path / "family_generalization_family_results.csv").exists()


def test_write_results_claim_requires_verifier_rl_tokens(tmp_path: Path) -> None:
    baseline_rows = [
        {
            "method": method,
            "candidate_count": 2,
            "verifier_calls": 2,
            "cad_audits": 2,
            "chrono_audits": 2,
            "verified_pass_rate": 0.5,
            "strict_score_pass_rate": 0.5,
            "best_verified_reward": 1.0,
            "n_tasks": 1,
            "adapter_updates": 0,
            "trained_tokens": 0,
            "rl_trained_tokens": 0,
            "n_rl_datums": 0,
            "no_procedural_fallback_rate": 1.0,
            **_PHYSICAL_ROW_METRICS,
            "families": [
                {
                    "family": "slider_crank",
                    "n_tasks": 1,
                    "verified_pass_rate": 0.5,
                    "strict_score_pass_rate": 0.5,
                    "mean_verified_reward": 0.5,
                    "best_verified_reward": 0.5,
                    "failure_count": 0,
                }
            ],
        }
        for method in REQUIRED_METHODS
        if method != "mechanical_evolve_ttrl"
    ]
    ttrl_row = {
        "method": "mechanical_evolve_ttrl",
        "base_model": PAPER_TTRL_BASE_MODEL,
        "ttrl_trainer": "trl_grpo",
        "ttrl_exact_grpo": True,
        "candidate_count": 2,
        "verifier_calls": 2,
        "cad_audits": 2,
        "chrono_audits": 2,
        "verified_pass_rate": 1.0,
        "strict_score_pass_rate": 1.0,
        "best_verified_reward": 1.0,
        "n_tasks": 1,
        "adapter_updates": 2,
        "trained_tokens": 512,
        "rl_trained_tokens": 0,
        "n_rl_datums": 0,
        "no_procedural_fallback_rate": 1.0,
        **_PHYSICAL_ROW_METRICS,
        "families": [
            {
                "family": "slider_crank",
                "n_tasks": 1,
                "verified_pass_rate": 1.0,
                "strict_score_pass_rate": 1.0,
                "mean_verified_reward": 1.0,
                "best_verified_reward": 1.0,
                "failure_count": 0,
            }
        ],
    }

    write_results(tmp_path, _split(), baseline_rows + [ttrl_row])
    payload = json.loads(
        (tmp_path / "family_generalization_results.json").read_text()
    )
    assert payload["mechanical_evolve_ttrl_beats_all"] is True
    assert payload["mechanical_evolve_ttrl_beats_required_baselines"] is True
    assert payload["mechanical_evolve_ttrl_adaptation_valid"] is False
    assert payload["claim_status"] == "does_not_yet_support_family_heldout_transfer"
    assert "ttrl_missing_verifier_derived_updates" in payload["claim_blockers"]

    ttrl_row["rl_trained_tokens"] = 256
    ttrl_row["n_rl_datums"] = 1
    write_results(tmp_path, _split(), baseline_rows + [ttrl_row])
    payload = json.loads(
        (tmp_path / "family_generalization_results.json").read_text()
    )
    assert payload["mechanical_evolve_ttrl_adaptation_valid"] is True
    assert payload["mechanical_evolve_ttrl_paper_model_valid"] is True
    assert payload["mechanical_evolve_ttrl_algorithm_valid"] is True
    assert payload["claim_status"] == "supports_family_heldout_transfer"

    baseline_rows[0]["no_procedural_fallback_rate"] = 0.0
    write_results(tmp_path, _split(), baseline_rows + [ttrl_row])
    payload = json.loads(
        (tmp_path / "family_generalization_results.json").read_text()
    )
    assert payload["claim_status"] == "does_not_yet_support_family_heldout_transfer"
    assert "procedural_fallback_not_proven_false" in payload["claim_blockers"]


def test_write_results_claim_requires_physical_metrics(tmp_path: Path) -> None:
    rows = [
        {
            "method": method,
            "base_model": PAPER_TTRL_BASE_MODEL if method == "mechanical_evolve_ttrl" else "",
            "ttrl_trainer": "trl_grpo" if method == "mechanical_evolve_ttrl" else "",
            "ttrl_exact_grpo": method == "mechanical_evolve_ttrl",
            "candidate_count": 2,
            "verifier_calls": 2,
            "cad_audits": 2,
            "chrono_audits": 2,
            "verified_pass_rate": 1.0 if method == "mechanical_evolve_ttrl" else 0.5,
            "strict_score_pass_rate": 1.0 if method == "mechanical_evolve_ttrl" else 0.5,
            "best_verified_reward": 1.0,
            "n_tasks": 1,
            "adapter_updates": 2 if method == "mechanical_evolve_ttrl" else 0,
            "trained_tokens": 512 if method == "mechanical_evolve_ttrl" else 0,
            "rl_trained_tokens": 256 if method == "mechanical_evolve_ttrl" else 0,
            "n_rl_datums": 1 if method == "mechanical_evolve_ttrl" else 0,
            "families": [
                {
                    "family": "slider_crank",
                    "n_tasks": 1,
                    "verified_pass_rate": 1.0 if method == "mechanical_evolve_ttrl" else 0.5,
                    "strict_score_pass_rate": 1.0 if method == "mechanical_evolve_ttrl" else 0.5,
                    "mean_verified_reward": 1.0 if method == "mechanical_evolve_ttrl" else 0.5,
                    "best_verified_reward": 1.0 if method == "mechanical_evolve_ttrl" else 0.5,
                    "failure_count": 0,
                }
            ],
        }
        for method in REQUIRED_METHODS
    ]

    write_results(tmp_path, _split(), rows)
    payload = json.loads(
        (tmp_path / "family_generalization_results.json").read_text()
    )

    assert payload["physical_metric_audit"]["complete_required_physical_metrics"] is False
    assert payload["claim_status"] == "does_not_yet_support_family_heldout_transfer"
    assert "missing_required_physical_metrics" in payload["claim_blockers"]


def test_write_results_claim_requires_positive_chrono_budget(tmp_path: Path) -> None:
    rows = [
        {
            "method": method,
            "base_model": PAPER_TTRL_BASE_MODEL if method == "mechanical_evolve_ttrl" else "",
            "ttrl_trainer": "trl_grpo" if method == "mechanical_evolve_ttrl" else "",
            "ttrl_exact_grpo": method == "mechanical_evolve_ttrl",
            "candidate_count": 2,
            "verifier_calls": 2,
            "cad_audits": 2,
            "chrono_audits": 0,
            "verified_pass_rate": 1.0 if method == "mechanical_evolve_ttrl" else 0.5,
            "strict_score_pass_rate": 1.0 if method == "mechanical_evolve_ttrl" else 0.5,
            "best_verified_reward": 1.0,
            "n_tasks": 1,
            "adapter_updates": 2 if method == "mechanical_evolve_ttrl" else 0,
            "trained_tokens": 512 if method == "mechanical_evolve_ttrl" else 0,
            "rl_trained_tokens": 256 if method == "mechanical_evolve_ttrl" else 0,
            "n_rl_datums": 1 if method == "mechanical_evolve_ttrl" else 0,
            "families": [],
        }
        for method in REQUIRED_METHODS
    ]

    write_results(tmp_path, _split(), rows)
    payload = json.loads(
        (tmp_path / "family_generalization_results.json").read_text()
    )

    assert payload["method_audit"]["equal_chrono_budget"] is True
    assert payload["method_audit"]["positive_chrono_budget"] is False
    assert payload["claim_status"] == "does_not_yet_support_family_heldout_transfer"
    assert "missing_positive_chrono_budget" in payload["claim_blockers"]


def test_write_results_claim_requires_positive_cad_budget(tmp_path: Path) -> None:
    rows = [
        {
            "method": method,
            "base_model": PAPER_TTRL_BASE_MODEL if method == "mechanical_evolve_ttrl" else "",
            "ttrl_trainer": "trl_grpo" if method == "mechanical_evolve_ttrl" else "",
            "ttrl_exact_grpo": method == "mechanical_evolve_ttrl",
            "candidate_count": 2,
            "verifier_calls": 2,
            "cad_audits": 0,
            "chrono_audits": 2,
            "verified_pass_rate": 1.0 if method == "mechanical_evolve_ttrl" else 0.5,
            "strict_score_pass_rate": 1.0 if method == "mechanical_evolve_ttrl" else 0.5,
            "best_verified_reward": 1.0,
            "n_tasks": 1,
            "adapter_updates": 2 if method == "mechanical_evolve_ttrl" else 0,
            "trained_tokens": 512 if method == "mechanical_evolve_ttrl" else 0,
            "rl_trained_tokens": 256 if method == "mechanical_evolve_ttrl" else 0,
            "n_rl_datums": 1 if method == "mechanical_evolve_ttrl" else 0,
            "families": [],
        }
        for method in REQUIRED_METHODS
    ]

    write_results(tmp_path, _split(), rows)
    payload = json.loads(
        (tmp_path / "family_generalization_results.json").read_text()
    )

    assert payload["method_audit"]["equal_cad_budget"] is True
    assert payload["method_audit"]["positive_cad_budget"] is False
    assert payload["claim_status"] == "does_not_yet_support_family_heldout_transfer"
    assert "missing_positive_cad_budget" in payload["claim_blockers"]


def test_write_results_claim_requires_paper_ttrl_model_and_exact_grpo(
    tmp_path: Path,
) -> None:
    rows = [
        {
            "method": method,
            "candidate_count": 2,
            "verifier_calls": 2,
            "chrono_audits": 0,
            "verified_pass_rate": 0.5,
            "strict_score_pass_rate": 0.5,
            "best_verified_reward": 1.0,
            "n_tasks": 1,
            "adapter_updates": 0,
            "trained_tokens": 0,
            "rl_trained_tokens": 0,
            "n_rl_datums": 0,
            "families": [],
        }
        for method in REQUIRED_METHODS
        if method != "mechanical_evolve_ttrl"
    ]
    rows.append({
        "method": "mechanical_evolve_ttrl",
        "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "ttrl_trainer": "worldlines_group_relative_weighted_ce",
        "ttrl_exact_grpo": False,
        "candidate_count": 2,
        "verifier_calls": 2,
        "chrono_audits": 0,
        "verified_pass_rate": 1.0,
        "strict_score_pass_rate": 1.0,
        "best_verified_reward": 1.0,
        "n_tasks": 1,
        "adapter_updates": 1,
        "trained_tokens": 512,
        "rl_trained_tokens": 256,
        "n_rl_datums": 1,
        "families": [],
    })

    write_results(tmp_path, _split(), rows)
    payload = json.loads(
        (tmp_path / "family_generalization_results.json").read_text()
    )

    assert payload["mechanical_evolve_ttrl_adaptation_valid"] is True
    assert payload["mechanical_evolve_ttrl_paper_model_valid"] is False
    assert payload["mechanical_evolve_ttrl_algorithm_valid"] is False
    assert payload["claim_status"] == "does_not_yet_support_family_heldout_transfer"
    assert "ttrl_wrong_base_model_for_paper_goal" in payload["claim_blockers"]
    assert "ttrl_not_exact_trl_grpo" in payload["claim_blockers"]


def test_build_true_grpo_cmd_uses_exact_trainer_and_split(tmp_path: Path) -> None:
    split_file = tmp_path / "train.txt"
    split_file.write_text("fourbar_path_t001\n")
    cmd = build_true_grpo_cmd(
        runner="uv run --extra training-grpo python",
        run_dir=tmp_path / "family_rlvr",
        base_model=PAPER_TTRL_BASE_MODEL,
        tasks_root="tasks",
        split_file=split_file,
        max_steps=6,
        samples_per_task=4,
        max_tokens=1536,
        max_context_tokens=8192,
        temperature=0.7,
        top_p=0.95,
        timeout=180.0,
        lora_rank=16,
        lr=1.0e-4,
        load_in_4bit=True,
        torch_dtype="bfloat16",
        device_map="auto",
        max_memory="0:32GiB,1:44GiB",
        trust_remote_code=True,
    )

    assert cmd[:4] == ["uv", "run", "--extra", "training-grpo"]
    assert cmd[4] == "python"
    assert "rl/train_true_grpo_trl.py" in cmd[5]
    assert cmd[cmd.index("--model") + 1] == PAPER_TTRL_BASE_MODEL
    assert cmd[cmd.index("--split-file") + 1] == str(split_file)
    assert cmd[cmd.index("--max-steps") + 1] == "6"
    assert cmd[cmd.index("--num-generations") + 1] == "4"
    assert "--load-in-4bit" in cmd
    assert cmd[cmd.index("--torch-dtype") + 1] == "bfloat16"
    assert cmd[cmd.index("--device-map") + 1] == "auto"
    assert cmd[cmd.index("--max-memory") + 1] == "0:32GiB,1:44GiB"
    assert "--trust-remote-code" in cmd


def test_build_peft_sft_cmd_uses_sft_trainer_and_split(tmp_path: Path) -> None:
    split_file = tmp_path / "train.txt"
    split_file.write_text("fourbar_path_t001\n")
    cmd = build_peft_sft_cmd(
        runner="uv run --extra training-grpo python",
        run_dir=tmp_path / "family_sft",
        base_model=PAPER_TTRL_BASE_MODEL,
        tasks_root="tasks",
        split_file=split_file,
        max_steps=6,
        max_context_tokens=8192,
        lora_rank=16,
        lr=1.0e-4,
        load_in_4bit=True,
        torch_dtype="bfloat16",
        device_map="auto",
        trust_remote_code=True,
    )

    assert cmd[:4] == ["uv", "run", "--extra", "training-grpo"]
    assert cmd[4] == "python"
    assert "rl/train_sft_peft.py" in cmd[5]
    assert cmd[cmd.index("--model") + 1] == PAPER_TTRL_BASE_MODEL
    assert cmd[cmd.index("--split-file") + 1] == str(split_file)
    assert cmd[cmd.index("--max-steps") + 1] == "6"
    assert cmd[cmd.index("--max-seq-length") + 1] == "8192"
    assert "--load-in-4bit" in cmd


def test_load_or_run_eval_summary_reuses_existing_summary(tmp_path: Path) -> None:
    report_dir = tmp_path / "eval_frozen_model"
    report_dir.mkdir()
    (report_dir / "smoke_summary.json").write_text(
        json.dumps({"n_samples": 1, "tasks": []}) + "\n"
    )

    summary = load_or_run_eval_summary(
        report_dir=report_dir,
        resume_existing=True,
        run_eval=lambda: (_ for _ in ()).throw(AssertionError("should not run")),
    )

    assert summary["n_samples"] == 1


def test_load_or_freeze_family_split_reuses_existing_manifest(
    tmp_path: Path,
) -> None:
    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    split_json = split_dir / "split_manifest.json"
    split = _split()
    split_json.write_text(json.dumps(split) + "\n")
    (split_dir / "train.txt").write_text("fourbar_path_t001\n")
    (split_dir / "test.txt").write_text("slider_crank_stroke_s0001\n")

    loaded = load_or_freeze_family_split(
        split_dir=split_dir,
        split_json=split_json,
        runner_python="/definitely/missing/python",
        tasks_root="tasks",
        seen_families="other_seen",
        unseen_families="other_unseen",
        split_seed=123,
        resume_existing=True,
    )

    assert loaded == split


def test_resolve_family_tasks_root_materializes_paper_tasks(
    tmp_path: Path,
) -> None:
    tasks_root = tmp_path / "tasks"
    source = tasks_root / "cam_follower_contact_stub_s0001"
    source.mkdir(parents=True)
    (source / "task.toml").write_text(
        """[task]
id = "cam_follower_contact_stub_s0001"
family = "cam_follower_contact_stub"
"""
    )
    (source / "eval_config.toml").write_text(
        """[[probes]]
id = "contact"
type = "contact_engagement"
adapter = "fake_contact_oracle"
required_pairs = ["cam:follower"]
"""
    )
    (source / "prompt.md").write_text("# Cam follower\n")

    out = resolve_family_tasks_root(
        tasks_root=str(tasks_root),
        out_dir=tmp_path / "run",
        seen_families="",
        unseen_families="cam_follower",
        materialize_paper_tasks=True,
        paper_tasks_root=None,
        paper_task_suffix="paper_verifier",
        paper_task_overwrite=False,
    )

    clone = out / "cam_follower_contact_stub_s0001_paper_verifier"
    assert clone.is_dir()
    assert "procedural_cycloidal_fallback = false" in (
        clone / "task.toml"
    ).read_text()
    eval_text = (clone / "eval_config.toml").read_text()
    assert 'adapter = "chrono_contact"' in eval_text
    assert 'type = "trusted_asset_preflight"' in eval_text


def test_fully_cached_resume_can_skip_worldlines_init(tmp_path: Path) -> None:
    for name in [
        "eval_frozen_model",
        "eval_verifier_gated",
        "eval_no_update_search",
        "eval_llm_evolve_no_update",
        "eval_sft_model",
        "eval_mechanical_evolve_ttrl",
    ]:
        path = tmp_path / name
        path.mkdir()
        (path / "smoke_summary.json").write_text("{}\n")
    for name in ("family_sft", "family_rlvr"):
        path = tmp_path / name
        path.mkdir()
        (path / "sampler_manifest.json").write_text("{}\n")

    assert can_skip_worldlines_init_for_resume(
        out_dir=tmp_path,
        resume_existing=True,
    ) is True
    (tmp_path / "eval_verifier_gated" / "smoke_summary.json").unlink()
    assert can_skip_worldlines_init_for_resume(
        out_dir=tmp_path,
        resume_existing=True,
    ) is False


def test_train_peft_sft_model_reuses_existing_manifest(tmp_path: Path) -> None:
    output_dir = tmp_path / "family_sft"
    output_dir.mkdir()
    (output_dir / "sampler_manifest.json").write_text(json.dumps({
        "path": "/tmp/existing-adapter",
        "step": 2,
        "adapter_updates": 2,
        "trained_tokens": 128,
        "rl_trained_tokens": 0,
        "n_rl_datums": 0,
    }) + "\n")

    model = train_peft_sft_model(
        run_dir=tmp_path / "train_sft",
        run_name="family_sft",
        base_model=PAPER_TTRL_BASE_MODEL,
        runner="/definitely/missing/python",
        tasks_root="tasks",
        split_file=tmp_path / "train.txt",
        max_steps=2,
        max_context_tokens=8192,
        train_timeout_s=1.0,
        lora_rank=16,
        lr=1.0e-4,
        load_in_4bit=False,
        torch_dtype=None,
        device_map=None,
        trust_remote_code=False,
        allow_zero_update_models=False,
        resume_existing=True,
    )

    assert model["path"] == "/tmp/existing-adapter"
    assert model["adapter_updates"] == 2


def test_family_run_uses_worldlines_only_for_worldlines_paths() -> None:
    assert family_run_uses_worldlines(
        eval_rollout_backend="sglang_chat",
        sft_trainer="peft_sft",
        ttrl_trainer="trl_grpo",
    ) is False

    assert family_run_uses_worldlines(
        eval_rollout_backend="worldlines_sampling",
        sft_trainer="peft_sft",
        ttrl_trainer="trl_grpo",
    ) is True

    assert family_run_uses_worldlines(
        eval_rollout_backend="sglang_chat",
        sft_trainer="worldlines_ce",
        ttrl_trainer="trl_grpo",
    ) is True

    assert family_run_uses_worldlines(
        eval_rollout_backend="sglang_chat",
        sft_trainer="peft_sft",
        ttrl_trainer="worldlines_ce",
    ) is True


def test_matched_eval_samples_per_task_reduces_multi_turn_samples() -> None:
    assert matched_eval_samples_per_task(
        base_samples_per_task=4,
        max_turns=2,
        match_planned_verifier_budget=True,
        method_name="mechanical_evolve_ttrl",
    ) == 2

    assert matched_eval_samples_per_task(
        base_samples_per_task=4,
        max_turns=1,
        match_planned_verifier_budget=True,
        method_name="frozen_model",
    ) == 4

    assert matched_eval_samples_per_task(
        base_samples_per_task=4,
        max_turns=2,
        match_planned_verifier_budget=False,
        method_name="debug",
    ) == 4


def test_matched_eval_samples_per_task_rejects_non_divisible_budget() -> None:
    try:
        matched_eval_samples_per_task(
            base_samples_per_task=5,
            max_turns=2,
            match_planned_verifier_budget=True,
            method_name="llm_evolve_no_update",
        )
    except SystemExit as exc:
        assert "not divisible" in str(exc)
    else:
        raise AssertionError("expected SystemExit")


def test_write_results_claim_requires_all_required_methods(tmp_path: Path) -> None:
    rows = [
        {
            "method": "frozen_model",
            "candidate_count": 2,
            "verifier_calls": 2,
            "chrono_audits": 0,
            "verified_pass_rate": 0.5,
            "strict_score_pass_rate": 0.5,
            "best_verified_reward": 1.0,
            "n_tasks": 1,
            "adapter_updates": 0,
            "trained_tokens": 0,
            "rl_trained_tokens": 0,
            "n_rl_datums": 0,
            "families": [],
        },
        {
            "method": "mechanical_evolve_ttrl",
            "candidate_count": 2,
            "verifier_calls": 2,
            "chrono_audits": 0,
            "verified_pass_rate": 1.0,
            "strict_score_pass_rate": 1.0,
            "best_verified_reward": 1.0,
            "n_tasks": 1,
            "adapter_updates": 2,
            "trained_tokens": 512,
            "rl_trained_tokens": 256,
            "n_rl_datums": 1,
            "families": [],
        },
    ]

    write_results(tmp_path, _split(), rows)
    payload = json.loads(
        (tmp_path / "family_generalization_results.json").read_text()
    )

    assert payload["mechanical_evolve_ttrl_beats_all"] is True
    assert payload["method_audit"]["required_methods_present"] is False
    assert payload["claim_status"] == "does_not_yet_support_family_heldout_transfer"
    assert "missing_required_methods" in payload["claim_blockers"]


def test_write_results_claim_requires_full_eval_coverage(tmp_path: Path) -> None:
    rows = [
        {
            "method": method,
            "candidate_count": 1,
            "verifier_calls": 1,
            "chrono_audits": 0,
            "n_tasks": 1 if method != REQUIRED_METHODS[0] else 0,
            "verified_pass_rate": 0.5,
            "strict_score_pass_rate": 0.5,
            "best_verified_reward": 1.0,
            "adapter_updates": 0,
            "trained_tokens": 0,
            "rl_trained_tokens": 0,
            "n_rl_datums": 0,
            "families": [],
        }
        for method in REQUIRED_METHODS
    ]
    for row in rows:
        if row["method"] == "mechanical_evolve_ttrl":
            row.update({
                "verified_pass_rate": 1.0,
                "adapter_updates": 2,
                "trained_tokens": 512,
                "rl_trained_tokens": 256,
                "n_rl_datums": 1,
            })

    write_results(tmp_path, _split(), rows)
    payload = json.loads(
        (tmp_path / "family_generalization_results.json").read_text()
    )

    assert payload["eval_coverage_audit"]["complete_required_eval_coverage"] is False
    assert payload["claim_status"] == "does_not_yet_support_family_heldout_transfer"
    assert "incomplete_eval_task_coverage" in payload["claim_blockers"]
