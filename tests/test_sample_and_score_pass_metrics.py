from __future__ import annotations

from types import SimpleNamespace

from rl.mech_bench_reward import (
    RewardResult,
    extract_no_procedural_fallback,
    extract_physical_metrics,
)
from rl.sample_and_score import (
    SampleOutcome,
    _reward_from_rollout_final,
    _rollout_verifier_calls,
)
from rl.verifier_audits import cad_audit_count, chrono_audit_count


def _outcome(reward: RewardResult) -> SampleOutcome:
    return SampleOutcome(
        task_id="cam_follower_contact_stub_s0001",
        family="cam_follower",
        tier="contact_dynamics",
        sample_idx=0,
        sample_duration_s=0.0,
        sample_tokens_in=0,
        sample_tokens_out=0,
        completion_chars=0,
        reward=reward,
        pass_threshold=1.0,
    )


def test_verifier_valid_pass_can_have_subunit_continuous_score() -> None:
    outcome = _outcome(
        RewardResult(
            score=0.8,
            verified_score=0.8,
            hard_gate_passed=True,
            evaluation_valid=True,
            failure_codes=[],
        )
    )

    assert not outcome.passed()
    assert outcome.verifier_valid_passed()
    assert outcome.to_dict()["strict_passed"] is False
    assert outcome.to_dict()["verifier_valid_passed"] is True


def test_verifier_valid_pass_rejects_failures_even_with_score() -> None:
    outcome = _outcome(
        RewardResult(
            score=0.8,
            verified_score=0.8,
            hard_gate_passed=True,
            evaluation_valid=True,
            failure_codes=["contact_missing"],
        )
    )

    assert not outcome.verifier_valid_passed()


def test_sample_outcome_serializes_verifier_calls() -> None:
    outcome = _outcome(
        RewardResult(
            score=1.0,
            verified_score=1.0,
            hard_gate_passed=True,
            evaluation_valid=True,
            failure_codes=[],
        )
    )
    outcome.verifier_calls = 3

    assert outcome.to_dict()["verifier_calls"] == 3


def test_sample_outcome_marks_repair_success_after_feedback_turns() -> None:
    outcome = _outcome(
        RewardResult(
            score=1.0,
            verified_score=1.0,
            hard_gate_passed=True,
            evaluation_valid=True,
            failure_codes=[],
        )
    )
    outcome.verifier_calls = 2

    assert outcome.repair_attempted() is True
    assert outcome.repair_succeeded() is True
    assert outcome.to_dict()["repair_attempted"] is True
    assert outcome.to_dict()["repair_succeeded"] is True


def test_sample_outcome_serializes_chrono_audits() -> None:
    outcome = _outcome(
        RewardResult(
            score=1.0,
            verified_score=1.0,
            hard_gate_passed=True,
            evaluation_valid=True,
            failure_codes=[],
            chrono_audits=1,
        )
    )
    outcome.chrono_audits = 1

    assert outcome.to_dict()["chrono_audits"] == 1


def test_sample_outcome_serializes_cad_audits() -> None:
    outcome = _outcome(
        RewardResult(
            score=1.0,
            verified_score=1.0,
            hard_gate_passed=True,
            evaluation_valid=True,
            failure_codes=[],
            cad_audits=1,
        )
    )
    outcome.cad_audits = 1

    assert outcome.to_dict()["cad_audits"] == 1


def test_rollout_verifier_calls_excludes_sampler_errors() -> None:
    rollout = SimpleNamespace(turns=[
        SimpleNamespace(failure_codes=[]),
        SimpleNamespace(failure_codes=["wrong_ratio"]),
        SimpleNamespace(failure_codes=["sampler_error"]),
    ])

    assert _rollout_verifier_calls(rollout) == 2


def test_reward_from_rollout_final_reuses_final_turn_score() -> None:
    rollout = SimpleNamespace(turns=[
        SimpleNamespace(
            dense_pct=75.0,
            score=50.0,
            passed=True,
            evaluation_valid=True,
            failure_codes=[],
            feedback=[],
            parsed_ok=True,
            cad_audits=1,
            chrono_audits=1,
            no_procedural_fallback=True,
        )
    ])

    reward = _reward_from_rollout_final(rollout)

    assert reward is not None
    assert reward.score == 0.75
    assert reward.verified_score == 0.5
    assert reward.hard_gate_passed is True
    assert reward.evaluation_valid is True
    assert reward.design_py_extracted is True
    assert reward.cad_audits == 1
    assert reward.chrono_audits == 1
    assert reward.no_procedural_fallback is True


def test_reward_from_rollout_final_ignores_sampler_error() -> None:
    rollout = SimpleNamespace(turns=[
        SimpleNamespace(failure_codes=["sampler_error"])
    ])

    assert _reward_from_rollout_final(rollout) is None


def test_chrono_audit_count_requires_real_chrono_attempt() -> None:
    assert chrono_audit_count({
        "timings": {"adapter.chrono_contact": 1.2},
        "feedback": [],
    }) == 1

    assert chrono_audit_count({
        "timings": {"adapter": {"chrono_contact": 1.2}},
        "feedback": [],
    }) == 1

    assert chrono_audit_count({
        "timings": {"adapter.fake_contact_oracle": 1.2},
        "feedback": [],
    }) == 0


def test_cad_audit_count_requires_trusted_cad_evidence() -> None:
    assert cad_audit_count({
        "metrics": {
            "trusted_asset_preflight.trusted_mass_properties_recomputed": 1.0,
        },
    }) == 1

    assert cad_audit_count({
        "metrics": {
            "trusted_asset_preflight.parts_with_trusted_mass_properties": 2.0,
        },
    }) == 1

    assert cad_audit_count({
        "metrics": {
            "trusted_asset_preflight": {
                "trusted_mass_properties_recomputed": 1.0,
            },
        },
    }) == 1

    assert cad_audit_count({
        "metrics": {
            "parts_with_trusted_mass_properties": 1.0,
        },
    }) == 1

    assert cad_audit_count({
        "timings": {"load_submission": 0.2},
        "metrics": {},
    }) == 0


def test_extract_physical_metrics_uses_canonical_aliases() -> None:
    metrics = extract_physical_metrics({
        "metrics": {
            "ratio.observed": 9.0,
            "ratio.error_pct": 1.5,
            "out_omega_med": 0.8,
            "max_penetration_mm": 0.2,
            "contact_force_rms_N": 12.0,
            "power_balance_error_pct": 4.0,
            "torque_ripple_pct": 8.0,
        }
    })

    assert metrics["ratio_observed"] == 9.0
    assert metrics["ratio_error_pct"] == 1.5
    assert metrics["out_omega_med"] == 0.8
    assert metrics["max_penetration_mm"] == 0.2
    assert metrics["contact_force_rms_N"] == 12.0

    prefixed = extract_physical_metrics({
        "metrics": {
            "chrono_contact.ratio_error_pct": 2.5,
            "chrono_contact.out_omega_med": 1.2,
            "chrono_contact.max_penetration_mm": 0.1,
        }
    })
    assert prefixed["ratio_error_pct"] == 2.5
    assert prefixed["out_omega_med"] == 1.2
    assert prefixed["max_penetration_mm"] == 0.1

    nested = extract_physical_metrics({
        "scalar_metrics": {
            "chrono_contact": {
                "contact_force_rms_N": 8.0,
            }
        }
    })
    assert nested["contact_force_rms_N"] == 8.0

    report_level = extract_physical_metrics({
        "metrics": {
            "contact.contact.cam:follower.rms_N": 16836.5,
            "swept_collision.max_penetration_mm": 0.025,
            "lockup.output_motion_rad": 50.0,
        }
    })
    assert report_level["contact_force_rms_N"] == 16836.5
    assert report_level["max_penetration_mm"] == 0.025
    assert report_level["out_omega_med"] == 50.0


def test_extract_no_procedural_fallback_uses_canonical_aliases() -> None:
    assert extract_no_procedural_fallback({
        "metrics": {"procedural_cycloidal_fallback": False}
    }) is True
    assert extract_no_procedural_fallback({
        "metrics": {"chrono.procedural_cycloidal_fallback": True}
    }) is False
    assert extract_no_procedural_fallback({
        "procedural_cycloidal_fallback": False,
    }) is True
    assert extract_no_procedural_fallback({
        "metrics": {"chrono_contact": {"procedural_cycloidal_fallback": False}}
    }) is True
    assert extract_no_procedural_fallback({
        "timings": {"adapter.chrono_contact": 1.2},
    }) is True
    assert extract_no_procedural_fallback({"metrics": {}}) is None

    assert chrono_audit_count({
        "timings": {"adapter.chrono_contact": 0.01},
        "feedback": [
            {
                "code": "capability_unavailable",
                "where": "adapter.chrono_contact",
                "message": "Adapter 'chrono_contact' unavailable.",
            }
        ],
    }) == 0
