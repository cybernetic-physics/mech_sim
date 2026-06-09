from __future__ import annotations

import json
from pathlib import Path

from scripts.analyze_mechanism_repair_results import (
    analyze_rows,
    build_benchmark_readiness,
    build_expected_coverage,
    build_claim_audit,
    normalize_rows,
)
from scripts.prepare_mechanism_repair_benchmark import REQUIRED_METHODS


def _support_rows(tmp_path: Path | None = None) -> list[dict]:
    rows = []
    families = ["cycloidal", "lead_screw", "planetary", "slider_crank"]
    for family in families:
        for seed in [1, 2]:
            task_id = f"{family}_task_{seed}"
            for method in REQUIRED_METHODS:
                if method == "mechanical_evolve_ttrl":
                    success = True
                    reward = 1.0
                elif method == "llm_evolve_no_update":
                    success = False
                    reward = 0.25
                else:
                    success = False
                    reward = 0.1
                row = {
                    "split": "A",
                    "family": family,
                    "task_id": task_id,
                    "seed": seed,
                    "method": method,
                    "verified_repair_success_at_32": success,
                    "best_verified_reward_at_32": reward,
                    "verifier_calls": 32,
                    "cad_audits": 0,
                    "chrono_audits": 0,
                    "failure_codes": "" if success else "wrong_ratio",
                    "adapter_updates": 0,
                    "trained_tokens": 0,
                    "rl_trained_tokens": 0,
                    "n_rl_datums": 0,
                    "first_valid_verifier_call": 4 if success else "",
                    "strict_score_pass_rate": 1.0 if success else 0.0,
                    "wrong_mobility_rate": 0.0 if success else 0.5,
                    "missing_port_rate": 0.0,
                    "ungrounded_port_rate": 0.0,
                    "invalid_topology_rate": 0.0 if success else 0.25,
                    "invalid_artifact_rate": 0.0,
                    "cad_pass_rate": 1.0,
                    "chrono_real_geometry_rate": 0.0,
                    "no_procedural_fallback_rate": 1.0,
                    "best_ratio_error_pct": 0.0 if success else 12.0,
                }
                if tmp_path is not None:
                    evidence_dir = tmp_path / method / task_id / str(seed)
                    evidence_dir.mkdir(parents=True, exist_ok=True)
                    raw = evidence_dir / "completion.txt"
                    verifier = evidence_dir / "verifier.json"
                    raw.write_text("module design\n")
                    verifier.write_text('{"verified_score": 1.0}\n')
                    row["raw_completion_paths"] = [str(raw)] * 32
                    row["verifier_output_paths"] = [str(verifier)] * 32
                    if method == "mechanical_evolve_ttrl":
                        reward_log = evidence_dir / "reward_log.jsonl"
                        reward_log.write_text('{"verified_score": 1.0}\n')
                        adapter_dir = evidence_dir / "final_adapter"
                        adapter_dir.mkdir()
                        (adapter_dir / "adapter_config.json").write_text(
                            '{"r": 16}\n'
                        )
                        row.update({
                            "trace_path": str(reward_log),
                            "adapter_path": str(adapter_dir),
                            "adapter_updates": 32,
                            "trained_tokens": 128,
                            "rl_trained_tokens": 128,
                            "n_rl_datums": 32,
                        })
                rows.append(row)
    return rows


def test_mechanism_repair_analysis_supports_primary_hypothesis(
    tmp_path: Path,
) -> None:
    rows = normalize_rows(_support_rows(tmp_path))
    stats = analyze_rows(rows, bootstrap_samples=200, seed=7)
    audit = build_claim_audit(stats)

    assert stats["required_methods_present"] is True
    assert stats["n_paired_cells"] == 8
    assert stats["evidence_audit"]["evidence_complete"] is True
    assert stats["learning_audit"]["ttrl_learning_evidence_complete"] is True
    assert stats["primary_comparison"]["success_delta_pct"] == 100.0
    assert stats["primary_comparison"]["success_sign_test_p_one_sided"] <= 0.05
    assert audit["claim_status"] == "supports_primary_hypothesis"
    assert audit["blockers"] == []


def test_mechanism_repair_analysis_rejects_unmatched_budget(
    tmp_path: Path,
) -> None:
    raw_rows = _support_rows(tmp_path)
    for row in raw_rows:
        if row["method"] == "mechanical_evolve_ttrl":
            row["verifier_calls"] = 64
    rows = normalize_rows(raw_rows)
    stats = analyze_rows(rows, bootstrap_samples=200, seed=7)
    audit = build_claim_audit(stats)

    assert stats["budget_audit"]["budget_matched"] is False
    assert audit["claim_status"] == "does_not_support_primary_hypothesis"
    assert any("budgets are not matched" in item for item in audit["blockers"])


def test_mechanism_repair_analysis_writes_rejectable_json_shape(
    tmp_path: Path,
) -> None:
    results = tmp_path / "results.json"
    results.write_text(json.dumps({"rows": _support_rows(tmp_path / "evidence")}))
    loaded = json.loads(results.read_text())["rows"]

    rows = normalize_rows(loaded)
    stats = analyze_rows(rows, bootstrap_samples=200, seed=7)

    assert set(stats["method_summary"]) == set(REQUIRED_METHODS)
    assert stats["reward_beats_all_required_baselines"] is True


def test_mechanism_repair_analysis_summarizes_secondary_metrics(
    tmp_path: Path,
) -> None:
    rows = normalize_rows(_support_rows(tmp_path))
    stats = analyze_rows(rows, bootstrap_samples=200, seed=7)

    ttrl_metrics = stats["method_summary"]["mechanical_evolve_ttrl"][
        "secondary_metrics"
    ]
    assert ttrl_metrics["first_valid_verifier_call"]["n_present"] == 8
    assert ttrl_metrics["strict_score_pass_rate"]["mean"] == 1.0
    assert ttrl_metrics["adapter_updates"]["mean"] == 32
    assert stats["family_method_summary"]
    family_row = next(
        row for row in stats["family_method_summary"]
        if row["method"] == "llm_evolve_no_update"
        and row["family"] == "cycloidal"
    )
    assert family_row["secondary_metrics"]["wrong_mobility_rate"]["mean"] == 0.5


def test_mechanism_repair_analysis_rejects_incomplete_expected_coverage(
    tmp_path: Path,
) -> None:
    benchmark_dir = tmp_path / "bench"
    benchmark_dir.mkdir()
    (benchmark_dir / "method_manifest.json").write_text(json.dumps({
        "required_methods": list(REQUIRED_METHODS),
        "eval_seeds": [1],
    }))
    for split in ["A", "B"]:
        (benchmark_dir / f"split_manifest_{split}.json").write_text(
            json.dumps({"splits": {"test": [f"{split}_heldout"]}})
        )

    rows = [
        {
            **row,
            "split": "A",
            "task_id": "A_heldout",
            "family": "cycloidal",
            "seed": 1,
        }
        for row in _support_rows(tmp_path / "evidence")
        if row["seed"] == 1 and row["task_id"].endswith("_1")
    ]
    rows = normalize_rows(rows)
    expected = build_expected_coverage(benchmark_dir)
    stats = analyze_rows(
        rows,
        bootstrap_samples=200,
        seed=7,
        expected_coverage=expected,
    )
    audit = build_claim_audit(stats)

    assert stats["coverage_audit"]["enforced"] is True
    assert stats["coverage_audit"]["complete_coverage"] is False
    assert stats["coverage_audit"]["n_missing_cells"] > 0
    assert audit["claim_status"] == "does_not_support_primary_hypothesis"
    assert any("incomplete expected" in item for item in audit["blockers"])


def test_mechanism_repair_analysis_rejects_missing_evidence() -> None:
    rows = normalize_rows(_support_rows())
    stats = analyze_rows(rows, bootstrap_samples=200, seed=7)
    audit = build_claim_audit(stats)

    assert stats["evidence_audit"]["evidence_complete"] is False
    assert audit["claim_status"] == "does_not_support_primary_hypothesis"
    assert any("evidence is incomplete" in item for item in audit["blockers"])


def test_mechanism_repair_analysis_rejects_missing_ttrl_learning(
    tmp_path: Path,
) -> None:
    raw_rows = _support_rows(tmp_path)
    for row in raw_rows:
        if row["method"] == "mechanical_evolve_ttrl":
            row["adapter_updates"] = 0
            row["rl_trained_tokens"] = 0
    rows = normalize_rows(raw_rows)
    stats = analyze_rows(rows, bootstrap_samples=200, seed=7)
    audit = build_claim_audit(stats)

    assert stats["learning_audit"]["ttrl_learning_evidence_complete"] is False
    assert audit["claim_status"] == "does_not_support_primary_hypothesis"
    assert any("TTRL learning evidence is incomplete" in item for item in audit["blockers"])


def _write_benchmark_readiness_fixture(
    root: Path,
    *,
    ready: bool = True,
) -> None:
    root.mkdir()
    families = [
        "belt",
        "chain",
        "rack_pinion",
        "lead_screw",
        "planetary",
        "fourbar",
        "slider_crank",
        "cycloidal",
    ]
    audit_tasks = []
    for family in families:
        for idx in range(5):
            audit_tasks.append({
                "task_id": f"{family}_{idx}",
                "canonical_family": family,
                "constraint_classes": (
                    ["topology_or_mobility"]
                    if not ready and family == "belt" and idx == 0
                    else ["topology_or_mobility", "functional_behavior"]
                ),
                "has_negative_control": ready or family != "chain" or idx != 0,
                "negative_control_count": 1 if ready or family != "chain" else 0,
                "uses_fake_contact_oracle": False,
                "validation": {
                    "reference_passed": True,
                    "reference_evaluation_valid": True,
                    "reference_hard_gate_passed": True,
                },
                "verifier_level": 1,
            })
    (root / "benchmark_manifest.json").write_text(json.dumps({
        "experiment_ready": ready,
        "primary_families": families,
        "task_count": 40,
        "audit": {
            "passes": ready,
            "blockers": [] if ready else ["synthetic blocker"],
            "task_count": 40,
            "family_counts": {family: 5 for family in families},
            "primary_families": families,
            "tasks": audit_tasks,
        },
    }))
    (root / "verifier_manifest.json").write_text(json.dumps({
        "main_claim_allows_fake_oracle": False,
        "fake_oracle_tasks": [],
        "verifier_levels": {"1": 40},
    }))
    split_specs = {
        "A": {
            "seen_families": ["belt", "chain", "rack_pinion", "fourbar"],
            "unseen_families": [
                "planetary",
                "lead_screw",
                "slider_crank",
                "cycloidal",
            ],
        },
        "B": {
            "seen_families": [
                "planetary",
                "lead_screw",
                "fourbar",
                "slider_crank",
            ],
            "unseen_families": ["belt", "chain", "rack_pinion", "cycloidal"],
        },
    }
    for split, spec in split_specs.items():
        (root / f"split_manifest_{split}.json").write_text(json.dumps({
            **spec,
            "splits": {
                "train": [f"{split}_train_{idx}" for idx in range(16)],
                "test": [f"{split}_test_{idx}" for idx in range(20)],
            },
        }))


def test_benchmark_readiness_accepts_contract_fixture(tmp_path: Path) -> None:
    bench = tmp_path / "bench"
    _write_benchmark_readiness_fixture(bench, ready=True)

    readiness = build_benchmark_readiness(bench)

    assert readiness["benchmark_ready"] is True
    assert readiness["blockers"] == []


def test_claim_audit_rejects_bad_benchmark_readiness(tmp_path: Path) -> None:
    bench = tmp_path / "bench"
    _write_benchmark_readiness_fixture(bench, ready=False)
    rows = normalize_rows(_support_rows(tmp_path / "evidence"))
    stats = analyze_rows(
        rows,
        bootstrap_samples=200,
        seed=7,
        benchmark_readiness=build_benchmark_readiness(bench),
    )
    audit = build_claim_audit(stats)

    assert stats["benchmark_readiness_audit"]["benchmark_ready"] is False
    assert audit["claim_status"] == "does_not_support_primary_hypothesis"
    assert any("benchmark/split/verifier readiness failed" in item for item in audit["blockers"])
