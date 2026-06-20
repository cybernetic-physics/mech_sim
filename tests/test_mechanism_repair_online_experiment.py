from __future__ import annotations

import json
import subprocess
from argparse import Namespace
from pathlib import Path

import pytest

from scripts import run_mechanism_repair_online_experiment as online
from scripts.run_mechanism_repair_online_experiment import (
    EvalMethod,
    build_plan,
    build_runtime_preflight,
    expensive_budget_caps_for_ttrl,
    order_methods_for_budget_dependencies,
    require_learning_manifest,
    reset_non_resume_outputs,
    reward_log_exceeds_expensive_caps,
    row_from_ttrl_reward_log,
    run_or_load_ttrl_cell,
    run_or_load_eval_summary,
    run_or_load_sft,
    rows_from_sample_summary,
    sample_rank,
    ttrl_steps_per_generation_for_budget,
)


def test_build_plan_records_ttrl_reward_channel(monkeypatch, tmp_path: Path) -> None:
    split_dir = tmp_path / "splits_A"
    split_dir.mkdir()
    (split_dir / "test.txt").write_text("task_a\ntask_b\n")
    monkeypatch.setattr(
        online,
        "build_expected_coverage",
        lambda _benchmark_dir: {"expected_cells": ["cell"]},
    )

    plan = build_plan(
        benchmark_dir=tmp_path,
        out_dir=tmp_path,
        splits=["A"],
        seeds=[20260607],
        methods=["mechanical_evolve_ttrl"],
        budget=32,
        feedback_turns=4,
        audit_retries=0,
        limit_tasks=1,
        init_online_from_sft=True,
        ttrl_steps=32,
        ttrl_generations=4,
        ttrl_steps_per_generation=4,
        ttrl_reward_channel="artifact_progress",
    )

    assert plan["ttrl_reward_channel"] == "artifact_progress"
    assert plan["ttrl_rollout_evaluations_per_cell"] == 32
    assert plan["ttrl_optimizer_steps"] == 32
    assert plan["ttrl_steps_per_generation"] == 4
    assert plan["split_tasks"] == {"A": ["task_a"]}
    assert plan["planned_cells"] == 1


def test_ttrl_steps_per_generation_for_budget_enforces_matched_rollouts() -> None:
    assert ttrl_steps_per_generation_for_budget(budget=32, num_generations=4) == 4
    with pytest.raises(SystemExit, match="must divide evenly"):
        ttrl_steps_per_generation_for_budget(budget=30, num_generations=4)


def test_runtime_preflight_rejects_missing_sglang_sampler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        online,
        "probe_openai_chat_server",
        lambda _base_url, *, timeout_s: (False, "connection refused"),
    )
    args = Namespace(
        rollout_backend="sglang_chat",
        sglang_base_url="http://127.0.0.1:30000",
        preflight_timeout_s=0.1,
        ttrl_rollout_openai=False,
    )

    report = build_runtime_preflight(
        args=args,
        requested_methods=["frozen_model"],
        needs_sft=False,
        method_contract={"is_physics": False},
    )

    assert report["ready"] is False
    assert "sglang_chat server unavailable" in report["blockers"][0]
    assert report["checks"]["sglang_chat"]["ok"] is False


def test_runtime_preflight_does_not_probe_sampler_when_not_needed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_probe(_base_url: str, *, timeout_s: float) -> tuple[bool, str]:
        raise AssertionError("sampler probe should not run")

    monkeypatch.setattr(online, "probe_openai_chat_server", fail_probe)
    args = Namespace(
        rollout_backend="sglang_chat",
        sglang_base_url="http://127.0.0.1:30000",
        preflight_timeout_s=0.1,
        ttrl_rollout_openai=False,
    )

    report = build_runtime_preflight(
        args=args,
        requested_methods=[],
        needs_sft=False,
        method_contract={"is_physics": False},
    )

    assert report["ready"] is True
    assert report["checks"]["sampler"] == {"required": False, "ok": True}


def test_runtime_preflight_accepts_local_transformers_sampler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_find_spec(name: str):
        return object() if name in {"torch", "transformers"} else None

    monkeypatch.setattr(online.importlib.util, "find_spec", fake_find_spec)
    args = Namespace(
        rollout_backend="transformers_local",
        local_device="cpu",
        local_torch_dtype="auto",
        ttrl_rollout_openai=False,
    )

    report = build_runtime_preflight(
        args=args,
        requested_methods=["frozen_model"],
        needs_sft=False,
        method_contract={"is_physics": False},
    )

    assert report["ready"] is True
    assert report["checks"]["transformers_local"]["ok"] is True
    assert report["checks"]["transformers_local"]["device"] == "cpu"


def test_build_plan_filters_to_shard_cells_and_normalizes_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    split_dir = tmp_path / "splits_A"
    split_dir.mkdir()
    task_a = tmp_path / "tasks" / "task_a"
    task_b = tmp_path / "tasks" / "task_b"
    task_a.mkdir(parents=True)
    task_b.mkdir()
    split_dir.joinpath("test.txt").write_text(f"{task_a}\n{task_b}\n")
    monkeypatch.setattr(
        online,
        "build_expected_coverage",
        lambda _benchmark_dir: {"expected_cells": ["cell"] * 8},
    )
    shard_path = tmp_path / "shard_0000.json"
    shard_cells = [
        {
            "split": "A",
            "task_id": "task_b",
            "seed": 20260610,
            "method": "mechanical_evolve_ttrl_tool_verified",
            "budget": 32,
            "_shard_file": str(shard_path),
        }
    ]

    plan = build_plan(
        benchmark_dir=tmp_path,
        out_dir=tmp_path,
        splits=["A"],
        seeds=[20260610, 20260611],
        methods=["frozen_model", "mechanical_evolve_ttrl_tool_verified"],
        budget=32,
        feedback_turns=4,
        audit_retries=0,
        limit_tasks=0,
        init_online_from_sft=True,
        ttrl_steps=32,
        ttrl_generations=4,
        ttrl_steps_per_generation=4,
        ttrl_reward_channel="artifact_progress",
        shard_cells=shard_cells,
    )

    assert plan["split_tasks"] == {"A": ["task_b"]}
    assert plan["planned_cells"] == 1
    assert plan["sharded_execution"] is True
    assert plan["cell_shard_file"] == str(shard_path.resolve())


def test_split_file_writers_preserve_variant_paths(tmp_path: Path) -> None:
    split_dir = tmp_path / "splits_hidden_perturbation"
    split_dir.mkdir()
    variant_a = tmp_path / "variants" / "hidden_perturbation" / "tasks" / "task_a"
    variant_b = tmp_path / "variants" / "hidden_perturbation" / "tasks" / "task_b"
    variant_a.mkdir(parents=True)
    variant_b.mkdir(parents=True)
    (split_dir / "test.txt").write_text(f"{variant_a}\n{variant_b}\n")

    limited = online.make_eval_split_file(
        split_dir=split_dir,
        run_root=tmp_path / "run",
        split="hidden_perturbation",
        limit=1,
    )
    one = online.write_one_task_split(
        tmp_path / "run",
        "hidden_perturbation",
        20260610,
        "task_a",
        task_entry=str(variant_a),
    )
    subset = online.write_task_subset_split(
        run_root=tmp_path / "run",
        split="hidden_perturbation",
        seed=20260610,
        method="frozen_model",
        task_ids=["task_b"],
        task_entry_by_id={"task_b": str(variant_b)},
    )

    assert limited.read_text().splitlines() == [str(variant_a)]
    assert one.read_text().splitlines() == [str(variant_a)]
    assert subset.read_text().splitlines() == [str(variant_b)]


def test_append_new_requested_rows_rejects_stale_summary_cells(
    tmp_path: Path,
) -> None:
    rows_path = tmp_path / "cell_results.jsonl"
    rows: list[dict] = []
    seen_keys: set[tuple[str, str, int, str, int]] = set()
    shard_filter = online.cell_filter_from_shard(
        [
            {
                "split": "A",
                "task_id": "task_keep",
                "seed": 20260610,
                "method": "frozen_model",
                "budget": 32,
            }
        ],
        budget=32,
    )
    keep = {
        "split": "A",
        "task_id": "task_keep",
        "seed": 20260610,
        "method": "frozen_model",
        "budget": 32,
    }
    extra_task = {
        "split": "A",
        "task_id": "task_extra",
        "seed": 20260610,
        "method": "frozen_model",
        "budget": 32,
    }
    wrong_budget = {
        "split": "A",
        "task_id": "task_keep",
        "seed": 20260610,
        "method": "frozen_model",
        "budget": 16,
    }

    counts = online.append_new_requested_rows(
        rows_path=rows_path,
        rows=rows,
        seen_keys=seen_keys,
        new_rows=[keep, extra_task, wrong_budget],
        shard_filter=shard_filter,
        budget=32,
    )

    assert counts == {
        "appended": 1,
        "duplicates": 0,
        "skipped_unrequested": 2,
    }
    assert rows == [keep]
    assert [json.loads(line) for line in rows_path.read_text().splitlines()] == [keep]


def test_append_new_requested_rows_allows_distinct_budgets_without_filter(
    tmp_path: Path,
) -> None:
    rows_path = tmp_path / "cell_results.jsonl"
    rows: list[dict] = []
    seen_keys: set[tuple[str, str, int, str, int]] = set()
    budget_16 = {
        "split": "A",
        "task_id": "task_keep",
        "seed": 20260610,
        "method": "frozen_model",
        "budget": 16,
    }
    budget_32 = {**budget_16, "budget": 32}

    counts = online.append_new_requested_rows(
        rows_path=rows_path,
        rows=rows,
        seen_keys=seen_keys,
        new_rows=[budget_16, budget_32, budget_32],
        shard_filter=None,
        budget=32,
    )

    assert counts == {
        "appended": 2,
        "duplicates": 1,
        "skipped_unrequested": 0,
    }
    assert rows == [budget_16, budget_32]
    assert [json.loads(line) for line in rows_path.read_text().splitlines()] == [
        budget_16,
        budget_32,
    ]


def test_physics_method_specs_and_reward_channels() -> None:
    gated = online.method_spec(
        "verifier_gated_search",
        budget=32,
        feedback_turns=4,
        sft_adapter=None,
        init_online_from_sft=False,
    )
    adaptive = online.method_spec(
        "adaptive_evolution",
        budget=32,
        feedback_turns=4,
        sft_adapter=None,
        init_online_from_sft=False,
    )
    sft = online.method_spec(
        "sft_seen_family",
        budget=32,
        feedback_turns=4,
        sft_adapter="/tmp/adapter",
        init_online_from_sft=False,
    )

    assert gated.samples_per_task == 32
    assert gated.max_turns == 1
    assert adaptive.max_turns == 4
    assert adaptive.temperature == 0.9
    assert sft.adapter_kind == "sft"
    assert (
        online.reward_channel_for_method(
            "mechanical_evolve_ttrl_confidence",
            default="artifact_progress",
        )
        == "verified_score"
    )


def test_physics_contract_defaults_include_anti_shortcut_splits(
    tmp_path: Path,
) -> None:
    (tmp_path / "method_manifest.json").write_text(
        json.dumps({
            "schema": "mechanism_repair_physics.method_manifest.v1",
            "required_methods": [
                "frozen_model",
                "sft_seen_family",
                "llm_evolve_no_update",
                "verifier_gated_search",
                "adaptive_evolution",
                "mechanical_evolve_ttrl",
                "mechanical_evolve_ttrl_tool_verified",
                "mechanical_evolve_ttrl_confidence",
            ],
            "eval_seeds": [20260610],
            "primary_budget_expensive_verifier_calls": 32,
        })
        + "\n"
    )

    contract = online.load_method_contract(tmp_path)

    assert contract["is_physics"] is True
    assert online.default_splits_for_contract(contract) == [
        "A",
        "B",
        "hidden_perturbation",
        "external_style",
        "isomorphic",
    ]


def test_legacy_contract_defaults_to_family_holdout_splits(
    tmp_path: Path,
) -> None:
    contract = online.load_method_contract(tmp_path)

    assert contract["is_physics"] is False
    assert online.default_splits_for_contract(contract) == ["A", "B"]


def test_validate_physics_benchmark_requires_anti_shortcut_manifests(
    tmp_path: Path,
) -> None:
    (tmp_path / "benchmark_manifest.json").write_text(
        json.dumps({"experiment_ready": True}) + "\n"
    )
    (tmp_path / "method_manifest.json").write_text(
        json.dumps({
            "schema": "mechanism_repair_physics.method_manifest.v1",
            "required_methods": [
                "frozen_model",
                "sft_seen_family",
                "llm_evolve_no_update",
                "verifier_gated_search",
                "adaptive_evolution",
                "mechanical_evolve_ttrl",
                "mechanical_evolve_ttrl_tool_verified",
                "mechanical_evolve_ttrl_confidence",
            ],
        })
        + "\n"
    )
    (tmp_path / "verifier_manifest.json").write_text("{}\n")
    (tmp_path / "split_manifest_A.json").write_text("{}\n")
    (tmp_path / "split_manifest_B.json").write_text("{}\n")
    (tmp_path / "split_manifest_hidden_perturbation.json").write_text("{}\n")
    (tmp_path / "split_manifest_external_style.json").write_text("{}\n")
    (tmp_path / "tasks").mkdir()

    with pytest.raises(SystemExit, match="split_manifest_isomorphic"):
        online.validate_benchmark(tmp_path)


def test_physics_anti_shortcut_splits_reuse_nonempty_sft_source(
    tmp_path: Path,
) -> None:
    for split in ("A", "hidden_perturbation", "external_style", "isomorphic"):
        split_dir = tmp_path / f"splits_{split}"
        split_dir.mkdir()
        (split_dir / "test.txt").write_text("task_eval\n")
        (split_dir / "train.txt").write_text(
            "task_train\n" if split == "A" else ""
        )

    mapping = online.resolve_sft_training_splits(
        benchmark_dir=tmp_path,
        splits=["A", "hidden_perturbation", "external_style", "isomorphic"],
        contract={"is_physics": True},
    )

    assert mapping == {
        "A": "A",
        "hidden_perturbation": "A",
        "external_style": "A",
        "isomorphic": "A",
    }


def test_empty_nonphysics_sft_split_is_rejected(tmp_path: Path) -> None:
    split_dir = tmp_path / "splits_A"
    split_dir.mkdir()
    (split_dir / "train.txt").write_text("")

    with pytest.raises(SystemExit, match="split A has no SFT train rows"):
        online.resolve_sft_training_split(
            benchmark_dir=tmp_path,
            split="A",
            contract={"is_physics": False},
        )


def test_build_plan_records_sft_training_split_mapping(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    split_dir = tmp_path / "splits_hidden_perturbation"
    split_dir.mkdir()
    (split_dir / "test.txt").write_text("task_a\n")
    monkeypatch.setattr(
        online,
        "build_expected_coverage",
        lambda _benchmark_dir: {"expected_cells": ["cell"]},
    )

    plan = build_plan(
        benchmark_dir=tmp_path,
        out_dir=tmp_path,
        splits=["hidden_perturbation"],
        seeds=[20260610],
        methods=["sft_seen_family"],
        budget=32,
        feedback_turns=4,
        audit_retries=0,
        limit_tasks=0,
        init_online_from_sft=True,
        ttrl_steps=32,
        ttrl_generations=4,
        ttrl_steps_per_generation=4,
        ttrl_reward_channel="artifact_progress",
        sft_training_splits={"hidden_perturbation": "A"},
    )

    assert plan["sft_training_splits"] == {"hidden_perturbation": "A"}


def test_method_order_runs_no_update_before_ttrl_for_caps() -> None:
    ordered = order_methods_for_budget_dependencies([
        "mechanical_evolve_ttrl_tool_verified",
        "frozen_model",
        "llm_evolve_no_update",
        "mechanical_evolve_ttrl",
    ])

    assert ordered.index("llm_evolve_no_update") < ordered.index(
        "mechanical_evolve_ttrl"
    )
    assert ordered.index("llm_evolve_no_update") < ordered.index(
        "mechanical_evolve_ttrl_tool_verified"
    )


def test_expensive_budget_caps_use_matching_no_update_row() -> None:
    rows = [
        {
            "split": "A",
            "task_id": "task",
            "seed": 1,
            "budget": 32,
            "method": "llm_evolve_no_update",
            "actual_cad_calls": 3,
            "actual_chrono_calls": 2,
        }
    ]

    assert expensive_budget_caps_for_ttrl(
        rows,
        split="A",
        task_id="task",
        seed=1,
        budget=32,
        required=True,
    ) == (3, 2)
    with pytest.raises(SystemExit, match="cap missing"):
        expensive_budget_caps_for_ttrl(
            rows,
            split="A",
            task_id="other",
            seed=1,
            budget=32,
            required=True,
        )


def test_prune_unusable_physics_resume_rows_drops_missing_evidence(
    tmp_path: Path,
) -> None:
    row = {
        "split": "A",
        "task_id": "level3_task",
        "seed": 20260610,
        "method": "frozen_model",
        "budget": 32,
        "actual_verifier_calls": 32,
        "actual_cad_calls": 0,
        "actual_chrono_calls": 0,
        "verifier_level": 3,
        "raw_completion_paths": [],
        "verifier_output_paths": [],
        "cad_artifact_paths": [],
        "chrono_output_paths": [],
    }

    kept, dropped = online.prune_unusable_resume_rows(
        [row],
        out_dir=tmp_path,
        budget=32,
        verifier_level_by_task={"level3_task": 3},
    )

    assert kept == []
    assert dropped == [
        {
            "key": "A/level3_task/seed20260610/frozen_model/budget32",
            "reasons": [
                "missing_cad_artifacts",
                "missing_chrono_outputs",
                "missing_raw_completions",
                "missing_verifier_outputs",
            ],
        }
    ]


def test_prune_unusable_physics_resume_rows_keeps_complete_row(
    tmp_path: Path,
) -> None:
    for relative in (
        "raw/complete.txt",
        "verifier/complete.json",
        "cad/complete.json",
        "chrono/complete.json",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n")
    row = {
        "split": "A",
        "task_id": "level3_task",
        "seed": 20260610,
        "method": "llm_evolve_no_update",
        "budget": 32,
        "actual_verifier_calls": 32,
        "actual_cad_calls": 1,
        "actual_chrono_calls": 1,
        "verifier_level": 3,
        "raw_completion_paths": ["raw/complete.txt"],
        "verifier_output_paths": ["verifier/complete.json"],
        "cad_artifact_paths": ["cad/complete.json"],
        "chrono_output_paths": ["chrono/complete.json"],
    }

    kept, dropped = online.prune_unusable_resume_rows(
        [row],
        out_dir=tmp_path,
        budget=32,
        verifier_level_by_task={"level3_task": 3},
    )

    assert kept == [row]
    assert dropped == []


def test_write_results_bundle_overwrites_empty_csvs(tmp_path: Path) -> None:
    (tmp_path / "results.csv").write_text("stale\n")
    (tmp_path / "cell_results.csv").write_text("stale\n")

    online.write_results_bundle(tmp_path, [])

    assert (tmp_path / "results.csv").read_text() == ""
    assert (tmp_path / "cell_results.csv").read_text() == ""
    assert json.loads((tmp_path / "results.json").read_text()) == {"rows": []}


def test_reward_log_exceeds_expensive_caps(tmp_path: Path) -> None:
    reward_log = tmp_path / "reward_log.jsonl"
    reward_log.write_text(
        json.dumps({"cad_audits": 1, "chrono_audits": 0}) + "\n"
        + json.dumps({"cad_audits": 1, "chrono_audits": 1}) + "\n"
    )

    assert reward_log_exceeds_expensive_caps(
        reward_log,
        max_cad_audits=1,
        max_chrono_audits=1,
    )
    assert not reward_log_exceeds_expensive_caps(
        reward_log,
        max_cad_audits=2,
        max_chrono_audits=1,
    )


def test_rows_from_sample_summary_uses_total_budget_and_canonical_family(
    tmp_path: Path,
) -> None:
    summary = {
        "max_turns": 4,
        "all_samples": [
            {
                "task_id": "planet_task",
                "family": "planetary_fixed_ring_ratio_analytic",
                "sample_idx": 0,
                "verified_score": 0.2,
                "evaluation_valid": True,
                "hard_gate_passed": False,
                "failure_codes": ["wrong_ratio"],
                "verifier_calls": 16,
                "cad_audits": 0,
                "chrono_audits": 0,
            },
            {
                "task_id": "planet_task",
                "family": "planetary_fixed_ring_ratio_analytic",
                "sample_idx": 1,
                "verified_score": 1.0,
                "evaluation_valid": True,
                "hard_gate_passed": True,
                "failure_codes": [],
                "verifier_calls": 16,
                "cad_audits": 0,
                "chrono_audits": 0,
            },
        ],
    }

    rows = rows_from_sample_summary(
        summary=summary,
        method="llm_evolve_no_update",
        split="A",
        seed=20260607,
        budget=32,
        trace_root=tmp_path / "trace",
        family_by_task={"planet_task": "planetary"},
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["family"] == "planetary"
    assert row["verified_repair_success_at_32"] is True
    assert row["best_verified_reward_at_32"] == 1.0
    assert row["verifier_calls"] == 32
    assert row["actual_budget_matches_primary"] is True


def test_rows_from_sample_summary_uses_explicit_sft_manifest(
    tmp_path: Path,
) -> None:
    shared_sft = tmp_path / "shared_sft" / "A" / "20260607" / "sft_train"
    adapter = shared_sft / "final_adapter"
    adapter.mkdir(parents=True)
    manifest = shared_sft / "run_manifest.json"
    manifest.write_text(
        json.dumps({
            "adapter_updates": 7,
            "trained_tokens": 1234,
            "final_adapter": str(adapter),
        })
    )
    summary = {
        "max_turns": 1,
        "all_samples": [
            {
                "task_id": "planet_task",
                "family": "planetary",
                "sample_idx": 0,
                "verified_score": 1.0,
                "evaluation_valid": True,
                "hard_gate_passed": True,
                "failure_codes": [],
                "verifier_calls": 32,
                "cad_audits": 0,
                "chrono_audits": 0,
            },
        ],
    }

    rows = rows_from_sample_summary(
        summary=summary,
        method="sft_seen_family",
        split="A",
        seed=20260607,
        budget=32,
        trace_root=(
            tmp_path / "online_runs" / "A" / "20260607" / "eval_sft_seen_family"
        ),
        family_by_task={"planet_task": "planetary"},
        sft_manifest=manifest,
    )

    assert rows[0]["adapter_updates"] == 7
    assert rows[0]["trained_tokens"] == 1234
    assert rows[0]["adapter_path"] == str(adapter)
    assert rows[0]["adapter_checkpoint_paths"] == [str(adapter)]
    assert rows[0]["training_log_paths"] == [str(manifest)]


def test_rows_from_sample_summary_materializes_multiturn_evidence(
    tmp_path: Path,
) -> None:
    summary = {
        "max_turns": 2,
        "all_samples": [
            {
                "task_id": "cycloidal_task",
                "family": "cycloidal",
                "sample_idx": 0,
                "verified_score": 0.25,
                "evaluation_valid": True,
                "hard_gate_passed": False,
                "failure_codes": ["wrong_ratio"],
                "verifier_calls": 2,
                "cad_audits": 0,
                "chrono_audits": 0,
                "turn_traces": [
                    {
                        "turn_idx": 0,
                        "assistant_text": "turn 0 code",
                        "dense_pct": 25.0,
                        "score": 0.0,
                        "passed": False,
                        "parsed_ok": True,
                        "evaluation_valid": True,
                        "failure_codes": ["wrong_ratio"],
                    },
                    {
                        "turn_idx": 1,
                        "assistant_text": "turn 1 code",
                        "dense_pct": 25.0,
                        "score": 0.0,
                        "passed": False,
                        "parsed_ok": True,
                        "evaluation_valid": True,
                        "failure_codes": ["wrong_ratio"],
                    },
                ],
            },
            {
                "task_id": "cycloidal_task",
                "family": "cycloidal",
                "sample_idx": 1,
                "verified_score": 1.0,
                "evaluation_valid": True,
                "hard_gate_passed": True,
                "failure_codes": [],
                "verifier_calls": 2,
                "cad_audits": 0,
                "chrono_audits": 0,
                "turn_traces": [
                    {
                        "turn_idx": 0,
                        "assistant_text": "turn 2 code",
                        "dense_pct": 50.0,
                        "score": 50.0,
                        "passed": False,
                        "parsed_ok": True,
                        "evaluation_valid": True,
                        "failure_codes": ["wrong_ratio"],
                    },
                    {
                        "turn_idx": 1,
                        "assistant_text": "turn 3 code",
                        "dense_pct": 100.0,
                        "score": 100.0,
                        "passed": True,
                        "parsed_ok": True,
                        "evaluation_valid": True,
                        "failure_codes": [],
                    },
                ],
            },
        ],
    }

    rows = rows_from_sample_summary(
        summary=summary,
        method="llm_evolve_no_update",
        split="A",
        seed=20260607,
        budget=4,
        trace_root=tmp_path / "trace",
        family_by_task={"cycloidal_task": "cycloidal"},
        evidence_root=tmp_path / "evidence",
    )

    row = rows[0]
    assert row["verifier_calls"] == 4
    assert len(row["raw_completion_paths"]) == 4
    assert len(row["verifier_output_paths"]) == 4
    assert all(Path(path).is_file() for path in row["raw_completion_paths"])
    assert all(Path(path).is_file() for path in row["verifier_output_paths"])
    assert Path(row["raw_completion_paths"][0]).read_text() == "turn 0 code"
    verifier = json.loads(Path(row["verifier_output_paths"][-1]).read_text())
    assert verifier["sample_idx"] == 1
    assert verifier["turn_idx"] == 1
    assert verifier["hard_gate_passed"] is True


def test_rows_from_sample_summary_materializes_bundled_evidence(
    tmp_path: Path,
) -> None:
    summary = {
        "max_turns": 2,
        "all_samples": [
            {
                "task_id": "cycloidal_task",
                "sample_idx": 0,
                "verified_score": 0.0,
                "evaluation_valid": True,
                "hard_gate_passed": False,
                "failure_codes": ["wrong_ratio"],
                "verifier_calls": 2,
                "cad_audits": 1,
                "chrono_audits": 1,
                "turn_traces": [
                    {
                        "turn_idx": 0,
                        "assistant_text": "turn 0 code",
                        "dense_pct": 10.0,
                        "score": 10.0,
                        "passed": False,
                        "parsed_ok": True,
                        "evaluation_valid": True,
                        "failure_codes": ["wrong_ratio"],
                        "cad_audits": 1,
                        "chrono_audits": 1,
                    },
                    {
                        "turn_idx": 1,
                        "assistant_text": "turn 1 code",
                        "dense_pct": 20.0,
                        "score": 20.0,
                        "passed": False,
                        "parsed_ok": True,
                        "evaluation_valid": True,
                        "failure_codes": ["wrong_ratio"],
                    },
                ],
            }
        ],
    }

    rows = rows_from_sample_summary(
        summary=summary,
        method="adaptive_evolution",
        split="A",
        seed=20260607,
        budget=2,
        trace_root=tmp_path / "trace",
        family_by_task={"cycloidal_task": "cycloidal"},
        evidence_root=tmp_path / "evidence",
        evidence_layout="bundled",
    )

    row = rows[0]
    assert len(row["raw_completion_paths"]) == 2
    assert len(row["verifier_output_paths"]) == 2
    assert len(set(row["raw_completion_paths"])) == 1
    assert len(set(row["verifier_output_paths"])) == 1
    raw_bundle = Path(row["raw_completion_paths"][0])
    verifier_bundle = Path(row["verifier_output_paths"][0])
    assert raw_bundle.is_file()
    assert verifier_bundle.is_file()
    assert len(raw_bundle.read_text().splitlines()) == 2
    assert len(verifier_bundle.read_text().splitlines()) == 2
    assert row["cad_artifact_paths"]
    assert row["chrono_output_paths"]
    assert Path(row["cad_artifact_paths"][0]).name == "cad_audits.jsonl"
    assert Path(row["chrono_output_paths"][0]).name == "chrono_audits.jsonl"


def test_rows_from_sample_summary_materializes_cad_chrono_evidence(
    tmp_path: Path,
) -> None:
    summary = {
        "max_turns": 1,
        "all_samples": [
            {
                "task_id": "rack_task",
                "family": "rack_pinion",
                "sample_idx": 0,
                "verified_score": 1.0,
                "evaluation_valid": True,
                "hard_gate_passed": True,
                "failure_codes": [],
                "verifier_calls": 1,
                "cad_audits": 1,
                "chrono_audits": 1,
                "sampler_http_400_count": 2,
                "sampler_retry_count": 2,
                "physical_metrics": {"contact_force_rms_N": 3.0},
            },
        ],
    }

    rows = rows_from_sample_summary(
        summary=summary,
        method="frozen_model",
        split="A",
        seed=20260610,
        budget=1,
        trace_root=tmp_path / "trace",
        family_by_task={"rack_task": "rack_pinion"},
        evidence_root=tmp_path / "evidence",
    )

    row = rows[0]
    assert row["actual_verifier_calls"] == 1
    assert row["actual_cad_calls"] == 1
    assert row["actual_chrono_calls"] == 1
    assert row["sampler_http_400_count"] == 2
    assert row["sampler_retry_count"] == 2
    assert len(row["cad_artifact_paths"]) == 1
    assert len(row["chrono_output_paths"]) == 1
    cad = json.loads(Path(row["cad_artifact_paths"][0]).read_text())
    chrono = json.loads(Path(row["chrono_output_paths"][0]).read_text())
    assert cad["kind"] == "cad"
    assert chrono["kind"] == "chrono"
    assert chrono["physical_metrics"]["contact_force_rms_N"] == 3.0


def test_rows_from_sample_summary_materializes_unreached_audit_obligations(
    tmp_path: Path,
) -> None:
    summary = {
        "max_turns": 1,
        "all_samples": [
            {
                "task_id": "geneva_task",
                "family": "geneva_indexer",
                "sample_idx": 0,
                "verified_score": 0.0,
                "evaluation_valid": False,
                "hard_gate_passed": False,
                "failure_codes": ["missing_port"],
                "verifier_calls": 32,
                "cad_audits": 0,
                "chrono_audits": 0,
            },
        ],
    }

    rows = rows_from_sample_summary(
        summary=summary,
        method="adaptive_evolution",
        split="A",
        seed=20260610,
        budget=32,
        trace_root=tmp_path / "trace",
        family_by_task={"geneva_task": "geneva_indexer"},
        verifier_level_by_task={"geneva_task": 3},
        evidence_root=tmp_path / "evidence",
        evidence_layout="bundled",
    )

    row = rows[0]
    assert row["actual_cad_calls"] == 0
    assert row["actual_chrono_calls"] == 0
    assert row["required_cad_audits"] == 32
    assert row["required_chrono_audits"] == 32
    assert len(row["cad_artifact_paths"]) == 1
    assert len(row["chrono_output_paths"]) == 1
    cad_record = json.loads(Path(row["cad_artifact_paths"][0]).read_text().splitlines()[0])
    chrono_record = json.loads(
        Path(row["chrono_output_paths"][0]).read_text().splitlines()[0]
    )
    assert cad_record["status"] == "precondition_failed_no_actual_audit"
    assert chrono_record["status"] == "precondition_failed_no_actual_audit"
    assert cad_record["required_audits"] == 32
    assert chrono_record["failure_code_counts"] == {"missing_port": 1}


def test_rows_from_sample_summary_keeps_retry_evidence_paths_unique(
    tmp_path: Path,
) -> None:
    summary = {
        "max_turns": 2,
        "all_samples": [
            {
                "task_id": "cycloidal_task",
                "family": "cycloidal",
                "sample_idx": 0,
                "verified_score": 0.0,
                "evaluation_valid": True,
                "hard_gate_passed": False,
                "failure_codes": ["wrong_ratio"],
                "verifier_calls": 2,
                "cad_audits": 0,
                "chrono_audits": 0,
                "turn_traces": [
                    {
                        "turn_idx": 0,
                        "assistant_text": "first retry turn",
                        "dense_pct": 0.0,
                        "score": 0.0,
                        "passed": False,
                        "parsed_ok": True,
                        "evaluation_valid": True,
                        "failure_codes": ["wrong_ratio"],
                        "sampler_attempt": 0,
                        "verifier_call_idx_within_sample": 0,
                    },
                    {
                        "turn_idx": 0,
                        "assistant_text": "second retry turn",
                        "dense_pct": 0.0,
                        "score": 0.0,
                        "passed": False,
                        "parsed_ok": True,
                        "evaluation_valid": True,
                        "failure_codes": ["wrong_ratio"],
                        "sampler_attempt": 1,
                        "verifier_call_idx_within_sample": 1,
                    },
                ],
            },
        ],
    }

    rows = rows_from_sample_summary(
        summary=summary,
        method="llm_evolve_no_update",
        split="A",
        seed=20260607,
        budget=2,
        trace_root=tmp_path / "trace",
        family_by_task={"cycloidal_task": "cycloidal"},
        evidence_root=tmp_path / "evidence",
    )

    row = rows[0]
    assert len(row["raw_completion_paths"]) == 2
    assert len(set(row["raw_completion_paths"])) == 2
    assert Path(row["raw_completion_paths"][0]).read_text() == "first retry turn"
    assert Path(row["raw_completion_paths"][1]).read_text() == "second retry turn"


def test_rows_from_sample_summary_counts_recovered_sampler_retry(
    tmp_path: Path,
) -> None:
    summary = {
        "max_turns": 1,
        "all_samples": [
            {
                "task_id": "cycloidal_task",
                "family": "cycloidal",
                "sample_idx": 0,
                "verified_score": 0.0,
                "evaluation_valid": True,
                "hard_gate_passed": False,
                "failure_codes": ["wrong_ratio"],
                "verifier_calls": 1,
                "cad_audits": 0,
                "chrono_audits": 0,
                "sampler_retry_count": 2,
                "turn_traces": [
                    {
                        "turn_idx": 0,
                        "assistant_text": "[sampler_error: 400]",
                        "failure_codes": ["sampler_error"],
                        "trace_kind": "sampler_error_retry",
                        "verifier_call_idx_within_sample": 0,
                    },
                    {
                        "turn_idx": 0,
                        "assistant_text": "scored code",
                        "failure_codes": ["wrong_ratio"],
                        "trace_kind": "scored_attempt",
                        "verifier_call_idx_within_sample": 1,
                    },
                ],
            },
        ],
    }

    rows = rows_from_sample_summary(
        summary=summary,
        method="adaptive_evolution",
        split="A",
        seed=20260607,
        budget=1,
        trace_root=tmp_path / "trace",
        family_by_task={"cycloidal_task": "cycloidal"},
        evidence_root=tmp_path / "evidence",
    )

    assert rows[0]["sampler_error_count"] == 1
    assert rows[0]["sampler_retry_count"] == 3
    assert len(rows[0]["raw_completion_paths"]) == 2
    assert len(rows[0]["verifier_output_paths"]) == 1


def test_rows_from_sample_summary_materializes_terminal_evidence_for_missing_trace(
    tmp_path: Path,
) -> None:
    trace_root = tmp_path / "trace"
    completion_dir = trace_root / "sample_0" / "cycloidal_task"
    completion_dir.mkdir(parents=True)
    (completion_dir / "completion.txt").write_text("terminal completion")
    summary = {
        "max_turns": 2,
        "all_samples": [
            {
                "task_id": "cycloidal_task",
                "family": "cycloidal",
                "sample_idx": 0,
                "sample_tokens_out": 17,
                "score": 0.0,
                "verified_score": 0.0,
                "evaluation_valid": True,
                "hard_gate_passed": False,
                "design_py_extracted": True,
                "failure_codes": ["wrong_ratio"],
                "verifier_calls": 2,
                "cad_audits": 0,
                "chrono_audits": 0,
                "turn_traces": [
                    {
                        "turn_idx": 0,
                        "assistant_text": "retry turn",
                        "dense_pct": 0.0,
                        "score": 0.0,
                        "passed": False,
                        "parsed_ok": True,
                        "evaluation_valid": True,
                        "failure_codes": ["wrong_ratio"],
                        "sampler_attempt": 0,
                        "verifier_call_idx_within_sample": 0,
                    },
                ],
            },
        ],
    }

    rows = rows_from_sample_summary(
        summary=summary,
        method="llm_evolve_no_update",
        split="A",
        seed=20260607,
        budget=2,
        trace_root=trace_root,
        family_by_task={"cycloidal_task": "cycloidal"},
        evidence_root=tmp_path / "evidence",
    )

    row = rows[0]
    assert len(row["raw_completion_paths"]) == 2
    assert len(row["verifier_output_paths"]) == 2
    assert Path(row["raw_completion_paths"][1]).read_text() == "terminal completion"
    verifier = json.loads(Path(row["verifier_output_paths"][1]).read_text())
    assert verifier["trace_kind"] == "terminal_sample_evidence"
    assert verifier["verifier_call_idx_within_sample"] == 1


def test_row_from_ttrl_reward_log_uses_reward_log_budget(
    tmp_path: Path,
) -> None:
    reward_log = tmp_path / "reward_log.jsonl"
    rows = []
    for idx in range(32):
        rows.append({
            "task_id": "cycloidal_task",
            "task_dir": str(tmp_path / "tasks" / "cycloidal_task"),
            "verified_score": 1.0 if idx == 31 else 0.25,
            "evaluation_valid": True,
            "hard_gate_passed": idx == 31,
            "failure_codes": [] if idx == 31 else ["wrong_ratio"],
            "cad_audits": 0,
            "chrono_audits": 0,
            "design_py_extracted": True,
        })
    reward_log.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n"
    )

    row = row_from_ttrl_reward_log(
        reward_log=reward_log,
        split="B",
        task_id="cycloidal_task",
        seed=20260609,
        budget=32,
        run_dir=tmp_path,
        family_by_task={"cycloidal_task": "cycloidal"},
    )

    assert row["method"] == "mechanical_evolve_ttrl"
    assert row["family"] == "cycloidal"
    assert row["verified_repair_success_at_32"] is True
    assert row["verifier_calls"] == 32
    assert row["actual_budget_matches_primary"] is True


def test_sample_rank_prefers_valid_structural_failure_over_invalid_artifact() -> None:
    invalid = {
        "verified_score": 0.0,
        "score": 0.0,
        "evaluation_valid": False,
        "hard_gate_passed": False,
        "failure_codes": ["invalid_artifact"],
        "design_py_extracted": True,
    }
    missing_port = {
        "verified_score": 0.0,
        "score": 0.0,
        "evaluation_valid": False,
        "hard_gate_passed": False,
        "failure_codes": ["missing_port"],
        "design_py_extracted": True,
    }
    valid_partial = {
        "verified_score": 0.0,
        "score": 0.0,
        "evaluation_valid": True,
        "hard_gate_passed": False,
        "failure_codes": ["missing_contact", "lockup"],
        "design_py_extracted": True,
    }

    assert sample_rank(missing_port) > sample_rank(invalid)
    assert sample_rank(valid_partial) > sample_rank(missing_port)


def test_eval_summary_runner_forces_declared_audit_retries(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, list[str]] = {}
    report_dir = tmp_path / "report"

    def fake_run(cmd: list[str], *, timeout: float) -> None:
        captured["cmd"] = cmd
        report_dir.mkdir(parents=True)
        (report_dir / "smoke_summary.json").write_text(
            json.dumps({"all_samples": []})
        )

    monkeypatch.setattr(online, "run", fake_run)

    args = Namespace(
        runner_python="python",
        sglang_base_url="http://127.0.0.1:30000",
        api_key="dummy",
        base_model="base",
        rollout_backend="sglang_chat",
        max_tokens=512,
        timeout=180.0,
        concurrency=2,
        audit_retries=0,
        eval_timeout_s=60.0,
    )

    run_or_load_eval_summary(
        args=args,
        method=EvalMethod("frozen_model", 32, 1, 0.2, 0.95),
        report_dir=report_dir,
        tasks_root=tmp_path / "tasks",
        test_file=tmp_path / "split.txt",
        seed=20260607,
        resume_existing=False,
    )

    cmd = captured["cmd"]
    assert "--audit-retries" in cmd
    assert cmd[cmd.index("--audit-retries") + 1] == "0"


def test_eval_summary_runner_caps_actual_verifier_calls_for_feedback_loop(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, list[str]] = {}
    report_dir = tmp_path / "report"

    def fake_run(cmd: list[str], *, timeout: float) -> None:
        captured["cmd"] = cmd
        report_dir.mkdir(parents=True)
        (report_dir / "smoke_summary.json").write_text(
            json.dumps({"all_samples": []})
        )

    monkeypatch.setattr(online, "run", fake_run)

    args = Namespace(
        runner_python="python",
        sglang_base_url="http://127.0.0.1:30000",
        api_key="dummy",
        base_model="base",
        rollout_backend="sglang_chat",
        max_tokens=512,
        timeout=180.0,
        concurrency=2,
        audit_retries=0,
        eval_timeout_s=60.0,
        budget=32,
    )

    run_or_load_eval_summary(
        args=args,
        method=EvalMethod("llm_evolve_no_update", 32, 4, 0.7, 0.95),
        report_dir=report_dir,
        tasks_root=tmp_path / "tasks",
        test_file=tmp_path / "split.txt",
        seed=20260607,
        resume_existing=False,
    )

    cmd = captured["cmd"]
    assert "--samples-per-task" in cmd
    assert cmd[cmd.index("--samples-per-task") + 1] == "32"
    assert "--max-turns" in cmd
    assert cmd[cmd.index("--max-turns") + 1] == "4"
    assert "--max-verifier-calls-per-task" in cmd
    assert cmd[cmd.index("--max-verifier-calls-per-task") + 1] == "32"


def test_eval_summary_runner_does_not_emit_none_budget(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, list[str]] = {}
    report_dir = tmp_path / "report"

    def fake_run(cmd: list[str], *, timeout: float) -> None:
        captured["cmd"] = cmd
        report_dir.mkdir(parents=True)
        (report_dir / "smoke_summary.json").write_text(
            json.dumps({"all_samples": []})
        )

    monkeypatch.setattr(online, "run", fake_run)

    args = Namespace(
        runner_python="python",
        sglang_base_url="http://127.0.0.1:30000",
        api_key="dummy",
        base_model="base",
        rollout_backend="sglang_chat",
        max_tokens=512,
        timeout=180.0,
        concurrency=2,
        audit_retries=0,
        eval_timeout_s=60.0,
        budget=None,
    )

    run_or_load_eval_summary(
        args=args,
        method=EvalMethod("adaptive_evolution", 32, 4, 0.9, 0.95),
        report_dir=report_dir,
        tasks_root=tmp_path / "tasks",
        test_file=tmp_path / "split.txt",
        seed=20260607,
        resume_existing=False,
    )

    cmd = captured["cmd"]
    assert "--max-verifier-calls-per-task" in cmd
    assert cmd[cmd.index("--max-verifier-calls-per-task") + 1] == "32"
    assert "None" not in cmd


def test_eval_summary_runner_passes_local_transformers_options(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, list[str]] = {}
    report_dir = tmp_path / "report"

    def fake_run(cmd: list[str], *, timeout: float) -> None:
        captured["cmd"] = cmd
        report_dir.mkdir(parents=True)
        (report_dir / "smoke_summary.json").write_text(
            json.dumps({"all_samples": []})
        )

    monkeypatch.setattr(online, "run", fake_run)

    args = Namespace(
        runner_python="python",
        sglang_base_url="http://127.0.0.1:30000",
        api_key="dummy",
        base_model="base",
        rollout_backend="transformers_local",
        local_device="cpu",
        local_torch_dtype="float32",
        local_trust_remote_code=True,
        max_tokens=512,
        timeout=180.0,
        concurrency=1,
        audit_retries=0,
        eval_timeout_s=60.0,
        budget=None,
    )

    run_or_load_eval_summary(
        args=args,
        method=EvalMethod("frozen_model", 32, 1, 0.2, 0.95),
        report_dir=report_dir,
        tasks_root=tmp_path / "tasks",
        test_file=tmp_path / "split.txt",
        seed=20260607,
        resume_existing=False,
    )

    cmd = captured["cmd"]
    assert cmd[cmd.index("--rollout-backend") + 1] == "transformers_local"
    assert cmd[cmd.index("--local-device") + 1] == "cpu"
    assert cmd[cmd.index("--local-torch-dtype") + 1] == "float32"
    assert "--local-trust-remote-code" in cmd


def test_eval_summary_runner_uses_explicit_sft_manifest(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, list[str]] = {}
    report_dir = (
        tmp_path / "online_runs" / "A" / "20260607" / "eval_sft_seen_family"
    )
    shared_sft = tmp_path / "shared_sft" / "A" / "20260607" / "sft_train"
    adapter = shared_sft / "final_adapter"
    adapter.mkdir(parents=True)
    manifest = shared_sft / "run_manifest.json"
    manifest.write_text(json.dumps({"final_adapter": str(adapter)}))

    def fake_run(cmd: list[str], *, timeout: float) -> None:
        captured["cmd"] = cmd
        report_dir.mkdir(parents=True)
        (report_dir / "smoke_summary.json").write_text(
            json.dumps({"all_samples": []})
        )

    monkeypatch.setattr(online, "run", fake_run)

    args = Namespace(
        runner_python="python",
        sglang_base_url="http://127.0.0.1:30000",
        api_key="dummy",
        base_model="base",
        rollout_backend="sglang_chat",
        max_tokens=512,
        timeout=180.0,
        concurrency=2,
        audit_retries=0,
        eval_timeout_s=60.0,
        budget=None,
    )

    run_or_load_eval_summary(
        args=args,
        method=EvalMethod("sft_seen_family", 32, 1, 0.2, 0.95, "sft"),
        report_dir=report_dir,
        tasks_root=tmp_path / "tasks",
        test_file=tmp_path / "split.txt",
        seed=20260607,
        resume_existing=False,
        sft_manifest=manifest,
    )

    cmd = captured["cmd"]
    assert "--sglang-lora-path" in cmd
    assert cmd[cmd.index("--sglang-lora-path") + 1] == str(adapter)


def test_resume_eval_summary_reruns_when_cache_misses_requested_tasks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    report_dir = tmp_path / "eval_frozen_model"
    report_dir.mkdir()
    (report_dir / "smoke_summary.json").write_text(
        json.dumps({"all_samples": [{"task_id": "task_old"}]})
    )
    test_file = tmp_path / "split.txt"
    test_file.write_text("task_new\n")
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], *, timeout: float) -> None:
        calls.append(cmd)
        (report_dir / "smoke_summary.json").write_text(
            json.dumps({"all_samples": [{"task_id": "task_new"}]})
        )

    monkeypatch.setattr(online, "run", fake_run)
    args = Namespace(
        runner_python="python",
        sglang_base_url="http://127.0.0.1:30000",
        api_key="dummy",
        base_model="base",
        rollout_backend="sglang_chat",
        max_tokens=512,
        timeout=180.0,
        concurrency=2,
        audit_retries=0,
        eval_timeout_s=60.0,
        budget=None,
    )

    summary = run_or_load_eval_summary(
        args=args,
        method=EvalMethod("frozen_model", 32, 1, 0.2, 0.95, "baseline"),
        report_dir=report_dir,
        tasks_root=tmp_path / "tasks",
        test_file=test_file,
        seed=20260610,
        resume_existing=True,
    )

    assert len(calls) == 1
    assert online.sample_summary_task_ids(summary) == {"task_new"}


def test_sft_runner_passes_kbit_preparation_flags(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, list[str]] = {}
    run_dir = tmp_path / "sft"

    def fake_run(cmd: list[str], *, timeout: float) -> None:
        captured["cmd"] = cmd
        adapter = run_dir / "final_adapter"
        adapter.mkdir(parents=True)
        (run_dir / "run_manifest.json").write_text(
            json.dumps({
                "adapter_updates": 4,
                "final_adapter": str(adapter),
                "optimizer_guard": {
                    "attempted_steps": 4,
                    "successful_steps": 4,
                },
            })
        )

    monkeypatch.setattr(online, "run", fake_run)

    args = Namespace(
        sft_runner="python",
        base_model="base",
        sft_max_steps=4,
        sft_learning_rate=5e-6,
        sft_max_grad_norm=1.0,
        sft_max_seq_length=512,
        sft_lora_rank=16,
        sft_load_in_4bit=True,
        sft_load_in_8bit=False,
        sft_prepare_kbit_training=True,
        sft_prepare_kbit_training_mode="lightweight",
        sft_use_cpu=True,
        sft_gradient_checkpointing=True,
        sft_trust_remote_code=True,
        sft_torch_dtype="bfloat16",
        sft_attn_implementation=None,
        sft_device_map="balanced",
        train_timeout_s=60.0,
    )

    adapter = run_or_load_sft(
        args=args,
        run_dir=run_dir,
        tasks_root=tmp_path / "tasks",
        train_file=tmp_path / "train.txt",
        seed=20260607,
        resume_existing=False,
    )

    cmd = captured["cmd"]
    assert adapter == str(run_dir / "final_adapter")
    assert "--load-in-4bit" in cmd
    assert "--prepare-kbit-training" in cmd
    assert "--prepare-kbit-training-mode" in cmd
    assert cmd[cmd.index("--prepare-kbit-training-mode") + 1] == "lightweight"
    assert "--max-grad-norm" in cmd
    assert cmd[cmd.index("--max-grad-norm") + 1] == "1.0"
    assert "--use-cpu" in cmd
    assert "--gradient-checkpointing" in cmd


def test_sft_resume_discards_incomplete_manifest(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, list[str]] = {}
    run_dir = tmp_path / "sft"
    run_dir.mkdir(parents=True)
    (run_dir / "run_manifest.json").write_text(
        json.dumps({"schema": "incomplete"}) + "\n"
    )

    def fake_run(cmd: list[str], *, timeout: float) -> None:
        captured["cmd"] = cmd
        adapter = run_dir / "final_adapter"
        adapter.mkdir(parents=True)
        (run_dir / "run_manifest.json").write_text(
            json.dumps({
                "adapter_updates": 4,
                "final_adapter": str(adapter),
                "optimizer_guard": {
                    "attempted_steps": 4,
                    "successful_steps": 4,
                },
            })
        )

    monkeypatch.setattr(online, "run", fake_run)
    args = Namespace(
        sft_runner="python",
        base_model="base",
        sft_max_steps=4,
        sft_learning_rate=5e-6,
        sft_max_grad_norm=1.0,
        sft_max_seq_length=512,
        sft_lora_rank=16,
        sft_load_in_4bit=False,
        sft_load_in_8bit=False,
        sft_prepare_kbit_training=False,
        sft_prepare_kbit_training_mode="peft",
        sft_use_cpu=True,
        sft_gradient_checkpointing=False,
        sft_trust_remote_code=False,
        sft_torch_dtype=None,
        sft_attn_implementation=None,
        sft_device_map=None,
        train_timeout_s=60.0,
    )

    adapter = run_or_load_sft(
        args=args,
        run_dir=run_dir,
        tasks_root=tmp_path / "tasks",
        train_file=tmp_path / "train.txt",
        seed=20260607,
        resume_existing=True,
    )

    assert captured["cmd"]
    assert adapter == str(run_dir / "final_adapter")


def test_non_resume_reset_removes_stale_experiment_outputs(tmp_path: Path) -> None:
    out_dir = tmp_path / "run"
    run_root = out_dir / "online_runs"
    stale_adapter = run_root / "A" / "20260607" / "sft_train" / "final_adapter"
    stale_adapter.mkdir(parents=True)
    (stale_adapter / "adapter_model.safetensors").write_text("bad")
    (out_dir / "cell_results.jsonl").parent.mkdir(parents=True, exist_ok=True)
    (out_dir / "cell_results.jsonl").write_text('{"stale": true}\n')
    (out_dir / "raw_completions" / "old").mkdir(parents=True)
    (out_dir / "verifier_outputs" / "old").mkdir(parents=True)

    reset_non_resume_outputs(out_dir=out_dir, run_root=run_root)

    assert run_root.is_dir()
    assert not stale_adapter.exists()
    assert not (out_dir / "cell_results.jsonl").exists()
    assert (out_dir / "raw_completions").is_dir()
    assert not (out_dir / "raw_completions" / "old").exists()
    assert (out_dir / "verifier_outputs").is_dir()
    assert not (out_dir / "verifier_outputs" / "old").exists()


def test_ttrl_runner_passes_lightweight_kbit_prepare_mode(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, list[str]] = {}
    run_dir = tmp_path / "ttrl"

    def fake_run(cmd: list[str], *, timeout: float) -> None:
        captured["cmd"] = cmd
        run_dir.mkdir(parents=True)
        rows = [
            {
                "task_id": "task",
                "verified_score": 1.0,
                "evaluation_valid": True,
                "hard_gate_passed": True,
                "failure_codes": [],
                "cad_audits": 0,
                "chrono_audits": 0,
                "design_py_extracted": True,
            }
            for _ in range(4)
        ]
        (run_dir / "reward_log.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n"
        )
        (run_dir / "run_manifest.json").write_text(
            json.dumps({
                "adapter_updates": 1,
                "trained_tokens": 16,
                "rl_trained_tokens": 16,
                "n_rl_datums": 4,
                "final_adapter": str(run_dir / "final_adapter"),
                "optimizer_guard": {
                    "attempted_steps": 1,
                    "successful_steps": 1,
                },
            })
        )

    monkeypatch.setattr(online, "run", fake_run)

    args = Namespace(
        ttrl_runner="python",
        ttrl_model=None,
        base_model="base",
        ttrl_learning_rate=5e-6,
        ttrl_max_grad_norm=1.0,
        max_context_tokens=512,
        max_tokens=128,
        timeout=30.0,
        ttrl_lora_rank=16,
        ttrl_save_adapter_dtype="bfloat16",
        ttrl_load_in_4bit=True,
        ttrl_load_in_8bit=False,
        ttrl_kbit_prepare_mode="lightweight",
        ttrl_use_cpu=True,
        ttrl_gradient_checkpointing=True,
        ttrl_trust_remote_code=True,
        ttrl_bf16=True,
        ttrl_fp16=False,
        ttrl_torch_dtype="bfloat16",
        ttrl_attn_implementation=None,
        ttrl_device_map="balanced",
        ttrl_max_memory=None,
        ttrl_rollout_openai=False,
        sglang_base_url="http://127.0.0.1:30000",
        base_url="http://127.0.0.1:30000",
        api_key="dummy",
        train_timeout_s=60.0,
    )

    row = run_or_load_ttrl_cell(
        args=args,
        run_dir=run_dir,
        benchmark_dir=tmp_path,
        split_file=tmp_path / "one.txt",
        split="A",
        task_id="task",
        seed=20260607,
        budget=4,
        ttrl_steps=1,
        ttrl_generations=4,
        ttrl_steps_per_generation=4,
        family_by_task={"task": "cycloidal"},
        evidence_root=None,
        init_adapter=None,
        resume_existing=False,
    )

    cmd = captured["cmd"]
    assert row["actual_budget_matches_primary"] is True
    assert "--kbit-prepare-mode" in cmd
    assert cmd[cmd.index("--kbit-prepare-mode") + 1] == "lightweight"
    assert "--use-cpu" in cmd
    assert "--steps-per-generation" in cmd
    assert cmd[cmd.index("--steps-per-generation") + 1] == "4"
    assert "--per-device-train-batch-size" in cmd
    assert cmd[cmd.index("--per-device-train-batch-size") + 1] == "1"
    assert "--max-grad-norm" in cmd
    assert cmd[cmd.index("--max-grad-norm") + 1] == "1.0"
    assert "--save-adapter-dtype" in cmd
    assert cmd[cmd.index("--save-adapter-dtype") + 1] == "bfloat16"
    assert "--bf16" in cmd
    assert "--fp16" not in cmd


def test_ttrl_runner_preserves_physics_method_and_reward_channel(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, list[str]] = {}
    run_dir = tmp_path / "ttrl_confidence"

    def fake_run(cmd: list[str], *, timeout: float) -> None:
        captured["cmd"] = cmd
        run_dir.mkdir(parents=True)
        rows = [
            {
                "task_id": "task",
                "verified_score": 1.0,
                "evaluation_valid": True,
                "hard_gate_passed": True,
                "failure_codes": [],
                "cad_audits": 0,
                "chrono_audits": 0,
                "design_py_extracted": True,
            }
            for _ in range(4)
        ]
        (run_dir / "reward_log.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n"
        )
        (run_dir / "run_manifest.json").write_text(
            json.dumps({
                "adapter_updates": 1,
                "trained_tokens": 16,
                "rl_trained_tokens": 16,
                "n_rl_datums": 4,
                "optimizer_guard": {
                    "attempted_steps": 1,
                    "successful_steps": 1,
                },
            })
        )

    monkeypatch.setattr(online, "run", fake_run)

    args = Namespace(
        ttrl_runner="python",
        ttrl_model=None,
        base_model="base",
        ttrl_learning_rate=5e-6,
        ttrl_max_grad_norm=1.0,
        max_context_tokens=512,
        max_tokens=128,
        timeout=30.0,
        ttrl_lora_rank=16,
        ttrl_save_adapter_dtype="native",
        ttrl_load_in_4bit=False,
        ttrl_load_in_8bit=False,
        ttrl_kbit_prepare_mode="none",
        ttrl_gradient_checkpointing=False,
        ttrl_trust_remote_code=True,
        ttrl_bf16=False,
        ttrl_fp16=False,
        ttrl_torch_dtype=None,
        ttrl_attn_implementation=None,
        ttrl_device_map=None,
        ttrl_max_memory=None,
        ttrl_rollout_openai=False,
        sglang_base_url="http://127.0.0.1:30000",
        base_url="http://127.0.0.1:30000",
        api_key="dummy",
        train_timeout_s=60.0,
    )
    method = "mechanical_evolve_ttrl_confidence"

    row = run_or_load_ttrl_cell(
        args=args,
        run_dir=run_dir,
        benchmark_dir=tmp_path,
        split_file=tmp_path / "one.txt",
        split="A",
        task_id="task",
        seed=20260610,
        budget=4,
        ttrl_steps=1,
        ttrl_generations=4,
        ttrl_steps_per_generation=4,
        family_by_task={"task": "cycloidal"},
        evidence_root=None,
        init_adapter=None,
        resume_existing=False,
        method=method,
        reward_channel=online.reward_channel_for_method(
            method,
            default="artifact_progress",
        ),
    )

    cmd = captured["cmd"]
    assert row["method"] == method
    assert row["method_variant"] == method
    assert "--reward-channel" in cmd
    assert cmd[cmd.index("--reward-channel") + 1] == "verified_score"


def test_ttrl_metadata_retention_keeps_checkpoint_provenance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "ttrl_metadata"

    def fake_run(cmd: list[str], *, timeout: float) -> None:
        del cmd, timeout
        run_dir.mkdir(parents=True)
        rows = [
            {
                "task_id": "task",
                "verified_score": 1.0,
                "evaluation_valid": True,
                "hard_gate_passed": True,
                "failure_codes": [],
                "cad_audits": 0,
                "chrono_audits": 0,
                "design_py_extracted": True,
            }
            for _ in range(4)
        ]
        (run_dir / "reward_log.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n"
        )
        adapter = run_dir / "final_adapter"
        adapter.mkdir()
        (adapter / "adapter_config.json").write_text('{"r": 8}\n')
        (adapter / "adapter_model.safetensors").write_bytes(b"weight-bytes")
        (adapter / "tokenizer.json").write_text('{"large": "shared tokenizer"}\n')
        (run_dir / "run_manifest.json").write_text(
            json.dumps({
                "adapter_updates": 1,
                "trained_tokens": 16,
                "rl_trained_tokens": 16,
                "n_rl_datums": 4,
                "final_adapter": str(adapter),
                "optimizer_guard": {
                    "attempted_steps": 1,
                    "successful_steps": 1,
                },
            })
        )

    monkeypatch.setattr(online, "run", fake_run)

    args = Namespace(
        ttrl_runner="python",
        ttrl_model=None,
        base_model="base",
        ttrl_learning_rate=5e-6,
        ttrl_max_grad_norm=1.0,
        max_context_tokens=512,
        max_tokens=128,
        timeout=30.0,
        ttrl_lora_rank=16,
        ttrl_save_adapter_dtype="native",
        ttrl_adapter_retention="metadata",
        ttrl_load_in_4bit=False,
        ttrl_load_in_8bit=False,
        ttrl_kbit_prepare_mode="none",
        ttrl_gradient_checkpointing=False,
        ttrl_trust_remote_code=True,
        ttrl_bf16=False,
        ttrl_fp16=False,
        ttrl_torch_dtype=None,
        ttrl_attn_implementation=None,
        ttrl_device_map=None,
        ttrl_max_memory=None,
        ttrl_rollout_openai=False,
        sglang_base_url="http://127.0.0.1:30000",
        base_url="http://127.0.0.1:30000",
        api_key="dummy",
        train_timeout_s=60.0,
    )

    row = run_or_load_ttrl_cell(
        args=args,
        run_dir=run_dir,
        benchmark_dir=tmp_path,
        split_file=tmp_path / "one.txt",
        split="A",
        task_id="task",
        seed=20260610,
        budget=4,
        ttrl_steps=1,
        ttrl_generations=4,
        ttrl_steps_per_generation=4,
        family_by_task={"task": "cycloidal"},
        evidence_root=tmp_path / "evidence",
        init_adapter=None,
        resume_existing=False,
    )

    adapter_path = Path(row["adapter_path"])
    manifest = json.loads((adapter_path / "checkpoint_manifest.json").read_text())
    source_files = {item["path"]: item for item in manifest["source_files"]}
    assert adapter_path.is_dir()
    assert (adapter_path / "adapter_config.json").is_file()
    assert not (adapter_path / "tokenizer.json").exists()
    assert (adapter_path / "training_run_manifest.json").is_file()
    assert source_files["adapter_model.safetensors"]["weight_file"] is True
    assert source_files["adapter_model.safetensors"]["bytes"] == len(b"weight-bytes")
    assert source_files["tokenizer.json"]["weight_file"] is False
    assert manifest["omitted_redundant_non_weight_files"] == ["tokenizer.json"]
    assert not (run_dir / "final_adapter").exists()
    assert row["adapter_checkpoint_paths"] == [str(adapter_path)]
    assert row["training_log_paths"] == [str(run_dir / "reward_log.jsonl")]


def test_optional_file_lock_enters_and_exits(tmp_path: Path) -> None:
    lock_path = tmp_path / "locks" / "sft.lock"

    with online.optional_file_lock(lock_path):
        assert lock_path.is_file()


def test_ttrl_runner_rejects_reward_log_over_budget(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "ttrl_over_budget"

    def fake_run(cmd: list[str], *, timeout: float) -> None:
        del cmd, timeout
        run_dir.mkdir(parents=True)
        rows = [
            {
                "task_id": "task",
                "verified_score": 0.0,
                "evaluation_valid": True,
                "hard_gate_passed": False,
                "failure_codes": ["invalid_artifact"],
                "cad_audits": 0,
                "chrono_audits": 0,
                "design_py_extracted": True,
            }
            for _ in range(5)
        ]
        (run_dir / "reward_log.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n"
        )
        (run_dir / "run_manifest.json").write_text(
            json.dumps({
                "adapter_updates": 1,
                "trained_tokens": 16,
                "rl_trained_tokens": 16,
                "n_rl_datums": 5,
                "optimizer_guard": {
                    "attempted_steps": 1,
                    "successful_steps": 1,
                },
            })
        )

    monkeypatch.setattr(online, "run", fake_run)

    args = Namespace(
        ttrl_runner="python",
        ttrl_model=None,
        base_model="base",
        ttrl_learning_rate=5e-6,
        ttrl_max_grad_norm=1.0,
        max_context_tokens=512,
        max_tokens=128,
        timeout=30.0,
        ttrl_lora_rank=16,
        ttrl_save_adapter_dtype="native",
        ttrl_load_in_4bit=False,
        ttrl_load_in_8bit=False,
        ttrl_kbit_prepare_mode="none",
        ttrl_gradient_checkpointing=False,
        ttrl_trust_remote_code=True,
        ttrl_bf16=False,
        ttrl_fp16=False,
        ttrl_torch_dtype=None,
        ttrl_attn_implementation=None,
        ttrl_device_map=None,
        ttrl_max_memory=None,
        ttrl_rollout_openai=False,
        sglang_base_url="http://127.0.0.1:30000",
        base_url="http://127.0.0.1:30000",
        api_key="dummy",
        train_timeout_s=60.0,
    )

    with pytest.raises(SystemExit, match="TTRL verifier budget mismatch"):
        run_or_load_ttrl_cell(
            args=args,
            run_dir=run_dir,
            benchmark_dir=tmp_path,
            split_file=tmp_path / "one.txt",
            split="A",
            task_id="task",
            seed=20260610,
            budget=4,
            ttrl_steps=1,
            ttrl_generations=4,
            ttrl_steps_per_generation=4,
            family_by_task={"task": "cycloidal"},
            evidence_root=None,
            init_adapter=None,
            resume_existing=False,
        )


def test_ttrl_resume_discards_partial_zero_update_manifest(
    monkeypatch,
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "ttrl"
    run_dir.mkdir()
    (run_dir / "reward_log.jsonl").write_text(
        json.dumps({"task_id": "task", "verified_score": 0.0}) + "\n"
    )
    (run_dir / "run_manifest.json").write_text(
        json.dumps({
            "adapter_updates": 0,
            "trained_tokens": 0,
            "rl_trained_tokens": 0,
            "n_rl_datums": 0,
            "optimizer_guard": {"attempted_steps": 1, "successful_steps": 0},
        })
    )

    calls = 0

    def fake_run(cmd: list[str], *, timeout: float) -> None:
        nonlocal calls
        calls += 1
        assert not (run_dir / "run_manifest.json").exists()
        run_dir.mkdir(parents=True, exist_ok=True)
        rows = [
            {
                "task_id": "task",
                "verified_score": 1.0,
                "evaluation_valid": True,
                "hard_gate_passed": True,
                "failure_codes": [],
                "cad_audits": 0,
                "chrono_audits": 0,
                "design_py_extracted": True,
            }
            for _ in range(4)
        ]
        (run_dir / "reward_log.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n"
        )
        (run_dir / "final_adapter").mkdir()
        (run_dir / "run_manifest.json").write_text(
            json.dumps({
                "adapter_updates": 1,
                "trained_tokens": 16,
                "rl_trained_tokens": 16,
                "n_rl_datums": 4,
                "final_adapter": str(run_dir / "final_adapter"),
                "optimizer_guard": {
                    "attempted_steps": 1,
                    "successful_steps": 1,
                },
            })
        )

    monkeypatch.setattr(online, "run", fake_run)

    args = Namespace(
        ttrl_runner="python",
        ttrl_model=None,
        base_model="base",
        ttrl_learning_rate=5e-6,
        ttrl_max_grad_norm=1.0,
        max_context_tokens=512,
        max_tokens=128,
        timeout=30.0,
        ttrl_lora_rank=16,
        ttrl_save_adapter_dtype="native",
        ttrl_load_in_4bit=False,
        ttrl_load_in_8bit=False,
        ttrl_kbit_prepare_mode="none",
        ttrl_gradient_checkpointing=False,
        ttrl_trust_remote_code=True,
        ttrl_bf16=False,
        ttrl_fp16=False,
        ttrl_torch_dtype="bfloat16",
        ttrl_attn_implementation=None,
        ttrl_device_map="none",
        ttrl_max_memory=None,
        ttrl_rollout_openai=False,
        sglang_base_url="http://127.0.0.1:30000",
        base_url="http://127.0.0.1:30000",
        api_key="dummy",
        train_timeout_s=60.0,
    )

    row = run_or_load_ttrl_cell(
        args=args,
        run_dir=run_dir,
        benchmark_dir=tmp_path,
        split_file=tmp_path / "one.txt",
        split="A",
        task_id="task",
        seed=20260607,
        budget=4,
        ttrl_steps=1,
        ttrl_generations=4,
        ttrl_steps_per_generation=4,
        family_by_task={"task": "cycloidal"},
        evidence_root=None,
        init_adapter=None,
        resume_existing=True,
    )

    assert calls == 1
    assert row["verified_repair_success_at_32"] is True
    assert row["adapter_updates"] == 1


def test_learning_manifest_rejects_zero_update_adapter(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="unusable adapter_updates=0"):
        require_learning_manifest(
            {
                "adapter_updates": 0,
                "optimizer_guard": {
                    "attempted_steps": 4,
                    "successful_steps": 0,
                },
            },
            manifest_path=tmp_path / "run_manifest.json",
            label="SFT",
            expected_adapter_updates=4,
        )


def test_learning_manifest_rejects_nonfinite_training_events(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="nonfinite training events"):
        require_learning_manifest(
            {
                "adapter_updates": 4,
                "optimizer_guard": {
                    "attempted_steps": 4,
                    "successful_steps": 4,
                    "skipped_nonfinite_gradient_steps": 1,
                },
            },
            manifest_path=tmp_path / "run_manifest.json",
            label="SFT",
            expected_adapter_updates=4,
        )


def test_learning_manifest_rejects_missing_ttrl_token_evidence(
    tmp_path: Path,
) -> None:
    with pytest.raises(SystemExit, match="trained_tokens=0"):
        require_learning_manifest(
            {
                "adapter_updates": 4,
                "n_rl_datums": 4,
                "trained_tokens": 0,
                "rl_trained_tokens": 4,
                "optimizer_guard": {
                    "attempted_steps": 4,
                    "successful_steps": 4,
                },
            },
            manifest_path=tmp_path / "run_manifest.json",
            label="TTRL task",
            expected_adapter_updates=4,
            min_rl_datums=4,
        )


def test_run_analysis_allows_negative_claim_when_artifacts_exist(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_run(
        cmd: list[str],
        *,
        timeout: float,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        for name in [
            "stats.json",
            "failure_analysis.json",
            "trace_pairs.json",
            "repair_taxonomy.json",
            "claim_audit.json",
        ]:
            (tmp_path / name).write_text("{}\n")
        return subprocess.CompletedProcess(cmd, 2)

    monkeypatch.setattr(online, "run", fake_run)

    online.run_analysis(out_dir=tmp_path, benchmark_dir=tmp_path)


def test_run_analysis_rejects_missing_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_run(
        cmd: list[str],
        *,
        timeout: float,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(online, "run", fake_run)

    with pytest.raises(SystemExit, match="analysis failed"):
        online.run_analysis(out_dir=tmp_path, benchmark_dir=tmp_path)
