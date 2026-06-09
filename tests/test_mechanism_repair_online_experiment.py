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
    require_learning_manifest,
    reset_non_resume_outputs,
    row_from_ttrl_reward_log,
    run_or_load_ttrl_cell,
    run_or_load_eval_summary,
    run_or_load_sft,
    rows_from_sample_summary,
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
        ttrl_reward_channel="artifact_progress",
    )

    assert plan["ttrl_reward_channel"] == "artifact_progress"
    assert plan["ttrl_rollout_evaluations_per_cell"] == 32
    assert plan["split_tasks"] == {"A": ["task_a"]}
    assert plan["planned_cells"] == 1


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
    assert len(row["cad_artifact_paths"]) == 1
    assert len(row["chrono_output_paths"]) == 1
    cad = json.loads(Path(row["cad_artifact_paths"][0]).read_text())
    chrono = json.loads(Path(row["chrono_output_paths"][0]).read_text())
    assert cad["kind"] == "cad"
    assert chrono["kind"] == "chrono"
    assert chrono["physical_metrics"]["contact_force_rms_N"] == 3.0


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
    assert "--gradient-checkpointing" in cmd


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
        ttrl_load_in_4bit=True,
        ttrl_load_in_8bit=False,
        ttrl_kbit_prepare_mode="lightweight",
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
        family_by_task={"task": "cycloidal"},
        evidence_root=None,
        init_adapter=None,
        resume_existing=False,
    )

    cmd = captured["cmd"]
    assert row["actual_budget_matches_primary"] is True
    assert "--kbit-prepare-mode" in cmd
    assert cmd[cmd.index("--kbit-prepare-mode") + 1] == "lightweight"
    assert "--per-device-train-batch-size" in cmd
    assert cmd[cmd.index("--per-device-train-batch-size") + 1] == "1"
    assert "--max-grad-norm" in cmd
    assert cmd[cmd.index("--max-grad-norm") + 1] == "1.0"
    assert "--bf16" in cmd
    assert "--fp16" not in cmd


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
