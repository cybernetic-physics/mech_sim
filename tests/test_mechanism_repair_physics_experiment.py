from __future__ import annotations

import json
import sys
from pathlib import Path

from scripts import run_mechanism_repair_physics_experiment as physics
from scripts.prepare_mechanism_repair_physics_benchmark import (
    EVAL_SEEDS,
    PRIMARY_BUDGET,
    REQUIRED_METHODS,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_fake_benchmark(root: Path) -> None:
    _write_json(
        root / "benchmark_manifest.json",
        {
            "schema": "mechanism_repair_physics.benchmark_manifest.v1",
            "experiment_ready": True,
        },
    )
    _write_json(
        root / "method_manifest.json",
        {
            "schema": "mechanism_repair_physics.method_manifest.v1",
            "required_methods": list(REQUIRED_METHODS),
        },
    )
    _write_json(
        root / "level_manifest.json",
        {
            "schema": "mechanism_repair_physics.level_manifest.v1",
            "tasks": [
                {
                    "task_id": "task_l2",
                    "family": "family_l2",
                    "verifier_level": 2,
                },
                {
                    "task_id": "task_l3",
                    "family": "family_l3",
                    "verifier_level": 3,
                },
            ],
        },
    )
    _write_json(root / "verifier_manifest.json", {"schema": "test"})
    _write_json(root / "hidden_variant_manifest.json", {"schema": "test"})
    (root / "tasks" / "task_l2").mkdir(parents=True)
    (root / "tasks" / "task_l3").mkdir(parents=True)
    for split in ("A", "B", "hidden_perturbation", "external_style"):
        split_dir = root / f"splits_{split}"
        split_dir.mkdir(parents=True)
        (split_dir / "test.txt").write_text(
            f"{root / 'tasks' / 'task_l3'}\n"
        )


def test_dry_run_writes_plan_and_incomplete_audit(
    monkeypatch,
    tmp_path: Path,
) -> None:
    benchmark = tmp_path / "benchmark"
    out_dir = tmp_path / "run"
    _write_fake_benchmark(benchmark)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_mechanism_repair_physics_experiment.py",
            "--benchmark-dir",
            str(benchmark),
            "--out-dir",
            str(out_dir),
            "--methods",
            "frozen_model",
            "--splits",
            "A",
            "--anti-shortcut-splits",
            "hidden_perturbation",
            "--eval-seeds",
            str(EVAL_SEEDS[0]),
            "--limit-tasks",
            "1",
            "--dry-run",
        ],
    )

    assert physics.main() == 0
    plan = json.loads((out_dir / "physics_experiment_plan.json").read_text())
    claim = json.loads((out_dir / "claim_audit.json").read_text())

    assert plan["planned_cells"] == 2
    assert (out_dir / "raw_completions").is_dir()
    assert (out_dir / "verifier_outputs").is_dir()
    assert claim["goal_complete"] is False
    assert "claim_status" not in claim
    assert "execute all required methods, not a smoke subset" in claim["blockers"]
    assert "execute all required evaluation seeds" in claim["blockers"]


def test_complete_synthetic_evidence_gets_binary_claim_status(
    tmp_path: Path,
) -> None:
    benchmark = tmp_path / "benchmark"
    out_dir = tmp_path / "run"
    _write_fake_benchmark(benchmark)
    for name in (
        "raw_completions",
        "verifier_outputs",
        "cad_artifacts",
        "chrono_outputs",
        "training_logs",
        "adapter_checkpoints",
    ):
        (out_dir / name).mkdir(parents=True)

    task_index = physics.load_task_index(benchmark)
    plan = physics.build_plan(
        benchmark_dir=benchmark,
        out_dir=out_dir,
        methods=list(REQUIRED_METHODS),
        splits=["A", "B"],
        anti_shortcut_splits=["hidden_perturbation", "external_style"],
        seeds=list(EVAL_SEEDS),
        budgets=[PRIMARY_BUDGET],
        limit_tasks=1,
        task_index=task_index,
    )
    rows = []
    for cell in plan["expected_cells"]:
        prefix = (
            f"{cell['split']}_{cell['task_id']}_{cell['seed']}_"
            f"{cell['method']}_{cell['budget']}"
        )
        raw = out_dir / "raw_completions" / f"{prefix}.txt"
        verifier = out_dir / "verifier_outputs" / f"{prefix}.json"
        cad = out_dir / "cad_artifacts" / f"{prefix}.step"
        chrono = out_dir / "chrono_outputs" / f"{prefix}.json"
        for path in (raw, verifier, cad, chrono):
            path.write_text("{}\n")
        row = {
            "split": cell["split"],
            "task_id": cell["task_id"],
            "seed": cell["seed"],
            "method": cell["method"],
            "budget": cell["budget"],
            "actual_verifier_calls": PRIMARY_BUDGET,
            "actual_cad_calls": 1,
            "actual_chrono_calls": 1,
            "raw_completion_paths": [str(raw.relative_to(out_dir))],
            "verifier_output_paths": [str(verifier.relative_to(out_dir))],
            "cad_artifact_paths": [str(cad.relative_to(out_dir))],
            "chrono_output_paths": [str(chrono.relative_to(out_dir))],
        }
        if cell["method"] in physics.TTRL_METHODS:
            log = out_dir / "training_logs" / f"{prefix}.log"
            ckpt = out_dir / "adapter_checkpoints" / prefix
            log.write_text("trained\n")
            ckpt.mkdir()
            row.update(
                {
                    "training_log_paths": [str(log.relative_to(out_dir))],
                    "adapter_checkpoint_paths": [str(ckpt.relative_to(out_dir))],
                    "adapter_updates": 1,
                }
            )
        rows.append(row)
    (out_dir / "cell_results.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    for name in ("failure_analysis.json", "trace_pairs.json", "repair_taxonomy.json"):
        _write_json(out_dir / name, {"schema": "test"})
    _write_json(
        out_dir / "stats.json",
        {
            "primary_comparison": {
                "success_delta_pct": 0.0,
                "success_delta_ci95": [0.0, 0.0],
                "success_sign_test_p_one_sided": 1.0,
            }
        },
    )

    audit = physics.audit_existing_experiment(out_dir=out_dir, plan=plan)

    assert audit["claim_audit"]["goal_complete"] is True
    assert (
        audit["claim_audit"]["claim_status"]
        == "does_not_support_primary_hypothesis"
    )
    assert audit["budget_audit"]["budget_matched"] is True
    assert audit["anti_shortcut_audit"]["anti_shortcut_executed"] is True


def test_unreached_cad_chrono_obligation_evidence_is_not_budget_mismatch(
    tmp_path: Path,
) -> None:
    benchmark = tmp_path / "benchmark"
    out_dir = tmp_path / "run"
    _write_fake_benchmark(benchmark)
    for name in (
        "raw_completions",
        "verifier_outputs",
        "cad_artifacts",
        "chrono_outputs",
        "training_logs",
        "adapter_checkpoints",
    ):
        (out_dir / name).mkdir(parents=True)

    task_index = physics.load_task_index(benchmark)
    plan = physics.build_plan(
        benchmark_dir=benchmark,
        out_dir=out_dir,
        methods=list(REQUIRED_METHODS),
        splits=["A", "B"],
        anti_shortcut_splits=["hidden_perturbation", "external_style"],
        seeds=list(EVAL_SEEDS),
        budgets=[PRIMARY_BUDGET],
        limit_tasks=1,
        task_index=task_index,
    )
    rows = []
    for cell in plan["expected_cells"]:
        prefix = (
            f"{cell['split']}_{cell['task_id']}_{cell['seed']}_"
            f"{cell['method']}_{cell['budget']}"
        )
        raw = out_dir / "raw_completions" / f"{prefix}.txt"
        verifier = out_dir / "verifier_outputs" / f"{prefix}.json"
        cad = out_dir / "cad_artifacts" / f"{prefix}.json"
        chrono = out_dir / "chrono_outputs" / f"{prefix}.json"
        for path in (raw, verifier, cad, chrono):
            path.write_text(
                json.dumps({
                    "status": "precondition_failed_no_actual_audit",
                    "failure_codes": ["missing_port"],
                }) + "\n"
            )
        row = {
            "split": cell["split"],
            "task_id": cell["task_id"],
            "seed": cell["seed"],
            "method": cell["method"],
            "budget": cell["budget"],
            "actual_verifier_calls": PRIMARY_BUDGET,
            "actual_cad_calls": 0,
            "actual_chrono_calls": 0,
            "required_cad_audits": PRIMARY_BUDGET,
            "required_chrono_audits": PRIMARY_BUDGET,
            "raw_completion_paths": [str(raw.relative_to(out_dir))],
            "verifier_output_paths": [str(verifier.relative_to(out_dir))],
            "cad_artifact_paths": [str(cad.relative_to(out_dir))],
            "chrono_output_paths": [str(chrono.relative_to(out_dir))],
        }
        if cell["method"] in physics.TTRL_METHODS:
            log = out_dir / "training_logs" / f"{prefix}.log"
            ckpt = out_dir / "adapter_checkpoints" / prefix
            log.write_text("trained\n")
            ckpt.mkdir()
            row.update(
                {
                    "training_log_paths": [str(log.relative_to(out_dir))],
                    "adapter_checkpoint_paths": [str(ckpt.relative_to(out_dir))],
                    "adapter_updates": 1,
                }
            )
        rows.append(row)
    (out_dir / "cell_results.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    for name in ("failure_analysis.json", "trace_pairs.json", "repair_taxonomy.json"):
        _write_json(out_dir / name, {"schema": "test"})
    _write_json(
        out_dir / "stats.json",
        {
            "primary_comparison": {
                "success_delta_pct": 0.0,
                "success_delta_ci95": [0.0, 0.0],
                "success_sign_test_p_one_sided": 1.0,
            }
        },
    )

    audit = physics.audit_existing_experiment(out_dir=out_dir, plan=plan)

    assert audit["budget_audit"]["budget_matched"] is True
    assert audit["budget_audit"]["budget_mismatch_count"] == 0
    assert audit["claim_audit"]["goal_complete"] is True


def test_shards_partition_full_experiment_plan(tmp_path: Path) -> None:
    benchmark = tmp_path / "benchmark"
    out_dir = tmp_path / "run"
    _write_fake_benchmark(benchmark)
    task_index = physics.load_task_index(benchmark)
    full = physics.build_plan(
        benchmark_dir=benchmark,
        out_dir=out_dir,
        methods=list(REQUIRED_METHODS),
        splits=["A", "B"],
        anti_shortcut_splits=["hidden_perturbation", "external_style"],
        seeds=list(EVAL_SEEDS),
        budgets=[PRIMARY_BUDGET],
        limit_tasks=1,
        task_index=task_index,
    )

    observed = []
    for shard_index in range(3):
        shard = physics.build_plan(
            benchmark_dir=benchmark,
            out_dir=out_dir,
            methods=list(REQUIRED_METHODS),
            splits=["A", "B"],
            anti_shortcut_splits=["hidden_perturbation", "external_style"],
            seeds=list(EVAL_SEEDS),
            budgets=[PRIMARY_BUDGET],
            limit_tasks=1,
            task_index=task_index,
            shard_index=shard_index,
            num_shards=3,
        )
        assert shard["full_planned_cells"] == full["planned_cells"]
        observed.extend(shard["expected_cells"])

    def key(cell: dict) -> tuple:
        return (
            cell["split"],
            cell["task_id"],
            cell["seed"],
            cell["method"],
            cell["budget"],
        )

    full_keys = {key(cell) for cell in full["expected_cells"]}
    observed_keys = [key(cell) for cell in observed]
    assert set(observed_keys) == full_keys
    assert len(observed_keys) == len(set(observed_keys))
