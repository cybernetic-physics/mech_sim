from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from scripts import run_mechanism_repair_physics_experiment as physics
from scripts.prepare_mechanism_repair_physics_benchmark import (
    EVAL_SEEDS,
    PRIMARY_BUDGET,
    REQUIRED_FAMILIES,
    REQUIRED_METHODS,
    materialize_benchmark,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
FROZEN_PHYSICS_BENCHMARK = REPO_ROOT / "runs" / "mechanism_repair_physics_final"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_complete_negative_stats(out_dir: Path) -> None:
    _write_json(
        out_dir / "stats.json",
        {
            "primary_comparison": {
                "success_delta_pct": 0.0,
                "success_delta_ci95": {"low": 0.0, "high": 0.0},
                "success_sign_test_p_one_sided": 1.0,
                "reward_delta_mean": 0.0,
            },
            "primary_result_table": [
                {
                    "method": method,
                    "level23_verified_repair_success_at_32": 0.0,
                    "hidden_variant_success_at_32": 0.0,
                    "anti_shortcut_pass_rate_at_32": 0.0,
                    "best_verified_reward_at_32": 0.0,
                    "actual_verifier_calls": PRIMARY_BUDGET,
                    "actual_cad_calls": 0.0,
                    "actual_chrono_calls": 0.0,
                }
                for method in REQUIRED_METHODS
            ],
            "split_deltas": [
                {
                    "split": "hidden_perturbation",
                    "success_delta": 0.0,
                    "success_delta_pct": 0.0,
                    "reward_delta": 0.0,
                }
            ],
            "anti_shortcut_comparison": {
                "anti_shortcut_pass_rate_delta": 0.0,
                "anti_shortcut_pass_rate_delta_pct": 0.0,
            },
            "paired_method_comparisons": {
                "adaptive_evolution": {"primary_beats_on_success": False},
                "verifier_gated_search": {"primary_beats_on_success": False},
            },
            "family_deltas": [
                {"family": family, "success_delta": 0.0}
                for family in REQUIRED_FAMILIES
            ],
            "leave_one_family_out": [
                {
                    "removed_family": family,
                    "keeps_positive_success_delta": False,
                }
                for family in REQUIRED_FAMILIES
            ],
            "analysis_claim_audit": {
                "claim_status": "does_not_support_primary_hypothesis",
                "blockers": ["synthetic negative result"],
            },
        },
    )


def _write_fake_benchmark(root: Path) -> None:
    tasks = []
    audit_tasks = []
    level_tasks = []
    hidden_tasks = []
    level3_families = set(REQUIRED_FAMILIES[:4])
    for family in REQUIRED_FAMILIES:
        verifier_level = 3 if family in level3_families else 2
        for idx in range(10):
            task_id = f"{family}_task_{idx:02d}"
            task_dir = root / "tasks" / task_id
            task_dir.mkdir(parents=True)
            tasks.append(
                {
                    "task_id": task_id,
                    "family": family,
                    "seed": 20260610 + idx,
                    "task_dir": str(task_dir),
                    "verifier_level": verifier_level,
                    "headline_eligible": True,
                }
            )
            audit_tasks.append(
                {
                    "task_id": task_id,
                    "family": family,
                    "task_dir": str(task_dir),
                    "verifier_level": verifier_level,
                    "headline_eligible": True,
                    "constraint_classes": [
                        "topology_mobility",
                        "interface",
                        "cad_artifact",
                    ],
                    "negative_control_count": 2,
                    "effective_negative_control_count": 2,
                    "has_hidden_variant": True,
                    "uses_fake_contact_oracle": False,
                    "validation": {
                        "reference_passed": True,
                        "reference_evaluation_valid": True,
                        "reference_hard_gate_passed": True,
                        "reference_oracle_is_synthetic": False,
                        "negative_failures": [],
                    },
                }
            )
            level_tasks.append(
                {
                    "task_id": task_id,
                    "family": family,
                    "verifier_level": verifier_level,
                    "headline_eligible": True,
                    "constraint_classes": [
                        "topology_mobility",
                        "interface",
                        "cad_artifact",
                    ],
                }
            )
            hidden_tasks.append(
                {
                    "task_id": task_id,
                    "family": family,
                    "verifier_level": verifier_level,
                    "hidden_variant_present": True,
                    "perturbations": ["rename", "retarget", "reframe"],
                    "isomorphic_variant_status": "manifested_not_executed",
                }
            )
    family_counts = {
        family: sum(1 for task in tasks if task["family"] == family)
        for family in REQUIRED_FAMILIES
    }
    level_counts = {
        "2": sum(1 for task in tasks if task["verifier_level"] == 2),
        "3": sum(1 for task in tasks if task["verifier_level"] == 3),
    }
    _write_json(
        root / "benchmark_manifest.json",
        {
            "schema": "mechanism_repair_physics.benchmark_manifest.v1",
            "experiment_ready": True,
            "task_count": len(tasks),
            "required_families": list(REQUIRED_FAMILIES),
            "level_counts": level_counts,
            "tasks": tasks,
            "audit": {
                "schema": "mechanism_repair_physics.benchmark_audit.v1",
                "experiment_ready": True,
                "task_count": len(tasks),
                "family_counts": family_counts,
                "level_counts": level_counts,
                "level2plus_headline_count": len(tasks),
                "level3_headline_count": level_counts["3"],
                "tasks": audit_tasks,
                "blockers": [],
                "paper_blockers": [],
                "structural_blockers": [],
                "validation_blockers": [],
            },
        },
    )
    _write_json(
        root / "method_manifest.json",
        {
            "schema": "mechanism_repair_physics.method_manifest.v1",
            "required_methods": list(REQUIRED_METHODS),
            "eval_seeds": list(EVAL_SEEDS),
            "primary_budget_expensive_verifier_calls": PRIMARY_BUDGET,
        },
    )
    _write_json(
        root / "level_manifest.json",
        {
            "schema": "mechanism_repair_physics.level_manifest.v1",
            "headline_levels": [2, 3],
            "level_counts": level_counts,
            "tasks": level_tasks,
        },
    )
    _write_json(
        root / "verifier_manifest.json",
        {
            "schema": "test",
            "main_claim_allows_fake_oracle": False,
            "requires_real_pychrono": True,
        },
    )
    _write_json(
        root / "hidden_variant_manifest.json",
        {
            "schema": "mechanism_repair_physics.hidden_variant_manifest.v1",
            "tasks": hidden_tasks,
        },
    )

    by_family = {
        family: [task for task in tasks if task["family"] == family]
        for family in REQUIRED_FAMILIES
    }
    split_specs = {
        "A": {
            "seen": list(REQUIRED_FAMILIES[:6]),
            "unseen": list(REQUIRED_FAMILIES[6:]),
        },
        "B": {
            "seen": list(REQUIRED_FAMILIES[::2]),
            "unseen": list(REQUIRED_FAMILIES[1::2]),
        },
        "hidden_perturbation": {"seen": [], "unseen": list(REQUIRED_FAMILIES)},
        "external_style": {"seen": [], "unseen": list(REQUIRED_FAMILIES[:3])},
    }
    for split, spec in split_specs.items():
        split_dir = root / f"splits_{split}"
        split_dir.mkdir(parents=True)
        split_tasks = [
            by_family[family][0]
            for family in spec["unseen"]
            if by_family.get(family)
        ]
        if not split_tasks:
            split_tasks = [tasks[0]]
        task_paths = [str(root / "tasks" / task["task_id"]) for task in split_tasks]
        (split_dir / "test.txt").write_text("\n".join(task_paths) + "\n")
        _write_json(
            root / f"split_manifest_{split}.json",
            {
                "schema": "test",
                "split_name": split,
                "seed": 20260610,
                "seen_families": spec["seen"],
                "unseen_families": spec["unseen"],
                "splits": {"test": task_paths},
            },
        )


def _write_complete_physics_rows(out_dir: Path, plan: dict, audit_counts) -> None:
    for name in (
        "raw_completions",
        "verifier_outputs",
        "cad_artifacts",
        "chrono_outputs",
        "training_logs",
        "adapter_checkpoints",
    ):
        (out_dir / name).mkdir(parents=True, exist_ok=True)

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
            path.write_text("{}\n")
        cad_calls, chrono_calls = audit_counts(cell)
        row = {
            "split": cell["split"],
            "task_id": cell["task_id"],
            "seed": cell["seed"],
            "method": cell["method"],
            "budget": cell["budget"],
            "actual_verifier_calls": PRIMARY_BUDGET,
            "actual_cad_calls": cad_calls,
            "actual_chrono_calls": chrono_calls,
            "raw_completion_paths": [str(raw.relative_to(out_dir))],
            "verifier_output_paths": [str(verifier.relative_to(out_dir))],
            "cad_artifact_paths": [str(cad.relative_to(out_dir))],
            "chrono_output_paths": [str(chrono.relative_to(out_dir))],
        }
        if cell["method"] in physics.LEARNING_METHODS:
            log = out_dir / "training_logs" / f"{prefix}.log"
            ckpt = out_dir / "adapter_checkpoints" / prefix
            log.write_text("trained\n")
            ckpt.mkdir()
            (ckpt / "adapter_model.safetensors").write_bytes(b"weights")
            row.update(
                {
                    "training_log_paths": [str(log.relative_to(out_dir))],
                    "adapter_checkpoint_paths": [str(ckpt.relative_to(out_dir))],
                    "adapter_updates": 1,
                    "trained_tokens": 16,
                }
            )
            if cell["method"] in physics.TTRL_METHODS:
                row.update({"n_rl_datums": 4, "rl_trained_tokens": 16})
        rows.append(row)
    (out_dir / "cell_results.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    for name in ("failure_analysis.json", "trace_pairs.json", "repair_taxonomy.json"):
        _write_json(out_dir / name, {"schema": "test"})
    _write_complete_negative_stats(out_dir)


def test_materialized_physics_prompts_are_family_specific(tmp_path: Path) -> None:
    tasks_root = tmp_path / "tasks"
    tasks_root.mkdir()
    manifest = materialize_benchmark(
        tasks_root=tasks_root,
        tasks_per_family=1,
        base_seed=20270610,
    )
    by_family = {row["family"]: row for row in manifest["tasks"]}

    assert set(by_family) == set(REQUIRED_FAMILIES)
    for family in REQUIRED_FAMILIES:
        prompt = Path(by_family[family]["task_dir"], "prompt.md").read_text()
        assert f"Canonical mechanism family: `{family}`" in prompt
        assert "DesignIR deliverable:" in prompt
        assert "`schema_version=\"design_ir.v2\"`" in prompt
        assert "`params[\"cad_mass_properties\"]`" in prompt
        assert "`ports` must be a dict keyed by port id, not a list" in prompt
        assert "`revolute_joint` and `prismatic_joint` values" in prompt
        assert "must reference joint ids" in prompt
        assert "Minimal trusted CAD/material evidence pattern:" in prompt
        assert "Do not return placeholder strings" in prompt
        assert "submissions fail if they call undefined helpers such as `cad(...)`" in prompt
        assert "`materials` must be a dict keyed by material id, not a list" in prompt
        assert "'elastic_modulus_pa': 205000000000.0" in prompt
        assert "'geometry': {'cad': _write_step" in prompt
        assert "'cad_mass_properties': _mass_props" in prompt
        assert "fake_contact_oracle" in prompt

    geneva_prompt = Path(
        by_family["geneva_indexer"]["task_dir"], "prompt.md"
    ).read_text()
    assert "Geneva indexing mechanism" in geneva_prompt
    assert "rotating driver wheel" in geneva_prompt
    assert "`driver`" in geneva_prompt
    assert "`geneva`" in geneva_prompt
    assert "`input_port`: kind `revolute_joint`" in geneva_prompt
    assert "`output_port`: kind `revolute_joint`" in geneva_prompt
    assert "Required contact pairs: `driver:geneva`" in geneva_prompt
    assert "Minimal Level-3 contact evidence pattern:" in geneva_prompt
    assert "'type': 'contact_pair'" in geneva_prompt
    assert "'parent': 'driver'" in geneva_prompt
    assert "'child': 'geneva'" in geneva_prompt
    assert "'chrono_collision'" in geneva_prompt
    assert "plain spur gear train" in geneva_prompt

    rack_prompt = Path(
        by_family["rack_pinion"]["task_dir"], "prompt.md"
    ).read_text()
    assert "rotating pinion meshes with a translating rack" in rack_prompt
    assert "Required contact pairs: `pinion:rack`" in rack_prompt
    assert "`output_port`: kind `prismatic_joint`" in rack_prompt


def test_frozen_physics_benchmark_prompts_are_family_specific() -> None:
    manifest = json.loads(
        (FROZEN_PHYSICS_BENCHMARK / "benchmark_manifest.json").read_text()
    )
    assert manifest["task_count"] == 120
    assert manifest["experiment_ready"] is True

    by_family = {row["family"]: row for row in manifest["tasks"]}
    assert set(by_family) == set(REQUIRED_FAMILIES)
    for family in REQUIRED_FAMILIES:
        prompt = Path(by_family[family]["task_dir"], "prompt.md").read_text()
        assert f"Canonical mechanism family: `{family}`" in prompt
        assert "DesignIR deliverable:" in prompt
        assert "Minimal trusted CAD/material evidence pattern:" in prompt
        assert "Do not return placeholder strings" in prompt
        assert "'geometry': {'cad': _write_step" in prompt
        assert "'cad_mass_properties': _mass_props" in prompt
        assert "fake_contact_oracle" in prompt

    geneva_prompt = Path(
        by_family["geneva_indexer"]["task_dir"], "prompt.md"
    ).read_text()
    assert "Geneva indexing mechanism" in geneva_prompt
    assert "Required contact pairs: `driver:geneva`" in geneva_prompt
    assert "Minimal Level-3 contact evidence pattern:" in geneva_prompt
    assert "'chrono_collision'" in geneva_prompt
    assert "plain spur gear train" in geneva_prompt

    rack_prompt = Path(
        by_family["rack_pinion"]["task_dir"], "prompt.md"
    ).read_text()
    assert "rotating pinion meshes with a translating rack" in rack_prompt
    assert "Required contact pairs: `pinion:rack`" in rack_prompt
    assert "`output_port`: kind `prismatic_joint`" in rack_prompt


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


def test_validate_benchmark_rejects_underfilled_contract(tmp_path: Path) -> None:
    benchmark = tmp_path / "benchmark"
    _write_fake_benchmark(benchmark)
    manifest_path = benchmark / "benchmark_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["tasks"] = manifest["tasks"][:1]
    manifest["task_count"] = 1
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    with pytest.raises(SystemExit, match="need at least 120"):
        physics.validate_benchmark(benchmark)


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
        if cell["method"] in physics.LEARNING_METHODS:
            log = out_dir / "training_logs" / f"{prefix}.log"
            ckpt = out_dir / "adapter_checkpoints" / prefix
            log.write_text("trained\n")
            ckpt.mkdir()
            (ckpt / "adapter_model.safetensors").write_bytes(b"weights")
            row.update(
                {
                    "training_log_paths": [str(log.relative_to(out_dir))],
                    "adapter_checkpoint_paths": [str(ckpt.relative_to(out_dir))],
                    "adapter_updates": 1,
                    "trained_tokens": 16,
                }
            )
            if cell["method"] in physics.TTRL_METHODS:
                row.update({"n_rl_datums": 4, "rl_trained_tokens": 16})
        rows.append(row)
    (out_dir / "cell_results.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    for name in ("failure_analysis.json", "trace_pairs.json", "repair_taxonomy.json"):
        _write_json(out_dir / name, {"schema": "test"})
    _write_complete_negative_stats(out_dir)

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
        if cell["method"] in physics.LEARNING_METHODS:
            log = out_dir / "training_logs" / f"{prefix}.log"
            ckpt = out_dir / "adapter_checkpoints" / prefix
            log.write_text("trained\n")
            ckpt.mkdir()
            (ckpt / "adapter_model.safetensors").write_bytes(b"weights")
            row.update(
                {
                    "training_log_paths": [str(log.relative_to(out_dir))],
                    "adapter_checkpoint_paths": [str(ckpt.relative_to(out_dir))],
                    "adapter_updates": 1,
                    "trained_tokens": 16,
                }
            )
            if cell["method"] in physics.TTRL_METHODS:
                row.update({"n_rl_datums": 4, "rl_trained_tokens": 16})
        rows.append(row)
    (out_dir / "cell_results.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    for name in ("failure_analysis.json", "trace_pairs.json", "repair_taxonomy.json"):
        _write_json(out_dir / name, {"schema": "test"})
    _write_complete_negative_stats(out_dir)

    audit = physics.audit_existing_experiment(out_dir=out_dir, plan=plan)

    assert audit["budget_audit"]["budget_matched"] is True
    assert audit["budget_audit"]["budget_mismatch_count"] == 0
    assert audit["claim_audit"]["goal_complete"] is True


def test_primary_fewer_cad_chrono_calls_than_baseline_is_not_budget_mismatch(
    tmp_path: Path,
) -> None:
    benchmark = tmp_path / "benchmark"
    out_dir = tmp_path / "run"
    _write_fake_benchmark(benchmark)

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

    def audit_counts(cell):
        if cell["method"] == physics.PRIMARY_METHOD:
            return 1, 1
        if cell["method"] == physics.PRIMARY_BASELINE:
            return 4, 4
        return 9, 9

    _write_complete_physics_rows(out_dir, plan, audit_counts)
    audit = physics.audit_existing_experiment(out_dir=out_dir, plan=plan)

    assert audit["budget_audit"]["budget_matched"] is True
    assert audit["budget_audit"]["primary_expensive_budget_excess_count"] == 0
    assert audit["claim_audit"]["goal_complete"] is True


def test_primary_cad_chrono_overspend_blocks_goal_completion(
    tmp_path: Path,
) -> None:
    benchmark = tmp_path / "benchmark"
    out_dir = tmp_path / "run"
    _write_fake_benchmark(benchmark)

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
    expected_groups = {
        (cell["split"], cell["task_id"], cell["seed"], cell["budget"])
        for cell in plan["expected_cells"]
    }

    def audit_counts(cell):
        if cell["method"] == physics.PRIMARY_METHOD:
            return 5, 6
        if cell["method"] == physics.PRIMARY_BASELINE:
            return 4, 5
        return 1, 1

    _write_complete_physics_rows(out_dir, plan, audit_counts)
    audit = physics.audit_existing_experiment(out_dir=out_dir, plan=plan)

    assert audit["budget_audit"]["budget_matched"] is False
    assert (
        audit["budget_audit"]["primary_expensive_budget_excess_count"]
        == 2 * len(expected_groups)
    )
    assert audit["claim_audit"]["goal_complete"] is False
    assert any(
        "actual verifier/CAD/Chrono budget" in item
        for item in audit["claim_audit"]["blockers"]
    )


def test_incomplete_stats_blocks_goal_completion(tmp_path: Path) -> None:
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
        if cell["method"] in physics.LEARNING_METHODS:
            log = out_dir / "training_logs" / f"{prefix}.log"
            ckpt = out_dir / "adapter_checkpoints" / prefix
            log.write_text("trained\n")
            ckpt.mkdir()
            (ckpt / "adapter_model.safetensors").write_bytes(b"weights")
            row.update({
                "training_log_paths": [str(log.relative_to(out_dir))],
                "adapter_checkpoint_paths": [str(ckpt.relative_to(out_dir))],
                "adapter_updates": 1,
                "trained_tokens": 16,
            })
            if cell["method"] in physics.TTRL_METHODS:
                row.update({"n_rl_datums": 4, "rl_trained_tokens": 16})
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

    assert audit["claim_audit"]["goal_complete"] is False
    assert audit["claim_audit"]["missing_analysis_requirements"]
    assert any(
        "compute required primary" in item
        for item in audit["claim_audit"]["blockers"]
    )


def test_ttrl_learning_evidence_requires_adapter_weights(tmp_path: Path) -> None:
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
        methods=["mechanical_evolve_ttrl"],
        splits=["A"],
        anti_shortcut_splits=[],
        seeds=[20260610],
        budgets=[PRIMARY_BUDGET],
        limit_tasks=1,
        task_index=task_index,
    )
    cell = plan["expected_cells"][0]
    prefix = (
        f"{cell['split']}_{cell['task_id']}_{cell['seed']}_"
        f"{cell['method']}_{cell['budget']}"
    )
    raw = out_dir / "raw_completions" / f"{prefix}.txt"
    verifier = out_dir / "verifier_outputs" / f"{prefix}.json"
    cad = out_dir / "cad_artifacts" / f"{prefix}.json"
    chrono = out_dir / "chrono_outputs" / f"{prefix}.json"
    for path in (raw, verifier, cad, chrono):
        path.write_text("{}\n")
    log = out_dir / "training_logs" / f"{prefix}.log"
    ckpt = out_dir / "adapter_checkpoints" / prefix
    log.write_text("trained\n")
    ckpt.mkdir()
    (ckpt / "checkpoint_manifest.json").write_text(
        json.dumps({"weights_retained": False}) + "\n"
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
        "raw_completion_paths": [str(raw.relative_to(out_dir))],
        "verifier_output_paths": [str(verifier.relative_to(out_dir))],
        "cad_artifact_paths": [str(cad.relative_to(out_dir))],
        "chrono_output_paths": [str(chrono.relative_to(out_dir))],
        "training_log_paths": [str(log.relative_to(out_dir))],
        "adapter_checkpoint_paths": [str(ckpt.relative_to(out_dir))],
        "adapter_updates": 1,
        "trained_tokens": 16,
        "n_rl_datums": 4,
        "rl_trained_tokens": 16,
    }
    (out_dir / "cell_results.jsonl").write_text(json.dumps(row) + "\n")
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

    assert audit["claim_audit"]["goal_complete"] is False
    assert audit["claim_audit"]["missing_learning_count"] == 1
    assert audit["claim_audit"]["sample_missing_learning"][0]["missing"] == [
        "adapter_checkpoint_weights"
    ]


def test_sft_learning_evidence_accepts_adjacent_manifest_for_legacy_rows(
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
    ):
        (out_dir / name).mkdir(parents=True)

    task_index = physics.load_task_index(benchmark)
    plan = physics.build_plan(
        benchmark_dir=benchmark,
        out_dir=out_dir,
        methods=["sft_seen_family"],
        splits=["A"],
        anti_shortcut_splits=[],
        seeds=[20260610],
        budgets=[PRIMARY_BUDGET],
        limit_tasks=1,
        task_index=task_index,
    )
    cell = plan["expected_cells"][0]
    prefix = (
        f"{cell['split']}_{cell['task_id']}_{cell['seed']}_"
        f"{cell['method']}_{cell['budget']}"
    )
    raw = out_dir / "raw_completions" / f"{prefix}.txt"
    verifier = out_dir / "verifier_outputs" / f"{prefix}.json"
    cad = out_dir / "cad_artifacts" / f"{prefix}.json"
    chrono = out_dir / "chrono_outputs" / f"{prefix}.json"
    for path in (raw, verifier, cad, chrono):
        path.write_text("{}\n")
    sft_dir = out_dir / "shared_sft" / "A" / "20260610" / "sft_train"
    ckpt = sft_dir / "final_adapter"
    ckpt.mkdir(parents=True)
    (ckpt / "adapter_model.safetensors").write_bytes(b"weights")
    (sft_dir / "run_manifest.json").write_text(
        json.dumps({"adapter_updates": 4, "trained_tokens": 16}) + "\n"
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
        "raw_completion_paths": [str(raw.relative_to(out_dir))],
        "verifier_output_paths": [str(verifier.relative_to(out_dir))],
        "cad_artifact_paths": [str(cad.relative_to(out_dir))],
        "chrono_output_paths": [str(chrono.relative_to(out_dir))],
        "adapter_checkpoint_paths": [str(ckpt.relative_to(out_dir))],
        "adapter_updates": 4,
        "trained_tokens": 16,
    }
    (out_dir / "cell_results.jsonl").write_text(json.dumps(row) + "\n")

    audit = physics.audit_existing_experiment(out_dir=out_dir, plan=plan)

    assert audit["claim_audit"]["missing_learning_count"] == 0


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


def test_shards_keep_methods_for_same_cell_group_together(tmp_path: Path) -> None:
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

    by_group: dict[tuple, set[int]] = {}
    for cell in full["expected_cells"]:
        group = (cell["split"], cell["task_id"], cell["seed"], cell["budget"])
        by_group.setdefault(group, set()).add(
            physics.cell_shard(cell, num_shards=5)
        )

    assert by_group
    assert all(len(shards) == 1 for shards in by_group.values())
