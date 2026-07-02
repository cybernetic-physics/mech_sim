from __future__ import annotations

import json
from pathlib import Path

from scripts.prepare_mechanism_repair_physics_benchmark import (
    EVAL_SEEDS,
    PRIMARY_BASELINE,
    PRIMARY_BUDGET,
    PRIMARY_METHOD,
    REQUIRED_FAMILIES,
)
from scripts.plan_mechanism_repair_physics_sentinel import (
    audit_sentinel_run,
    build_sentinel_plan,
    materialize_benchmark_scaffold,
    write_sentinel_artifacts,
)


def test_sentinel_plan_writes_staged_primary_pair_and_calibrators(
    tmp_path: Path,
) -> None:
    benchmark = tmp_path / "benchmark"
    out_dir = tmp_path / "sentinel"
    _write_fake_benchmark(benchmark)

    plan = build_sentinel_plan(
        benchmark_dir=benchmark,
        out_dir=out_dir,
        splits=["hidden_perturbation", "isomorphic"],
        seeds=list(EVAL_SEEDS[:2]),
        primary_method=PRIMARY_METHOD,
        baseline_method=PRIMARY_BASELINE,
        calibrator_methods=["frozen_model", "sft_seen_family"],
        budget=PRIMARY_BUDGET,
        tasks_per_family_per_split=1,
        min_effect_pp=5.0,
    )
    materialize_benchmark_scaffold(source_dir=benchmark, out_dir=out_dir)
    write_sentinel_artifacts(out_dir=out_dir, plan=plan)

    assert plan["planned_cells"] == 192
    assert "USE_PREPLANNED_SHARDS=1" in plan["first_stage_submit_command"]
    assert "RESUME_EXISTING=1" in plan["first_stage_submit_command"]
    assert "SHARD_INDICES=0" in plan["first_stage_resume_command"]
    assert "RESTAGE_REMOTE_REPO=0" in plan["first_stage_resume_command"]
    assert "REFRESH_REMOTE_CODE=1" in plan["first_stage_resume_command"]
    assert "ALLOW_DESTRUCTIVE_RESTAGE=1" not in plan["first_stage_resume_command"]
    assert "SHARD_INDICES=0" in plan["stage_submit_commands"][0]["command"]
    assert "RESTAGE_REMOTE_REPO=1" in plan["stage_submit_commands"][0]["command"]
    assert "ALLOW_DESTRUCTIVE_RESTAGE=1" in plan["stage_submit_commands"][0]["command"]
    assert "SHARD_INDICES=0" in plan["stage_submit_commands"][0]["resume_command"]
    assert "RESTAGE_REMOTE_REPO=0" in plan["stage_submit_commands"][0]["resume_command"]
    assert "REFRESH_REMOTE_CODE=1" in plan["stage_submit_commands"][0]["resume_command"]
    assert "ALLOW_DESTRUCTIVE_RESTAGE=1" not in plan["stage_submit_commands"][0]["resume_command"]
    assert "SHARD_INDICES=1" in plan["stage_submit_commands"][1]["command"]
    assert "RESTAGE_REMOTE_REPO=0" in plan["stage_submit_commands"][1]["command"]
    assert "REFRESH_REMOTE_CODE=1" in plan["stage_submit_commands"][1]["command"]
    assert "RESUME_EXISTING=1" in plan["stage_submit_commands"][1]["command"]
    assert plan["stage_submit_commands"][1]["resume_command"] == (
        plan["stage_submit_commands"][1]["command"]
    )
    assert "ALLOW_DESTRUCTIVE_RESTAGE=1" not in plan["stage_submit_commands"][1]["command"]
    assert "sync_mechanism_repair_physics_sentinel_from_matx.sh" in (
        plan["sync_audit_command"]
    )
    assert [stage["planned_cells"] for stage in plan["stages"]] == [48, 48, 96]
    assert plan["selected_task_summary"]["by_verifier_level"] == {"2": 18, "3": 6}
    assert (out_dir / "tasks").is_dir()
    assert (out_dir / "split_manifest_hidden_perturbation.json").is_file()

    first_shard = json.loads(
        (out_dir / "experiment_shards" / "shard_0000.json").read_text(
            encoding="utf-8"
        )
    )
    assert first_shard["schema"] == "mechanism_repair_physics.experiment_shard.v1"
    assert first_shard["sentinel_schema"] == (
        "mechanism_repair_physics.sentinel_shard.v1"
    )
    assert first_shard["planned_cells"] == 48
    methods_by_group: dict[tuple[str, str, int, int], set[str]] = {}
    for cell in first_shard["cells"]:
        group = (cell["split"], cell["task_id"], cell["seed"], cell["budget"])
        methods_by_group.setdefault(group, set()).add(cell["method"])
    assert methods_by_group
    assert all(
        methods == {PRIMARY_BASELINE, PRIMARY_METHOD}
        for methods in methods_by_group.values()
    )


def test_sentinel_audit_reports_primary_delta(tmp_path: Path) -> None:
    benchmark = tmp_path / "benchmark"
    out_dir = tmp_path / "sentinel"
    _write_fake_benchmark(benchmark)
    plan = build_sentinel_plan(
        benchmark_dir=benchmark,
        out_dir=out_dir,
        splits=["hidden_perturbation"],
        seeds=[EVAL_SEEDS[0]],
        primary_method=PRIMARY_METHOD,
        baseline_method=PRIMARY_BASELINE,
        calibrator_methods=[],
        budget=PRIMARY_BUDGET,
        tasks_per_family_per_split=1,
        min_effect_pp=5.0,
    )
    rows = []
    for index, cell in enumerate(plan["expected_cells"]):
        row = {
            "split": cell["split"],
            "task_id": cell["task_id"],
            "seed": cell["seed"],
            "method": cell["method"],
            "budget": cell["budget"],
            "verified_repair_success_at_32": (
                cell["method"] == PRIMARY_METHOD
                and index % 4 in {0, 1}
            )
            or (
                cell["method"] == PRIMARY_BASELINE
                and index % 8 == 0
            ),
        }
        rows.append(row)
    shard_dir = out_dir / "shard_runs" / "shard_0000"
    shard_dir.mkdir(parents=True)
    (shard_dir / "cell_results.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    audit = audit_sentinel_run(
        out_dir=out_dir,
        planned_cells=plan["expected_cells"],
        primary_method=PRIMARY_METHOD,
        baseline_method=PRIMARY_BASELINE,
        budget=PRIMARY_BUDGET,
        min_effect_pp=5.0,
    )

    summary = audit["primary_pair_summary"]
    assert summary["paired_count"] == 12
    assert summary["delta_success_rate"] > 0
    assert audit["decision"]["status"] == "promising"


def test_sentinel_audit_groups_missing_cells_and_pair_completion(
    tmp_path: Path,
) -> None:
    benchmark = tmp_path / "benchmark"
    out_dir = tmp_path / "sentinel"
    _write_fake_benchmark(benchmark)
    plan = build_sentinel_plan(
        benchmark_dir=benchmark,
        out_dir=out_dir,
        splits=["hidden_perturbation"],
        seeds=[EVAL_SEEDS[0]],
        primary_method=PRIMARY_METHOD,
        baseline_method=PRIMARY_BASELINE,
        calibrator_methods=[],
        budget=PRIMARY_BUDGET,
        tasks_per_family_per_split=1,
        min_effect_pp=5.0,
    )
    first_baseline = next(
        cell
        for cell in plan["expected_cells"]
        if cell["method"] == PRIMARY_BASELINE
    )
    row = {
        "split": first_baseline["split"],
        "task_id": first_baseline["task_id"],
        "seed": first_baseline["seed"],
        "method": first_baseline["method"],
        "budget": first_baseline["budget"],
        "verified_repair_success_at_32": False,
    }
    shard_dir = out_dir / "shard_runs" / "shard_0000"
    shard_dir.mkdir(parents=True)
    (shard_dir / "cell_results.jsonl").write_text(
        json.dumps(row, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    audit = audit_sentinel_run(
        out_dir=out_dir,
        planned_cells=plan["expected_cells"],
        primary_method=PRIMARY_METHOD,
        baseline_method=PRIMARY_BASELINE,
        budget=PRIMARY_BUDGET,
        min_effect_pp=5.0,
    )

    missing = audit["missing_cell_summary"]
    pair_completion = audit["primary_pair_completion_summary"]
    assert audit["observed_cell_count"] == 1
    assert audit["missing_cell_count"] == len(plan["expected_cells"]) - 1
    assert missing["by_method"][PRIMARY_METHOD] == 12
    assert missing["by_method"][PRIMARY_BASELINE] == 11
    assert pair_completion["planned_pair_count"] == 12
    assert pair_completion["complete_pair_count"] == 0
    assert pair_completion["baseline_only_count"] == 1
    assert pair_completion["primary_only_count"] == 0
    assert pair_completion["neither_count"] == 11


def test_sentinel_audit_reports_partial_artifacts_without_counting_rows(
    tmp_path: Path,
) -> None:
    benchmark = tmp_path / "benchmark"
    out_dir = tmp_path / "sentinel"
    _write_fake_benchmark(benchmark)
    plan = build_sentinel_plan(
        benchmark_dir=benchmark,
        out_dir=out_dir,
        splits=["hidden_perturbation"],
        seeds=[EVAL_SEEDS[0]],
        primary_method=PRIMARY_METHOD,
        baseline_method=PRIMARY_BASELINE,
        calibrator_methods=[],
        budget=PRIMARY_BUDGET,
        tasks_per_family_per_split=1,
        min_effect_pp=5.0,
    )
    task_id = plan["expected_cells"][0]["task_id"]
    eval_dir = (
        out_dir
        / "shard_runs"
        / "shard_0000"
        / "online_runs"
        / "hidden_perturbation"
        / str(EVAL_SEEDS[0])
        / f"eval_{PRIMARY_BASELINE}"
    )
    task_dir = eval_dir / "sample_0" / task_id
    task_dir.mkdir(parents=True)
    (task_dir / "completion.txt").write_text("terminal code", encoding="utf-8")
    (task_dir / "sample_outcome.json").write_text(
        json.dumps({
            "version": "mech_bench.sample_outcome_checkpoint.v1",
            "outcome": {"task_id": task_id, "sample_idx": 0},
        }),
        encoding="utf-8",
    )
    (eval_dir / "smoke_summary.json").write_text(
        json.dumps({"complete": False, "all_samples": []}),
        encoding="utf-8",
    )

    audit = audit_sentinel_run(
        out_dir=out_dir,
        planned_cells=plan["expected_cells"],
        primary_method=PRIMARY_METHOD,
        baseline_method=PRIMARY_BASELINE,
        budget=PRIMARY_BUDGET,
        min_effect_pp=5.0,
    )

    partial = audit["partial_artifact_summary"]
    assert audit["observed_cell_count"] == 0
    assert audit["missing_cell_count"] == len(plan["expected_cells"])
    assert partial["smoke_summary_count"] == 1
    assert partial["incomplete_smoke_summary_count"] == 1
    assert partial["sample_outcome_checkpoint_count"] == 1
    assert partial["terminal_completion_count"] == 1
    assert partial["terminal_completion_task_count"] == 1


def _write_fake_benchmark(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    tasks = []
    level_tasks = []
    hidden_tasks = []
    task_paths_by_family: dict[str, list[str]] = {}
    level3_families = set(REQUIRED_FAMILIES[:3])
    for family_index, family in enumerate(REQUIRED_FAMILIES):
        verifier_level = 3 if family in level3_families else 2
        task_paths_by_family[family] = []
        for task_index in range(2):
            task_id = f"{family}_t{task_index:02d}_s{family_index}"
            task_dir = root / "tasks" / task_id
            task_dir.mkdir(parents=True)
            (task_dir / "prompt.md").write_text("repair mechanism\n", encoding="utf-8")
            task = {
                "task_id": task_id,
                "task_dir": str(task_dir),
                "family": family,
                "verifier_level": verifier_level,
                "headline_eligible": True,
                "has_hidden_variant": True,
            }
            tasks.append(task)
            level_tasks.append({
                "task_id": task_id,
                "family": family,
                "verifier_level": verifier_level,
                "headline_eligible": True,
            })
            hidden_tasks.append({
                "task_id": task_id,
                "family": family,
                "verifier_level": verifier_level,
                "hidden_variant_present": True,
                "perturbations": ["rename", "retarget", "reframe"],
            })
            task_paths_by_family[family].append(str(task_dir))

    _write_json(
        root / "benchmark_manifest.json",
        {
            "schema": "mechanism_repair_physics.benchmark_manifest.v1",
            "experiment_ready": True,
            "task_count": len(tasks),
            "tasks": tasks,
        },
    )
    _write_json(
        root / "method_manifest.json",
        {
            "schema": "mechanism_repair_physics.method_manifest.v1",
            "required_methods": [
                "frozen_model",
                "sft_seen_family",
                PRIMARY_BASELINE,
                "verifier_gated_search",
                "adaptive_evolution",
                "mechanical_evolve_ttrl",
                PRIMARY_METHOD,
                "mechanical_evolve_ttrl_confidence",
            ],
            "eval_seeds": list(EVAL_SEEDS),
            "primary_method": PRIMARY_METHOD,
            "primary_baseline": PRIMARY_BASELINE,
            "primary_budget_expensive_verifier_calls": PRIMARY_BUDGET,
        },
    )
    _write_json(
        root / "level_manifest.json",
        {
            "schema": "mechanism_repair_physics.level_manifest.v1",
            "headline_levels": [2, 3],
            "tasks": level_tasks,
        },
    )
    _write_json(
        root / "verifier_manifest.json",
        {
            "schema": "mechanism_repair_physics.verifier_manifest.v1",
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
    for split in ("A", "B", "external_style", "hidden_perturbation", "isomorphic"):
        split_dir = root / f"splits_{split}"
        split_dir.mkdir()
        paths = [paths[0] for _family, paths in task_paths_by_family.items()]
        (split_dir / "test.txt").write_text("\n".join(paths) + "\n", encoding="utf-8")
        (split_dir / "train.txt").write_text("\n".join(paths[:3]) + "\n", encoding="utf-8")
        _write_json(
            root / f"split_manifest_{split}.json",
            {
                "schema": "mechanism_repair_physics.split_manifest.v1",
                "split_name": split,
                "splits": {"test": paths, "train": paths[:3]},
            },
        )


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
