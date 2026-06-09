from __future__ import annotations

import importlib.util
import json
import tomllib
from pathlib import Path

from mech_bench.evaluator import evaluate
from scripts.prepare_mechanism_repair_physics_benchmark import (
    REQUIRED_FAMILIES,
    REQUIRED_METHODS,
    SPLITS,
    audit_benchmark,
    build_method_manifest,
    freeze_required_splits,
    materialize_benchmark,
)


def _codes(report) -> set[str]:
    return {
        item.code.value if hasattr(item.code, "value") else str(item.code)
        for item in report.feedback
    }


def _family_task(tasks_root: Path, family: str) -> Path:
    for task_dir in sorted(tasks_root.iterdir()):
        if not (task_dir / "task.toml").is_file():
            continue
        data = tomllib.loads((task_dir / "task.toml").read_text())
        if data["task"]["family"] == family:
            return task_dir
    raise AssertionError(f"missing family {family}")


def test_physics_preflight_materializes_required_families_and_manifests(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "mechanism_repair_physics"
    tasks_root = out_dir / "tasks"

    manifest = materialize_benchmark(
        tasks_root=tasks_root,
        tasks_per_family=1,
        base_seed=20260610,
    )
    audit = audit_benchmark(
        tasks_root=tasks_root,
        min_tasks_per_family=1,
        validate=False,
        validate_level3=False,
        scratch_root=out_dir / "scratch",
    )
    splits = freeze_required_splits(
        tasks_root=tasks_root,
        out_dir=out_dir,
        split_seed=20260610,
    )
    methods = build_method_manifest()

    assert manifest["task_count"] == len(REQUIRED_FAMILIES)
    assert audit["structural_passes"] is True
    assert audit["passes"] is True
    assert audit["experiment_ready"] is False
    assert audit["paper_blockers"] == [
        "chrono_level3_validation_not_run",
        "reference_and_negative_validation_not_run",
    ]
    assert audit["family_counts"] == {family: 1 for family in REQUIRED_FAMILIES}
    assert audit["level_counts"] == {"2": 8, "3": 4}
    assert all(
        len(task["constraint_classes"]) >= 3
        and task["negative_control_count"] >= 2
        and task["effective_negative_control_count"] >= 2
        and task["has_hidden_variant"]
        and not task["uses_fake_contact_oracle"]
        for task in audit["tasks"]
    )

    assert set(splits) == {"A", "B", "hidden_perturbation", "external_style"}
    for split_name in ("A", "B"):
        split_manifest = json.loads(Path(splits[split_name]["manifest_path"]).read_text())
        assert split_manifest["seen_families"] == sorted(SPLITS[split_name]["seen"])
        assert split_manifest["unseen_families"] == sorted(SPLITS[split_name]["unseen"])
        assert not (
            set(split_manifest["seen_families"])
            & set(split_manifest["unseen_families"])
        )
        assert split_manifest["splits"]["test"]

    assert methods["required_methods"] == list(REQUIRED_METHODS)
    assert methods["primary_method"] == "mechanical_evolve_ttrl_tool_verified"


def test_level2_reference_passes_trusted_asset_gate_and_negative_fails(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "mechanism_repair_physics"
    tasks_root = out_dir / "tasks"
    materialize_benchmark(
        tasks_root=tasks_root,
        tasks_per_family=1,
        base_seed=20260610,
    )

    task_dir = _family_task(tasks_root, "belt_drive")
    reference = evaluate(
        task_dir,
        task_dir / "reference_solution",
        scratch_dir=out_dir / "reference",
    )
    assert reference.evaluation_valid
    assert reference.hard_gate_passed
    assert reference.score > 0.5
    assert "invalid_mass_properties" not in _codes(reference)
    assert not reference.oracle_is_synthetic

    negative = evaluate(
        task_dir,
        task_dir / "negative_solutions" / "missing_trusted_mass_physics",
        scratch_dir=out_dir / "missing_trusted_mass",
    )
    assert not negative.hard_gate_passed
    assert "invalid_mass_properties" in _codes(negative)


def test_rack_pinion_reference_places_rack_body_on_contact_line(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "mechanism_repair_physics"
    tasks_root = out_dir / "tasks"
    materialize_benchmark(
        tasks_root=tasks_root,
        tasks_per_family=1,
        base_seed=20260610,
    )

    task_dir = _family_task(tasks_root, "rack_pinion")
    design_path = task_dir / "reference_solution" / "design.py"
    spec = importlib.util.spec_from_file_location("rack_reference", design_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    ir = module.build_design(tmp_path / "rack_build")

    rack = next(part for part in ir["parts"] if part["id"] == "rack")
    rack_params = rack["params"]
    assert tuple(rack_params["initial_pose_mm"]) == (0.0, 13.0, 0.0)
    assert tuple(rack_params["chrono_collision"]["center_mm"]) == (
        0.0,
        0.0,
        0.0,
    )
    output_axis = next(joint for joint in ir["joints"] if joint["id"] == "output_axis")
    assert tuple(output_axis["anchor_world_mm"]) == (0.0, 13.0, 0.0)
