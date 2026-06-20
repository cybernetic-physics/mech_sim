from __future__ import annotations

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
    assert "chrono_level3_validation_not_run" in audit["paper_blockers"]
    assert "reference_and_negative_validation_not_run" in audit["paper_blockers"]
    assert audit["family_counts"] == {family: 1 for family in REQUIRED_FAMILIES}
    assert audit["headline_task_count"] == 12
    assert audit["headline_family_counts"] == {
        "belt_drive": 1,
        "cam_follower": 1,
        "chain_drive": 1,
        "cycloidal_reducer": 1,
        "fourbar_linkage": 1,
        "geneva_indexer": 1,
        "lead_screw": 1,
        "planetary_reducer": 1,
        "rack_pinion": 1,
        "shaft_bearing_coupling": 1,
        "slider_crank": 1,
        "spur_compound_gear_train": 1,
    }
    assert audit["diagnostic_task_count"] == 0
    assert audit["level_counts"] == {"2": 9, "3": 3}
    assert all(
        len(task["constraint_classes"]) >= 3
        and task["negative_control_count"] >= 2
        and task["effective_negative_control_count"] >= 2
        and task["has_hidden_variant"]
        and not task["uses_fake_contact_oracle"]
        for task in audit["tasks"]
    )

    assert set(splits) == {
        "A",
        "B",
        "hidden_perturbation",
        "external_style",
        "isomorphic",
    }
    for split_name in ("A", "B"):
        split_manifest = json.loads(Path(splits[split_name]["manifest_path"]).read_text())
        assert split_manifest["seen_families"] == sorted(SPLITS[split_name]["seen"])
        assert split_manifest["unseen_families"] == sorted(SPLITS[split_name]["unseen"])
        assert not (
            set(split_manifest["seen_families"])
            & set(split_manifest["unseen_families"])
        )
        assert split_manifest["headline_only"] is True
        if split_name == "A":
            test_names = {
                Path(path).name for path in split_manifest["splits"]["test"]
            }
            assert len(test_names) == 6
            assert any(name.startswith("planetary_reducer") for name in test_names)
            assert any(name.startswith("lead_screw") for name in test_names)
            assert any(name.startswith("slider_crank") for name in test_names)
            assert any(name.startswith("cycloidal_reducer") for name in test_names)
            assert any(name.startswith("cam_follower") for name in test_names)
            assert any(name.startswith("geneva_indexer") for name in test_names)
        else:
            test_names = {
                Path(path).name for path in split_manifest["splits"]["test"]
            }
            assert len(test_names) == 6
            assert any(name.startswith("belt_drive") for name in test_names)
            assert any(name.startswith("chain_drive") for name in test_names)
            assert any(name.startswith("rack_pinion") for name in test_names)
            assert any(name.startswith("cycloidal_reducer") for name in test_names)
            assert any(
                name.startswith("spur_compound_gear_train")
                for name in test_names
            )
            assert any(name.startswith("geneva_indexer") for name in test_names)

    isomorphic_manifest = json.loads(
        Path(splits["isomorphic"]["manifest_path"]).read_text()
    )
    assert isomorphic_manifest["split_name"] == "isomorphic"
    assert isomorphic_manifest["headline_only"] is True
    assert len(isomorphic_manifest["splits"]["test"]) == len(audit["tasks"])
    assert isomorphic_manifest["isomorphic_transform_contract"]
    for split_name in ("hidden_perturbation", "isomorphic"):
        split_manifest = json.loads(
            Path(splits[split_name]["manifest_path"]).read_text()
        )
        variant_path = Path(split_manifest["splits"]["test"][0])
        assert "anti_shortcut_variants" in variant_path.parts
        assert variant_path.name in {
            Path(task["task_dir"]).name for task in audit["tasks"]
        }
        assert (variant_path / "anti_shortcut_variant.json").is_file()
        active_eval = (variant_path / "eval_config.toml").read_text()
        assert "hidden_variant = true" in active_eval

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


def test_rack_pinion_reference_uses_pitch_radius_velocity_probe(
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
    reference = evaluate(
        task_dir,
        task_dir / "reference_solution",
        scratch_dir=out_dir / "rack_reference",
    )
    assert reference.evaluation_valid
    assert reference.hard_gate_passed
    assert reference.score > 0.999999

    negative = evaluate(
        task_dir,
        task_dir / "negative_solutions" / "wrong_pinion_geometry",
        scratch_dir=out_dir / "rack_wrong_pinion_geometry",
    )
    assert negative.evaluation_valid
    assert negative.hard_gate_passed
    assert "wrong_ratio" in _codes(negative)


def test_gear_family_references_use_tooth_geometry_velocity_probes(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "mechanism_repair_physics"
    tasks_root = out_dir / "tasks"
    materialize_benchmark(
        tasks_root=tasks_root,
        tasks_per_family=1,
        base_seed=20260610,
    )

    cases = [
        ("spur_compound_gear_train", "wrong_output_gear_geometry"),
        ("planetary_reducer", "wrong_ring_geometry"),
    ]
    for family, negative_name in cases:
        task_dir = _family_task(tasks_root, family)
        reference = evaluate(
            task_dir,
            task_dir / "reference_solution",
            scratch_dir=out_dir / f"{family}_reference",
        )
        assert reference.evaluation_valid
        assert reference.hard_gate_passed
        assert reference.score > 0.99

        negative = evaluate(
            task_dir,
            task_dir / "negative_solutions" / negative_name,
            scratch_dir=out_dir / f"{family}_{negative_name}",
        )
        assert negative.evaluation_valid
        assert negative.hard_gate_passed
        assert "wrong_ratio" in _codes(negative)


def test_shaft_bearing_coupling_reference_uses_geometry_velocity_probe(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "mechanism_repair_physics"
    tasks_root = out_dir / "tasks"
    materialize_benchmark(
        tasks_root=tasks_root,
        tasks_per_family=1,
        base_seed=20260610,
    )

    task_dir = _family_task(tasks_root, "shaft_bearing_coupling")
    reference = evaluate(
        task_dir,
        task_dir / "reference_solution",
        scratch_dir=out_dir / "shaft_reference",
    )
    assert reference.evaluation_valid
    assert reference.hard_gate_passed
    assert reference.score > 0.99

    negative = evaluate(
        task_dir,
        task_dir / "negative_solutions" / "wrong_output_shaft_geometry",
        scratch_dir=out_dir / "shaft_wrong_geometry",
    )
    assert negative.evaluation_valid
    assert negative.hard_gate_passed
    assert {"invalid_artifact", "simulator_divergence"} & _codes(negative)
