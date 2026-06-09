from __future__ import annotations

import json
from pathlib import Path

from mech_bench.evaluator import evaluate
from mech_bench.generators.base import write_task_directory
from mech_bench.generators.gear_train import CycloidalLayoutRatioGenerator
from scripts.prepare_mechanism_repair_benchmark import (
    PRIMARY_FAMILIES,
    SPLITS,
    audit_benchmark,
    freeze_required_splits,
    materialize_benchmark,
)


def _codes(report) -> set[str]:
    return {
        item.code.value if hasattr(item.code, "value") else str(item.code)
        for item in report.feedback
    }


def test_cycloidal_layout_ratio_reference_and_negatives(tmp_path: Path) -> None:
    task_dir = write_task_directory(
        CycloidalLayoutRatioGenerator().generate(seed=11),
        tmp_path / "tasks",
    )

    reference = evaluate(
        task_dir,
        task_dir / "reference_solution",
        scratch_dir=tmp_path / "reference",
    )
    assert reference.evaluation_valid
    assert reference.hard_gate_passed
    assert reference.score > 0.99

    expected = json.loads((task_dir / "expected_failures.json").read_text())
    for control in expected["controls"]:
        report = evaluate(
            task_dir,
            task_dir / control["submission"],
            scratch_dir=tmp_path / control["id"],
        )
        assert set(control["expected_failure_codes"]).issubset(_codes(report))
        if "expected_hard_gate_passed" in control:
            assert report.hard_gate_passed is bool(
                control["expected_hard_gate_passed"]
            )
        if "expected_score_below" in control:
            assert report.score < float(control["expected_score_below"])


def test_mechanism_repair_preflight_materializes_non_toy_splits(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "mechanism_repair"
    tasks_root = out_dir / "tasks"

    manifest = materialize_benchmark(
        tasks_root=tasks_root,
        tasks_per_family=1,
        base_seed=20260607,
    )
    audit = audit_benchmark(
        tasks_root=tasks_root,
        min_tasks_per_family=1,
        validate=True,
        scratch_root=out_dir / "scratch",
    )
    splits = freeze_required_splits(
        tasks_root=tasks_root,
        out_dir=out_dir,
        split_seed=20260607,
    )

    assert manifest["task_count"] == len(PRIMARY_FAMILIES)
    assert audit["passes"] is True
    assert audit["blockers"] == []
    assert audit["family_counts"] == {family: 1 for family in PRIMARY_FAMILIES}
    assert all(
        len(task["constraint_classes"]) >= 2
        for task in audit["tasks"]
    )
    assert not any(task["uses_fake_contact_oracle"] for task in audit["tasks"])

    assert set(splits) == set(SPLITS)
    for split_name, split in splits.items():
        split_manifest = json.loads(Path(split["manifest_path"]).read_text())
        assert split_manifest["seen_families"] == sorted(
            SPLITS[split_name]["seen"]
        )
        assert split_manifest["unseen_families"] == sorted(
            SPLITS[split_name]["unseen"]
        )
        assert not (
            set(split_manifest["seen_families"])
            & set(split_manifest["unseen_families"])
        )
        assert len(split_manifest["splits"]["test"]) == 4
