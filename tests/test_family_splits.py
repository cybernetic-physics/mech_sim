from __future__ import annotations

import json
from pathlib import Path

from mech_bench.family_splits import (
    build_family_split_manifest,
    canonical_mechanism_family,
    write_family_split_files,
)


def _write_task(task_dir: Path, task_id: str, family: str) -> None:
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "task.toml").write_text(
        f"""[task]\nid = \"{task_id}\"\nfamily = \"{family}\"\n"""
    )


def test_canonical_family_mapping_covers_mechanical_families() -> None:
    assert canonical_mechanism_family("planar_4bar") == "fourbar"
    assert canonical_mechanism_family("fourbar_path") == "fourbar"
    assert canonical_mechanism_family("slider_crank_stroke") == "slider_crank"
    assert canonical_mechanism_family("timing_belt_center_distance") == "belt"
    assert canonical_mechanism_family("chain_sprocket_ratio") == "chain"
    assert canonical_mechanism_family("lead_screw_linear_travel") == "lead_screw"


def test_family_split_manifest_freezes_seen_and_unseen_sets(tmp_path: Path) -> None:
    tasks_root = tmp_path / "tasks"
    _write_task(tasks_root / "cycloidal_lowN_stub_s0001", "cycloidal_lowN_stub_s0001", "cycloidal_lowN_stub")
    _write_task(tasks_root / "belt_pulley_ratio_s0001", "belt_pulley_ratio_s0001", "belt_pulley_ratio")
    _write_task(tasks_root / "chain_sprocket_ratio_s0001", "chain_sprocket_ratio_s0001", "chain_sprocket_ratio")
    _write_task(tasks_root / "rack_pinion_conversion_s0001", "rack_pinion_conversion_s0001", "rack_pinion_conversion")
    _write_task(tasks_root / "fourbar_path_t001", "fourbar_path_t001", "planar_4bar")
    _write_task(tasks_root / "planetary_fixed_ring_ratio_analytic_s0001", "planetary_fixed_ring_ratio_analytic_s0001", "planetary_fixed_ring_ratio_analytic")
    _write_task(tasks_root / "lead_screw_linear_travel_s0001", "lead_screw_linear_travel_s0001", "lead_screw_linear_travel")
    _write_task(tasks_root / "cam_follower_contact_stub_s0001", "cam_follower_contact_stub_s0001", "cam_follower_contact_stub")
    _write_task(tasks_root / "slider_crank_stroke_s0001", "slider_crank_stroke_s0001", "slider_crank_stroke")

    manifest = build_family_split_manifest(
        tasks_root=tasks_root,
        seen_families=["cycloidal", "belt", "chain", "rack_pinion", "fourbar"],
        unseen_families=["planetary", "lead_screw", "cam_follower", "slider_crank"],
        seed=7,
    )
    out_dir = tmp_path / "splits"
    write_family_split_files(manifest, out_dir)

    assert (out_dir / "train.txt").is_file()
    assert (out_dir / "val.txt").is_file()
    assert (out_dir / "test.txt").is_file()

    train_ids = (out_dir / "train.txt").read_text().split()
    val_ids = (out_dir / "val.txt").read_text().split()
    test_ids = (out_dir / "test.txt").read_text().split()

    assert "cycloidal_lowN_stub_s0001" in train_ids
    assert "belt_pulley_ratio_s0001" in train_ids
    assert "chain_sprocket_ratio_s0001" in train_ids
    assert "rack_pinion_conversion_s0001" in train_ids
    assert "fourbar_path_t001" in train_ids
    assert "planetary_fixed_ring_ratio_analytic_s0001" in test_ids
    assert "lead_screw_linear_travel_s0001" in test_ids
    assert "cam_follower_contact_stub_s0001" in test_ids
    assert "slider_crank_stroke_s0001" in test_ids
    assert not set(train_ids) & set(test_ids)
    assert not set(val_ids) & set(test_ids)

    loaded = json.loads((out_dir / "split_manifest.json").read_text())
    assert loaded["seen_families"] == ["belt", "chain", "cycloidal", "fourbar", "rack_pinion"]
    assert loaded["unseen_families"] == ["cam_follower", "lead_screw", "planetary", "slider_crank"]
    assert loaded["family_manifest"]["fourbar"]["role"] == "seen"
    assert loaded["family_manifest"]["planetary"]["role"] == "unseen"
