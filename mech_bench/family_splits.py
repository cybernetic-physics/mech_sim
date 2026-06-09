"""Family-held-out split helpers for mechanism benchmarks.

The mechanical-design paper claim depends on a frozen family split, not a
task-id split alone.  This module turns the existing ``tasks/*/task.toml``
inventory into canonical mechanism families plus train/val/test task-id files.
"""

from __future__ import annotations

import json
import math
import random
import tomllib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


CANONICAL_FAMILY_RULES: tuple[tuple[str, str], ...] = (
    ("cycloidal", "cycloidal"),
    ("planetary", "planetary"),
    ("rack_pinion", "rack_pinion"),
    ("lead_screw", "lead_screw"),
    ("fourbar", "fourbar"),
    ("planar_4bar", "fourbar"),
    ("slider_crank", "slider_crank"),
    ("cam_follower", "cam_follower"),
    ("belt", "belt"),
    ("timing_belt", "belt"),
    ("chain", "chain"),
)


@dataclass(frozen=True)
class TaskRecord:
    task_id: str
    raw_family: str
    canonical_family: str
    task_dir: Path
    has_chrono_contact_config: bool = False
    chrono_procedural_fallback_disabled: bool = False
    has_trusted_asset_preflight: bool = False
    requires_trusted_mass_properties: bool = False


def canonical_mechanism_family(raw_family: str) -> str:
    value = str(raw_family or "").strip().lower()
    for prefix, canonical in CANONICAL_FAMILY_RULES:
        if value.startswith(prefix):
            return canonical
    return value


def _probe_specs(eval_config: dict[str, Any]) -> list[dict[str, Any]]:
    probes = eval_config.get("probes") or []
    if not isinstance(probes, list):
        return []
    return [probe for probe in probes if isinstance(probe, dict)]


def _task_has_chrono_contact_config(
    task_data: dict[str, Any],
    eval_config: dict[str, Any],
) -> bool:
    if isinstance(task_data.get("chrono_contact"), dict):
        return True
    return any(
        str(probe.get("adapter") or "") == "chrono_contact"
        for probe in _probe_specs(eval_config)
    )


def _chrono_procedural_fallback_disabled(task_data: dict[str, Any]) -> bool:
    chrono = task_data.get("chrono_contact")
    if isinstance(chrono, dict) and chrono.get("procedural_cycloidal_fallback") is False:
        return True
    return False


def _eval_chrono_procedural_fallback_disabled(eval_config: dict[str, Any]) -> bool:
    adapters = eval_config.get("adapters") or {}
    if not isinstance(adapters, dict):
        return False
    chrono = adapters.get("chrono_contact")
    if not isinstance(chrono, dict):
        return False
    return chrono.get("procedural_cycloidal_fallback") is False


def _task_has_trusted_asset_preflight(eval_config: dict[str, Any]) -> bool:
    return any(
        str(probe.get("type") or "") == "trusted_asset_preflight"
        for probe in _probe_specs(eval_config)
    )


def _requires_trusted_mass_properties(eval_config: dict[str, Any]) -> bool:
    return any(
        str(probe.get("type") or "") == "trusted_asset_preflight"
        and probe.get("require_trusted_mass_properties") is True
        for probe in _probe_specs(eval_config)
    )


def load_task_records(tasks_root: Path) -> list[TaskRecord]:
    tasks_root = Path(tasks_root)
    records: list[TaskRecord] = []
    for task_dir in sorted(tasks_root.iterdir()):
        task_toml = task_dir / "task.toml"
        if not task_toml.is_file():
            continue
        data = tomllib.loads(task_toml.read_text())
        eval_config_path = task_dir / "eval_config.toml"
        eval_config = (
            tomllib.loads(eval_config_path.read_text())
            if eval_config_path.is_file()
            else {}
        )
        task = data.get("task", {})
        task_id = str(task.get("id", task_dir.name))
        raw_family = str(task.get("family", task_dir.name))
        records.append(TaskRecord(
            task_id=task_id,
            raw_family=raw_family,
            canonical_family=canonical_mechanism_family(raw_family),
            task_dir=task_dir,
            has_chrono_contact_config=_task_has_chrono_contact_config(
                data, eval_config
            ),
            chrono_procedural_fallback_disabled=(
                _chrono_procedural_fallback_disabled(data)
                or _eval_chrono_procedural_fallback_disabled(eval_config)
            ),
            has_trusted_asset_preflight=(
                _task_has_trusted_asset_preflight(eval_config)
            ),
            requires_trusted_mass_properties=(
                _requires_trusted_mass_properties(eval_config)
            ),
        ))
    return records


def build_family_split_manifest(
    *,
    tasks_root: Path,
    seen_families: Iterable[str],
    unseen_families: Iterable[str],
    seed: int = 20260528,
    seen_train_ratio: float = 0.8,
    seen_val_ratio: float = 0.2,
) -> dict[str, Any]:
    tasks_root = Path(tasks_root)
    if not tasks_root.exists():
        raise FileNotFoundError(tasks_root)

    seen = [canonical_mechanism_family(f) for f in seen_families]
    unseen = [canonical_mechanism_family(f) for f in unseen_families]
    seen_set = set(seen)
    unseen_set = set(unseen)
    overlap = seen_set & unseen_set
    if overlap:
        raise ValueError(
            f"seen and unseen families must be disjoint; overlap={sorted(overlap)}"
        )

    ratio_sum = float(seen_train_ratio + seen_val_ratio)
    if not math.isclose(ratio_sum, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("seen_train_ratio + seen_val_ratio must equal 1.0")

    records = load_task_records(tasks_root)
    by_family: dict[str, list[TaskRecord]] = defaultdict(list)
    for record in records:
        by_family[record.canonical_family].append(record)

    missing = [family for family in sorted(seen_set | unseen_set)
               if family not in by_family]
    if missing:
        raise ValueError(f"no tasks found for canonical families: {missing}")

    rng = random.Random(seed)
    train: list[TaskRecord] = []
    val: list[TaskRecord] = []
    test: list[TaskRecord] = []
    family_manifest: dict[str, Any] = {}

    for family in sorted(seen_set):
        items = sorted(by_family[family], key=lambda r: r.task_id)
        rng.shuffle(items)
        n = len(items)
        n_train = int(round(n * seen_train_ratio))
        n_val = n - n_train
        if n > 1 and n_train == 0:
            n_train = 1
            n_val = n - 1
        if n > 1 and n_val == 0:
            n_val = 1
            n_train = n - 1
        if n_train + n_val != n:
            n_train = max(0, n - n_val)
        train.extend(items[:n_train])
        val.extend(items[n_train:n_train + n_val])
        family_manifest[family] = {
            "role": "seen",
            "n_tasks": n,
            "train_task_ids": [r.task_id for r in items[:n_train]],
            "val_task_ids": [r.task_id for r in items[n_train:n_train + n_val]],
        }

    for family in sorted(unseen_set):
        items = sorted(by_family[family], key=lambda r: r.task_id)
        test.extend(items)
        family_manifest[family] = {
            "role": "unseen",
            "n_tasks": len(items),
            "test_task_ids": [r.task_id for r in items],
        }

    train_ids = sorted(r.task_id for r in train)
    val_ids = sorted(r.task_id for r in val)
    test_ids = sorted(r.task_id for r in test)
    used_families = seen_set | unseen_set
    excluded_records = [r for r in records if r.canonical_family not in used_families]

    return {
        "schema": "mech_bench.family_generalization_split_manifest.v1",
        "seed": int(seed),
        "tasks_root": str(tasks_root),
        "seen_families": sorted(seen_set),
        "unseen_families": sorted(unseen_set),
        "n_tasks_total": len(records),
        "n_tasks_used": len(train_ids) + len(val_ids) + len(test_ids),
        "n_tasks_excluded": len(excluded_records),
        "excluded_families": sorted({r.canonical_family for r in excluded_records}),
        "excluded_task_ids": sorted(r.task_id for r in excluded_records),
        "splits": {
            "train": train_ids,
            "val": val_ids,
            "test": test_ids,
        },
        "family_manifest": family_manifest,
        "task_index": {
            r.task_id: {
                "raw_family": r.raw_family,
                "canonical_family": r.canonical_family,
                "task_dir": str(r.task_dir),
                "has_chrono_contact_config": r.has_chrono_contact_config,
                "chrono_procedural_fallback_disabled": (
                    r.chrono_procedural_fallback_disabled
                ),
                "has_trusted_asset_preflight": r.has_trusted_asset_preflight,
                "requires_trusted_mass_properties": (
                    r.requires_trusted_mass_properties
                ),
            }
            for r in records
        },
    }


def write_family_split_files(manifest: dict[str, Any], out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = manifest["splits"]
    aliases = {
        "train": ["train", "seen_train"],
        "val": ["val", "seen_val"],
        "test": ["test", "unseen_test"],
    }
    for split_name, task_ids in splits.items():
        text = "\n".join(task_ids) + "\n"
        for alias in aliases.get(split_name, [split_name]):
            (out_dir / f"{alias}.txt").write_text(text)

    (out_dir / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
