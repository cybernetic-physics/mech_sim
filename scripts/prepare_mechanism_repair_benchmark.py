#!/usr/bin/env python3
"""Prepare and audit the MechanismRepair-TTRL benchmark.

This is the experiment-facing benchmark preflight for ``goals.md``. It
materializes a balanced non-static mechanism task suite, validates that tasks
are not one-parameter toys, optionally runs reference and negative-control
checks, and freezes the two required family-held-out splits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tomllib
from collections import defaultdict
from pathlib import Path
from typing import Any

from mech_bench.evaluator import evaluate
from mech_bench.family_splits import (
    build_family_split_manifest,
    canonical_mechanism_family,
    load_task_records,
    write_family_split_files,
)
from mech_bench.generators.base import TaskGenerator, write_task_directory
from mech_bench.generators.benchmark_suite import generator_for


PRIMARY_FAMILIES = (
    "belt",
    "chain",
    "rack_pinion",
    "lead_screw",
    "planetary",
    "fourbar",
    "slider_crank",
    "cycloidal",
)

REQUIRED_METHODS = (
    "frozen_model",
    "verifier_gated",
    "no_update_search",
    "llm_evolve_no_update",
    "sft_model",
    "mechanical_evolve_ttrl",
)

PRIMARY_BASELINE = "llm_evolve_no_update"
PRIMARY_METHOD = "mechanical_evolve_ttrl"
PRIMARY_BUDGET = 32
EVAL_SEEDS = (20260607, 20260608, 20260609)
SUCCESS_DELTA_PCT = 15.0

SPLITS = {
    "A": {
        "seen": ("belt", "chain", "rack_pinion", "fourbar"),
        "unseen": ("planetary", "lead_screw", "slider_crank", "cycloidal"),
    },
    "B": {
        "seen": ("planetary", "lead_screw", "fourbar", "slider_crank"),
        "unseen": ("belt", "chain", "rack_pinion", "cycloidal"),
    },
}

GENERATOR_FAMILIES = {
    "belt": (
        "belt_pulley_ratio",
        "timing_belt_center_distance",
    ),
    "chain": (
        "chain_sprocket_ratio",
    ),
    "rack_pinion": (
        "rack_pinion_conversion",
        "rack_pinion_force_direction",
    ),
    "lead_screw": (
        "lead_screw_linear_travel",
    ),
    "planetary": (
        "planetary_fixed_ring_ratio_analytic",
        "planetary_fixed_sun_ratio_analytic",
    ),
    "fourbar": (
        "fourbar_path",
        "fourbar_wiper_arc",
        "fourbar_straight_line_approx",
        "fourbar_dwell_path",
        "fourbar_pump_handle",
    ),
    "slider_crank": (
        "slider_crank_stroke",
        "slider_crank_stroke_precision",
        "slider_crank_quick_return_proxy",
    ),
    "cycloidal": (
        "cycloidal_layout_ratio",
    ),
}

TOPOLOGY_PROBES = {
    "dof_grubler",
    "required_ports",
}
FUNCTIONAL_PROBES = {
    "analytic_param_check",
    "path_trace_chamfer",
    "port_velocity_ratio",
}
ARTIFACT_PROBES = {
    "trusted_asset_preflight",
    "printability_dfam",
    "safety_factor",
    "swept_collision",
}
CONTACT_PROBES = {
    "contact_engagement",
    "lockup",
    "torque_load_trial",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="runs/mechanism_repair_ttrl_final")
    parser.add_argument("--tasks-per-family", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=20260607)
    parser.add_argument("--split-seed", type=int, default=20260607)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-validation", action="store_true",
                        help="materialize and structurally audit only; final "
                             "experiment preflight must validate references")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks_root = out_dir / "tasks"
    if tasks_root.exists() and args.overwrite:
        shutil.rmtree(tasks_root)
    tasks_root.mkdir(parents=True, exist_ok=True)

    manifest = materialize_benchmark(
        tasks_root=tasks_root,
        tasks_per_family=max(1, int(args.tasks_per_family)),
        base_seed=int(args.base_seed),
    )
    audit = audit_benchmark(
        tasks_root=tasks_root,
        min_tasks_per_family=max(1, int(args.tasks_per_family)),
        validate=not args.skip_validation,
        scratch_root=out_dir / "preflight_scratch",
    )
    split_manifests = freeze_required_splits(
        tasks_root=tasks_root,
        out_dir=out_dir,
        split_seed=int(args.split_seed),
    )
    verifier_manifest = build_verifier_manifest(
        tasks_root=tasks_root,
        audit=audit,
    )

    manifest.update({
        "audit": audit,
        "splits": {
            key: split["manifest_path"]
            for key, split in split_manifests.items()
        },
        "verifier_manifest_path": str(
            out_dir / "verifier_manifest.json"
        ),
        "method_manifest_path": str(
            out_dir / "method_manifest.json"
        ),
        "experiment_ready": bool(audit["passes"]),
    })
    write_json(out_dir / "benchmark_manifest.json", manifest)
    write_json(out_dir / "verifier_manifest.json", verifier_manifest)
    write_json(out_dir / "method_manifest.json", build_method_manifest())

    print(json.dumps({
        "benchmark_manifest": str(out_dir / "benchmark_manifest.json"),
        "verifier_manifest": str(out_dir / "verifier_manifest.json"),
        "method_manifest": str(out_dir / "method_manifest.json"),
        "tasks_root": str(tasks_root),
        "passes": audit["passes"],
        "blockers": audit["blockers"],
        "family_counts": audit["family_counts"],
    }, indent=2, sort_keys=True))
    return 0 if audit["passes"] else 2


def materialize_benchmark(
    *,
    tasks_root: Path,
    tasks_per_family: int,
    base_seed: int,
) -> dict[str, Any]:
    task_rows: list[dict[str, Any]] = []
    for fam_index, family in enumerate(PRIMARY_FAMILIES):
        generator_names = GENERATOR_FAMILIES[family]
        for task_index in range(tasks_per_family):
            generator_name = generator_names[task_index % len(generator_names)]
            generator_cls = generator_for(generator_name)
            generator: TaskGenerator = generator_cls()
            seed = base_seed + fam_index * 1000 + task_index
            task = generator.generate(seed=seed)
            task_dir = write_task_directory(task, tasks_root)
            task_rows.append({
                "task_id": task.task_id,
                "canonical_family": family,
                "raw_family": task.family,
                "generator": (
                    f"{generator_cls.__module__}.{generator_cls.__name__}"
                ),
                "seed": seed,
                "tier": generator.tier,
                "task_dir": str(task_dir),
            })
    return {
        "schema": "mechanism_repair_ttrl.benchmark_manifest.v1",
        "primary_families": list(PRIMARY_FAMILIES),
        "tasks_per_family_target": int(tasks_per_family),
        "base_seed": int(base_seed),
        "tasks_root": str(tasks_root),
        "task_count": len(task_rows),
        "tasks": task_rows,
    }


def audit_benchmark(
    *,
    tasks_root: Path,
    min_tasks_per_family: int,
    validate: bool,
    scratch_root: Path,
) -> dict[str, Any]:
    records = load_task_records(tasks_root)
    family_counts: dict[str, int] = defaultdict(int)
    task_audits: list[dict[str, Any]] = []
    blockers: list[str] = []

    for record in records:
        family_counts[record.canonical_family] += 1
        task_audit = audit_task(record.task_dir)
        task_audit.update({
            "task_id": record.task_id,
            "raw_family": record.raw_family,
            "canonical_family": record.canonical_family,
            "task_dir": str(record.task_dir),
        })
        if validate:
            validation = validate_task(record.task_dir, scratch_root)
            task_audit["validation"] = validation
        task_audits.append(task_audit)

    for family in PRIMARY_FAMILIES:
        count = int(family_counts.get(family, 0))
        if count < min_tasks_per_family:
            blockers.append(
                f"{family}: only {count} tasks; need {min_tasks_per_family}"
            )

    for task in task_audits:
        if len(task["constraint_classes"]) < 2:
            blockers.append(
                f"{task['task_id']}: not non-toy; constraint_classes="
                f"{task['constraint_classes']}"
            )
        if not task["has_negative_control"]:
            blockers.append(f"{task['task_id']}: no negative control")
        if validate:
            validation = task.get("validation", {})
            if not validation.get("reference_passed", False):
                blockers.append(
                    f"{task['task_id']}: reference failed "
                    f"{validation.get('reference_codes')}"
                )
            for item in validation.get("negative_failures", []):
                blockers.append(f"{task['task_id']}: {item}")

    return {
        "schema": "mechanism_repair_ttrl.benchmark_audit.v1",
        "tasks_root": str(tasks_root),
        "validate": bool(validate),
        "primary_families": list(PRIMARY_FAMILIES),
        "min_tasks_per_family": int(min_tasks_per_family),
        "family_counts": dict(sorted(family_counts.items())),
        "task_count": len(task_audits),
        "tasks": sorted(task_audits, key=lambda row: row["task_id"]),
        "blockers": blockers,
        "passes": not blockers,
    }


def audit_task(task_dir: Path) -> dict[str, Any]:
    task_toml = tomllib.loads((task_dir / "task.toml").read_text())
    eval_config = tomllib.loads((task_dir / "eval_config.toml").read_text())
    expected_path = task_dir / "expected_failures.json"
    expected = (
        json.loads(expected_path.read_text())
        if expected_path.is_file()
        else {}
    )
    probes = [
        probe for probe in eval_config.get("probes", [])
        if isinstance(probe, dict)
    ]
    probe_types = [str(probe.get("type") or "") for probe in probes]
    classes = classify_constraint_classes(eval_config)
    task_id = str((task_toml.get("task") or {}).get("id", task_dir.name))
    return {
        "task_id": task_id,
        "tier": str((task_toml.get("task") or {}).get("tier", "")),
        "probe_types": probe_types,
        "constraint_classes": sorted(classes),
        "has_negative_control": bool(expected.get("controls")),
        "negative_control_count": len(expected.get("controls", []) or []),
        "task_hash": hash_task_contract(task_dir),
        "verifier_level": infer_verifier_level(eval_config),
        "uses_fake_contact_oracle": uses_fake_contact_oracle(eval_config),
    }


def classify_constraint_classes(eval_config: dict[str, Any]) -> set[str]:
    classes: set[str] = set()
    probes = [
        probe for probe in eval_config.get("probes", [])
        if isinstance(probe, dict)
    ]
    for probe in probes:
        probe_type = str(probe.get("type") or "")
        if probe_type in TOPOLOGY_PROBES:
            classes.add("topology_or_mobility")
        if probe_type in FUNCTIONAL_PROBES:
            classes.add("functional_behavior")
        if probe_type in ARTIFACT_PROBES:
            classes.add("artifact_validity")
        if probe_type in CONTACT_PROBES:
            classes.add("contact_or_dynamics")
    return classes


def infer_verifier_level(eval_config: dict[str, Any]) -> int:
    classes = classify_constraint_classes(eval_config)
    if "contact_or_dynamics" in classes:
        return 3
    if "artifact_validity" in classes:
        return 2
    return 1


def uses_fake_contact_oracle(eval_config: dict[str, Any]) -> bool:
    adapters = eval_config.get("adapters") or {}
    if isinstance(adapters, dict):
        fake_cfg = adapters.get("fake_contact_oracle") or {}
        if isinstance(fake_cfg, dict) and fake_cfg.get("enabled") is True:
            return True
    for probe in eval_config.get("probes", []) or []:
        if isinstance(probe, dict) and probe.get("adapter") == "fake_contact_oracle":
            return True
    return False


def validate_task(task_dir: Path, scratch_root: Path) -> dict[str, Any]:
    scratch_root.mkdir(parents=True, exist_ok=True)
    reference = evaluate(
        task_dir,
        task_dir / "reference_solution",
        scratch_dir=scratch_root / f"{task_dir.name}_reference",
    )
    reference_codes = feedback_codes(reference)
    negative_failures: list[str] = []
    expected = json.loads((task_dir / "expected_failures.json").read_text())
    for control in expected.get("controls", []) or []:
        submission = task_dir / str(control["submission"])
        report = evaluate(
            task_dir,
            submission,
            scratch_dir=scratch_root / f"{task_dir.name}_{control['id']}",
        )
        codes = set(feedback_codes(report))
        expected_codes = set(control.get("expected_failure_codes", []) or [])
        if not expected_codes.issubset(codes):
            negative_failures.append(
                f"{control['id']} expected codes "
                f"{sorted(expected_codes)} got {sorted(codes)}"
            )
        if "expected_hard_gate_passed" in control:
            expected_gate = bool(control["expected_hard_gate_passed"])
            if bool(report.hard_gate_passed) != expected_gate:
                negative_failures.append(
                    f"{control['id']} expected hard_gate={expected_gate} "
                    f"got {bool(report.hard_gate_passed)}"
                )
        if "expected_score_below" in control:
            score_limit = float(control["expected_score_below"])
            if float(report.score) >= score_limit:
                negative_failures.append(
                    f"{control['id']} expected score < {score_limit} "
                    f"got {float(report.score)}"
                )
    return {
        "reference_passed": (
            bool(reference.evaluation_valid)
            and bool(reference.hard_gate_passed)
            and float(reference.score) > 0.5
        ),
        "reference_evaluation_valid": bool(reference.evaluation_valid),
        "reference_hard_gate_passed": bool(reference.hard_gate_passed),
        "reference_score": float(reference.score),
        "reference_codes": reference_codes,
        "reference_oracle_is_synthetic": bool(reference.oracle_is_synthetic),
        "negative_failures": negative_failures,
    }


def feedback_codes(report: Any) -> list[str]:
    return [
        item.code.value if hasattr(item.code, "value") else str(item.code)
        for item in report.feedback
    ]


def freeze_required_splits(
    *,
    tasks_root: Path,
    out_dir: Path,
    split_seed: int,
) -> dict[str, dict[str, Any]]:
    split_outputs: dict[str, dict[str, Any]] = {}
    for name, spec in SPLITS.items():
        split_dir = out_dir / f"splits_{name}"
        manifest = build_family_split_manifest(
            tasks_root=tasks_root,
            seen_families=spec["seen"],
            unseen_families=spec["unseen"],
            seed=split_seed + (0 if name == "A" else 1),
        )
        write_family_split_files(manifest, split_dir)
        manifest_path = out_dir / f"split_manifest_{name}.json"
        write_json(manifest_path, manifest)
        split_outputs[name] = {
            "manifest_path": str(manifest_path),
            "split_dir": str(split_dir),
            "seen_families": list(spec["seen"]),
            "unseen_families": list(spec["unseen"]),
            "n_train": len(manifest["splits"]["train"]),
            "n_val": len(manifest["splits"]["val"]),
            "n_test": len(manifest["splits"]["test"]),
        }
    return split_outputs


def build_verifier_manifest(
    *,
    tasks_root: Path,
    audit: dict[str, Any],
) -> dict[str, Any]:
    level_counts: dict[str, int] = defaultdict(int)
    fake_oracle_tasks: list[str] = []
    for task in audit["tasks"]:
        level_counts[str(task["verifier_level"])] += 1
        if task["uses_fake_contact_oracle"]:
            fake_oracle_tasks.append(task["task_id"])
    return {
        "schema": "mechanism_repair_ttrl.verifier_manifest.v1",
        "tasks_root": str(tasks_root),
        "verifier_levels": dict(sorted(level_counts.items())),
        "constraint_class_policy": {
            "non_toy_min_classes": 2,
            "classes": [
                "topology_or_mobility",
                "functional_behavior",
                "artifact_validity",
                "contact_or_dynamics",
            ],
        },
        "fake_oracle_tasks": sorted(fake_oracle_tasks),
        "main_claim_allows_fake_oracle": False,
    }


def build_method_manifest() -> dict[str, Any]:
    return {
        "schema": "mechanism_repair_ttrl.method_manifest.v1",
        "required_methods": list(REQUIRED_METHODS),
        "primary_method": PRIMARY_METHOD,
        "primary_baseline": PRIMARY_BASELINE,
        "primary_budget_verifier_calls": PRIMARY_BUDGET,
        "budget_curve": [8, 16, 32, 64],
        "eval_seeds": list(EVAL_SEEDS),
        "success_threshold": {
            "verified_repair_success_abs_delta_pct": SUCCESS_DELTA_PCT,
            "paired_reward_delta_must_be_positive": True,
            "paired_statistical_support": (
                "bootstrap_ci95_low_gt_0_or_paired_test_p_lte_0.05"
            ),
        },
        "causal_contrast": {
            "same_base_model": True,
            "same_tasks": True,
            "same_task_order": True,
            "same_prompts": True,
            "same_verifier_feedback": True,
            "same_actual_verifier_calls": True,
            "difference": "mechanical_evolve_ttrl updates LoRA weights; "
                          "llm_evolve_no_update does not",
        },
        "method_roles": {
            "frozen_model": "base model, no updates",
            "verifier_gated": "low-temperature verifier-gated no-update",
            "no_update_search": "high-temperature best-of-K no-update",
            "llm_evolve_no_update": "primary no-update feedback baseline",
            "sft_model": "seen-family supervised adapter",
            "mechanical_evolve_ttrl": "verifier-derived GRPO/LoRA updates",
        },
    }


def hash_task_contract(task_dir: Path) -> str:
    h = hashlib.sha256()
    for name in (
        "task.toml",
        "eval_config.toml",
        "eval_config.public.toml",
        "eval_config.hidden.toml",
        "prompt.md",
        "expected_failures.json",
    ):
        path = task_dir / name
        if path.is_file():
            h.update(name.encode("utf-8"))
            h.update(b"\0")
            h.update(path.read_bytes())
            h.update(b"\0")
    return h.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
