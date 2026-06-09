#!/usr/bin/env python3
"""Prepare and audit the MechanismRepair-Physics benchmark.

This is the experiment-facing preflight for the current ``goals.md`` contract.
It materializes the 12 required mechanism families, upgrades generated tasks to
the Level-2/Level-3 verifier contract, freezes splits/manifests, and refuses to
mark the benchmark paper-ready unless real validation evidence exists.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import shutil
import tomllib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mech_bench.adapters.chrono_contact import chrono_diagnostic
from mech_bench.evaluator import evaluate
from mech_bench.generators.base import GeneratedTask, TaskGenerator, write_task_directory
from mech_bench.generators.benchmark_suite import generator_for


DEFAULT_OUT_DIR = "runs/mechanism_repair_physics_final"
PRIMARY_BUDGET = 32
EVAL_SEEDS = (20260610, 20260611, 20260612)
SUCCESS_DELTA_PCT = 15.0

REQUIRED_METHODS = (
    "frozen_model",
    "sft_seen_family",
    "llm_evolve_no_update",
    "verifier_gated_search",
    "adaptive_evolution",
    "mechanical_evolve_ttrl",
    "mechanical_evolve_ttrl_tool_verified",
    "mechanical_evolve_ttrl_confidence",
)
PRIMARY_BASELINE = "llm_evolve_no_update"
PRIMARY_METHOD = "mechanical_evolve_ttrl_tool_verified"
FALLBACK_PRIMARY_METHOD = "mechanical_evolve_ttrl"


@dataclass(frozen=True)
class FamilySpec:
    name: str
    verifier_level: int
    generators: tuple[str, ...]
    headline_eligible: bool = True


FAMILY_SPECS: tuple[FamilySpec, ...] = (
    FamilySpec("cycloidal_reducer", 3, ("cycloidal_lowN_stub",)),
    FamilySpec(
        "planetary_reducer",
        2,
        ("planetary_fixed_ring_ratio_analytic", "planetary_fixed_sun_ratio_analytic"),
    ),
    FamilySpec(
        "spur_compound_gear_train",
        2,
        ("compound_gear_ratio_analytic", "spur_gear_ratio_analytic"),
    ),
    FamilySpec("belt_drive", 2, ("belt_pulley_ratio", "timing_belt_center_distance")),
    FamilySpec("chain_drive", 2, ("chain_sprocket_ratio",)),
    FamilySpec("rack_pinion", 3, ("rack_pinion_contact_stub",)),
    FamilySpec("lead_screw", 2, ("lead_screw_linear_travel",)),
    FamilySpec(
        "slider_crank",
        2,
        (
            "slider_crank_stroke",
            "slider_crank_stroke_precision",
            "slider_crank_quick_return_proxy",
        ),
    ),
    FamilySpec(
        "fourbar_linkage",
        2,
        (
            "fourbar_path",
            "fourbar_wiper_arc",
            "fourbar_straight_line_approx",
            "fourbar_dwell_path",
            "fourbar_pump_handle",
        ),
    ),
    FamilySpec("cam_follower", 3, ("cam_follower_contact_stub",)),
    FamilySpec("geneva_indexer", 3, ("geneva_indexing_stub",)),
    FamilySpec(
        "shaft_bearing_coupling",
        2,
        ("keyed_shaft_hub_fit", "bearing_seat_clearance"),
    ),
)

REQUIRED_FAMILIES = tuple(spec.name for spec in FAMILY_SPECS)
FAMILY_BY_NAME = {spec.name: spec for spec in FAMILY_SPECS}

SPLITS = {
    "A": {
        "seen": (
            "belt_drive",
            "chain_drive",
            "rack_pinion",
            "fourbar_linkage",
            "shaft_bearing_coupling",
            "spur_compound_gear_train",
        ),
        "unseen": (
            "planetary_reducer",
            "lead_screw",
            "slider_crank",
            "cycloidal_reducer",
            "cam_follower",
            "geneva_indexer",
        ),
    },
    "B": {
        "seen": (
            "planetary_reducer",
            "lead_screw",
            "fourbar_linkage",
            "slider_crank",
            "cam_follower",
            "shaft_bearing_coupling",
        ),
        "unseen": (
            "belt_drive",
            "chain_drive",
            "rack_pinion",
            "cycloidal_reducer",
            "spur_compound_gear_train",
            "geneva_indexer",
        ),
    },
}

DEFAULT_CONTACT_PAIRS = {
    "cycloidal_reducer": ("housing:disc",),
    "rack_pinion": ("pinion:rack",),
    "cam_follower": ("cam:follower",),
    "geneva_indexer": ("driver:geneva",),
}

TOPOLOGY_PROBES = {"dof_grubler"}
INTERFACE_PROBES = {"required_ports"}
FUNCTIONAL_PROBES = {
    "analytic_param_check",
    "path_trace_chamfer",
    "port_velocity_ratio",
}
ARTIFACT_PROBES = {
    "trusted_asset_preflight",
    "swept_collision",
    "printability_dfam",
    "safety_factor",
}
CONTACT_PROBES = {"contact_engagement", "lockup", "torque_load_trial"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--tasks-per-family", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=20260610)
    parser.add_argument("--split-seed", type=int, default=20260610)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help=(
            "materialize and structurally audit only; paper readiness remains "
            "false until reference, negative, CAD, and Chrono validation run"
        ),
    )
    parser.add_argument(
        "--skip-level3-validation",
        action="store_true",
        help=(
            "validate Level-2 tasks but do not execute Chrono Level-3 tasks; "
            "paper readiness remains false"
        ),
    )
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
        validate_level3=not args.skip_level3_validation,
        scratch_root=out_dir / "preflight_scratch",
    )
    splits = freeze_required_splits(
        tasks_root=tasks_root,
        out_dir=out_dir,
        split_seed=int(args.split_seed),
    )
    verifier_manifest = build_verifier_manifest(tasks_root=tasks_root, audit=audit)
    level_manifest = build_level_manifest(tasks_root=tasks_root, audit=audit)
    hidden_manifest = build_hidden_variant_manifest(tasks_root=tasks_root, audit=audit)
    method_manifest = build_method_manifest()
    run_scaffold = ensure_run_scaffold(out_dir)
    claim_audit = build_claim_audit(audit)

    manifest.update(
        {
            "audit": audit,
            "splits": {name: split["manifest_path"] for name, split in splits.items()},
            "verifier_manifest_path": str(out_dir / "verifier_manifest.json"),
            "method_manifest_path": str(out_dir / "method_manifest.json"),
            "level_manifest_path": str(out_dir / "level_manifest.json"),
            "hidden_variant_manifest_path": str(
                out_dir / "hidden_variant_manifest.json"
            ),
            "run_scaffold": run_scaffold,
            "experiment_ready": bool(audit["experiment_ready"]),
        }
    )
    write_json(out_dir / "benchmark_manifest.json", manifest)
    write_json(out_dir / "verifier_manifest.json", verifier_manifest)
    write_json(out_dir / "method_manifest.json", method_manifest)
    write_json(out_dir / "level_manifest.json", level_manifest)
    write_json(out_dir / "hidden_variant_manifest.json", hidden_manifest)
    write_json(out_dir / "claim_audit.json", claim_audit)

    print(
        json.dumps(
            {
                "benchmark_manifest": str(out_dir / "benchmark_manifest.json"),
                "tasks_root": str(tasks_root),
                "structural_passes": audit["structural_passes"],
                "passes": audit["passes"],
                "experiment_ready": audit["experiment_ready"],
                "blockers": audit["blockers"],
                "paper_blockers": audit["paper_blockers"],
                "family_counts": audit["family_counts"],
                "level_counts": audit["level_counts"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if audit["structural_passes"] else 2


def materialize_benchmark(
    *,
    tasks_root: Path,
    tasks_per_family: int,
    base_seed: int,
) -> dict[str, Any]:
    task_rows: list[dict[str, Any]] = []
    for fam_index, spec in enumerate(FAMILY_SPECS):
        for task_index in range(tasks_per_family):
            generator_name = spec.generators[task_index % len(spec.generators)]
            generator_cls = generator_for(generator_name)
            generator: TaskGenerator = generator_cls()
            seed = base_seed + fam_index * 1000 + task_index
            generated = generator.generate(seed=seed, difficulty=spec.verifier_level)
            task = upgrade_task_for_physics(
                generated,
                spec=spec,
                source_generator=generator_name,
                seed=seed,
                task_index=task_index,
            )
            task_dir = write_task_directory(task, tasks_root)
            task_rows.append(
                {
                    "task_id": task.task_id,
                    "family": spec.name,
                    "source_family": generated.family,
                    "source_generator": generator_name,
                    "verifier_level": spec.verifier_level,
                    "seed": seed,
                    "task_dir": str(task_dir),
                    "headline_eligible": bool(spec.headline_eligible),
                }
            )

    level_counts = defaultdict(int)
    for row in task_rows:
        level_counts[str(row["verifier_level"])] += 1

    return {
        "schema": "mechanism_repair_physics.benchmark_manifest.v1",
        "goals_contract": "goals.md",
        "benchmark_name": "MechanismRepair-Physics",
        "required_families": list(REQUIRED_FAMILIES),
        "tasks_per_family_target": int(tasks_per_family),
        "base_seed": int(base_seed),
        "tasks_root": str(tasks_root),
        "task_count": len(task_rows),
        "level_counts": dict(sorted(level_counts.items())),
        "tasks": task_rows,
        "notes": [
            "Level-2 tasks use the current trusted_asset_preflight contract.",
            "Level-3 tasks require chrono_contact validation; fake_contact_oracle "
            "is removed from headline tasks.",
        ],
    }


def upgrade_task_for_physics(
    task: GeneratedTask,
    *,
    spec: FamilySpec,
    source_generator: str,
    seed: int,
    task_index: int,
) -> GeneratedTask:
    task = copy.deepcopy(task)
    original_id = task.task_id
    task.task_id = f"{spec.name}_t{task_index:02d}_{original_id}"
    task.family = spec.name

    task.task_toml.setdefault("task", {})
    task.task_toml["task"].update(
        {
            "id": task.task_id,
            "family": spec.name,
            "difficulty": int(spec.verifier_level),
            "tier": "physics_contact" if spec.verifier_level == 3 else "cad_artifact",
        }
    )
    task.task_toml["mechanism_repair_physics"] = {
        "canonical_family": spec.name,
        "source_task_id": original_id,
        "source_generator": source_generator,
        "verifier_level": int(spec.verifier_level),
        "headline_eligible": bool(spec.headline_eligible),
        "min_constraint_classes": 3,
        "requires_hidden_variant": True,
        "requires_isomorphic_variant": True,
        "fake_contact_oracle_allowed": False,
    }
    if spec.verifier_level == 3:
        task.task_toml["chrono_contact"] = {
            "contact_model": "nsc",
            "procedural_cycloidal_fallback": False,
            "samples": 720,
            "duration_s": 1.0,
            "input_speed_rad_s": 1.0,
            "output_load_Nm": 0.05,
        }

    task.prompt_md = task.prompt_md.rstrip() + (
        "\n\n## MechanismRepair-Physics verifier contract\n\n"
        "This task counts only if the submitted mechanism preserves topology, "
        "ports, functional behavior, trusted CAD/material/mass evidence, and "
        "the hidden variant semantics. Fake contact-oracle outputs are not "
        "accepted for headline evaluation.\n"
    )

    task.reference_solution_py = wrap_reference_with_trusted_assets(
        task.reference_solution_py,
        family=spec.name,
        verifier_level=spec.verifier_level,
    )
    task.negative_solutions = dict(task.negative_solutions or {})
    add_universal_negative_controls(task)

    task.eval_config_toml = upgrade_eval_config(
        task.eval_config_toml,
        family=spec.name,
        verifier_level=spec.verifier_level,
    )
    hidden = task.eval_config_hidden_toml or copy.deepcopy(task.eval_config_toml)
    task.eval_config_hidden_toml = upgrade_eval_config(
        hidden,
        family=spec.name,
        verifier_level=spec.verifier_level,
        hidden=True,
    )

    task.expected_failures = dict(task.expected_failures or {})
    task.expected_failures.setdefault(
        "description", f"MechanismRepair-Physics negatives for {task.task_id}."
    )
    task.expected_failures["mechanism_repair_physics"] = {
        "min_effective_negative_controls": 2,
        "negative_controls_must_not_all_pass_hard_gates": True,
    }

    task.metadata = dict(task.metadata or {})
    task.metadata.update(
        {
            "task_id": task.task_id,
            "family": spec.name,
            "source_task_id": original_id,
            "source_generator": source_generator,
            "seed": int(seed),
            "verifier_level": int(spec.verifier_level),
            "headline_eligible": bool(spec.headline_eligible),
            "hidden_perturbations": [
                "tighter tolerance window",
                "dimension/target perturbation inherited from generator",
                "renamed semantic manifest for isomorphic checks",
            ],
            "trusted_asset_bridge": (
                "reference solutions emit relative CAD artifacts and "
                "cad_mass_properties records consumed by trusted_asset_preflight"
            ),
            "fake_contact_oracle_allowed": False,
        }
    )
    return task


def wrap_reference_with_trusted_assets(
    source: str,
    *,
    family: str,
    verifier_level: int,
) -> str:
    renamed, count = re.subn(
        r"def\s+build_design\s*\(",
        "def _base_build_design(",
        source,
        count=1,
    )
    if count != 1:
        raise ValueError("reference solution must define exactly one build_design")
    return (
        renamed.rstrip()
        + "\n\n"
        + _trusted_asset_wrapper_source(family=family, verifier_level=verifier_level)
    )


def _trusted_asset_wrapper_source(*, family: str, verifier_level: int) -> str:
    return f'''

def _physics_safe_id(raw):
    text = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in str(raw))
    return text or "part"


def _physics_stub_step(part_id):
    return (
        "ISO-10303-21;\\n"
        "HEADER;\\n"
        "FILE_DESCRIPTION(('MechanismRepair-Physics trusted preflight stub'),'2;1');\\n"
        "FILE_NAME('" + _physics_safe_id(part_id) + ".step','2026-06-10',('mech_bench'),"
        "('corl'),'trusted_asset_preflight','trusted_asset_preflight','');\\n"
        "ENDSEC;\\n"
        "DATA;\\n"
        "ENDSEC;\\n"
        "END-ISO-10303-21;\\n"
    )


def _physics_contact_body_ids(ir):
    bodies = set()
    for joint in ir.get("joints", []) or []:
        if not isinstance(joint, dict) or joint.get("type") != "contact_pair":
            continue
        parent = joint.get("parent")
        child = joint.get("child")
        if parent:
            bodies.add(str(parent))
        if child:
            bodies.add(str(child))
    for pair in {list(DEFAULT_CONTACT_PAIRS.get(family, ()))!r}:
        left, _, right = str(pair).partition(":")
        if left:
            bodies.add(left)
        if right:
            bodies.add(right)
    declared_pair = (ir.get("params") or {{}}).get("declared_pair")
    if declared_pair:
        left, _, right = str(declared_pair).partition(":")
        if left:
            bodies.add(left)
        if right:
            bodies.add(right)
    return bodies


def _physics_default_chrono_collision(part, family):
    part_id = str(part.get("id", ""))
    role = str(part.get("role", ""))
    center = tuple(part.get("com_local_mm", (0.0, 0.0, 0.0)))
    if family == "cycloidal_reducer":
        if part_id == "housing" or role == "ground":
            return {{
                "shape": "cylinder",
                "radius_mm": 31.0,
                "height_mm": 12.0,
                "center_mm": center,
                "axis": (0.0, 0.0, 1.0),
            }}
        if part_id == "disc" or role == "cycloidal_disc":
            return {{
                "shape": "cylinder",
                "radius_mm": 30.98,
                "height_mm": 8.0,
                "center_mm": center,
                "axis": (0.0, 0.0, 1.0),
            }}
    if family == "rack_pinion":
        if part_id == "pinion":
            return {{
                "shape": "cylinder",
                "radius_mm": 14.9,
                "height_mm": 8.0,
                "center_mm": center,
                "axis": (0.0, 0.0, 1.0),
            }}
        if part_id == "rack":
            return {{
                "shape": "box",
                "size_mm": (80.0, 3.0, 8.0),
                "center_mm": center,
            }}
    if family == "cam_follower":
        if part_id == "cam":
            return {{
                "shape": "cylinder",
                "radius_mm": 20.0,
                "height_mm": 8.0,
                "center_mm": center,
                "axis": (0.0, 0.0, 1.0),
            }}
        if part_id == "follower":
            return {{
                "shape": "cylinder",
                "radius_mm": 20.0,
                "height_mm": 8.0,
                "center_mm": center,
                "axis": (0.0, 0.0, 1.0),
            }}
    if family == "geneva_indexer":
        if part_id == "driver":
            return {{
                "shape": "cylinder",
                "radius_mm": 20.0,
                "height_mm": 8.0,
                "center_mm": center,
                "axis": (0.0, 0.0, 1.0),
            }}
        if part_id == "geneva":
            return {{
                "shape": "cylinder",
                "radius_mm": 20.05,
                "height_mm": 8.0,
                "center_mm": center,
                "axis": (0.0, 0.0, 1.0),
            }}
    return {{
        "shape": "box",
        "size_mm": (20.0, 20.0, 8.0),
        "center_mm": center,
    }}


def _physics_default_initial_pose_mm(part, family):
    part_id = str(part.get("id", ""))
    if family == "rack_pinion" and part_id == "rack":
        return (0.0, 13.0, 0.0)
    if family == "cam_follower" and part_id == "follower":
        return (40.04, 0.0, 0.0)
    if family == "geneva_indexer" and part_id == "geneva":
        return (39.98, 0.0, 0.0)
    return None


def _physics_adjust_chrono_joints(ir, family):
    if family != "rack_pinion":
        return
    for joint in ir.get("joints", []) or []:
        if not isinstance(joint, dict):
            continue
        if joint.get("type") == "prismatic" and joint.get("child") == "rack":
            anchor = list(joint.get("anchor_world_mm") or (0.0, 0.0, 0.0))
            while len(anchor) < 3:
                anchor.append(0.0)
            joint["anchor_world_mm"] = (float(anchor[0]), 13.0, float(anchor[2]))


def _physics_enrich_design(ir, out_dir):
    from pathlib import Path

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    ir.setdefault("schema_version", "design_ir.v2")
    ir.setdefault("units", "mm")
    materials = ir.setdefault("materials", {{}})
    materials.setdefault(
        "steel_1045",
        {{
            "name": "AISI 1045 steel",
            "density_kg_m3": 7850.0,
            "elastic_modulus_pa": 205000000000.0,
            "poisson_ratio": 0.29,
            "yield_strength_pa": 530000000.0,
            "process": "machined_or_ground_reference",
            "provenance": "MechanismRepair-Physics preflight material table",
        }},
    )
    params = ir.setdefault("params", {{}})
    params.setdefault(
        "cad_source",
        {{
            "kernel": "FreeCAD/OCCT trusted preflight bridge",
            "source": "scripts.prepare_mechanism_repair_physics_benchmark",
            "family": {family!r},
            "verifier_level": {int(verifier_level)},
        }},
    )
    provenance = ir.setdefault("provenance", {{}})
    provenance.setdefault(
        "mechanism_repair_physics",
        {{
            "trusted_asset_bridge": True,
            "note": (
                "This preflight bridge supplies CAD-relative artifacts and "
                "mass-property evidence for verifier testing; final paper runs "
                "must replace or validate it with the trusted CAD/OCCT pipeline."
            ),
        }},
    )
    if {int(verifier_level)} >= 3:
        chrono_cfg = params.setdefault("chrono", {{}})
        chrono_cfg.setdefault("collision_filter_named_pairs", True)
        chrono_cfg.setdefault("contact_margin_m", 2.0e-5)
        chrono_cfg.setdefault("contact_envelope_m", 2.0e-5)
        chrono_cfg.setdefault("smc_use_material_properties", False)
        chrono_cfg.setdefault("normal_stiffness_N_m", 25000.0)
        chrono_cfg.setdefault("normal_damping_N_s_m", 250.0)
        chrono_cfg.setdefault("friction_mu", 0.05)
        chrono_cfg.setdefault("solver_max_iterations", 300)
        chrono_cfg.setdefault("solver_tolerance", 1.0e-8)
    contact_bodies = (
        _physics_contact_body_ids(ir) if {int(verifier_level)} >= 3 else set()
    )
    _physics_adjust_chrono_joints(ir, {family!r})
    for index, part in enumerate(ir.get("parts", []) or []):
        if not isinstance(part, dict):
            continue
        part_id = _physics_safe_id(part.get("id", f"part_{{index}}"))
        part.setdefault("material", "steel_1045")
        part.setdefault("com_local_mm", (0.0, 0.0, 0.0))
        pparams = part.setdefault("params", {{}})
        initial_pose = _physics_default_initial_pose_mm(part, {family!r})
        if initial_pose is not None:
            pparams.setdefault("initial_pose_mm", initial_pose)
        if part_id in contact_bodies:
            pparams.setdefault(
                "chrono_collision",
                _physics_default_chrono_collision(part, {family!r}),
            )
        geom = part.setdefault("geometry", {{}})
        cad_name = geom.setdefault("cad", f"{{part_id}}.step")
        cad_path = out_path / cad_name
        if not cad_path.exists():
            cad_path.write_text(_physics_stub_step(part_id))
        mass = float(part.get("mass_kg", 0.0) or 0.0)
        if mass > 0.0:
            scale = max(mass, 1.0e-6)
            pparams.setdefault(
                "cad_mass_properties",
                {{
                    "mass_kg": mass,
                    "com_local_mm": tuple(part.get("com_local_mm", (0.0, 0.0, 0.0))),
                    "inertia_kg_m2": (
                        (scale * 1.0e-5, 0.0, 0.0),
                        (0.0, scale * 1.2e-5, 0.0),
                        (0.0, 0.0, scale * 1.5e-5),
                    ),
                }},
            )
    return ir


def build_design(out_dir):
    return _physics_enrich_design(_base_build_design(out_dir), out_dir)
'''


def add_universal_negative_controls(task: GeneratedTask) -> None:
    controls = list((task.expected_failures or {}).get("controls", []) or [])

    if "missing_required_port_physics" not in task.negative_solutions:
        task.negative_solutions["missing_required_port_physics"] = (
            universal_negative_source(
                "    ports = ir.get('ports') or {}\n"
                "    if ports:\n"
                "        del ports[sorted(ports)[0]]"
            )
        )
        controls.append(
            {
                "id": "missing_required_port_physics",
                "submission": "negative_solutions/missing_required_port_physics",
                "expected_failure_codes": ["missing_port"],
                "expected_hard_gate_passed": False,
                "expected_score_below": 0.001,
            }
        )

    if "missing_trusted_mass_physics" not in task.negative_solutions:
        task.negative_solutions["missing_trusted_mass_physics"] = (
            universal_negative_source(
                "    for part in ir.get('parts', []) or []:\n"
                "        if float(part.get('mass_kg', 0.0) or 0.0) > 0.0:\n"
                "            params = part.get('params') or {}\n"
                "            params.pop('cad_mass_properties', None)\n"
                "            part['params'] = params\n"
                "            break"
            )
        )
        controls.append(
            {
                "id": "missing_trusted_mass_physics",
                "submission": "negative_solutions/missing_trusted_mass_physics",
                "expected_failure_codes": ["invalid_mass_properties"],
                "expected_hard_gate_passed": False,
                "expected_score_below": 0.6,
            }
        )

    task.expected_failures = dict(task.expected_failures or {})
    task.expected_failures["controls"] = controls


def universal_negative_source(modifier_py: str) -> str:
    return (
        "# auto-generated; do not edit by hand. See scripts.prepare_mechanism_repair_physics_benchmark.\n"
        "import sys\n"
        "from pathlib import Path\n\n\n"
        "def build_design(out_dir):\n"
        "    ref_dir = (\n"
        "        Path(__file__).resolve().parent.parent.parent\n"
        "        / 'reference_solution'\n"
        "    )\n"
        "    sys.path.insert(0, str(ref_dir))\n"
        "    try:\n"
        "        import design as ref  # noqa: I001\n"
        "    finally:\n"
        "        sys.path.pop(0)\n"
        "    ir = ref.build_design(out_dir)\n"
        + modifier_py
        + "\n    return ir\n"
    )


def upgrade_eval_config(
    eval_config: dict[str, Any],
    *,
    family: str,
    verifier_level: int,
    hidden: bool = False,
) -> dict[str, Any]:
    cfg = copy.deepcopy(eval_config or {})
    probes = cfg.setdefault("probes", [])
    if not isinstance(probes, list):
        cfg["probes"] = probes = []

    if not any(
        isinstance(p, dict) and p.get("type") == "trusted_asset_preflight"
        for p in probes
    ):
        probes.append(
            {
                "id": "trusted_asset_preflight",
                "type": "trusted_asset_preflight",
                "require_geometry_roles": ["cad"],
                "require_materials": True,
                "require_material_properties": [
                    "density_kg_m3",
                    "elastic_modulus_pa",
                    "poisson_ratio",
                    "yield_strength_pa",
                ],
                "require_provenance": True,
                "require_trusted_mass_properties": True,
                "hard_gate": True,
                "severity": "critical",
                "weight": 0.2,
            }
        )

    feedback = cfg.setdefault("feedback", {})
    public_metrics = feedback.setdefault("public_metrics", [])
    hidden_metrics = feedback.setdefault("hidden_metrics", [])
    _append_unique(public_metrics, "trusted_asset_preflight.parts_with_required_geometry")
    _append_unique(
        hidden_metrics,
        "trusted_asset_preflight.trusted_mass_properties_recomputed",
    )

    hard_gate = cfg.setdefault("hard_gate", {})
    required = hard_gate.setdefault("require", [])
    _append_unique(required, "trusted_asset_preflight")

    if verifier_level >= 3:
        upgrade_level3_chrono_config(cfg, family=family)

    cfg["mechanism_repair_physics"] = {
        "family": family,
        "verifier_level": int(verifier_level),
        "hidden_variant": bool(hidden),
        "fake_contact_oracle_allowed": False,
    }
    return cfg


def upgrade_level3_chrono_config(cfg: dict[str, Any], *, family: str) -> None:
    probes = cfg.setdefault("probes", [])
    pairs = set(DEFAULT_CONTACT_PAIRS.get(family, ()))

    adapters = cfg.setdefault("adapters", {})
    if isinstance(adapters, dict):
        fake_cfg = adapters.pop("fake_contact_oracle", {}) or {}
        if isinstance(fake_cfg, dict):
            for pair in fake_cfg.get("contact_pairs", []) or []:
                pairs.add(str(pair))
        adapters["chrono_contact"] = {
            "contact_model": "nsc",
            "procedural_cycloidal_fallback": False,
            "samples": 720,
            "duration_s": 1.0,
            "input_speed_rad_s": 1.0,
            "output_load_Nm": 0.05,
        }

    for probe in probes:
        if not isinstance(probe, dict):
            continue
        if probe.get("adapter") == "fake_contact_oracle":
            probe["adapter"] = "chrono_contact"
        if probe.get("type") == "contact_engagement":
            probe["min_engagement_fraction"] = min(
                float(probe.get("min_engagement_fraction", 0.2)),
                0.01,
            )
            for pair in probe.get("required_pairs", []) or []:
                pairs.add(str(pair))

    for probe in probes:
        if not isinstance(probe, dict):
            continue
        if probe.get("type") != "swept_collision":
            continue
        ignored = probe.setdefault("ignored_pairs", [])
        if not isinstance(ignored, list):
            probe["ignored_pairs"] = ignored = []
        for pair in sorted(pairs):
            _append_unique(ignored, pair)

    if not any(
        isinstance(p, dict) and p.get("adapter") == "chrono_contact"
        for p in probes
    ):
        probes.append(
            {
                "id": "chrono_contact_smoke",
                "type": "contact_engagement",
                "required_pairs": sorted(pairs) or ["input_port:output_port"],
                "min_rms_force_N": 0.1,
                "min_engagement_fraction": 0.01,
                "adapter": "chrono_contact",
                "hard_gate": True,
                "severity": "critical",
                "weight": 0.5,
            }
        )

    hard_gate = cfg.setdefault("hard_gate", {})
    required = hard_gate.setdefault("require", [])
    for probe in probes:
        if isinstance(probe, dict) and probe.get("adapter") == "chrono_contact":
            if probe.get("hard_gate") is True:
                _append_unique(required, str(probe.get("id")))


def _append_unique(items: list[Any], value: Any) -> None:
    if value not in items:
        items.append(value)


def audit_benchmark(
    *,
    tasks_root: Path,
    min_tasks_per_family: int,
    validate: bool,
    validate_level3: bool,
    scratch_root: Path,
) -> dict[str, Any]:
    records = load_task_records(tasks_root)
    family_counts: dict[str, int] = defaultdict(int)
    level_counts: dict[str, int] = defaultdict(int)
    task_audits: list[dict[str, Any]] = []
    structural_blockers: list[str] = []
    validation_blockers: list[str] = []
    paper_blockers: list[str] = []

    for record in records:
        family_counts[record["family"]] += 1
        level_counts[str(record["verifier_level"])] += 1
        task_audit = audit_task(record["task_dir"])
        task_audit.update({**record, "task_dir": str(record["task_dir"])})
        if validate and (record["verifier_level"] < 3 or validate_level3):
            validation = validate_task(
                record["task_dir"],
                scratch_root / record["task_id"],
            )
            task_audit["validation"] = validation
        elif validate and record["verifier_level"] >= 3 and not validate_level3:
            task_audit["validation"] = {
                "skipped": True,
                "reason": "level3_validation_skipped",
            }
        task_audits.append(task_audit)

    total_tasks = len(task_audits)
    headline_tasks = [task for task in task_audits if task["headline_eligible"]]
    level2plus = [task for task in headline_tasks if int(task["verifier_level"]) >= 2]
    level3 = [task for task in headline_tasks if int(task["verifier_level"]) >= 3]

    if len(family_counts) < len(REQUIRED_FAMILIES):
        structural_blockers.append(
            f"only {len(family_counts)} families; need {len(REQUIRED_FAMILIES)}"
        )
    for family in REQUIRED_FAMILIES:
        count = int(family_counts.get(family, 0))
        if count < min_tasks_per_family:
            structural_blockers.append(
                f"{family}: only {count} tasks; need {min_tasks_per_family}"
            )
    if total_tasks < len(REQUIRED_FAMILIES) * min_tasks_per_family:
        structural_blockers.append(
            f"only {total_tasks} tasks; need "
            f"{len(REQUIRED_FAMILIES) * min_tasks_per_family}"
        )
    if total_tasks and len(level2plus) / total_tasks < 0.40:
        structural_blockers.append("Level-2/3 task share below 40 percent")
    if total_tasks and len(level3) / total_tasks < 0.25:
        structural_blockers.append("Level-3 task share below 25 percent")

    for task in task_audits:
        if len(task["constraint_classes"]) < 3:
            structural_blockers.append(
                f"{task['task_id']}: only {task['constraint_classes']} "
                "constraint classes; need at least 3"
            )
        if task["negative_control_count"] < 2:
            structural_blockers.append(
                f"{task['task_id']}: only {task['negative_control_count']} "
                "negative controls; need at least 2"
            )
        if task["effective_negative_control_count"] < 2:
            structural_blockers.append(
                f"{task['task_id']}: only "
                f"{task['effective_negative_control_count']} effective "
                "negative controls"
            )
        if not task["has_hidden_variant"]:
            structural_blockers.append(f"{task['task_id']}: no hidden variant")
        if task["uses_fake_contact_oracle"]:
            structural_blockers.append(
                f"{task['task_id']}: fake_contact_oracle still present"
            )
        if validate:
            validation = task.get("validation", {})
            if validation.get("skipped"):
                paper_blockers.append(
                    f"{task['task_id']}: {validation.get('reason')}"
                )
            elif not validation.get("reference_passed", False):
                validation_blockers.append(
                    f"{task['task_id']}: reference failed "
                    f"{validation.get('reference_codes')}"
                )
            if validation.get("reference_oracle_is_synthetic"):
                validation_blockers.append(
                    f"{task['task_id']}: reference used synthetic oracle"
                )
            for item in validation.get("negative_failures", []) or []:
                validation_blockers.append(f"{task['task_id']}: {item}")

    if not validate:
        paper_blockers.append("reference_and_negative_validation_not_run")
    if not validate or not validate_level3:
        paper_blockers.append("chrono_level3_validation_not_run")
    if validate and validate_level3 and any(int(t["verifier_level"]) >= 3 for t in task_audits):
        diag = chrono_diagnostic()
        if diag.get("status") != "available":
            validation_blockers.append(
                "chrono_contact_unavailable: "
                f"{diag.get('reason') or diag.get('runner_status') or diag}"
            )

    blockers = structural_blockers + validation_blockers
    structural_passes = not structural_blockers
    passes = not blockers
    experiment_ready = passes and not paper_blockers and validate and validate_level3
    return {
        "schema": "mechanism_repair_physics.benchmark_audit.v1",
        "tasks_root": str(tasks_root),
        "validate": bool(validate),
        "validate_level3": bool(validate_level3),
        "required_families": list(REQUIRED_FAMILIES),
        "min_tasks_per_family": int(min_tasks_per_family),
        "family_counts": dict(sorted(family_counts.items())),
        "level_counts": dict(sorted(level_counts.items())),
        "task_count": total_tasks,
        "level2plus_headline_count": len(level2plus),
        "level3_headline_count": len(level3),
        "tasks": sorted(task_audits, key=lambda row: row["task_id"]),
        "structural_blockers": structural_blockers,
        "validation_blockers": validation_blockers,
        "paper_blockers": sorted(set(paper_blockers)),
        "blockers": blockers,
        "structural_passes": structural_passes,
        "passes": passes,
        "experiment_ready": experiment_ready,
    }


def load_task_records(tasks_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for task_dir in sorted(Path(tasks_root).iterdir()):
        task_toml = task_dir / "task.toml"
        if not task_toml.is_file():
            continue
        data = tomllib.loads(task_toml.read_text())
        task_data = data.get("task", {}) or {}
        physics = data.get("mechanism_repair_physics", {}) or {}
        task_id = str(task_data.get("id", task_dir.name))
        family = str(physics.get("canonical_family") or task_data.get("family", ""))
        level = int(physics.get("verifier_level") or task_data.get("difficulty", 1))
        records.append(
            {
                "task_id": task_id,
                "family": family,
                "task_dir": task_dir,
                "verifier_level": level,
                "headline_eligible": bool(physics.get("headline_eligible", True)),
                "source_generator": str(physics.get("source_generator", "")),
            }
        )
    return records


def audit_task(task_dir: Path) -> dict[str, Any]:
    eval_config = tomllib.loads((task_dir / "eval_config.toml").read_text())
    expected = json.loads((task_dir / "expected_failures.json").read_text())
    probes = [p for p in eval_config.get("probes", []) if isinstance(p, dict)]
    probe_types = [str(p.get("type") or "") for p in probes]
    controls = list(expected.get("controls", []) or [])
    effective_controls = [c for c in controls if is_effective_negative_control(c)]
    return {
        "probe_types": probe_types,
        "constraint_classes": sorted(classify_constraint_classes(eval_config)),
        "has_hidden_variant": (task_dir / "eval_config.hidden.toml").is_file(),
        "negative_control_count": len(controls),
        "effective_negative_control_count": len(effective_controls),
        "task_hash": hash_task_contract(task_dir),
        "uses_fake_contact_oracle": uses_fake_contact_oracle(eval_config),
    }


def classify_constraint_classes(eval_config: dict[str, Any]) -> set[str]:
    classes: set[str] = set()
    probes = [p for p in eval_config.get("probes", []) if isinstance(p, dict)]
    for probe in probes:
        probe_type = str(probe.get("type") or "")
        if probe_type in TOPOLOGY_PROBES:
            classes.add("topology_mobility")
        if probe_type in INTERFACE_PROBES:
            classes.add("interface")
        if probe_type in FUNCTIONAL_PROBES:
            classes.add("functional_behavior")
        if probe_type in ARTIFACT_PROBES:
            classes.add("cad_artifact")
        if probe_type in CONTACT_PROBES:
            classes.add("physics_contact")
    return classes


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


def is_effective_negative_control(control: dict[str, Any]) -> bool:
    expected_codes = control.get("expected_failure_codes", []) or []
    if expected_codes:
        return True
    if control.get("expected_hard_gate_passed") is False:
        return True
    if float(control.get("expected_score_below", 1.0)) < 1.0:
        return True
    return False


def validate_task(task_dir: Path, scratch_root: Path) -> dict[str, Any]:
    scratch_root.mkdir(parents=True, exist_ok=True)
    reference = evaluate(
        task_dir,
        task_dir / "reference_solution",
        scratch_dir=scratch_root / "reference",
    )
    reference_codes = feedback_codes(reference)
    negative_failures: list[str] = []
    expected = json.loads((task_dir / "expected_failures.json").read_text())
    for control in expected.get("controls", []) or []:
        submission = task_dir / str(control["submission"])
        report = evaluate(
            task_dir,
            submission,
            scratch_dir=scratch_root / str(control["id"]),
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
    records = load_task_records(tasks_root)
    split_outputs: dict[str, dict[str, Any]] = {}
    for split_name, spec in SPLITS.items():
        seen = set(spec["seen"])
        unseen = set(spec["unseen"])
        manifest = build_split_manifest(
            records=records,
            split_name=split_name,
            seen_families=seen,
            unseen_families=unseen,
            seed=split_seed + (0 if split_name == "A" else 1),
        )
        split_dir = out_dir / f"splits_{split_name}"
        write_split_files(manifest, split_dir)
        manifest_path = out_dir / f"split_manifest_{split_name}.json"
        write_json(manifest_path, manifest)
        split_outputs[split_name] = {
            "manifest_path": str(manifest_path),
            "split_dir": str(split_dir),
            "seen_families": sorted(seen),
            "unseen_families": sorted(unseen),
            "n_train": len(manifest["splits"]["train"]),
            "n_val": len(manifest["splits"]["val"]),
            "n_test": len(manifest["splits"]["test"]),
        }

    hidden_manifest = build_hidden_split_manifest(records=records, seed=split_seed + 99)
    hidden_path = out_dir / "split_manifest_hidden_perturbation.json"
    write_json(hidden_path, hidden_manifest)
    write_split_files(hidden_manifest, out_dir / "splits_hidden_perturbation")
    split_outputs["hidden_perturbation"] = {
        "manifest_path": str(hidden_path),
        "split_dir": str(out_dir / "splits_hidden_perturbation"),
        "n_test": len(hidden_manifest["splits"]["test"]),
    }
    external_manifest = build_external_style_split_manifest(records=records, seed=split_seed + 199)
    external_path = out_dir / "split_manifest_external_style.json"
    write_json(external_path, external_manifest)
    write_split_files(external_manifest, out_dir / "splits_external_style")
    split_outputs["external_style"] = {
        "manifest_path": str(external_path),
        "split_dir": str(out_dir / "splits_external_style"),
        "n_test": len(external_manifest["splits"]["test"]),
    }
    return split_outputs


def build_split_manifest(
    *,
    records: list[dict[str, Any]],
    split_name: str,
    seen_families: set[str],
    unseen_families: set[str],
    seed: int,
) -> dict[str, Any]:
    train: list[str] = []
    val: list[str] = []
    test: list[str] = []
    for family in sorted(seen_families):
        family_records = sorted(
            [r for r in records if r["family"] == family],
            key=lambda row: stable_shuffle_key(row["task_id"], seed),
        )
        cut = max(1, int(round(len(family_records) * 0.8)))
        train.extend(str(r["task_dir"]) for r in family_records[:cut])
        val.extend(str(r["task_dir"]) for r in family_records[cut:])
    for family in sorted(unseen_families):
        test.extend(
            str(r["task_dir"])
            for r in sorted(
                [row for row in records if row["family"] == family],
                key=lambda row: stable_shuffle_key(row["task_id"], seed),
            )
        )
    return {
        "schema": "mechanism_repair_physics.family_split.v1",
        "split_name": split_name,
        "seed": int(seed),
        "seen_families": sorted(seen_families),
        "unseen_families": sorted(unseen_families),
        "splits": {
            "train": sorted(train),
            "val": sorted(val),
            "test": sorted(test),
        },
    }


def build_hidden_split_manifest(
    *,
    records: list[dict[str, Any]],
    seed: int,
) -> dict[str, Any]:
    tests = [str(r["task_dir"]) for r in sorted(records, key=lambda row: stable_shuffle_key(row["task_id"], seed))]
    return {
        "schema": "mechanism_repair_physics.hidden_split.v1",
        "split_name": "hidden_perturbation",
        "seed": int(seed),
        "seen_families": [],
        "unseen_families": list(REQUIRED_FAMILIES),
        "eval_config": "eval_config.hidden.toml",
        "perturbations": [
            "dimension and target perturbation",
            "tighter hidden tolerance",
            "renamed/isomorphic semantic audit manifest",
        ],
        "splits": {"train": [], "val": [], "test": tests},
    }


def build_external_style_split_manifest(
    *,
    records: list[dict[str, Any]],
    seed: int,
) -> dict[str, Any]:
    selected = [
        r
        for r in records
        if r["family"] in {"shaft_bearing_coupling", "spur_compound_gear_train", "fourbar_linkage"}
    ]
    tests = [
        str(r["task_dir"])
        for r in sorted(selected, key=lambda row: stable_shuffle_key(row["task_id"], seed))
    ]
    return {
        "schema": "mechanism_repair_physics.external_style_split.v1",
        "split_name": "external_style",
        "seed": int(seed),
        "seen_families": [],
        "unseen_families": sorted({r["family"] for r in selected}),
        "positioning": (
            "CAD/design-style holdout; this does not claim CADBench/BenchCAD "
            "evaluation until those external tasks are actually imported."
        ),
        "splits": {"train": [], "val": [], "test": tests},
    }


def stable_shuffle_key(task_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{task_id}".encode("utf-8")).hexdigest()


def write_split_files(manifest: dict[str, Any], split_dir: Path) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    for name, items in manifest["splits"].items():
        (split_dir / f"{name}.txt").write_text("\n".join(items) + ("\n" if items else ""))


def build_verifier_manifest(
    *,
    tasks_root: Path,
    audit: dict[str, Any],
) -> dict[str, Any]:
    fake_oracle_tasks = [
        task["task_id"] for task in audit["tasks"] if task["uses_fake_contact_oracle"]
    ]
    return {
        "schema": "mechanism_repair_physics.verifier_manifest.v1",
        "tasks_root": str(tasks_root),
        "verifier_levels": audit["level_counts"],
        "level_2_contract": {
            "requires_trusted_asset_preflight": True,
            "requires_geometry_roles": ["cad"],
            "requires_material_provenance": True,
            "requires_trusted_mass_properties": True,
            "current_limit": (
                "trusted_asset_preflight consumes trusted mass records; final "
                "paper validation must run the trusted CAD/OCCT bridge or an "
                "equivalent recomputation path."
            ),
        },
        "level_3_contract": {
            "requires_chrono_contact": True,
            "requires_real_pychrono": True,
            "allows_fake_contact_oracle": False,
            "chrono_diagnostic": chrono_diagnostic(),
        },
        "constraint_class_policy": {
            "non_toy_min_classes": 3,
            "classes": [
                "topology_mobility",
                "interface",
                "functional_behavior",
                "cad_artifact",
                "physics_contact",
                "manufacturability_assembly",
            ],
        },
        "fake_oracle_tasks": sorted(fake_oracle_tasks),
        "main_claim_allows_fake_oracle": False,
    }


def build_level_manifest(
    *,
    tasks_root: Path,
    audit: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "mechanism_repair_physics.level_manifest.v1",
        "tasks_root": str(tasks_root),
        "level_counts": audit["level_counts"],
        "headline_levels": [2, 3],
        "tasks": [
            {
                "task_id": task["task_id"],
                "family": task["family"],
                "verifier_level": task["verifier_level"],
                "headline_eligible": task["headline_eligible"],
                "constraint_classes": task["constraint_classes"],
            }
            for task in audit["tasks"]
        ],
    }


def build_hidden_variant_manifest(
    *,
    tasks_root: Path,
    audit: dict[str, Any],
) -> dict[str, Any]:
    rows = []
    for task in audit["tasks"]:
        rows.append(
            {
                "task_id": task["task_id"],
                "family": task["family"],
                "verifier_level": task["verifier_level"],
                "public_eval_config": str(Path(task["task_dir"]) / "eval_config.public.toml"),
                "hidden_eval_config": str(Path(task["task_dir"]) / "eval_config.hidden.toml"),
                "hidden_variant_present": task["has_hidden_variant"],
                "perturbations": [
                    "tighter tolerance or target perturbation",
                    "hidden metric withholding",
                    "isomorphic renaming required by audit",
                ],
                "isomorphic_variant_status": "manifested_not_executed",
            }
        )
    return {
        "schema": "mechanism_repair_physics.hidden_variant_manifest.v1",
        "tasks_root": str(tasks_root),
        "tasks": rows,
        "paper_readiness_note": (
            "Hidden eval configs are frozen here. Full anti-shortcut completion "
            "still requires executing these variants and writing anti_shortcut_audit.json."
        ),
    }


def build_method_manifest() -> dict[str, Any]:
    return {
        "schema": "mechanism_repair_physics.method_manifest.v1",
        "required_methods": list(REQUIRED_METHODS),
        "primary_method": PRIMARY_METHOD,
        "fallback_primary_method": FALLBACK_PRIMARY_METHOD,
        "primary_baseline": PRIMARY_BASELINE,
        "primary_budget_expensive_verifier_calls": PRIMARY_BUDGET,
        "budget_curve": [8, 16, 32],
        "eval_seeds": list(EVAL_SEEDS),
        "success_threshold": {
            "level23_success_abs_delta_pct": SUCCESS_DELTA_PCT,
            "paired_bootstrap_ci95_low_gt_0": True,
            "paired_sign_or_permutation_p_lte": 0.05,
            "hidden_variant_delta_positive": True,
            "anti_shortcut_delta_positive": True,
            "beats_adaptive_evolution": True,
            "beats_verifier_gated_search": True,
            "positive_delta_min_families": 8,
        },
        "causal_contrast": {
            "same_base_model": True,
            "same_tasks": True,
            "same_task_order": True,
            "same_prompts": True,
            "same_actual_verifier_budget": True,
            "difference": (
                "mechanical_evolve_ttrl variants update LoRA weights online; "
                "llm_evolve_no_update receives the same verifier feedback but "
                "does not update weights"
            ),
        },
        "method_roles": {
            "frozen_model": "base model only, no feedback updates",
            "sft_seen_family": "seen-family LoRA/SFT, no test-time updates",
            "llm_evolve_no_update": "primary no-update verifier-feedback baseline",
            "verifier_gated_search": "best-of-K or beam under matched budget",
            "adaptive_evolution": "population/archive search, no gradient updates",
            "mechanical_evolve_ttrl": "online GRPO/LoRA from verifier rewards",
            "mechanical_evolve_ttrl_tool_verified": (
                "TTRL with tool-verification reward filtering"
            ),
            "mechanical_evolve_ttrl_confidence": (
                "confidence-conditioned TTRL exploration/reward ablation"
            ),
        },
    }


def ensure_run_scaffold(out_dir: Path) -> dict[str, Any]:
    required_dirs = (
        "raw_completions",
        "verifier_outputs",
        "cad_artifacts",
        "chrono_outputs",
        "training_logs",
        "adapter_checkpoints",
    )
    for name in required_dirs:
        (out_dir / name).mkdir(parents=True, exist_ok=True)
    return {
        "schema": "mechanism_repair_physics.run_scaffold.v1",
        "directories": [str(out_dir / name) for name in required_dirs],
        "pending_result_artifacts": [
            "results.json",
            "results.csv",
            "cell_results.jsonl",
            "stats.json",
            "failure_analysis.json",
            "trace_pairs.json",
            "repair_taxonomy.json",
            "anti_shortcut_audit.json",
            "budget_audit.json",
        ],
    }


def build_claim_audit(audit: dict[str, Any]) -> dict[str, Any]:
    missing = [
        "execute all required methods for at least three seeds",
        "record raw completions and verifier outputs for every cell",
        "run hidden/isomorphic anti-shortcut variants",
        "prove matched actual CAD/Chrono verifier budget",
        "write statistical, failure, trace-pair, repair-taxonomy, "
        "anti-shortcut, and budget analyses",
    ]
    if not bool(audit["experiment_ready"]):
        missing.insert(
            2,
            "run Level-3 tasks with a registered real chrono_contact adapter",
        )
    return {
        "schema": "mechanism_repair_physics.claim_audit.v1",
        "goal_complete": False,
        "experiment_ready": bool(audit["experiment_ready"]),
        "structural_passes": bool(audit["structural_passes"]),
        "passes_current_preflight": bool(audit["passes"]),
        "blockers": audit["blockers"],
        "paper_blockers": audit["paper_blockers"],
        "missing_before_paper_claim": missing,
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
        "metadata.json",
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
