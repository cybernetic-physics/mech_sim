#!/usr/bin/env python3
"""Generate paper-facing proof for the real-geometry Chrono reducer path.

The proof intentionally exercises the non-procedural path:

1. FreeCAD/OCCT exports named STEP/STL bodies through CycloidGearBox.
2. DesignIR is built from the manifest with Chrono collision for contact bodies
   backed by exported CAD meshes and CAD-derived feature frames.
3. Chrono runs NSC and SMC with ``procedural_cycloidal_fallback=false``.

The emitted JSON stores hashes and sizes for the CAD exports plus the scalar
metrics that prove Chrono consumed the generated collision geometry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mech_bench.adapters import _chrono_impl
from mech_bench.geometry.cycloidal_freecad import (
    BODY_NAMES,
    CycloidalCadExportError,
    CycloidalReducerAssets,
    build_chrono_design_ir_from_assets,
    find_cycloid_gearbox_path,
    find_freecad_command,
    generate_cycloidal_reducer_assets,
)
from mech_bench.trusted_assets import build_trusted_asset_manifest


METRIC_KEYS = (
    "lockup_detected",
    "ratio_observed",
    "ratio_error_pct",
    "in_omega_med",
    "out_omega_med",
    "max_penetration_mm",
    "max_constraint_error_mm",
    "n_contacts_max",
    "top_contact_pairs",
    "contact_force_rms_N",
    "input_power_W_mean",
    "output_power_W_mean",
    "power_balance_error_pct",
    "power_balance_residual_pct",
    "mechanical_efficiency_pct",
    "unaccounted_power_W_mean",
    "kinetic_energy_rate_W_mean",
    "torque_ripple_pct",
    "out_omega_med_raw",
    "out_omega_fit_rad_s",
    "contact_pair_max_penetration_mm",
    "contact_pair_rms_force_N",
)

NUMERIC_METRIC_KEYS = (
    "lockup_detected",
    "ratio_observed",
    "ratio_error_pct",
    "in_omega_med",
    "out_omega_med",
    "max_penetration_mm",
    "max_constraint_error_mm",
    "n_contacts_max",
    "contact_force_rms_N",
    "input_power_W_mean",
    "output_power_W_mean",
    "power_balance_error_pct",
    "power_balance_residual_pct",
    "mechanical_efficiency_pct",
    "unaccounted_power_W_mean",
    "kinetic_energy_rate_W_mean",
    "torque_ripple_pct",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        default="/tmp/mech_bench_cycloidal_real_geometry_proof",
        help="Directory where FreeCAD should write generated assets.",
    )
    parser.add_argument(
        "--proof-json",
        default=None,
        help="Path for the proof JSON. Defaults to <out-dir>/proof.json.",
    )
    parser.add_argument(
        "--freecad-cmd",
        default=None,
        help="FreeCAD command. Defaults to MECH_BENCH_FREECAD_CMD or PATH.",
    )
    parser.add_argument(
        "--cycloid-gearbox-path",
        default=None,
        help="CycloidGearBox checkout. Defaults to MECH_BENCH_CYCLOID_GEARBOX_PATH.",
    )
    parser.add_argument(
        "--allow-failure",
        action="store_true",
        help="Write proof JSON and exit 0 even when an acceptance check fails.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    proof_json = (
        Path(args.proof_json).expanduser().resolve()
        if args.proof_json
        else out_dir / "proof.json"
    )
    proof: dict[str, Any] = _base_proof(out_dir, proof_json, args)

    try:
        assets = generate_cycloidal_reducer_assets(
            out_dir,
            {
                "pins": 10,
                "line_segment_count": 42,
                "clearance": 0.6,
                "driver_pin_collision_shrink_mm": 0.68,
            },
            freecad_cmd=args.freecad_cmd,
            cycloid_gearbox_path=args.cycloid_gearbox_path,
            timeout_s=300.0,
        )
        proof["asset_generation"] = _asset_proof(assets)

        ir = build_chrono_design_ir_from_assets(
            assets,
            collision_sweep_radius_m=2.0e-5,
        )
        trusted_manifest = build_trusted_asset_manifest(
            ir, build_root=assets.root)
        proof["trusted_asset_manifest"] = trusted_manifest.to_dict()
        proof["acceptance"]["trusted_cad_mass_properties"] = (
            trusted_manifest.trusted_mass_properties_recomputed
        )
        proof["design_ir"] = _design_ir_proof(ir, assets)
        collision_issue = _collision_shape_issue(proof["design_ir"], assets)
        if collision_issue:
            _mark_failed(proof, "collision mesh import", collision_issue)
        else:
            proof["runs"] = {
                "nsc": _run_chrono(ir, assets, "nsc", output_load_nm=0.75),
                "smc": _run_chrono(ir, assets, "smc", output_load_nm=0.75),
                "smc_unloaded": _run_chrono(
                    ir,
                    assets,
                    "smc",
                    output_load_nm=0.0,
                    max_power_error_pct=1.0e12,
                    max_torque_ripple_pct=1.0e12,
                ),
            }
            proof["convergence"] = _run_smc_sample_convergence(ir, assets)
            _evaluate_runs(proof)
    except CycloidalCadExportError as exc:
        _mark_failed(proof, exc.stage, str(exc))
    except Exception as exc:  # noqa: BLE001 - proof boundary
        _mark_failed(proof, "solver dynamics", f"{type(exc).__name__}: {exc}")

    safe_proof = _json_safe(proof)
    _write_json(proof_json, safe_proof)
    print(json.dumps(safe_proof, indent=2, sort_keys=True, allow_nan=False))
    return 0 if proof.get("ok") or args.allow_failure else 1


def _base_proof(out_dir: Path, proof_json: Path, args: argparse.Namespace) -> dict[str, Any]:
    freecad_cmd = _freecad_argv(args.freecad_cmd)
    cycloid_path = (
        Path(args.cycloid_gearbox_path).expanduser().resolve()
        if args.cycloid_gearbox_path
        else find_cycloid_gearbox_path()
    )
    return {
        "schema": "mech_bench.cycloidal_real_geometry_chrono_proof.v1",
        "claim": "FreeCAD/OCCT geometry-backed cycloidal reducer runs in Chrono NSC and SMC without procedural fallback.",
        "validation_scope": "real-geometry solver acceptance; not empirical hardware calibration",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "ok": False,
        "missing_bridge": None,
        "proof_json": str(proof_json),
        "asset_out_dir": str(out_dir),
        "environment": {
            "cwd": str(Path.cwd()),
            "python": sys.version,
            "platform": platform.platform(),
            "repo_commit": _git(["rev-parse", "HEAD"]),
            "repo_branch": _git(["branch", "--show-current"]),
            "freecad_cmd": freecad_cmd,
            "freecad_version": _command_output(freecad_cmd + ["--version"])
            if freecad_cmd
            else None,
            "cycloid_gearbox_path": str(cycloid_path) if cycloid_path else None,
            "cycloid_gearbox_commit": _git(
                ["-C", str(cycloid_path), "rev-parse", "HEAD"]
            )
            if cycloid_path
            else None,
        },
        "acceptance": {
            "freecad_export_ran": False,
            "manifest_written": False,
            "named_bodies_exported": False,
            "nonempty_step_stl_exports": False,
            "cad_datums_present": False,
            "cad_static_contact_audit_passed": False,
            "chrono_procedural_fallback_false": False,
            "chrono_nsc_real_geometry_metrics": False,
            "chrono_smc_real_geometry_metrics": False,
            "trusted_cad_mass_properties": False,
            "nsc_bad_regime_observed": False,
            "smc_minimum_success_threshold": False,
            "smc_unloaded_ratio_near_declared": False,
            "smc_ratio_convergence": False,
        },
    }


def _asset_proof(assets: CycloidalReducerAssets) -> dict[str, Any]:
    manifest = _file_record(assets.manifest_path)
    bodies: dict[str, Any] = {}
    for name in BODY_NAMES:
        body = assets.bodies.get(name, {})
        bodies[name] = {
            "step": _file_record(assets.root / str(body.get("step", ""))),
            "stl": _file_record(assets.root / str(body.get("stl", ""))),
            "bbox_mm": body.get("bbox_mm"),
        }
    collision_meshes: dict[str, Any] = {}
    for name, mesh in sorted(assets.collision_meshes.items()):
        collision_meshes[name] = {
            "step": _file_record(assets.root / str(mesh.get("step", ""))),
            "stl": _file_record(assets.root / str(mesh.get("stl", ""))),
            "bbox_mm": mesh.get("bbox_mm"),
        }
    all_body_exports = [
        rec[ext]["exists"] and rec[ext]["bytes"] > 0
        for rec in bodies.values()
        for ext in ("step", "stl")
    ]
    return {
        "ran": True,
        "root": str(assets.root),
        "manifest": manifest,
        "manifest_schema": json.loads(assets.manifest_path.read_text()).get(
            "schema_version"
        ),
        "source": assets.source,
        "parameters": {
            key: assets.parameters.get(key)
            for key in (
                "ring_pin_count",
                "tooth_count",
                "declared_ratio",
                "clearance",
                "driver_disk_hole_count",
                "roller_diameter",
                "roller_circle_diameter",
            )
        },
        "body_names": sorted(assets.bodies),
        "bodies": bodies,
        "collision_meshes": collision_meshes,
        "collision_primitives": sorted(assets.collision_primitives),
        "feature_frame_counts": {
            key: len(value)
            for key, value in assets.feature_frames.items()
            if isinstance(value, list)
        },
        "static_audit": assets.static_audit,
        "all_named_bodies_present": sorted(assets.bodies) == sorted(BODY_NAMES),
        "all_body_step_stl_nonempty": all(all_body_exports),
    }


def _design_ir_proof(ir: Any, assets: CycloidalReducerAssets) -> dict[str, Any]:
    collision_shapes: dict[str, Any] = {}
    for part in ir.parts:
        params = part.params or {}
        shape = (part.params or {}).get("chrono_collision")
        if not isinstance(shape, dict):
            continue
        mesh = shape.get("mesh") or shape.get("stl") or shape.get("collision_mesh")
        asset = params.get("chrono_collision_asset")
        asset_mesh = asset.get("mesh") if isinstance(asset, dict) else None
        collision_shapes[part.id] = {
            "shape": shape.get("shape"),
            "mesh": mesh,
            "mesh_file": _file_record(assets.root / str(mesh)) if mesh else None,
            "asset_mesh": asset_mesh,
            "asset_mesh_file": (
                _file_record(assets.root / str(asset_mesh)) if asset_mesh else None
            ),
            "is_static": shape.get("is_static"),
            "is_convex": shape.get("is_convex"),
            "sweep_sphere_radius_m": shape.get("sweep_sphere_radius_m"),
            "child_count": len(shape.get("children", []))
            if isinstance(shape.get("children"), list)
            else 0,
        }
    return {
        "parts": [part.id for part in ir.parts],
        "joints": [joint.id for joint in ir.joints],
        "ports": sorted(ir.ports),
        "cad_assets_manifest": ir.params.get("cad_assets_manifest"),
        "cad_source": ir.params.get("cad_source"),
        "collision_shapes": collision_shapes,
        "contact_bodies_expected_chrono_collision": [
            "pinDisk",
            "driverDisk",
            "cycloidalDisk1",
        ],
    }


def _collision_shape_issue(
    design_ir: dict[str, Any], assets: CycloidalReducerAssets,
) -> str | None:
    shapes = design_ir.get("collision_shapes", {})
    for body_name in design_ir["contact_bodies_expected_chrono_collision"]:
        shape = shapes.get(body_name)
        if not isinstance(shape, dict):
            return f"{body_name} has no Chrono collision shape"
        shape_kind = shape.get("shape")
        if shape_kind == "mesh":
            mesh = shape.get("mesh")
            if not mesh:
                return f"{body_name} mesh collision path is missing"
            mesh_path = assets.root / str(mesh)
            if not mesh_path.is_file() or mesh_path.stat().st_size <= 0:
                return f"{body_name} mesh collision file is missing or empty: {mesh_path}"
        elif shape_kind == "compound":
            if int(shape.get("child_count") or 0) <= 0:
                return f"{body_name} compound collision has no children"
            asset_mesh = shape.get("asset_mesh")
            if body_name in {"pinDisk", "driverDisk"} and not asset_mesh:
                return f"{body_name} compound collision has no exported CAD collision mesh"
            if asset_mesh:
                mesh_path = assets.root / str(asset_mesh)
                if not mesh_path.is_file() or mesh_path.stat().st_size <= 0:
                    return f"{body_name} exported collision mesh is missing or empty: {mesh_path}"
        else:
            return f"{body_name} collision shape is unsupported: {shape_kind!r}"
    return None


def _run_chrono(
    ir: Any,
    assets: CycloidalReducerAssets,
    contact_model: str,
    *,
    output_load_nm: float = 0.75,
    samples: int = 61,
    max_power_error_pct: float = 25.0,
    max_torque_ripple_pct: float = 30.0,
) -> dict[str, Any]:
    cfg = _chrono_config(
        assets,
        contact_model,
        output_load_nm=output_load_nm,
        samples=samples,
        max_power_error_pct=max_power_error_pct,
        max_torque_ripple_pct=max_torque_ripple_pct,
    )
    out = _chrono_impl.run(ir, cfg)
    metadata = out.get("metadata", {})
    metrics = out.get("scalar_metrics", {})
    result = {
        "ok": False,
        "contact_model_requested": contact_model,
        "procedural_cycloidal_fallback": False,
        "capability_unavailable": bool(out.get("__capability_unavailable__")),
        "metadata_contact_model": metadata.get("contact_model"),
        "metadata_contact_method": metadata.get("contact_method"),
        "execution_mode": metadata.get("execution_mode"),
        "chrono_version": metadata.get("chrono_version"),
        "preflight_issues": metadata.get("preflight_issues", []),
        "build_meta": metadata.get("build_meta", {}),
        "contact_config": metadata.get("config", {}),
        "passed": bool(out.get("passed", False)),
        "output_load_Nm": output_load_nm,
        "samples": samples,
        "metrics": {key: metrics.get(key) for key in METRIC_KEYS},
        "failure_mode": metrics.get("failure_mode"),
        "solver_diverged": metrics.get("solver_diverged"),
        "missing_metric_keys": [
            key for key in METRIC_KEYS if key not in metrics
        ],
    }
    result["ok"] = _run_has_real_geometry_metrics(result)
    return result


def _chrono_config(
    assets: CycloidalReducerAssets,
    contact_model: str,
    *,
    output_load_nm: float,
    samples: int,
    max_power_error_pct: float,
    max_torque_ripple_pct: float,
) -> dict[str, Any]:
    return {
        "samples": samples,
        "duration_s": 0.2,
        "timestep": 5.0e-5,
        "contact_model": contact_model,
        "contact_method": contact_model.upper(),
        "procedural_cycloidal_fallback": False,
        "contact_margin": 2.0e-5,
        "contact_envelope": 5.0e-5,
        "friction": 0.02,
        "restitution": 0.0,
        "young_modulus": 5.0e6,
        "normal_stiffness": 5.0e7,
        "damping": 2000.0,
        "solver_iterations": 500,
        "solver_max_iterations": 500,
        "_mech_bench": {
            "build_root": str(assets.root),
            "task": {
                "id": "cycloidal_lowN_stub_s0001",
                "family": "cycloidal_lowN_stub",
                "difficulty": 3,
                "units": "mm",
            },
            "probe_specs": [
                {
                    "id": "torque",
                    "type": "torque_load_trial",
                    "config": {
                        "input_port": "input_port",
                        "output_port": "output_port",
                        "input_speed_rad_s": 10.0,
                        "output_load_Nm": float(output_load_nm),
                        "min_output_speed_rad_s": 0.5,
                        "max_power_error_pct": float(max_power_error_pct),
                        "max_torque_ripple_pct": float(max_torque_ripple_pct),
                    },
                }
            ],
        },
    }


def _run_smc_sample_convergence(
    ir: Any, assets: CycloidalReducerAssets,
) -> dict[str, Any]:
    runs = [
        _run_chrono(
            ir,
            assets,
            "smc",
            output_load_nm=0.0,
            samples=samples,
            max_power_error_pct=1.0e12,
            max_torque_ripple_pct=1.0e12,
        )
        for samples in (41, 61, 81)
    ]
    ratios = [
        _metric_float(run.get("metrics", {}), "ratio_observed", math.inf)
        for run in runs
    ]
    errors = [
        _metric_float(run.get("metrics", {}), "ratio_error_pct", math.inf)
        for run in runs
    ]
    penetrations = [
        _metric_float(run.get("metrics", {}), "max_penetration_mm", math.inf)
        for run in runs
    ]
    lockups = [
        _metric_float(run.get("metrics", {}), "lockup_detected", 1.0)
        for run in runs
    ]
    ratio_span = max(ratios) - min(ratios) if ratios else math.inf
    return {
        "samples": [run["samples"] for run in runs],
        "runs": runs,
        "ratio_observed_values": ratios,
        "ratio_error_pct_max": max(errors) if errors else math.inf,
        "ratio_observed_span": ratio_span,
        "max_penetration_mm_max": max(penetrations) if penetrations else math.inf,
        "ok": (
            all(run.get("ok") for run in runs)
            and max(errors, default=math.inf) <= 15.0
            and ratio_span <= 1.5
            and max(penetrations, default=math.inf) < 1.0
            and max(lockups, default=1.0) == 0.0
        ),
    }


def _run_has_real_geometry_metrics(run: dict[str, Any]) -> bool:
    if run["capability_unavailable"]:
        return False
    if run["execution_mode"] == "procedural_cycloidal_contact_fallback":
        return False
    if run["metadata_contact_model"] != run["contact_model_requested"]:
        return False
    if run["preflight_issues"]:
        return False
    if run["missing_metric_keys"]:
        return False
    metrics = run["metrics"]
    for key in NUMERIC_METRIC_KEYS:
        try:
            value = float(metrics[key])
        except (TypeError, ValueError):
            return False
        if not math.isfinite(value):
            return False
    if float(metrics["n_contacts_max"]) <= 0.0:
        return False
    if float(metrics["contact_force_rms_N"]) <= 0.0:
        return False
    return bool(metrics.get("top_contact_pairs"))


def _evaluate_runs(proof: dict[str, Any]) -> None:
    acceptance = proof["acceptance"]
    assets = proof["asset_generation"]
    acceptance["freecad_export_ran"] = bool(assets["ran"])
    acceptance["manifest_written"] = bool(
        assets["manifest"]["exists"] and assets["manifest"]["bytes"] > 0
    )
    acceptance["named_bodies_exported"] = bool(assets["all_named_bodies_present"])
    acceptance["nonempty_step_stl_exports"] = bool(assets["all_body_step_stl_nonempty"])
    acceptance["cad_datums_present"] = _cad_datums_present(assets)
    acceptance["cad_static_contact_audit_passed"] = (
        _cad_static_contact_audit_passed(assets)
    )
    acceptance["chrono_procedural_fallback_false"] = all(
        run.get("procedural_cycloidal_fallback") is False
        and run.get("execution_mode") != "procedural_cycloidal_contact_fallback"
        for run in proof.get("runs", {}).values()
    )
    acceptance["chrono_nsc_real_geometry_metrics"] = bool(
        proof["runs"]["nsc"]["ok"]
    )
    acceptance["chrono_smc_real_geometry_metrics"] = bool(
        proof["runs"]["smc"]["ok"]
    )
    nsc_metrics = proof["runs"]["nsc"].get("metrics", {})
    smc_metrics = proof["runs"]["smc"].get("metrics", {})
    nsc_pen = _metric_float(nsc_metrics, "max_penetration_mm", 0.0)
    nsc_contacts = _metric_float(nsc_metrics, "n_contacts_max", 0.0)
    smc_contacts = _metric_float(smc_metrics, "n_contacts_max", math.inf)
    acceptance["nsc_bad_regime_observed"] = (
        nsc_pen > 1.0 and nsc_contacts > smc_contacts
    )
    smc_failure = proof["runs"]["smc"].get("failure_mode")
    smc_lockup = _metric_float(smc_metrics, "lockup_detected", 1.0)
    smc_out = _metric_float(smc_metrics, "out_omega_med", 0.0)
    smc_ratio = _metric_float(smc_metrics, "ratio_observed", math.inf)
    smc_pen = _metric_float(smc_metrics, "max_penetration_mm", math.inf)
    acceptance["smc_minimum_success_threshold"] = (
        smc_lockup == 0.0
        and abs(smc_out) > 0.5
        and math.isfinite(smc_ratio)
        and smc_pen < 1.0
        and smc_contacts < nsc_contacts
        and smc_failure not in {"lockup_mechanism_jammed", "solver_diverged"}
    )
    unloaded_metrics = proof["runs"]["smc_unloaded"].get("metrics", {})
    acceptance["smc_unloaded_ratio_near_declared"] = (
        proof["runs"]["smc_unloaded"].get("ok")
        and _metric_float(unloaded_metrics, "lockup_detected", 1.0) == 0.0
        and _metric_float(unloaded_metrics, "ratio_error_pct", math.inf) <= 15.0
        and _metric_float(unloaded_metrics, "max_penetration_mm", math.inf) < 1.0
    )
    acceptance["smc_ratio_convergence"] = bool(
        proof.get("convergence", {}).get("ok"))
    if all(acceptance.values()):
        proof["ok"] = True
        proof["missing_bridge"] = None
        return

    for model in ("nsc", "smc"):
        run = proof["runs"].get(model, {})
        if run.get("capability_unavailable") or run.get("preflight_issues"):
            _mark_failed(proof, "Chrono contact setup", f"{model}: {run}")
            return
        if not run.get("ok"):
            _mark_failed(proof, "solver dynamics", f"{model}: {run}")
            return
    _mark_failed(proof, "solver dynamics", "acceptance gate failed")


def _cad_datums_present(assets: dict[str, Any]) -> bool:
    counts = assets.get("feature_frame_counts", {})
    params = assets.get("parameters", {})
    if not isinstance(counts, dict) or not isinstance(params, dict):
        return False
    ring_expected = _int_value(params.get("ring_pin_count"))
    driver_expected = _int_value(params.get("driver_disk_hole_count"))
    ring_count = _int_value(counts.get("ring_pins"))
    driver_count = _int_value(counts.get("driver_pins"))
    disk1_holes = _int_value(counts.get("cycloidalDisk1_output_holes"))
    disk2_holes = _int_value(counts.get("cycloidalDisk2_output_holes"))
    return (
        ring_expected is not None
        and driver_expected is not None
        and ring_count == ring_expected
        and driver_count == driver_expected
        and disk1_holes == driver_expected
        and disk2_holes == driver_expected
    )


def _cad_static_contact_audit_passed(assets: dict[str, Any]) -> bool:
    audit = assets.get("static_audit", {})
    counts = assets.get("feature_frame_counts", {})
    if not isinstance(audit, dict) or not isinstance(counts, dict):
        return False
    ring_distance = _finite_nonnegative(
        audit.get("ring_pins_to_cycloidalDisk1_distance_mm"))
    driver_distance = _finite_nonnegative(
        audit.get("driver_pins_to_cycloidalDisk1_distance_mm"))
    hole = audit.get("driver_pins_to_cycloidalDisk1_output_holes", {})
    if not isinstance(hole, dict):
        return False
    pair_count = _int_value(hole.get("pair_count"))
    driver_count = _int_value(counts.get("driver_pins"))
    min_clearance = _float_value(hole.get("min_radial_clearance_mm"))
    mean_clearance = _float_value(hole.get("mean_radial_clearance_mm"))
    return (
        ring_distance
        and driver_distance
        and hole.get("status") == "ok"
        and pair_count is not None
        and driver_count is not None
        and pair_count == driver_count
        and min_clearance is not None
        and min_clearance > 0.0
        and mean_clearance is not None
        and mean_clearance > 0.0
    )


def _finite_nonnegative(value: Any) -> bool:
    numeric = _float_value(value)
    return numeric is not None and numeric >= 0.0


def _float_value(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _int_value(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _metric_float(metrics: dict[str, Any], key: str, default: float) -> float:
    try:
        return float(metrics.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def _mark_failed(proof: dict[str, Any], stage: str, message: str) -> None:
    proof["ok"] = False
    proof["missing_bridge"] = stage
    proof["error"] = message


def _file_record(path: Path) -> dict[str, Any]:
    exists = path.is_file()
    size = path.stat().st_size if exists else 0
    return {
        "path": str(path),
        "exists": exists,
        "bytes": size,
        "sha256": _sha256(path) if exists and size > 0 else None,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _freecad_argv(configured: str | None) -> list[str] | None:
    if configured:
        return shlex.split(configured)
    found = find_freecad_command()
    return found if found else None


def _command_output(argv: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            argv,
            text=True,
            capture_output=True,
            timeout=10.0,
            check=False,
        )
    except Exception:
        return None
    text = "\n".join(
        x.strip() for x in (completed.stdout, completed.stderr) if x.strip()
    )
    return text or None


def _git(args: list[str]) -> str | None:
    return _command_output(["git", *args])


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    raise SystemExit(main())
