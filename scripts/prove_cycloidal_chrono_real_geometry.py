#!/usr/bin/env python3
"""Generate paper-facing proof for the real-geometry Chrono reducer path.

The proof intentionally exercises the non-procedural path:

1. FreeCAD/OCCT exports named STEP/STL bodies through CycloidGearBox.
2. DesignIR is built from the manifest with mesh collision for contact bodies.
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


METRIC_KEYS = (
    "lockup_detected",
    "ratio_observed",
    "out_omega_med",
    "max_penetration_mm",
    "n_contacts_max",
    "contact_force_rms_N",
    "top_contact_pairs",
)

NUMERIC_METRIC_KEYS = (
    "lockup_detected",
    "ratio_observed",
    "out_omega_med",
    "max_penetration_mm",
    "n_contacts_max",
    "contact_force_rms_N",
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
            {"pins": 10, "line_segment_count": 42, "clearance": 0.6},
            freecad_cmd=args.freecad_cmd,
            cycloid_gearbox_path=args.cycloid_gearbox_path,
            timeout_s=300.0,
        )
        proof["asset_generation"] = _asset_proof(assets)

        ir = build_chrono_design_ir_from_assets(
            assets,
            collision_sweep_radius_m=2.0e-5,
            use_primitive_pin_collision=False,
            use_cad_collision_primitives=False,
        )
        proof["design_ir"] = _design_ir_proof(ir, assets)
        mesh_issue = _mesh_collision_issue(proof["design_ir"], assets)
        if mesh_issue:
            _mark_failed(proof, "collision mesh import", mesh_issue)
        else:
            proof["runs"] = {
                "nsc": _run_chrono(ir, assets, "nsc"),
                "smc": _run_chrono(ir, assets, "smc"),
            }
            _evaluate_runs(proof)
    except CycloidalCadExportError as exc:
        _mark_failed(proof, exc.stage, str(exc))
    except Exception as exc:  # noqa: BLE001 - proof boundary
        _mark_failed(proof, "solver dynamics", f"{type(exc).__name__}: {exc}")

    _write_json(proof_json, proof)
    print(json.dumps(proof, indent=2, sort_keys=True, allow_nan=False))
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
        "validation_scope": "real-geometry solver smoke; not empirical hardware calibration and not a reducer-ratio accuracy claim",
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
            "chrono_procedural_fallback_false": False,
            "chrono_nsc_real_geometry_metrics": False,
            "chrono_smc_real_geometry_metrics": False,
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
                "roller_diameter",
                "roller_circle_diameter",
            )
        },
        "body_names": sorted(assets.bodies),
        "bodies": bodies,
        "collision_meshes": collision_meshes,
        "all_named_bodies_present": sorted(assets.bodies) == sorted(BODY_NAMES),
        "all_body_step_stl_nonempty": all(all_body_exports),
    }


def _design_ir_proof(ir: Any, assets: CycloidalReducerAssets) -> dict[str, Any]:
    collision_shapes: dict[str, Any] = {}
    for part in ir.parts:
        shape = (part.params or {}).get("chrono_collision")
        if not isinstance(shape, dict):
            continue
        mesh = shape.get("mesh") or shape.get("stl") or shape.get("collision_mesh")
        collision_shapes[part.id] = {
            "shape": shape.get("shape"),
            "mesh": mesh,
            "mesh_file": _file_record(assets.root / str(mesh)) if mesh else None,
            "is_static": shape.get("is_static"),
            "is_convex": shape.get("is_convex"),
            "sweep_sphere_radius_m": shape.get("sweep_sphere_radius_m"),
        }
    return {
        "parts": [part.id for part in ir.parts],
        "joints": [joint.id for joint in ir.joints],
        "ports": sorted(ir.ports),
        "cad_assets_manifest": ir.params.get("cad_assets_manifest"),
        "cad_source": ir.params.get("cad_source"),
        "collision_shapes": collision_shapes,
        "contact_bodies_expected_mesh_collision": [
            "pinDisk",
            "driverDisk",
            "cycloidalDisk1",
        ],
    }


def _mesh_collision_issue(design_ir: dict[str, Any], assets: CycloidalReducerAssets) -> str | None:
    shapes = design_ir.get("collision_shapes", {})
    for body_name in design_ir["contact_bodies_expected_mesh_collision"]:
        shape = shapes.get(body_name)
        if not isinstance(shape, dict):
            return f"{body_name} has no Chrono collision shape"
        if shape.get("shape") != "mesh":
            return f"{body_name} collision shape is {shape.get('shape')!r}, not 'mesh'"
        mesh = shape.get("mesh")
        if not mesh:
            return f"{body_name} mesh collision path is missing"
        mesh_path = assets.root / str(mesh)
        if not mesh_path.is_file() or mesh_path.stat().st_size <= 0:
            return f"{body_name} mesh collision file is missing or empty: {mesh_path}"
    return None


def _run_chrono(ir: Any, assets: CycloidalReducerAssets, contact_model: str) -> dict[str, Any]:
    cfg = _chrono_config(assets, contact_model)
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
        "passed": bool(out.get("passed", False)),
        "metrics": {key: metrics.get(key) for key in METRIC_KEYS},
        "failure_mode": metrics.get("failure_mode"),
        "solver_diverged": metrics.get("solver_diverged"),
        "missing_metric_keys": [
            key for key in METRIC_KEYS if key not in metrics
        ],
    }
    result["ok"] = _run_has_real_geometry_metrics(result)
    return result


def _chrono_config(assets: CycloidalReducerAssets, contact_model: str) -> dict[str, Any]:
    return {
        "samples": 5,
        "duration_s": 5.0e-5,
        "dt": 1.0e-5,
        "timestep": 1.0e-5,
        "contact_model": contact_model,
        "contact_method": contact_model.upper(),
        "procedural_cycloidal_fallback": False,
        "contact_margin": 0.0,
        "contact_envelope": 0.0,
        "friction": 0.01,
        "restitution": 0.0,
        "young_modulus": 1.0e3,
        "normal_stiffness": 10.0,
        "damping": 0.1,
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
                        "input_speed_rad_s": 0.01,
                        "output_load_Nm": 1.0e-5,
                        "min_output_speed_rad_s": 1.0e-12,
                        "max_power_error_pct": 1.0e12,
                        "max_torque_ripple_pct": 1.0e12,
                    },
                }
            ],
        },
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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    raise SystemExit(main())
