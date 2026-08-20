"""Headless FreeCAD/OCCT bridge for cycloidal reducer assets.

The reducer geometry comes from the external CycloidGearBox FreeCAD
workbench. This module only maps reducer parameters, invokes FreeCAD in a
subprocess, records provenance, and converts the exported named bodies into
DesignIR parts that Chrono can consume as mesh collision geometry.
"""

from __future__ import annotations

import json
import math
import os
import shlex
import shutil
import struct
import subprocess
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mech_bench.schema import DesignIR, Joint, Part, Port


BODY_NAMES = (
    "pinDisk",
    "driverDisk",
    "inputShaft",
    "cycloidalDisk1",
    "cycloidalDisk2",
    "eccentricKey",
    "outputShaft",
)

DEFAULT_SOURCE_CANDIDATES = (
    str(Path(__file__).resolve().parents[2] / ".external" / "src" / "CycloidGearBox"),
    "/opt/CycloidGearBox",
    "/workspace/third_party/CycloidGearBox",
    "/tmp/CycloidGearBox",
)

MASS_KG_DECIMALS = 12
COM_MM_DECIMALS = 9
INERTIA_KG_M2_DECIMALS = 15


class CycloidalCadExportError(RuntimeError):
    """Raised when the FreeCAD asset bridge cannot produce usable assets."""

    def __init__(self, stage: str, message: str) -> None:
        self.stage = stage
        super().__init__(f"{stage}: {message}")


@dataclass(frozen=True)
class CycloidalReducerAssets:
    """Manifest-backed CAD exports for one cycloidal reducer assembly."""

    root: Path
    manifest_path: Path
    bodies: dict[str, dict[str, Any]]
    collision_meshes: dict[str, dict[str, Any]]
    collision_primitives: dict[str, dict[str, Any]]
    feature_frames: dict[str, Any]
    static_audit: dict[str, Any]
    parameters: dict[str, Any]
    source: dict[str, Any]

    @classmethod
    def from_manifest(cls, manifest_path: str | Path) -> "CycloidalReducerAssets":
        path = Path(manifest_path).resolve()
        data = json.loads(path.read_text())
        root = Path(data.get("root", path.parent)).resolve()
        return cls(
            root=root,
            manifest_path=path,
            bodies=dict(data.get("bodies", {})),
            collision_meshes=dict(data.get("collision_meshes", {})),
            collision_primitives=dict(data.get("collision_primitives", {})),
            feature_frames=dict(data.get("feature_frames", {})),
            static_audit=dict(data.get("static_audit", {})),
            parameters=dict(data.get("parameters", {})),
            source=dict(data.get("source", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "manifest_path": str(self.manifest_path),
            "bodies": self.bodies,
            "collision_meshes": self.collision_meshes,
            "collision_primitives": self.collision_primitives,
            "feature_frames": self.feature_frames,
            "static_audit": self.static_audit,
            "parameters": self.parameters,
            "source": self.source,
        }


def find_freecad_command() -> list[str] | None:
    """Return the configured FreeCAD command as argv, if available."""

    configured = os.environ.get("MECH_BENCH_FREECAD_CMD")
    if configured:
        return shlex.split(configured)
    for name in ("FreeCADCmd", "freecadcmd", "freecad"):
        found = shutil.which(name)
        if found:
            return [found]
    return None


def find_cycloid_gearbox_path() -> Path | None:
    """Return the CycloidGearBox source tree path, if available."""

    configured = os.environ.get("MECH_BENCH_CYCLOID_GEARBOX_PATH")
    candidates = ([configured] if configured else []) + list(DEFAULT_SOURCE_CANDIDATES)
    for raw in candidates:
        if not raw:
            continue
        path = Path(raw).expanduser().resolve()
        if (path / "cycloidFun.py").is_file():
            return path
    return None


def generate_cycloidal_reducer_assets(
    out_dir: str | Path,
    parameters: dict[str, Any] | None = None,
    *,
    freecad_cmd: list[str] | str | None = None,
    cycloid_gearbox_path: str | Path | None = None,
    timeout_s: float = 300.0,
) -> CycloidalReducerAssets:
    """Generate named STEP/STL reducer assets with headless FreeCAD.

    Parameters may use either CycloidGearBox's native names or the benchmark
    aliases ``pins`` / ``ring_pin_count``. ``pins`` means fixed ring pins, so
    it maps to the workbench's ``tooth_count = pins - 1``.
    """

    root = Path(out_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    params = _normalized_cycloid_parameters(parameters or {})
    source_path = (
        Path(cycloid_gearbox_path).expanduser().resolve()
        if cycloid_gearbox_path is not None
        else find_cycloid_gearbox_path()
    )
    if source_path is None or not (source_path / "cycloidFun.py").is_file():
        raise CycloidalCadExportError(
            "CAD export",
            "CycloidGearBox source is not available. Set "
            "MECH_BENCH_CYCLOID_GEARBOX_PATH to a checkout containing "
            "cycloidFun.py.",
        )

    cmd = _freecad_argv(freecad_cmd)
    if cmd is None:
        raise CycloidalCadExportError(
            "CAD export",
            "FreeCAD command is not available. Set MECH_BENCH_FREECAD_CMD "
            "or install FreeCADCmd/freecadcmd.",
        )

    script_path = root / "_generate_cycloidal_freecad_assets.py"
    params_path = root / "cycloidal_parameters.json"
    manifest_path = root / "cycloidal_assets_manifest.json"
    script_path.write_text(_FREECAD_SCRIPT)
    params_path.write_text(json.dumps(params, indent=2, sort_keys=True))

    env = os.environ.copy()
    env.update({
        "MECH_BENCH_CYCLOID_PARAMS_JSON": str(params_path),
        "MECH_BENCH_CYCLOID_MANIFEST_JSON": str(manifest_path),
        "MECH_BENCH_CYCLOID_GEARBOX_PATH": str(source_path),
        "MECH_BENCH_CYCLOID_SOURCE_COMMIT": _source_commit(source_path),
    })

    completed = subprocess.run(
        [*cmd, str(script_path)],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        timeout=float(timeout_s),
        check=False,
    )
    if completed.returncode != 0:
        detail = "\n".join(
            x for x in (completed.stdout.strip(), completed.stderr.strip()) if x
        )
        raise CycloidalCadExportError(
            "CAD export",
            f"FreeCAD asset generation failed with exit code "
            f"{completed.returncode}.\n{detail[-4000:]}",
        )
    if not manifest_path.exists():
        raise CycloidalCadExportError(
            "CAD export",
            "FreeCAD completed but did not write cycloidal_assets_manifest.json.",
        )
    assets = CycloidalReducerAssets.from_manifest(manifest_path)
    missing = [
        name for name in BODY_NAMES
        if name not in assets.bodies
        or not (assets.root / str(assets.bodies[name].get("stl", ""))).is_file()
        or not (assets.root / str(assets.bodies[name].get("step", ""))).is_file()
    ]
    if missing:
        raise CycloidalCadExportError(
            "body naming",
            "FreeCAD export did not produce all required named bodies: "
            + ", ".join(missing),
        )
    return assets


def build_chrono_design_ir_from_assets(
    assets: CycloidalReducerAssets | dict[str, Any] | str | Path,
    *,
    include_secondary_disc: bool = True,
    collidable_body_names: set[str] | None = None,
    collision_sweep_radius_m: float = 2.0e-4,
    use_primitive_pin_collision: bool = True,
    use_cad_collision_primitives: bool = True,
    use_cad_eccentric_body_frames: bool = False,
    use_cad_outer_sidewall_collision: bool = False,
    cad_outer_sidewall_thickness_mm: float = 0.75,
    cad_outer_sidewall_max_hulls: int = 128,
) -> DesignIR:
    """Build a DesignIR that feeds generated STL meshes to Chrono."""

    if not isinstance(assets, CycloidalReducerAssets):
        if isinstance(assets, (str, Path)):
            assets = CycloidalReducerAssets.from_manifest(assets)
        else:
            manifest = assets.get("manifest_path")
            if manifest:
                assets = CycloidalReducerAssets.from_manifest(manifest)
            else:
                root = Path(str(assets["root"])).resolve()
                assets = CycloidalReducerAssets(
                    root=root,
                    manifest_path=root / "cycloidal_assets_manifest.json",
                    bodies=dict(assets.get("bodies", {})),
                    collision_meshes=dict(assets.get("collision_meshes", {})),
                    collision_primitives=dict(assets.get("collision_primitives", {})),
                    feature_frames=dict(assets.get("feature_frames", {})),
                    static_audit=dict(assets.get("static_audit", {})),
                    parameters=dict(assets.get("parameters", {})),
                    source=dict(assets.get("source", {})),
                )

    selected = list(BODY_NAMES)
    if not include_secondary_disc:
        selected.remove("cycloidalDisk2")
    if collidable_body_names is None:
        collidable_body_names = {"pinDisk", "driverDisk", "cycloidalDisk1"}
        if include_secondary_disc:
            collidable_body_names.add("cycloidalDisk2")

    roles = {
        "pinDisk": "ground",
        "driverDisk": "output_pin_driver",
        "inputShaft": "input_shaft",
        "cycloidalDisk1": "cycloidal_disc",
        "cycloidalDisk2": "cycloidal_disc",
        "eccentricKey": "eccentric",
        "outputShaft": "carrier",
    }
    masses = {
        "pinDisk": 1.0,
        "driverDisk": 0.2,
        "inputShaft": 0.15,
        "cycloidalDisk1": 0.12,
        "cycloidalDisk2": 0.12,
        "eccentricKey": 0.08,
        "outputShaft": 0.2,
    }

    parts: list[Part] = []
    collision_meshes = _collision_meshes_by_body(assets)
    collision_primitives = (
        _collision_primitives_by_body(assets) if use_primitive_pin_collision else {}
    )
    if use_cad_collision_primitives:
        collision_primitives.update(_cad_collision_primitives_by_body(assets))
    density_kg_m3 = float(assets.parameters.get("density_kg_m3", 7850.0))
    for name in selected:
        body = assets.bodies.get(name)
        if not body:
            raise CycloidalCadExportError(
                "body naming", f"asset manifest is missing body {name!r}")
        stl = str(body.get("stl", ""))
        if not stl:
            raise CycloidalCadExportError(
                "collision mesh import",
                f"asset manifest body {name!r} has no STL path",
            )
        fixed = name == "pinDisk"
        params = {
            "cad_body_name": name,
            "initial_pose_mm": (0.0, 0.0, 0.0),
        }
        mass_kg, com_mm, inertia = _mass_properties_for_body(
            body, density_kg_m3, masses.get(name, 0.1))
        params["cad_mass_properties"] = {
            "mass_kg": mass_kg,
            "com_local_mm": com_mm,
            "inertia_kg_m2": inertia,
        }
        collision_body = collision_meshes.get(name, body)
        cad_frame_origin = (
            _cad_eccentric_axis_origin(assets, name)
            if use_cad_eccentric_body_frames
            else None
        )
        if cad_frame_origin is not None:
            params["initial_pose_mm"] = cad_frame_origin

        if name in collidable_body_names:
            if name in collision_primitives:
                shape = dict(collision_primitives[name])
                if cad_frame_origin is not None:
                    shape = _offset_collision_shape(
                        shape, tuple(-v for v in cad_frame_origin))
                params["chrono_collision"] = shape
                params["chrono_collision_asset"] = {
                    "mesh": str(collision_body.get("stl", stl)),
                    "step": str(collision_body.get("step", body.get("step", ""))),
                }
            else:
                shape = {
                    "shape": "mesh",
                    "mesh": str(collision_body.get("stl", stl)),
                    "is_static": fixed,
                    "is_convex": False,
                    "sweep_sphere_radius_m": float(collision_sweep_radius_m),
                }
                if cad_frame_origin is not None:
                    shape["center_mm"] = tuple(-v for v in cad_frame_origin)
                sidewall = (
                    _cad_outer_sidewall_collision_shape(
                        assets,
                        name,
                        shell_thickness_mm=cad_outer_sidewall_thickness_mm,
                        max_hulls=cad_outer_sidewall_max_hulls,
                    )
                    if use_cad_outer_sidewall_collision
                    else None
                )
                if sidewall is not None:
                    if cad_frame_origin is not None:
                        sidewall = _offset_collision_shape(
                            sidewall, tuple(-v for v in cad_frame_origin))
                    shape = {
                        "shape": "compound",
                        "source": "cad_mesh_plus_outer_sidewall",
                        "children": [shape, *sidewall["children"]],
                    }
                params["chrono_collision"] = shape
        parts.append(Part(
            id=name,
            role=roles.get(name, ""),
            fixed=fixed,
            mass_kg=mass_kg,
            com_local_mm=com_mm,
            inertia_kg_m2=inertia,
            geometry={"mesh": stl, "step": str(body.get("step", ""))},
            params=params,
        ))

    eccentricity = float(assets.parameters.get("eccentricity", 2.0))
    disk1_axis = (
        _cad_eccentric_axis_origin(assets, "cycloidalDisk1")
        if use_cad_eccentric_body_frames
        else None
    )
    disk2_axis = (
        _cad_eccentric_axis_origin(assets, "cycloidalDisk2")
        if use_cad_eccentric_body_frames
        else None
    )
    joints = [
        Joint(
            id="input_revolute",
            type="revolute",
            parent="pinDisk",
            child="inputShaft",
            axis_world=(0.0, 0.0, 1.0),
            anchor_world_mm=(0.0, 0.0, 0.0),
        ),
        Joint(
            id="eccentric_disc",
            type="revolute",
            parent="inputShaft",
            child="cycloidalDisk1",
            axis_world=(0.0, 0.0, 1.0),
            anchor_world_mm=disk1_axis or (eccentricity, 0.0, 0.0),
        ),
        Joint(
            id="output_revolute",
            type="revolute",
            parent="pinDisk",
            child="outputShaft",
            axis_world=(0.0, 0.0, 1.0),
            anchor_world_mm=(0.0, 0.0, 0.0),
        ),
        Joint(
            id="ring_contact",
            type="contact_pair",
            parent="pinDisk",
            child="cycloidalDisk1",
            axis_world=(0.0, 0.0, 1.0),
            anchor_world_mm=(0.0, 0.0, 0.0),
        ),
        Joint(
            id="output_pin_contact",
            type="contact_pair",
            parent="driverDisk",
            child="cycloidalDisk1",
            axis_world=(0.0, 0.0, 1.0),
            anchor_world_mm=(0.0, 0.0, 0.0),
        ),
    ]
    if "driverDisk" in selected:
        joints.append(Joint(
            id="driver_output_fixed",
            type="fixed",
            parent="outputShaft",
            child="driverDisk",
            axis_world=(0.0, 0.0, 1.0),
            anchor_world_mm=(0.0, 0.0, 0.0),
        ))
    if "eccentricKey" in selected:
        joints.append(Joint(
            id="eccentric_key_fixed",
            type="fixed",
            parent="inputShaft",
            child="eccentricKey",
            axis_world=(0.0, 0.0, 1.0),
            anchor_world_mm=(0.0, 0.0, 0.0),
        ))
    if "cycloidalDisk2" in selected:
        joints.append(Joint(
            id="eccentric_disc_2",
            type="revolute",
            parent="inputShaft",
            child="cycloidalDisk2",
            axis_world=(0.0, 0.0, 1.0),
            anchor_world_mm=disk2_axis or (-eccentricity, 0.0, 0.0),
        ))
        joints.append(Joint(
            id="ring_contact_2",
            type="contact_pair",
            parent="pinDisk",
            child="cycloidalDisk2",
            axis_world=(0.0, 0.0, 1.0),
            anchor_world_mm=(0.0, 0.0, 0.0),
        ))
        joints.append(Joint(
            id="output_pin_contact_2",
            type="contact_pair",
            parent="driverDisk",
            child="cycloidalDisk2",
            axis_world=(0.0, 0.0, 1.0),
            anchor_world_mm=(0.0, 0.0, 0.0),
        ))

    ring_pins = int(assets.parameters.get("ring_pin_count", 0) or 0)
    if ring_pins <= 0:
        ring_pins = int(assets.parameters.get("tooth_count", 9)) + 1

    return DesignIR(
        schema_version="design_ir.v2",
        parts=parts,
        joints=joints,
        ports={
            "input_port": Port(
                id="input_port",
                part="input_revolute",
                kind="revolute_joint",
            ),
            "output_port": Port(
                id="output_port",
                part="output_revolute",
                kind="revolute_joint",
            ),
        },
        params={
            "pins": ring_pins,
            "declared_ratio": float(assets.parameters.get(
                "declared_ratio", max(1, ring_pins - 1))),
            "cad_assets_manifest": str(assets.manifest_path),
            "cad_source": assets.source,
            "cad_feature_frames": assets.feature_frames,
            "cad_static_audit": assets.static_audit,
        },
    )


def audit_cycloidal_static_geometry(
    assets: CycloidalReducerAssets | dict[str, Any] | str | Path,
) -> dict[str, Any]:
    """Return CAD-exported static geometry checks for the reducer assembly."""

    if not isinstance(assets, CycloidalReducerAssets):
        if isinstance(assets, (str, Path)):
            assets = CycloidalReducerAssets.from_manifest(assets)
        else:
            manifest = assets.get("manifest_path")
            if manifest:
                assets = CycloidalReducerAssets.from_manifest(manifest)
            else:
                root = Path(str(assets["root"])).resolve()
                assets = CycloidalReducerAssets(
                    root=root,
                    manifest_path=root / "cycloidal_assets_manifest.json",
                    bodies=dict(assets.get("bodies", {})),
                    collision_meshes=dict(assets.get("collision_meshes", {})),
                    collision_primitives=dict(assets.get("collision_primitives", {})),
                    feature_frames=dict(assets.get("feature_frames", {})),
                    static_audit=dict(assets.get("static_audit", {})),
                    parameters=dict(assets.get("parameters", {})),
                    source=dict(assets.get("source", {})),
                )
    audit = dict(assets.static_audit or {})
    audit["feature_frame_counts"] = {
        key: len(value)
        for key, value in (assets.feature_frames or {}).items()
        if isinstance(value, list)
    }
    return audit


def _collision_meshes_by_body(
    assets: CycloidalReducerAssets,
) -> dict[str, dict[str, Any]]:
    raw = getattr(assets, "collision_meshes", None)
    if raw is None:
        data = json.loads(assets.manifest_path.read_text())
        raw = data.get("collision_meshes", {})
    if not isinstance(raw, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    mapping = {
        "ringPins": "pinDisk",
        "driverPins": "driverDisk",
    }
    for mesh_name, body_name in mapping.items():
        spec = raw.get(mesh_name)
        if isinstance(spec, dict) and spec.get("stl"):
            out[body_name] = dict(spec)
    return out


def _cad_collision_primitives_by_body(
    assets: CycloidalReducerAssets,
) -> dict[str, dict[str, Any]]:
    raw = getattr(assets, "collision_primitives", None)
    if raw is None:
        data = json.loads(assets.manifest_path.read_text())
        raw = data.get("collision_primitives", {})
    if not isinstance(raw, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for body_name in ("cycloidalDisk1", "cycloidalDisk2"):
        spec = raw.get(body_name)
        children = spec.get("children") if isinstance(spec, dict) else None
        if (
            isinstance(spec, dict)
            and spec.get("shape") == "compound"
            and isinstance(children, list)
            and children
        ):
            out[body_name] = dict(spec)
    return out


def _cad_outer_sidewall_collision_shape(
    assets: CycloidalReducerAssets,
    body_name: str,
    *,
    shell_thickness_mm: float,
    max_hulls: int,
) -> dict[str, Any] | None:
    if not body_name.startswith("cycloidalDisk"):
        return None
    body = assets.bodies.get(body_name)
    if not isinstance(body, dict):
        return None
    rel = body.get("stl")
    if not rel:
        return None
    path = assets.root / str(rel)
    if not path.is_file():
        return None

    bbox = [float(v) for v in list(body.get("bbox_mm", ()))[:6]]
    if len(bbox) < 6:
        return None
    zmin, zmax = bbox[2], bbox[5]
    height = max(zmax - zmin, 1.0e-9)
    params = assets.parameters
    try:
        cutoff_radius = (
            float(params["driver_circle_diameter"]) / 2.0
            + float(params["driver_hole_diameter"]) / 2.0
            + float(params.get("clearance", 0.0))
            + 0.75
        )
    except (KeyError, TypeError, ValueError):
        cutoff_radius = max(
            math.hypot(bbox[0], bbox[1]),
            math.hypot(bbox[3], bbox[4]),
        ) * 0.78

    children: list[dict[str, Any]] = []
    thickness = max(float(shell_thickness_mm), 1.0e-6)
    for triangle in _read_stl_triangles_mm(path):
        zs = [p[2] for p in triangle]
        if max(zs) - min(zs) < 0.45 * height:
            continue
        cx = sum(p[0] for p in triangle) / 3.0
        cy = sum(p[1] for p in triangle) / 3.0
        radius = math.hypot(cx, cy)
        if radius < cutoff_radius:
            continue
        ux, uy = (-cx / radius, -cy / radius) if radius > 1.0e-9 else (0.0, 0.0)
        points = [[p[0], p[1], p[2]] for p in triangle]
        points.extend(
            [p[0] + ux * thickness, p[1] + uy * thickness, p[2]]
            for p in triangle
        )
        children.append({"shape": "convex_hull", "points_mm": points})

    if not children:
        return None
    max_hulls = max(int(max_hulls), 1)
    if len(children) > max_hulls:
        stride = max(1, len(children) // max_hulls)
        children = children[::stride][:max_hulls]
    return {
        "shape": "compound",
        "source": "cad_stl_outer_sidewall",
        "children": children,
        "shell_thickness_mm": thickness,
        "radial_cutoff_mm": cutoff_radius,
    }


def _read_stl_triangles_mm(
    path: Path,
) -> list[tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]]:
    data = path.read_bytes()
    if len(data) >= 84:
        count = struct.unpack_from("<I", data, 80)[0]
        expected = 84 + count * 50
        if expected == len(data):
            triangles = []
            offset = 84
            for _ in range(count):
                vals = struct.unpack_from("<12fH", data, offset)
                offset += 50
                triangles.append((
                    (vals[3] * 1000.0, vals[4] * 1000.0, vals[5] * 1000.0),
                    (vals[6] * 1000.0, vals[7] * 1000.0, vals[8] * 1000.0),
                    (vals[9] * 1000.0, vals[10] * 1000.0, vals[11] * 1000.0),
                ))
            return triangles

    vertices: list[tuple[float, float, float]] = []
    for line in data.decode("utf-8", errors="ignore").splitlines():
        parts = line.strip().split()
        if len(parts) == 4 and parts[0].lower() == "vertex":
            try:
                vertices.append(tuple(float(v) * 1000.0 for v in parts[1:4]))
            except ValueError:
                continue
    return [
        (vertices[i], vertices[i + 1], vertices[i + 2])
        for i in range(0, len(vertices) - 2, 3)
    ]


def _mass_properties_for_body(
    body: dict[str, Any],
    density_kg_m3: float,
    fallback_mass_kg: float,
) -> tuple[
    float,
    tuple[float, float, float],
    tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]],
]:
    cad_mass: float | None = None
    cad_com: tuple[float, float, float] | None = None
    cad_inertia: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ] | None = None
    props = body.get("mass_properties")
    if isinstance(props, dict):
        try:
            mass = float(props["mass_kg"])
            com_vals = [float(v) for v in list(props["com_mm"])[:3]]
            if len(com_vals) == 3 and math.isfinite(mass) and mass > 0.0:
                cad_mass = mass
                cad_com = tuple(com_vals)  # type: ignore[assignment]
        except (KeyError, TypeError, ValueError):
            pass
        try:
            inertia_rows = tuple(
                tuple(float(v) for v in list(row)[:3])
                for row in list(props["inertia_kg_m2"])[:3]
            )
            if (
                len(inertia_rows) == 3
                and all(len(row) == 3 for row in inertia_rows)
                and all(math.isfinite(v) for row in inertia_rows for v in row)
            ):
                cad_inertia = inertia_rows  # type: ignore[assignment]
        except (KeyError, TypeError, ValueError):
            pass

    if cad_mass is not None and cad_com is not None and cad_inertia is not None:
        return _canonical_chrono_mass_properties(
            cad_mass,
            cad_com,
            cad_inertia,
        )

    bbox = [float(v) for v in list(body.get("bbox_mm", (0, 0, 0, 1, 1, 1)))[:6]]
    while len(bbox) < 6:
        bbox.append(0.0)
    sx = max((bbox[3] - bbox[0]) / 1000.0, 1e-6)
    sy = max((bbox[4] - bbox[1]) / 1000.0, 1e-6)
    sz = max((bbox[5] - bbox[2]) / 1000.0, 1e-6)
    volume_m3 = sx * sy * sz
    mass = (
        float(cad_mass)
        if cad_mass is not None
        else max(float(fallback_mass_kg), density_kg_m3 * volume_m3 * 0.35)
    )
    com = cad_com or (
        (bbox[0] + bbox[3]) / 2.0,
        (bbox[1] + bbox[4]) / 2.0,
        (bbox[2] + bbox[5]) / 2.0,
    )
    ixx = mass * (sy * sy + sz * sz) / 12.0
    iyy = mass * (sx * sx + sz * sz) / 12.0
    izz = mass * (sx * sx + sy * sy) / 12.0
    return _canonical_chrono_mass_properties(
        mass,
        com,
        ((ixx, 0.0, 0.0), (0.0, iyy, 0.0), (0.0, 0.0, izz)),
    )


def _canonical_chrono_mass_properties(
    mass_kg: float,
    com_mm: tuple[float, float, float],
    inertia_kg_m2: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ],
) -> tuple[
    float,
    tuple[float, float, float],
    tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]],
]:
    """Suppress FreeCAD/OCCT export jitter before Chrono consumes the body."""

    mass = round(float(mass_kg), MASS_KG_DECIMALS)
    com = tuple(round(float(v), COM_MM_DECIMALS) for v in com_mm[:3])
    inertia = tuple(
        tuple(round(float(v), INERTIA_KG_M2_DECIMALS) for v in row[:3])
        for row in inertia_kg_m2[:3]
    )
    return mass, com, inertia  # type: ignore[return-value]


def _collision_primitives_by_body(
    assets: CycloidalReducerAssets,
) -> dict[str, dict[str, Any]]:
    """Return exact primitive collision shapes for cylindrical CAD features."""

    params = assets.parameters
    try:
        tooth_count = int(params["tooth_count"])
        ring_count = tooth_count + 1
        disk_height = float(params["disk_height"])
        base_height = float(params["base_height"])
        clearance = float(params.get("clearance", 0.0))
        roller_circle_diameter = float(params["roller_circle_diameter"])
        roller_diameter = float(params["roller_diameter"])
        driver_count = int(params["driver_disk_hole_count"])
        driver_circle_diameter = float(params["driver_circle_diameter"])
        driver_hole_diameter = float(params["driver_hole_diameter"])
    except (KeyError, TypeError, ValueError):
        return {}

    ring_radius = roller_circle_diameter / 2.0 + clearance
    ring_pin_radius = roller_diameter / 2.0
    ring_height = base_height + disk_height * 3.0
    driver_radius = driver_circle_diameter / 2.0
    shrink = float(params.get(
        "driver_pin_collision_shrink_mm",
        min(max(clearance, 0.0), 0.5),
    ))
    # Keep the SMC contact proxy inside the CAD clearance band so generated
    # pin/hole assemblies do not start with artificial interference.
    driver_pin_radius = max(driver_hole_diameter / 2.0 - shrink, 0.1)
    driver_z = base_height - disk_height
    driver_height = disk_height * 4.0

    return {
        "pinDisk": {
            "shape": "compound",
            "children": _cylinder_pattern(
                count=ring_count,
                radius_mm=ring_radius,
                cylinder_radius_mm=ring_pin_radius,
                height_mm=ring_height,
                z_base_mm=0.0,
            ),
        },
        "driverDisk": {
            "shape": "compound",
            "children": _cylinder_pattern(
                count=driver_count,
                radius_mm=driver_radius,
                cylinder_radius_mm=driver_pin_radius,
                height_mm=driver_height,
                z_base_mm=driver_z,
            ),
        },
    }


def _cylinder_pattern(
    *,
    count: int,
    radius_mm: float,
    cylinder_radius_mm: float,
    height_mm: float,
    z_base_mm: float,
) -> list[dict[str, Any]]:
    shapes: list[dict[str, Any]] = []
    if count <= 0:
        return shapes
    for idx in range(count):
        angle = 2.0 * math.pi * idx / count
        shapes.append({
            "shape": "cylinder",
            "radius_mm": float(cylinder_radius_mm),
            "height_mm": float(height_mm),
            "center_mm": (
                float(radius_mm * math.cos(angle)),
                float(radius_mm * math.sin(angle)),
                float(z_base_mm + height_mm / 2.0),
            ),
            "axis": (0.0, 0.0, 1.0),
        })
    return shapes


def _cad_eccentric_axis_origin(
    assets: CycloidalReducerAssets,
    body_name: str,
) -> tuple[float, float, float] | None:
    axes = assets.feature_frames.get("axes")
    if not isinstance(axes, dict):
        return None
    feature = axes.get(f"{body_name}_eccentric_axis")
    if not isinstance(feature, dict):
        return None
    if feature.get("body") not in (None, body_name):
        return None
    center = feature.get("center_mm")
    try:
        vals = [float(v) for v in list(center)[:3]]
    except (TypeError, ValueError):
        return None
    if len(vals) != 3 or not all(math.isfinite(v) for v in vals):
        return None
    return (vals[0], vals[1], vals[2])


def _offset_collision_shape(
    shape: dict[str, Any],
    offset_mm: tuple[float, float, float],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in shape.items():
        if isinstance(value, dict):
            out[key] = _offset_collision_shape(value, offset_mm)
        elif key in {"children", "shapes"} and isinstance(value, list):
            out[key] = [
                _offset_collision_shape(child, offset_mm)
                if isinstance(child, dict) else child
                for child in value
            ]
        else:
            out[key] = value

    for key in ("center_mm", "pos_mm", "position_mm"):
        if key in out:
            out[key] = _offset_vec3(out[key], offset_mm)
            break
    else:
        if str(out.get("shape", "")).lower() == "mesh":
            out["center_mm"] = tuple(float(v) for v in offset_mm)
    if isinstance(out.get("points_mm"), list):
        out["points_mm"] = [
            _offset_vec3(point, offset_mm) for point in out["points_mm"]
        ]
    if isinstance(out.get("points_m"), list):
        offset_m = tuple(v * 0.001 for v in offset_mm)
        out["points_m"] = [
            _offset_vec3(point, offset_m) for point in out["points_m"]
        ]
    return out


def _offset_vec3(
    raw: Any,
    offset_mm: tuple[float, float, float],
) -> tuple[float, float, float]:
    vals = [float(v) for v in list(raw)[:3]]
    while len(vals) < 3:
        vals.append(0.0)
    return (
        vals[0] + offset_mm[0],
        vals[1] + offset_mm[1],
        vals[2] + offset_mm[2],
    )


def _freecad_argv(cmd: list[str] | str | None) -> list[str] | None:
    if cmd is None:
        return find_freecad_command()
    if isinstance(cmd, str):
        return shlex.split(cmd)
    return [str(x) for x in cmd]


def _normalized_cycloid_parameters(raw: dict[str, Any]) -> dict[str, Any]:
    params = dict(raw)
    pins_raw = params.pop("ring_pin_count", params.get("pins"))
    if pins_raw is not None:
        ring_pins = max(4, int(pins_raw))
        params["ring_pin_count"] = ring_pins
        params["tooth_count"] = ring_pins - 1
        params.setdefault("declared_ratio", float(ring_pins - 1))
    elif "tooth_count" in params:
        tooth_count = max(3, int(params["tooth_count"]))
        params["tooth_count"] = tooth_count
        params["ring_pin_count"] = tooth_count + 1
        params.setdefault("declared_ratio", float(tooth_count))
    else:
        params["ring_pin_count"] = 12
        params.setdefault("declared_ratio", 11.0)

    tooth_count = max(3, int(params["tooth_count"]))
    params["tooth_count"] = tooth_count
    params["line_segment_count"] = max(
        int(params.get("line_segment_count", 0) or 0),
        max(42, tooth_count * 4),
    )
    return params


def _source_commit(path: Path) -> str:
    configured = os.environ.get("MECH_BENCH_CYCLOID_GEARBOX_REF")
    if configured:
        return configured
    try:
        completed = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            capture_output=True,
            timeout=5.0,
            check=False,
        )
    except Exception:
        return "unknown"
    if completed.returncode == 0:
        return completed.stdout.strip()
    return "unknown"


_FREECAD_SCRIPT = textwrap.dedent(r'''
    import json
    import math
    import os
    import sys
    import traceback
    from pathlib import Path

    source_path = Path(os.environ["MECH_BENCH_CYCLOID_GEARBOX_PATH"]).resolve()
    params_path = Path(os.environ["MECH_BENCH_CYCLOID_PARAMS_JSON"]).resolve()
    manifest_path = Path(os.environ["MECH_BENCH_CYCLOID_MANIFEST_JSON"]).resolve()
    source_commit = os.environ.get("MECH_BENCH_CYCLOID_SOURCE_COMMIT", "unknown")

    sys.path.insert(0, str(source_path))

    BODY_NAMES = (
        "pinDisk",
        "driverDisk",
        "inputShaft",
        "cycloidalDisk1",
        "cycloidalDisk2",
        "eccentricKey",
        "outputShaft",
    )

    def _fail(stage, message):
        print(json.dumps({"stage": stage, "error": message}), file=sys.stderr)
        sys.exit(2)

    try:
        import FreeCAD as App
        import Mesh
        import Part
        try:
            import Import
        except Exception:
            Import = None
        import cycloidFun
    except Exception:
        _fail("CAD export", traceback.format_exc())

    def _export_step(objs, path):
        if Import is not None:
            try:
                Import.export(objs, str(path))
                return
            except Exception:
                pass
        Part.export(objs, str(path))

    def _export_stl(obj, path, *, scale=0.001):
        vertices, facets = obj.Shape.tessellate(0.15)
        mesh = Mesh.Mesh()
        for i, j, k in facets:
            vi = vertices[i]
            vj = vertices[j]
            vk = vertices[k]
            mesh.addFacet(
                App.Vector(vi.x * scale, vi.y * scale, vi.z * scale),
                App.Vector(vj.x * scale, vj.y * scale, vj.z * scale),
                App.Vector(vk.x * scale, vk.y * scale, vk.z * scale),
            )
        mesh.write(str(path))

    def _feature_from_shape(doc, name, shape):
        obj = doc.addObject("Part::Feature", name)
        obj.Shape = shape
        return obj

    def _cylinder_compound(cylinders):
        shapes = []
        for cyl in cylinders:
            shapes.append(Part.makeCylinder(
                cyl["radius"],
                cyl["height"],
                App.Vector(cyl["x"], cyl["y"], cyl["z"]),
                App.Vector(0, 0, 1),
            ))
        return Part.makeCompound(shapes)

    def _export_collision_feature(doc, name, shape, exports_dir, root):
        obj = _feature_from_shape(doc, name, shape)
        step_path = exports_dir / f"{name}.step"
        stl_path = exports_dir / f"{name}.stl"
        _export_step([obj], step_path)
        _export_stl(obj, stl_path)
        if not step_path.exists() or step_path.stat().st_size <= 0:
            _fail("CAD export", f"empty STEP export for {name}")
        if not stl_path.exists() or stl_path.stat().st_size <= 0:
            _fail("CAD export", f"empty STL export for {name}")
        bb = shape.BoundBox
        return {
            "step": str(step_path.relative_to(root)),
            "stl": str(stl_path.relative_to(root)),
            "stl_units": "m",
            "bbox_mm": [
                float(bb.XMin), float(bb.YMin), float(bb.ZMin),
                float(bb.XMax), float(bb.YMax), float(bb.ZMax),
            ],
        }

    def _bbox_inertia_kg_m2(shape, mass_kg):
        bb = shape.BoundBox
        sx = max(float(bb.XLength) / 1000.0, 1.0e-9)
        sy = max(float(bb.YLength) / 1000.0, 1.0e-9)
        sz = max(float(bb.ZLength) / 1000.0, 1.0e-9)
        ixx = mass_kg * (sy * sy + sz * sz) / 12.0
        iyy = mass_kg * (sx * sx + sz * sz) / 12.0
        izz = mass_kg * (sx * sx + sy * sy) / 12.0
        return [[ixx, 0.0, 0.0], [0.0, iyy, 0.0], [0.0, 0.0, izz]]

    def _shape_mass_properties(shape, density_kg_m3):
        volume_mm3 = max(float(getattr(shape, "Volume", 0.0)), 0.0)
        mass_kg = max(volume_mm3 * 1.0e-9 * density_kg_m3, 1.0e-9)
        com = getattr(shape, "CenterOfMass", App.Vector(0, 0, 0))
        inertia = None
        moi = getattr(shape, "MatrixOfInertia", None)
        if moi is not None:
            try:
                vals = [
                    [float(moi.A11), float(moi.A12), float(moi.A13)],
                    [float(moi.A21), float(moi.A22), float(moi.A23)],
                    [float(moi.A31), float(moi.A32), float(moi.A33)],
                ]
                c = [float(com.x), float(com.y), float(com.z)]
                c2 = c[0] * c[0] + c[1] * c[1] + c[2] * c[2]
                shift = [
                    [c2 - c[0] * c[0], -c[0] * c[1], -c[0] * c[2]],
                    [-c[1] * c[0], c2 - c[1] * c[1], -c[1] * c[2]],
                    [-c[2] * c[0], -c[2] * c[1], c2 - c[2] * c[2]],
                ]
                inertia = [
                    [
                        (vals[r][cidx] - volume_mm3 * shift[r][cidx])
                        * density_kg_m3 * 1.0e-15
                        for cidx in range(3)
                    ]
                    for r in range(3)
                ]
                if min(inertia[0][0], inertia[1][1], inertia[2][2]) <= 0.0:
                    inertia = None
            except Exception:
                inertia = None
        if inertia is None:
            inertia = _bbox_inertia_kg_m2(shape, mass_kg)
        return {
            "volume_mm3": volume_mm3,
            "density_kg_m3": float(density_kg_m3),
            "mass_kg": mass_kg,
            "com_mm": [float(com.x), float(com.y), float(com.z)],
            "inertia_kg_m2": inertia,
        }

    def _triangle_area_xy(a, b, c):
        return abs(
            (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
        ) * 0.5

    def _convex_prism_decomposition(shape, *, tessellation_mm=0.6):
        zmin = float(shape.BoundBox.ZMin)
        zmax = float(shape.BoundBox.ZMax)
        vertices, facets = shape.tessellate(float(tessellation_mm))
        children = []
        tol = max(0.02, (zmax - zmin) * 0.02)
        for i, j, k in facets:
            tri = (vertices[i], vertices[j], vertices[k])
            if not all(abs(float(v.z) - zmax) <= tol for v in tri):
                continue
            if _triangle_area_xy(*tri) <= 0.01:
                continue
            points = []
            for z in (zmin, zmax):
                for v in tri:
                    points.append([float(v.x), float(v.y), float(z)])
            children.append({
                "shape": "convex_hull",
                "points_mm": points,
            })
        return {
            "shape": "compound",
            "source": "freecad_occt_top_face_prism_decomposition",
            "children": children,
        }

    def _world_point(obj, x, y, z):
        p = obj.Placement.multVec(App.Vector(float(x), float(y), float(z)))
        return [float(p.x), float(p.y), float(p.z)]

    def _point_feature(name, body, center_mm, radius_mm=None):
        out = {
            "name": str(name),
            "body": str(body),
            "center_mm": [float(v) for v in center_mm],
            "axis": [0.0, 0.0, 1.0],
        }
        if radius_mm is not None:
            out["radius_mm"] = float(radius_mm)
        return out

    def _circular_pattern_features(
        *,
        name,
        body,
        count,
        radius_mm,
        feature_radius_mm,
        z_local_mm,
        obj=None,
        origin_x_mm=0.0,
        origin_y_mm=0.0,
    ):
        features = []
        for idx in range(int(count)):
            angle = 2.0 * math.pi * idx / int(count)
            x = float(origin_x_mm) + float(radius_mm) * math.cos(angle)
            y = float(origin_y_mm) + float(radius_mm) * math.sin(angle)
            center = (
                _world_point(obj, x, y, z_local_mm)
                if obj is not None
                else [float(x), float(y), float(z_local_mm)]
            )
            item = _point_feature(
                f"{name}_{idx:02d}",
                body,
                center,
                feature_radius_mm,
            )
            item["index"] = idx
            item["angle_rad"] = float(angle)
            features.append(item)
        return features

    def _shape_distance_mm(shape_a, shape_b):
        try:
            result = shape_a.distToShape(shape_b)
            return float(result[0])
        except Exception:
            return None

    def _pin_hole_clearance_audit(pin_features, hole_features):
        pairs = []
        for pin in pin_features:
            pc = pin["center_mm"]
            pr = float(pin.get("radius_mm", 0.0))
            best = None
            for hole in hole_features:
                hc = hole["center_mm"]
                hr = float(hole.get("radius_mm", 0.0))
                dx = float(pc[0]) - float(hc[0])
                dy = float(pc[1]) - float(hc[1])
                dist = math.sqrt(dx * dx + dy * dy)
                clearance = hr - pr - dist
                row = {
                    "pin": pin["name"],
                    "hole": hole["name"],
                    "center_distance_mm": float(dist),
                    "radial_clearance_mm": float(clearance),
                }
                if best is None or dist < best["center_distance_mm"]:
                    best = row
            if best is not None:
                pairs.append(best)
        clearances = [p["radial_clearance_mm"] for p in pairs]
        if not clearances:
            return {
                "status": "missing_features",
                "pair_count": 0,
                "pairs": [],
            }
        return {
            "status": "ok",
            "pair_count": len(pairs),
            "min_radial_clearance_mm": float(min(clearances)),
            "max_radial_clearance_mm": float(max(clearances)),
            "mean_radial_clearance_mm": float(sum(clearances) / len(clearances)),
            "pairs": pairs,
        }

    def _cycloidal_feature_frames(doc, params):
        tooth_count = int(params["tooth_count"])
        ring_count = tooth_count + 1
        disk_height = float(params["disk_height"])
        base_height = float(params["base_height"])
        clearance = float(params.get("clearance", 0.0))
        eccentricity = float(params["eccentricity"])
        driver_count = int(params["driver_disk_hole_count"])
        driver_radius = float(params["driver_circle_diameter"]) / 2.0
        driver_hole_radius = float(params["driver_hole_diameter"]) / 2.0
        disk_output_hole_radius = driver_hole_radius + eccentricity
        ring_radius = float(params["roller_circle_diameter"]) / 2.0 + clearance
        ring_pin_radius = float(params["roller_diameter"]) / 2.0
        driver_shrink = float(params.get(
            "driver_pin_collision_shrink_mm",
            min(max(clearance, 0.0), 0.5),
        ))
        driver_collision_radius = max(driver_hole_radius - driver_shrink, 0.1)

        driver_obj = doc.getObject("driverDisk")
        disk1_obj = doc.getObject("cycloidalDisk1")
        disk2_obj = doc.getObject("cycloidalDisk2")
        driver_z_local = disk_height * 2.0
        disk_z_local = disk_height / 2.0
        return {
            "ring_pins": _circular_pattern_features(
                name="ring_pin",
                body="pinDisk",
                count=ring_count,
                radius_mm=ring_radius,
                feature_radius_mm=ring_pin_radius,
                z_local_mm=(base_height + disk_height * 3.0) / 2.0,
            ),
            "driver_pins": _circular_pattern_features(
                name="driver_pin",
                body="driverDisk",
                count=driver_count,
                radius_mm=driver_radius,
                feature_radius_mm=driver_collision_radius,
                z_local_mm=driver_z_local,
                obj=driver_obj,
            ),
            "driver_pin_nominal_radius_mm": float(driver_hole_radius),
            "driver_pin_collision_radius_mm": float(driver_collision_radius),
            "cycloidalDisk1_output_holes": _circular_pattern_features(
                name="cycloidalDisk1_output_hole",
                body="cycloidalDisk1",
                count=driver_count,
                radius_mm=driver_radius,
                feature_radius_mm=disk_output_hole_radius,
                z_local_mm=disk_z_local,
                obj=disk1_obj,
                origin_x_mm=eccentricity,
            ),
            "cycloidalDisk2_output_holes": _circular_pattern_features(
                name="cycloidalDisk2_output_hole",
                body="cycloidalDisk2",
                count=driver_count,
                radius_mm=driver_radius,
                feature_radius_mm=disk_output_hole_radius,
                z_local_mm=disk_z_local,
                obj=disk2_obj,
                origin_x_mm=eccentricity,
            ),
            "axes": {
                "input_axis": _point_feature(
                    "input_axis", "inputShaft", [0.0, 0.0, base_height / 2.0]),
                "output_axis": _point_feature(
                    "output_axis", "outputShaft",
                    [0.0, 0.0, base_height + disk_height * 3.0]),
                "eccentric_axis": _point_feature(
                    "eccentric_axis", "inputShaft",
                    [eccentricity, 0.0, base_height + disk_height / 2.0]),
                "cycloidalDisk1_eccentric_axis": _point_feature(
                    "cycloidalDisk1_eccentric_axis",
                    "cycloidalDisk1",
                    _world_point(disk1_obj, eccentricity, 0.0, disk_z_local),
                ),
                "cycloidalDisk2_eccentric_axis": _point_feature(
                    "cycloidalDisk2_eccentric_axis",
                    "cycloidalDisk2",
                    _world_point(disk2_obj, eccentricity, 0.0, disk_z_local),
                ),
            },
        }

    def _ready_part(doc, name):
        part = cycloidFun.ready_part(doc, name)
        return part

    def _generate_parts(doc, params):
        cycloidFun.validate_parameters(params)
        minr, maxr = cycloidFun.calculate_min_max_radii(params)
        params["min_rad"] = minr
        params["max_rad"] = maxr
        steps = (
            ("pinDisk", cycloidFun.generate_pin_disk_part, ()),
            ("driverDisk", cycloidFun.generate_driver_disk_part, ()),
            ("inputShaft", cycloidFun.generate_input_shaft_part, ()),
            ("cycloidalDisk1", cycloidFun.generate_cycloidal_disk_part, (True,)),
            ("cycloidalDisk2", cycloidFun.generate_cycloidal_disk_part, (False,)),
            ("eccentricKey", cycloidFun.generate_eccentric_key_part, ()),
            ("outputShaft", cycloidFun.generate_output_shaft_part, ()),
        )
        for name, fn, extra in steps:
            part = _ready_part(doc, name)
            fn(part, params, *extra)
        doc.recompute()

    def _apply_cycloidal_disk_phase_alignment(doc, params):
        if not bool(params.get("align_output_pin_holes", True)):
            return
        base_height = float(params["base_height"])
        disk_height = float(params["disk_height"])
        phases = {
            "cycloidalDisk1": float(params.get(
                "cycloidal_disk1_phase_deg", 180.0)),
            "cycloidalDisk2": float(params.get(
                "cycloidal_disk2_phase_deg", 0.0)),
        }
        z_offsets = {
            "cycloidalDisk1": base_height,
            "cycloidalDisk2": base_height + disk_height,
        }
        for name, phase_deg in phases.items():
            obj = doc.getObject(name)
            if obj is None:
                continue
            obj.Placement = App.Placement(
                App.Vector(0, 0, z_offsets[name]),
                App.Rotation(App.Vector(0, 0, 1), phase_deg),
            )
        doc.recompute()

    try:
        overrides = json.loads(params_path.read_text())
        doc = App.newDocument("CycloidalReducer")
        params = cycloidFun.generate_default_parameters()
        for key, value in overrides.items():
            if key in {"pins"}:
                continue
            if key == "ring_pin_count":
                params[key] = value
                continue
            params[key] = value
        density_kg_m3 = float(params.get("density_kg_m3", 7850.0))
        _generate_parts(doc, params)
        _apply_cycloidal_disk_phase_alignment(doc, params)

        root = manifest_path.parent
        exports_dir = root / "cycloidal_freecad_assets"
        exports_dir.mkdir(parents=True, exist_ok=True)
        bodies = {}
        collision_meshes = {}
        collision_primitives = {}
        feature_frames = _cycloidal_feature_frames(doc, params)
        static_audit = {}
        exported_objects = []
        for name in BODY_NAMES:
            obj = doc.getObject(name)
            if obj is None:
                _fail("body naming", f"missing FreeCAD object {name}")
            shape = getattr(obj, "Shape", None)
            if shape is None or shape.isNull():
                _fail("body naming", f"FreeCAD object {name} has no shape")
            step_path = exports_dir / f"{name}.step"
            stl_path = exports_dir / f"{name}.stl"
            _export_step([obj], step_path)
            _export_stl(obj, stl_path)
            if not step_path.exists() or step_path.stat().st_size <= 0:
                _fail("CAD export", f"empty STEP export for {name}")
            if not stl_path.exists() or stl_path.stat().st_size <= 0:
                _fail("CAD export", f"empty STL export for {name}")
            bb = shape.BoundBox
            bodies[name] = {
                "step": str(step_path.relative_to(root)),
                "stl": str(stl_path.relative_to(root)),
                "stl_units": "m",
                "bbox_mm": [
                    float(bb.XMin), float(bb.YMin), float(bb.ZMin),
                    float(bb.XMax), float(bb.YMax), float(bb.ZMax),
                ],
                "mass_properties": _shape_mass_properties(shape, density_kg_m3),
            }
            if name in {"cycloidalDisk1", "cycloidalDisk2"}:
                collision_primitives[name] = _convex_prism_decomposition(shape)
            exported_objects.append(obj)

        tooth_count = int(params["tooth_count"])
        ring_count = tooth_count + 1
        disk_height = float(params["disk_height"])
        base_height = float(params["base_height"])
        clearance = float(params.get("clearance", 0.0))
        ring_radius = float(params["roller_circle_diameter"]) / 2.0 + clearance
        ring_pin_radius = float(params["roller_diameter"]) / 2.0
        ring_pin_height = base_height + disk_height * 3.0
        ring_cylinders = []
        for idx in range(ring_count):
            angle = 2.0 * 3.141592653589793 * idx / ring_count
            ring_cylinders.append({
                "x": ring_radius * math.cos(angle),
                "y": ring_radius * math.sin(angle),
                "z": 0.0,
                "radius": ring_pin_radius,
                "height": ring_pin_height,
            })
        ring_shape = _cylinder_compound(ring_cylinders)
        collision_meshes["ringPins"] = _export_collision_feature(
            doc, "ringPinsCollision",
            ring_shape, exports_dir, root)

        driver_count = int(params["driver_disk_hole_count"])
        driver_radius = float(params["driver_circle_diameter"]) / 2.0
        driver_pin_radius = float(params["driver_hole_diameter"]) / 2.0
        driver_z = base_height - disk_height
        driver_height = disk_height * 4.0
        driver_cylinders = []
        for idx in range(driver_count):
            angle = 2.0 * 3.141592653589793 * idx / driver_count
            driver_cylinders.append({
                "x": driver_radius * math.cos(angle),
                "y": driver_radius * math.sin(angle),
                "z": driver_z,
                "radius": driver_pin_radius,
                "height": driver_height,
            })
        driver_shape = _cylinder_compound(driver_cylinders)
        collision_meshes["driverPins"] = _export_collision_feature(
            doc, "driverPinsCollision",
            driver_shape, exports_dir, root)

        disk1_shape = doc.getObject("cycloidalDisk1").Shape
        disk2_shape = doc.getObject("cycloidalDisk2").Shape
        static_audit = {
            "ring_pins_to_cycloidalDisk1_distance_mm": _shape_distance_mm(
                ring_shape, disk1_shape),
            "ring_pins_to_cycloidalDisk2_distance_mm": _shape_distance_mm(
                ring_shape, disk2_shape),
            "driver_pins_to_cycloidalDisk1_distance_mm": _shape_distance_mm(
                driver_shape, disk1_shape),
            "driver_pins_to_cycloidalDisk2_distance_mm": _shape_distance_mm(
                driver_shape, disk2_shape),
            "driver_pins_to_cycloidalDisk1_output_holes": (
                _pin_hole_clearance_audit(
                    feature_frames["driver_pins"],
                    feature_frames["cycloidalDisk1_output_holes"],
                )
            ),
            "driver_pins_to_cycloidalDisk2_output_holes": (
                _pin_hole_clearance_audit(
                    feature_frames["driver_pins"],
                    feature_frames["cycloidalDisk2_output_holes"],
                )
            ),
        }

        assembly_step = exports_dir / "cycloidal_reducer_assembly.step"
        _export_step(exported_objects, assembly_step)
        doc_path = exports_dir / "cycloidal_reducer.FCStd"
        doc.saveAs(str(doc_path))

        manifest = {
            "schema_version": "cycloidal_freecad_assets.v1",
            "root": str(root),
            "body_names": list(BODY_NAMES),
            "bodies": bodies,
            "collision_meshes": collision_meshes,
            "collision_primitives": collision_primitives,
            "feature_frames": feature_frames,
            "static_audit": static_audit,
            "assembly_step": str(assembly_step.relative_to(root)),
            "freecad_document": str(doc_path.relative_to(root)),
            "parameters": params,
            "source": {
                "generator": "iplayfast/CycloidGearBox",
                "url": "https://github.com/iplayfast/CycloidGearBox",
                "commit": source_commit,
                "path": str(source_path),
                "kernel": "FreeCAD/OCCT",
            },
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    except SystemExit:
        raise
    except Exception:
        _fail("CAD export", traceback.format_exc())
''').lstrip()
