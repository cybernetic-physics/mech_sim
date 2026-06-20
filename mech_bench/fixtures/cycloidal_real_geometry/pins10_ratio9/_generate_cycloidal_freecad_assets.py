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
