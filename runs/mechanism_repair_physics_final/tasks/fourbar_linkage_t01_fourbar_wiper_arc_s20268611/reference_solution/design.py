# auto-generated; do not edit by hand. See mech_bench.generators.
from pathlib import Path


def _base_build_design(out_dir: Path) -> dict:
    parts = [
    {
        'id': 'ground',
        'role': 'ground',
        'mass_kg': 0.0,
        'fixed': True,
        'com_local_mm': (55.0, 0.0, 0.0),
    },
    {
        'id': 'crank',
        'role': 'crank',
        'mass_kg': 0.02,
        'com_local_mm': (15.36, 0.0, 0.0),
    },
    {
        'id': 'coupler',
        'role': 'coupler',
        'mass_kg': 0.06,
        'com_local_mm': (43.47, 0.0, 0.0),
    },
    {
        'id': 'rocker',
        'role': 'rocker',
        'mass_kg': 0.05,
        'com_local_mm': (45.14, 0.0, 0.0),
    },
]
    joints = [
    {
        'id': 'joint_input',
        'type': 'revolute',
        'parent': 'ground',
        'child': 'crank',
        'axis_world': (0.0, 0.0, 1.0),
        'anchor_world_mm': (0.0, 0.0, 0.0),
    },
    {
        'id': 'joint_bc',
        'type': 'revolute',
        'parent': 'crank',
        'child': 'coupler',
        'axis_world': (0.0, 0.0, 1.0),
        'anchor_world_mm': (30.72, 0.0, 0.0),
    },
    {
        'id': 'joint_cd',
        'type': 'revolute',
        'parent': 'coupler',
        'child': 'rocker',
        'axis_world': (0.0, 0.0, 1.0),
        'anchor_world_mm': (66.62693491422806, 79.17863111386424, 0.0),
    },
    {
        'id': 'joint_output',
        'type': 'revolute',
        'parent': 'ground',
        'child': 'rocker',
        'axis_world': (0.0, 0.0, 1.0),
        'anchor_world_mm': (110.0, 0.0, 0.0),
    },
]
    ports = {
    'input_port': {
        'id': 'input_port',
        'part': 'joint_input',
        'kind': 'revolute_joint',
        'pose_local_mm': (0.0, 0.0, 0.0),
    },
    'output_port': {
        'id': 'output_port',
        'part': 'joint_output',
        'kind': 'revolute_joint',
        'pose_local_mm': (0.0, 0.0, 0.0),
    },
    'coupler_point': {
        'id': 'coupler_point',
        'part': 'coupler',
        'kind': 'frame',
        'pose_local_mm': (46.5, 26.64, 0.0),
    },
}
    params = {
    'link_lengths_mm': {
        'ground': 110.0,
        'crank': 30.72,
        'coupler': 86.94,
        'rocker': 90.28,
    },
}
    ir = {
        'schema_version': 'design_ir.v2',
        'parts': parts,
        'joints': joints,
        'ports': ports,
        'params': params,
    }
    return ir



def _physics_safe_id(raw):
    text = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in str(raw))
    return text or "part"


def _physics_stub_step(part_id):
    return (
        "ISO-10303-21;\n"
        "HEADER;\n"
        "FILE_DESCRIPTION(('MechanismRepair-Physics trusted preflight stub'),'2;1');\n"
        "FILE_NAME('" + _physics_safe_id(part_id) + ".step','2026-06-10',('mech_bench'),"
        "('corl'),'trusted_asset_preflight','trusted_asset_preflight','');\n"
        "ENDSEC;\n"
        "DATA;\n"
        "ENDSEC;\n"
        "END-ISO-10303-21;\n"
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
    for pair in []:
        left, _, right = str(pair).partition(":")
        if left:
            bodies.add(left)
        if right:
            bodies.add(right)
    declared_pair = (ir.get("params") or {}).get("declared_pair")
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
            return {
                "shape": "cylinder",
                "radius_mm": 31.0,
                "height_mm": 12.0,
                "center_mm": center,
                "axis": (0.0, 0.0, 1.0),
            }
        if part_id == "disc" or role == "cycloidal_disc":
            return {
                "shape": "cylinder",
                "radius_mm": 30.98,
                "height_mm": 8.0,
                "center_mm": center,
                "axis": (0.0, 0.0, 1.0),
            }
    if family == "rack_pinion":
        if part_id == "pinion":
            return {
                "shape": "cylinder",
                "radius_mm": 14.9,
                "height_mm": 8.0,
                "center_mm": center,
                "axis": (0.0, 0.0, 1.0),
            }
        if part_id == "rack":
            return {
                "shape": "box",
                "size_mm": (80.0, 3.0, 8.0),
                "center_mm": center,
            }
    if family == "cam_follower":
        if part_id == "cam":
            return {
                "shape": "cylinder",
                "radius_mm": 20.0,
                "height_mm": 8.0,
                "center_mm": center,
                "axis": (0.0, 0.0, 1.0),
            }
        if part_id == "follower":
            return {
                "shape": "cylinder",
                "radius_mm": 20.0,
                "height_mm": 8.0,
                "center_mm": center,
                "axis": (0.0, 0.0, 1.0),
            }
    if family == "geneva_indexer":
        if part_id == "driver":
            return {
                "shape": "cylinder",
                "radius_mm": 20.0,
                "height_mm": 8.0,
                "center_mm": center,
                "axis": (0.0, 0.0, 1.0),
            }
        if part_id == "geneva":
            return {
                "shape": "cylinder",
                "radius_mm": 20.05,
                "height_mm": 8.0,
                "center_mm": center,
                "axis": (0.0, 0.0, 1.0),
            }
    return {
        "shape": "box",
        "size_mm": (20.0, 20.0, 8.0),
        "center_mm": center,
    }


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
    materials = ir.setdefault("materials", {})
    materials.setdefault(
        "steel_1045",
        {
            "name": "AISI 1045 steel",
            "density_kg_m3": 7850.0,
            "elastic_modulus_pa": 205000000000.0,
            "poisson_ratio": 0.29,
            "yield_strength_pa": 530000000.0,
            "process": "machined_or_ground_reference",
            "provenance": "MechanismRepair-Physics preflight material table",
        },
    )
    params = ir.setdefault("params", {})
    params.setdefault(
        "cad_source",
        {
            "kernel": "FreeCAD/OCCT trusted preflight bridge",
            "source": "scripts.prepare_mechanism_repair_physics_benchmark",
            "family": 'fourbar_linkage',
            "verifier_level": 2,
        },
    )
    provenance = ir.setdefault("provenance", {})
    provenance.setdefault(
        "mechanism_repair_physics",
        {
            "trusted_asset_bridge": True,
            "note": (
                "This preflight bridge supplies CAD-relative artifacts and "
                "mass-property evidence for verifier testing; final paper runs "
                "must replace or validate it with the trusted CAD/OCCT pipeline."
            ),
        },
    )
    if 2 >= 3:
        chrono_cfg = params.setdefault("chrono", {})
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
        _physics_contact_body_ids(ir) if 2 >= 3 else set()
    )
    _physics_adjust_chrono_joints(ir, 'fourbar_linkage')
    for index, part in enumerate(ir.get("parts", []) or []):
        if not isinstance(part, dict):
            continue
        part_id = _physics_safe_id(part.get("id", f"part_{index}"))
        part.setdefault("material", "steel_1045")
        part.setdefault("com_local_mm", (0.0, 0.0, 0.0))
        pparams = part.setdefault("params", {})
        initial_pose = _physics_default_initial_pose_mm(part, 'fourbar_linkage')
        if initial_pose is not None:
            pparams.setdefault("initial_pose_mm", initial_pose)
        if part_id in contact_bodies:
            pparams.setdefault(
                "chrono_collision",
                _physics_default_chrono_collision(part, 'fourbar_linkage'),
            )
        geom = part.setdefault("geometry", {})
        cad_name = geom.setdefault("cad", f"{part_id}.step")
        cad_path = out_path / cad_name
        if not cad_path.exists():
            cad_path.write_text(_physics_stub_step(part_id))
        mass = float(part.get("mass_kg", 0.0) or 0.0)
        if mass > 0.0:
            scale = max(mass, 1.0e-6)
            pparams.setdefault(
                "cad_mass_properties",
                {
                    "mass_kg": mass,
                    "com_local_mm": tuple(part.get("com_local_mm", (0.0, 0.0, 0.0))),
                    "inertia_kg_m2": (
                        (scale * 1.0e-5, 0.0, 0.0),
                        (0.0, scale * 1.2e-5, 0.0),
                        (0.0, 0.0, scale * 1.5e-5),
                    ),
                },
            )
    return ir


def build_design(out_dir):
    return _physics_enrich_design(_base_build_design(out_dir), out_dir)
