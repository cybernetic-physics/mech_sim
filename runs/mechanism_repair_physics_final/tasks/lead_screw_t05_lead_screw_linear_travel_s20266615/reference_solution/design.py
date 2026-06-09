# auto-generated; do not edit by hand. See mech_bench.generators.
from pathlib import Path


def _base_build_design(out_dir: Path) -> dict:
    parts = [
    {
        'id': 'frame',
        'role': 'ground',
        'mass_kg': 0.0,
        'fixed': True,
        'com_local_mm': (0.0, 0.0, 0.0),
    },
    {
        'id': 'screw',
        'role': 'input',
        'mass_kg': 0.02,
        'com_local_mm': (0.0, 0.0, 0.0),
    },
    {
        'id': 'nut',
        'role': 'slider',
        'mass_kg': 0.08,
        'com_local_mm': (0.0, 0.0, 0.0),
    },
]
    joints = [
    {
        'id': 'input_axis',
        'type': 'revolute',
        'parent': 'frame',
        'child': 'screw',
        'axis_world': (0.0, 0.0, 1.0),
        'anchor_world_mm': (0.0, 0.0, 0.0),
    },
    {
        'id': 'output_axis',
        'type': 'prismatic',
        'parent': 'frame',
        'child': 'nut',
        'axis_world': (1.0, 0.0, 0.0),
        'anchor_world_mm': (0.0, 0.0, 0.0),
    },
]
    ports = {
    'input_port': {
        'id': 'input_port',
        'part': 'input_axis',
        'kind': 'revolute_joint',
        'pose_local_mm': (0.0, 0.0, 0.0),
    },
    'output_port': {
        'id': 'output_port',
        'part': 'output_axis',
        'kind': 'prismatic_joint',
        'pose_local_mm': (0.0, 0.0, 0.0),
    },
}
    params = {
    'lead_mm': 6.197,
    'declared_travel_per_rev_mm': 6.197,
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
            "family": 'lead_screw',
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
    for index, part in enumerate(ir.get("parts", []) or []):
        if not isinstance(part, dict):
            continue
        part_id = _physics_safe_id(part.get("id", f"part_{index}"))
        part.setdefault("material", "steel_1045")
        part.setdefault("com_local_mm", (0.0, 0.0, 0.0))
        geom = part.setdefault("geometry", {})
        cad_name = geom.setdefault("cad", f"{part_id}.step")
        cad_path = out_path / cad_name
        if not cad_path.exists():
            cad_path.write_text(_physics_stub_step(part_id))
        mass = float(part.get("mass_kg", 0.0) or 0.0)
        if mass > 0.0:
            pparams = part.setdefault("params", {})
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
