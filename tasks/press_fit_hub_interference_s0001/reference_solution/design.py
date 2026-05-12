# auto-generated; do not edit by hand. See mech_bench.generators.
from pathlib import Path


def build_design(out_dir: Path) -> dict:
    parts = [
    {
        'id': 'hub',
        'role': 'ground',
        'mass_kg': 0.0,
        'fixed': True,
        'com_local_mm': (0.0, 0.0, 0.0),
    },
    {
        'id': 'shaft',
        'role': 'shaft',
        'mass_kg': 0.05,
        'com_local_mm': (0.0, 0.0, 0.0),
    },
]
    joints = [
    {
        'id': 'press',
        'type': 'fixed',
        'parent': 'hub',
        'child': 'shaft',
        'axis_world': (0.0, 0.0, 1.0),
        'anchor_world_mm': (0.0, 0.0, 0.0),
    },
]
    ports = {
    'hub_face': {
        'id': 'hub_face',
        'part': 'hub',
        'kind': 'frame',
        'pose_local_mm': (0.0, 0.0, 0.0),
    },
    'shaft_origin': {
        'id': 'shaft_origin',
        'part': 'shaft',
        'kind': 'frame',
        'pose_local_mm': (0.0, 0.0, 0.0),
    },
}
    params = {
    'nominal_diameter_mm': 21.338,
    'interference_mm': 0.0222,
}
    ir = {
        'schema_version': 'design_ir.v2',
        'parts': parts,
        'joints': joints,
        'ports': ports,
        'params': params,
    }
    return ir
