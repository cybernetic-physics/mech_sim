# auto-generated; do not edit by hand. See mech_bench.generators.
from pathlib import Path


def build_design(out_dir: Path) -> dict:
    parts = [
    {
        'id': 'base',
        'role': 'ground',
        'mass_kg': 0.0,
        'fixed': True,
        'com_local_mm': (0.0, 0.0, 0.0),
    },
    {
        'id': 'lid',
        'role': 'lid',
        'mass_kg': 0.04,
        'com_local_mm': (0.0, 0.0, 0.0),
    },
]
    joints = [
    {
        'id': 'register',
        'type': 'fixed',
        'parent': 'base',
        'child': 'lid',
        'axis_world': (0.0, 0.0, 1.0),
        'anchor_world_mm': (0.0, 0.0, 0.0),
    },
]
    ports = {
    'base_frame': {
        'id': 'base_frame',
        'part': 'base',
        'kind': 'frame',
        'pose_local_mm': (0.0, 0.0, 0.0),
    },
    'lid_frame': {
        'id': 'lid_frame',
        'part': 'lid',
        'kind': 'frame',
        'pose_local_mm': (0.0, 0.0, 0.0),
    },
}
    params = {
    'register_clearance_mm': 0.204,
}
    ir = {
        'schema_version': 'design_ir.v2',
        'parts': parts,
        'joints': joints,
        'ports': ports,
        'params': params,
    }
    return ir
