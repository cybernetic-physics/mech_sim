# auto-generated; do not edit by hand. See mech_bench.generators.
from pathlib import Path


def build_design(out_dir: Path) -> dict:
    parts = [
    {
        'id': 'body',
        'role': 'ground',
        'mass_kg': 0.0,
        'fixed': True,
        'com_local_mm': (0.0, 0.0, 0.0),
    },
    {
        'id': 'tab',
        'role': 'tab',
        'mass_kg': 0.02,
        'com_local_mm': (0.0, 0.0, 0.0),
    },
]
    joints = [
    {
        'id': 'snap',
        'type': 'fixed',
        'parent': 'body',
        'child': 'tab',
        'axis_world': (0.0, 0.0, 1.0),
        'anchor_world_mm': (0.0, 0.0, 0.0),
    },
]
    ports = {
    'body_face': {
        'id': 'body_face',
        'part': 'body',
        'kind': 'frame',
        'pose_local_mm': (0.0, 0.0, 0.0),
    },
    'tab_face': {
        'id': 'tab_face',
        'part': 'tab',
        'kind': 'frame',
        'pose_local_mm': (0.0, 0.0, 0.0),
    },
}
    params = {
    'gap_mm': 0.45,
    'min_wall_mm': 2.865,
}
    ir = {
        'schema_version': 'design_ir.v2',
        'parts': parts,
        'joints': joints,
        'ports': ports,
        'params': params,
    }
    return ir
