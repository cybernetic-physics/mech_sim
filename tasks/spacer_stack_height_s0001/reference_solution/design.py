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
        'id': 'stack',
        'role': 'stack',
        'mass_kg': 0.04,
        'com_local_mm': (0.0, 0.0, 0.0),
    },
]
    joints = [
    {
        'id': 'stack_fix',
        'type': 'fixed',
        'parent': 'base',
        'child': 'stack',
        'axis_world': (0.0, 0.0, 1.0),
        'anchor_world_mm': (0.0, 0.0, 0.0),
    },
]
    ports = {
    'base_face': {
        'id': 'base_face',
        'part': 'base',
        'kind': 'frame',
        'pose_local_mm': (0.0, 0.0, 0.0),
    },
    'stack_top': {
        'id': 'stack_top',
        'part': 'stack',
        'kind': 'frame',
        'pose_local_mm': (0.0, 0.0, 15.726),
    },
}
    params = {
    'spacer_count': 3,
    'spacer_height_mm': 5.242,
    'declared_stack_height_mm': 15.726,
}
    ir = {
        'schema_version': 'design_ir.v2',
        'parts': parts,
        'joints': joints,
        'ports': ports,
        'params': params,
    }
    return ir
