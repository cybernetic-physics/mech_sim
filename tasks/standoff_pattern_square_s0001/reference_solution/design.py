# auto-generated; do not edit by hand. See mech_bench.generators.
from pathlib import Path


def build_design(out_dir: Path) -> dict:
    parts = [
    {
        'id': 'plate',
        'role': 'ground',
        'mass_kg': 0.0,
        'fixed': True,
        'com_local_mm': (0.0, 0.0, 0.0),
    },
]
    joints = []
    ports = {
    'so_1': {
        'id': 'so_1',
        'part': 'plate',
        'kind': 'frame',
        'pose_local_mm': (0.0, 0.0, 0.0),
    },
    'so_2': {
        'id': 'so_2',
        'part': 'plate',
        'kind': 'frame',
        'pose_local_mm': (58.874, 0.0, 0.0),
    },
    'so_3': {
        'id': 'so_3',
        'part': 'plate',
        'kind': 'frame',
        'pose_local_mm': (58.874, 58.874, 0.0),
    },
    'so_4': {
        'id': 'so_4',
        'part': 'plate',
        'kind': 'frame',
        'pose_local_mm': (0.0, 58.874, 0.0),
    },
}
    params = {
    'side_length_mm': 58.874,
    'standoff_count': 4,
}
    ir = {
        'schema_version': 'design_ir.v2',
        'parts': parts,
        'joints': joints,
        'ports': ports,
        'params': params,
    }
    return ir
