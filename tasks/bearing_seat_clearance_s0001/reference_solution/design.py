# auto-generated; do not edit by hand. See mech_bench.generators.
from pathlib import Path


def build_design(out_dir: Path) -> dict:
    parts = [
    {
        'id': 'housing',
        'role': 'ground',
        'mass_kg': 0.0,
        'fixed': True,
        'com_local_mm': (0.0, 0.0, 0.0),
    },
    {
        'id': 'bearing',
        'role': 'bearing',
        'mass_kg': 0.04,
        'com_local_mm': (0.0, 0.0, 0.0),
    },
]
    joints = [
    {
        'id': 'press',
        'type': 'fixed',
        'parent': 'housing',
        'child': 'bearing',
        'axis_world': (0.0, 0.0, 1.0),
        'anchor_world_mm': (0.0, 0.0, 0.0),
    },
]
    ports = {
    'bore_face': {
        'id': 'bore_face',
        'part': 'housing',
        'kind': 'frame',
        'pose_local_mm': (0.0, 0.0, 0.0),
    },
    'bearing_seat': {
        'id': 'bearing_seat',
        'part': 'bearing',
        'kind': 'frame',
        'pose_local_mm': (0.0, 0.0, 0.0),
    },
}
    params = {
    'bore_diameter_mm': 29.923,
    'bearing_od_mm': 29.873,
    'clearance_mm': 0.05,
}
    ir = {
        'schema_version': 'design_ir.v2',
        'parts': parts,
        'joints': joints,
        'ports': ports,
        'params': params,
    }
    return ir
