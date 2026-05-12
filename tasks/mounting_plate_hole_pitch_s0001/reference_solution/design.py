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
    'mount_a': {
        'id': 'mount_a',
        'part': 'plate',
        'kind': 'frame',
        'pose_local_mm': (0.0, 0.0, 0.0),
    },
    'mount_b': {
        'id': 'mount_b',
        'part': 'plate',
        'kind': 'frame',
        'pose_local_mm': (40.566, 0.0, 0.0),
    },
}
    params = {
    'hole_diameter_mm': 3.659,
    'declared_pitch_mm': 40.566,
}
    ir = {
        'schema_version': 'design_ir.v2',
        'parts': parts,
        'joints': joints,
        'ports': ports,
        'params': params,
    }
    return ir
