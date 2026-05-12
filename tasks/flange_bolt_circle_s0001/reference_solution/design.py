# auto-generated; do not edit by hand. See mech_bench.generators.
from pathlib import Path


def build_design(out_dir: Path) -> dict:
    parts = [
    {
        'id': 'flange',
        'role': 'ground',
        'mass_kg': 0.0,
        'fixed': True,
        'com_local_mm': (0.0, 0.0, 0.0),
    },
]
    joints = []
    ports = {
    'flange_axis': {
        'id': 'flange_axis',
        'part': 'flange',
        'kind': 'frame',
        'pose_local_mm': (0.0, 0.0, 0.0),
    },
    'bolt_ref': {
        'id': 'bolt_ref',
        'part': 'flange',
        'kind': 'frame',
        'pose_local_mm': (52.5555, 0.0, 0.0),
    },
}
    params = {
    'declared_bolt_circle_mm': 105.111,
    'declared_bolt_count': 4,
}
    ir = {
        'schema_version': 'design_ir.v2',
        'parts': parts,
        'joints': joints,
        'ports': ports,
        'params': params,
    }
    return ir
