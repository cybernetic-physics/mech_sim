# auto-generated; do not edit by hand. See mech_bench.generators.
from pathlib import Path


def build_design(out_dir: Path) -> dict:
    parts = [
    {
        'id': 'frame',
        'role': 'ground',
        'mass_kg': 0.0,
        'fixed': True,
        'com_local_mm': (0.0, 0.0, 0.0),
    },
    {
        'id': 'pinion',
        'role': 'input',
        'mass_kg': 0.04,
        'com_local_mm': (0.0, 0.0, 0.0),
    },
    {
        'id': 'rack',
        'role': 'output',
        'mass_kg': 0.08,
        'com_local_mm': (0.0, 0.0, 0.0),
    },
]
    joints = [
    {
        'id': 'input_axis',
        'type': 'revolute',
        'parent': 'frame',
        'child': 'pinion',
        'axis_world': (0.0, 0.0, 1.0),
        'anchor_world_mm': (0.0, 0.0, 0.0),
    },
    {
        'id': 'output_axis',
        'type': 'prismatic',
        'parent': 'frame',
        'child': 'rack',
        'axis_world': (1.0, 0.0, 0.0),
        'anchor_world_mm': (0.0, 0.0, 0.0),
    },
    {
        'id': 'pinion_rack_contact',
        'type': 'contact_pair',
        'parent': 'pinion',
        'child': 'rack',
        'axis_world': (0.0, 0.0, 1.0),
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
    'declared_pair': 'pinion:rack',
    'pitch_radius_mm': 21.018,
    'declared_travel_per_rev_mm': 132.06,
}
    ir = {
        'schema_version': 'design_ir.v2',
        'parts': parts,
        'joints': joints,
        'ports': ports,
        'params': params,
    }
    return ir
