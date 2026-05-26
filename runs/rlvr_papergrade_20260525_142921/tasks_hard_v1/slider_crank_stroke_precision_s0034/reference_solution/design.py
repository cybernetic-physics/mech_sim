# auto-generated; do not edit by hand. See mech_bench.generators.
from pathlib import Path


def build_design(out_dir: Path) -> dict:
    parts = [
    {
        'id': 'ground',
        'role': 'ground',
        'mass_kg': 0.0,
        'fixed': True,
        'com_local_mm': (0.0, 0.0, 0.0),
    },
    {
        'id': 'crank',
        'role': 'crank',
        'mass_kg': 0.02,
        'com_local_mm': (12.66, 0.0, 0.0),
    },
    {
        'id': 'coupler',
        'role': 'coupler',
        'mass_kg': 0.05,
        'com_local_mm': (32.96, 0.0, 0.0),
    },
    {
        'id': 'slider',
        'role': 'slider',
        'mass_kg': 0.08,
        'com_local_mm': (0.0, 0.0, 0.0),
    },
]
    joints = [
    {
        'id': 'joint_input',
        'type': 'revolute',
        'parent': 'ground',
        'child': 'crank',
        'axis_world': (0.0, 0.0, 1.0),
        'anchor_world_mm': (0.0, 0.0, 0.0),
    },
    {
        'id': 'joint_bc',
        'type': 'revolute',
        'parent': 'crank',
        'child': 'coupler',
        'axis_world': (0.0, 0.0, 1.0),
        'anchor_world_mm': (25.32, 0.0, 0.0),
    },
    {
        'id': 'joint_cs',
        'type': 'revolute',
        'parent': 'coupler',
        'child': 'slider',
        'axis_world': (0.0, 0.0, 1.0),
        'anchor_world_mm': (91.24000000000001, 0.0, 0.0),
    },
    {
        'id': 'joint_slide',
        'type': 'prismatic',
        'parent': 'ground',
        'child': 'slider',
        'axis_world': (1.0, 0.0, 0.0),
        'anchor_world_mm': (0.0, 0.0, 0.0),
    },
]
    ports = {
    'input_port': {
        'id': 'input_port',
        'part': 'joint_input',
        'kind': 'revolute_joint',
        'pose_local_mm': (0.0, 0.0, 0.0),
    },
    'output_port': {
        'id': 'output_port',
        'part': 'joint_slide',
        'kind': 'prismatic_joint',
        'pose_local_mm': (0.0, 0.0, 0.0),
    },
}
    params = {
    'crank_mm': 25.32,
    'coupler_mm': 65.92,
    'declared_stroke_mm': 50.64,
}
    ir = {
        'schema_version': 'design_ir.v2',
        'parts': parts,
        'joints': joints,
        'ports': ports,
        'params': params,
    }
    return ir
