# auto-generated; do not edit by hand. See mech_bench.generators.
from pathlib import Path


def build_design(out_dir: Path) -> dict:
    parts = [
    {
        'id': 'ground',
        'role': 'ground',
        'mass_kg': 0.0,
        'fixed': True,
        'com_local_mm': (51.735, 0.0, 0.0),
    },
    {
        'id': 'crank',
        'role': 'crank',
        'mass_kg': 0.02,
        'com_local_mm': (14.99, 0.0, 0.0),
    },
    {
        'id': 'coupler',
        'role': 'coupler',
        'mass_kg': 0.06,
        'com_local_mm': (43.9, 0.0, 0.0),
    },
    {
        'id': 'rocker',
        'role': 'rocker',
        'mass_kg': 0.05,
        'com_local_mm': (42.325, 0.0, 0.0),
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
        'anchor_world_mm': (29.98, 0.0, 0.0),
    },
    {
        'id': 'joint_cd',
        'type': 'revolute',
        'parent': 'coupler',
        'child': 'rocker',
        'axis_world': (0.0, 0.0, 1.0),
        'anchor_world_mm': (70.42085998094977, 77.9318730943971, 0.0),
    },
    {
        'id': 'joint_output',
        'type': 'revolute',
        'parent': 'ground',
        'child': 'rocker',
        'axis_world': (0.0, 0.0, 1.0),
        'anchor_world_mm': (103.47, 0.0, 0.0),
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
        'part': 'joint_output',
        'kind': 'revolute_joint',
        'pose_local_mm': (0.0, 0.0, 0.0),
    },
    'coupler_point': {
        'id': 'coupler_point',
        'part': 'coupler',
        'kind': 'frame',
        'pose_local_mm': (43.9, 2.28, 0.0),
    },
}
    params = {
    'link_lengths_mm': {
        'ground': 103.47,
        'crank': 29.98,
        'coupler': 87.8,
        'rocker': 84.65,
    },
    'output_rocker_min_rad': -0.785,
    'output_rocker_max_rad': 0.785,
}
    ir = {
        'schema_version': 'design_ir.v2',
        'parts': parts,
        'joints': joints,
        'ports': ports,
        'params': params,
    }
    return ir
