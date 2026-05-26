# auto-generated; do not edit by hand. See mech_bench.generators.
from pathlib import Path


def build_design(out_dir: Path) -> dict:
    PITCH_R = 16.649
    LIN_PER_REV = 104.6088
    parts = [
        {'id': 'frame', 'role': 'ground', 'mass_kg': 0.0, 'fixed': True, 'com_local_mm': (0.0, 0.0, 0.0)},
        {'id': 'pinion', 'role': 'pinion', 'mass_kg': 0.02, 'com_local_mm': (0.0, 0.0, 0.0), 'params': {'pitch_radius_mm': PITCH_R}},
        {'id': 'rack', 'role': 'rack', 'mass_kg': 0.04, 'com_local_mm': (0.0, 0.0, 0.0)},
    ]
    joints = [
        {'id': 'pinion_axis', 'type': 'revolute', 'parent': 'frame', 'child': 'pinion', 'axis_world': (0.0, 0.0, 1.0), 'anchor_world_mm': (0.0, 0.0, 0.0)},
        {'id': 'rack_slide', 'type': 'prismatic', 'parent': 'frame', 'child': 'rack', 'axis_world': (1.0, 0.0, 0.0), 'anchor_world_mm': (0.0, -PITCH_R, 0.0)},
    ]
    ports = {
        'input_port': {'id': 'input_port', 'part': 'pinion_axis', 'kind': 'revolute_joint', 'pose_local_mm': (0.0, 0.0, 0.0)},
        'output_port': {'id': 'output_port', 'part': 'rack_slide', 'kind': 'prismatic_joint', 'pose_local_mm': (0.0, 0.0, 0.0)},
    }
    return {
        'schema_version': 'design_ir.v2',
        'parts': parts,
        'joints': joints,
        'ports': ports,
        'params': {
            'pitch_radius_mm': PITCH_R,
            'declared_linear_per_rev_mm': LIN_PER_REV,
        },
    }
