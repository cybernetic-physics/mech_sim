# auto-generated; do not edit by hand. See mech_bench.generators.
from pathlib import Path


def build_design(out_dir: Path) -> dict:
    CRANK = 24.76
    COUPLER = 66.54
    STROKE = 49.52
    parts = [
        {'id': 'ground', 'role': 'ground', 'mass_kg': 0.0, 'fixed': True, 'com_local_mm': (0.0, 0.0, 0.0)},
        {'id': 'crank', 'role': 'crank', 'mass_kg': 0.02, 'com_local_mm': (CRANK / 2, 0.0, 0.0)},
        {'id': 'coupler', 'role': 'coupler', 'mass_kg': 0.05, 'com_local_mm': (COUPLER / 2, 0.0, 0.0)},
        {'id': 'slider', 'role': 'slider', 'mass_kg': 0.08, 'com_local_mm': (0.0, 0.0, 0.0)},
    ]
    joints = [
        {'id': 'joint_input', 'type': 'revolute', 'parent': 'ground', 'child': 'crank', 'axis_world': (0.0, 0.0, 1.0), 'anchor_world_mm': (0.0, 0.0, 0.0)},
        {'id': 'joint_bc', 'type': 'revolute', 'parent': 'crank', 'child': 'coupler', 'axis_world': (0.0, 0.0, 1.0), 'anchor_world_mm': (CRANK, 0.0, 0.0)},
        {'id': 'joint_cs', 'type': 'revolute', 'parent': 'coupler', 'child': 'slider', 'axis_world': (0.0, 0.0, 1.0), 'anchor_world_mm': (CRANK + COUPLER, 0.0, 0.0)},
        {'id': 'joint_slide', 'type': 'prismatic', 'parent': 'ground', 'child': 'slider', 'axis_world': (1.0, 0.0, 0.0), 'anchor_world_mm': (0.0, 0.0, 0.0)},
    ]
    ports = {
        'input_port': {'id': 'input_port', 'part': 'joint_input', 'kind': 'revolute_joint', 'pose_local_mm': (0.0, 0.0, 0.0)},
        'output_port': {'id': 'output_port', 'part': 'joint_slide', 'kind': 'prismatic_joint', 'pose_local_mm': (0.0, 0.0, 0.0)},
    }
    return {
        'schema_version': 'design_ir.v2',
        'parts': parts,
        'joints': joints,
        'ports': ports,
        'params': {
            'crank_mm': CRANK,
            'coupler_mm': COUPLER,
            'declared_stroke_mm': STROKE,
        },
    }
