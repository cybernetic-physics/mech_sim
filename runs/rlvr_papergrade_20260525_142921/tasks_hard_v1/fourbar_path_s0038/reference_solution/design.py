# auto-generated; do not edit by hand. See mech_bench.generators.
import math
from pathlib import Path


def build_design(out_dir: Path) -> dict:
    GROUND = 108.69
    CRANK = 31.35
    COUPLER = 97.56
    ROCKER = 82.2
    OFF_X = 43.85
    OFF_Y = 23.23
    A = (0.0, 0.0, 0.0)
    D = (GROUND, 0.0, 0.0)
    B = (CRANK, 0.0, 0.0)
    BDx = D[0] - B[0]
    BDy = D[1] - B[1]
    d = math.hypot(BDx, BDy)
    a = (COUPLER ** 2 - ROCKER ** 2 + d ** 2) / (2.0 * d)
    h = math.sqrt(COUPLER ** 2 - a ** 2)
    Mx = B[0] + a * BDx / d
    My = B[1] + a * BDy / d
    perp = (-BDy / d, BDx / d)
    C = (Mx + h * perp[0], My + h * perp[1], 0.0)
    parts = [
        {'id': 'ground', 'role': 'ground', 'mass_kg': 0.0, 'fixed': True, 'com_local_mm': (GROUND / 2, 0.0, 0.0)},
        {'id': 'crank', 'role': 'crank', 'mass_kg': 0.02, 'com_local_mm': (CRANK / 2, 0.0, 0.0)},
        {'id': 'coupler', 'role': 'coupler', 'mass_kg': 0.06, 'com_local_mm': (COUPLER / 2, 0.0, 0.0)},
        {'id': 'rocker', 'role': 'rocker', 'mass_kg': 0.05, 'com_local_mm': (ROCKER / 2, 0.0, 0.0)},
    ]
    joints = [
        {'id': 'joint_input', 'type': 'revolute', 'parent': 'ground', 'child': 'crank', 'axis_world': (0.0, 0.0, 1.0), 'anchor_world_mm': A},
        {'id': 'joint_bc', 'type': 'revolute', 'parent': 'crank', 'child': 'coupler', 'axis_world': (0.0, 0.0, 1.0), 'anchor_world_mm': B},
        {'id': 'joint_cd', 'type': 'revolute', 'parent': 'coupler', 'child': 'rocker', 'axis_world': (0.0, 0.0, 1.0), 'anchor_world_mm': C},
        {'id': 'joint_output', 'type': 'revolute', 'parent': 'ground', 'child': 'rocker', 'axis_world': (0.0, 0.0, 1.0), 'anchor_world_mm': D},
    ]
    ports = {
        'input_port': {'id': 'input_port', 'part': 'joint_input', 'kind': 'revolute_joint', 'pose_local_mm': (0.0, 0.0, 0.0)},
        'output_port': {'id': 'output_port', 'part': 'joint_output', 'kind': 'revolute_joint', 'pose_local_mm': (0.0, 0.0, 0.0)},
        'coupler_point': {'id': 'coupler_point', 'part': 'coupler', 'kind': 'frame', 'pose_local_mm': (OFF_X, OFF_Y, 0.0)},
    }
    return {
        'schema_version': 'design_ir.v2',
        'parts': parts,
        'joints': joints,
        'ports': ports,
        'params': {
            'link_lengths_mm': {
                'ground': GROUND, 'crank': CRANK,
                'coupler': COUPLER, 'rocker': ROCKER,
            }
        },
    }
