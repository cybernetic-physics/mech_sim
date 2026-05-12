# auto-generated; do not edit by hand. See mech_bench.generators.
from pathlib import Path


def build_design(out_dir: Path) -> dict:
    N_PINS = 10
    RATIO = 9.0
    parts = [
        {'id': 'housing', 'role': 'ground', 'mass_kg': 0.0, 'fixed': True, 'com_local_mm': (0.0, 0.0, 0.0)},
        {'id': 'eccentric', 'role': 'eccentric', 'mass_kg': 0.05, 'com_local_mm': (0.0, 0.0, 0.0)},
        {'id': 'disc', 'role': 'cycloidal_disc', 'mass_kg': 0.08, 'com_local_mm': (0.0, 0.0, 0.0), 'params': {'pins': N_PINS}},
        {'id': 'carrier', 'role': 'carrier', 'mass_kg': 0.04, 'com_local_mm': (0.0, 0.0, 0.0)},
    ]
    joints = [
        {'id': 'input_revolute', 'type': 'revolute', 'parent': 'housing', 'child': 'eccentric', 'axis_world': (0.0, 0.0, 1.0), 'anchor_world_mm': (0.0, 0.0, 0.0)},
        {'id': 'eccentric_disc', 'type': 'revolute', 'parent': 'eccentric', 'child': 'disc', 'axis_world': (0.0, 0.0, 1.0), 'anchor_world_mm': (1.0, 0.0, 0.0)},
        {'id': 'output_revolute', 'type': 'revolute', 'parent': 'housing', 'child': 'carrier', 'axis_world': (0.0, 0.0, 1.0), 'anchor_world_mm': (0.0, 0.0, 0.0)},
        {'id': 'ring_contact', 'type': 'contact_pair', 'parent': 'housing', 'child': 'disc', 'axis_world': (0.0, 0.0, 1.0), 'anchor_world_mm': (0.0, 0.0, 0.0)},
    ]
    ports = {
        'input_port': {'id': 'input_port', 'part': 'input_revolute', 'kind': 'revolute_joint', 'pose_local_mm': (0.0, 0.0, 0.0)},
        'output_port': {'id': 'output_port', 'part': 'output_revolute', 'kind': 'revolute_joint', 'pose_local_mm': (0.0, 0.0, 0.0)},
    }
    return {
        'schema_version': 'design_ir.v2',
        'parts': parts,
        'joints': joints,
        'ports': ports,
        'params': {
            'pins': N_PINS,
            'declared_ratio': RATIO,
        },
    }
