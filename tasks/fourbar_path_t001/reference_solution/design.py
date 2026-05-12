"""Reference solution for fourbar_path_t001.

A Grashof crank-rocker with link lengths:
    ground (AD)  = 100 mm
    crank  (AB)  = 30 mm
    coupler (BC) = 90 mm
    rocker (CD)  = 80 mm

The coupler point is offset 35 mm along the coupler's local x-axis
(from B toward C) and 18 mm perpendicular (y_local).
"""

from __future__ import annotations

from pathlib import Path


def build_design(out_dir: Path) -> dict:
    A = (0.0, 0.0, 0.0)
    D = (100.0, 0.0, 0.0)
    # Initial pose: theta_2 = 0, so B is at (30, 0, 0). C is then
    # the intersection of circle(B, 90) and circle(D, 80) above the
    # x-axis. (Adapter recomputes; here we just give a consistent
    # rest-pose anchor for joint dimensions.)
    B = (30.0, 0.0, 0.0)
    # |BC| = 90, |CD| = 80, |BD| = 70 → solvable. C above x-axis.
    # See ARCHITECTURE.md and the adapter implementation for the
    # forward-kinematics derivation.
    import math
    BDx = D[0] - B[0]
    BDy = D[1] - B[1]
    d = math.hypot(BDx, BDy)
    a = (90.0 ** 2 - 80.0 ** 2 + d ** 2) / (2.0 * d)
    h = math.sqrt(90.0 ** 2 - a ** 2)
    Mx = B[0] + a * BDx / d
    My = B[1] + a * BDy / d
    perp = (-BDy / d, BDx / d)
    C = (Mx + h * perp[0], My + h * perp[1], 0.0)

    parts = [
        {
            "id": "ground",
            "role": "ground",
            "mass_kg": 0.0,
            "com_local_mm": (50.0, 0.0, 0.0),
            "fixed": True,
        },
        {
            "id": "crank",
            "role": "crank",
            "mass_kg": 0.02,
            "com_local_mm": (15.0, 0.0, 0.0),
        },
        {
            "id": "coupler",
            "role": "coupler",
            "mass_kg": 0.06,
            "com_local_mm": (45.0, 0.0, 0.0),
        },
        {
            "id": "rocker",
            "role": "rocker",
            "mass_kg": 0.05,
            "com_local_mm": (40.0, 0.0, 0.0),
        },
    ]

    joints = [
        {
            "id": "joint_input",
            "type": "revolute",
            "parent": "ground",
            "child": "crank",
            "axis_world": (0.0, 0.0, 1.0),
            "anchor_world_mm": A,
        },
        {
            "id": "joint_bc",
            "type": "revolute",
            "parent": "crank",
            "child": "coupler",
            "axis_world": (0.0, 0.0, 1.0),
            "anchor_world_mm": B,
        },
        {
            "id": "joint_cd",
            "type": "revolute",
            "parent": "coupler",
            "child": "rocker",
            "axis_world": (0.0, 0.0, 1.0),
            "anchor_world_mm": C,
        },
        {
            "id": "joint_output",
            "type": "revolute",
            "parent": "ground",
            "child": "rocker",
            "axis_world": (0.0, 0.0, 1.0),
            "anchor_world_mm": D,
        },
    ]

    ports = {
        "input_port": {
            "id": "input_port",
            "part": "joint_input",
            "kind": "revolute_joint",
            "pose_local_mm": (0.0, 0.0, 0.0),
        },
        "output_port": {
            "id": "output_port",
            "part": "joint_output",
            "kind": "revolute_joint",
            "pose_local_mm": (0.0, 0.0, 0.0),
        },
        "coupler_point": {
            "id": "coupler_point",
            "part": "coupler",
            "kind": "frame",
            # In the coupler local frame: origin at B, x-axis from
            # B toward C. Offset 35 mm along x, 18 mm along y.
            "pose_local_mm": (35.0, 18.0, 0.0),
        },
    }

    return {
        "schema_version": "design_ir.v2",
        "parts": parts,
        "joints": joints,
        "ports": ports,
        "params": {
            "link_lengths_mm": {
                "ground": 100.0,
                "crank": 30.0,
                "coupler": 90.0,
                "rocker": 80.0,
            }
        },
    }
