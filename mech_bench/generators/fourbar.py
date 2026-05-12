"""Tier 1 generator: planar 4-bar coupler-path tasks.

The generator picks a Grashof crank-rocker (ground ≥ crank, satisfies
the Grashof inequality so a full revolution is possible) and a random
coupler-point offset, then traces the resulting coupler curve and
stores it as ``fixtures/target_path.csv``. The reference solution
reconstructs the same crank-rocker; negative controls perturb either
the coupler offset (path error) or the mobility (extra fixed joint).
"""

from __future__ import annotations

import io
import math
import random
from typing import Any

from mech_bench.generators.base import (
    GeneratedTask,
    TaskGenerator,
    common_metadata,
    make_task_id,
)
from mech_bench.generators.static_fit import _negative_overlay


_PUBLIC_HEAD = (
    "# auto-generated; do not edit by hand. See mech_bench.generators.\n"
)


def _grashof_crank_rocker(rng: random.Random
                           ) -> tuple[float, float, float, float]:
    """Pick (ground, crank, coupler, rocker) lengths satisfying Grashof.

    The shortest must be the crank, and s + l < p + q.
    """
    for _ in range(200):
        crank = round(rng.uniform(25.0, 40.0), 2)
        coupler = round(rng.uniform(80.0, 100.0), 2)
        rocker = round(rng.uniform(70.0, 95.0), 2)
        ground = round(rng.uniform(95.0, 115.0), 2)
        lengths = [crank, coupler, rocker, ground]
        s = min(lengths)
        l = max(lengths)
        rest = sum(lengths) - s - l
        if s != crank:
            continue
        if s + l < rest:
            return ground, crank, coupler, rocker
    # Fallback to a known Grashof set.
    return 100.0, 30.0, 90.0, 80.0


def _trace_coupler(ground: float, crank: float, coupler: float,
                     rocker: float, off_x: float, off_y: float,
                     n: int = 360) -> list[tuple[float, float]]:
    A = (0.0, 0.0)
    D = (ground, 0.0)
    out: list[tuple[float, float]] = []
    prev_C: tuple[float, float] | None = None
    for i in range(n):
        theta = 2.0 * math.pi * i / n
        Bx = A[0] + crank * math.cos(theta)
        By = A[1] + crank * math.sin(theta)
        # intersect circles around B and D
        BDx = D[0] - Bx
        BDy = D[1] - By
        d = math.hypot(BDx, BDy)
        if d <= 0:
            continue
        if d > coupler + rocker or d < abs(coupler - rocker):
            continue
        a = (coupler * coupler - rocker * rocker + d * d) / (2.0 * d)
        h_sq = coupler * coupler - a * a
        if h_sq < 0:
            continue
        h = math.sqrt(h_sq)
        ux = BDx / d
        uy = BDy / d
        Mx = Bx + a * ux
        My = By + a * uy
        # two candidates; pick one closer to previous C for continuity
        p1 = (Mx + h * (-uy), My + h * ux)
        p2 = (Mx - h * (-uy), My - h * ux)
        if prev_C is None:
            C = p1
        else:
            d1 = (p1[0] - prev_C[0]) ** 2 + (p1[1] - prev_C[1]) ** 2
            d2 = (p2[0] - prev_C[0]) ** 2 + (p2[1] - prev_C[1]) ** 2
            C = p1 if d1 <= d2 else p2
        prev_C = C
        # coupler point in world: B + off_x * ex + off_y * ey
        vx = C[0] - Bx
        vy = C[1] - By
        norm = math.hypot(vx, vy)
        if norm < 1e-9:
            continue
        ex = (vx / norm, vy / norm)
        ey = (-ex[1], ex[0])
        Px = Bx + off_x * ex[0] + off_y * ey[0]
        Py = By + off_x * ex[1] + off_y * ey[1]
        out.append((Px, Py))
    return out


def _csv_text(rows: list[tuple[float, float]]) -> str:
    buf = io.StringIO()
    buf.write("x_mm,y_mm\n")
    for x, y in rows:
        buf.write(f"{x:.6f},{y:.6f}\n")
    return buf.getvalue()


def _fourbar_reference_py(ground: float, crank: float, coupler: float,
                            rocker: float, off_x: float, off_y: float) -> str:
    return (
        _PUBLIC_HEAD
        + "import math\n"
        + "from pathlib import Path\n\n\n"
        + "def build_design(out_dir: Path) -> dict:\n"
        + f"    GROUND = {ground}\n"
        + f"    CRANK = {crank}\n"
        + f"    COUPLER = {coupler}\n"
        + f"    ROCKER = {rocker}\n"
        + f"    OFF_X = {off_x}\n"
        + f"    OFF_Y = {off_y}\n"
        + "    A = (0.0, 0.0, 0.0)\n"
        + "    D = (GROUND, 0.0, 0.0)\n"
        + "    B = (CRANK, 0.0, 0.0)\n"
        + "    BDx = D[0] - B[0]\n"
        + "    BDy = D[1] - B[1]\n"
        + "    d = math.hypot(BDx, BDy)\n"
        + "    a = (COUPLER ** 2 - ROCKER ** 2 + d ** 2) / (2.0 * d)\n"
        + "    h = math.sqrt(COUPLER ** 2 - a ** 2)\n"
        + "    Mx = B[0] + a * BDx / d\n"
        + "    My = B[1] + a * BDy / d\n"
        + "    perp = (-BDy / d, BDx / d)\n"
        + "    C = (Mx + h * perp[0], My + h * perp[1], 0.0)\n"
        + "    parts = [\n"
        + "        {'id': 'ground', 'role': 'ground', 'mass_kg': 0.0, "
        "'fixed': True, 'com_local_mm': (GROUND / 2, 0.0, 0.0)},\n"
        + "        {'id': 'crank', 'role': 'crank', 'mass_kg': 0.02, "
        "'com_local_mm': (CRANK / 2, 0.0, 0.0)},\n"
        + "        {'id': 'coupler', 'role': 'coupler', 'mass_kg': 0.06, "
        "'com_local_mm': (COUPLER / 2, 0.0, 0.0)},\n"
        + "        {'id': 'rocker', 'role': 'rocker', 'mass_kg': 0.05, "
        "'com_local_mm': (ROCKER / 2, 0.0, 0.0)},\n"
        + "    ]\n"
        + "    joints = [\n"
        + "        {'id': 'joint_input', 'type': 'revolute', "
        "'parent': 'ground', 'child': 'crank', "
        "'axis_world': (0.0, 0.0, 1.0), 'anchor_world_mm': A},\n"
        + "        {'id': 'joint_bc', 'type': 'revolute', "
        "'parent': 'crank', 'child': 'coupler', "
        "'axis_world': (0.0, 0.0, 1.0), 'anchor_world_mm': B},\n"
        + "        {'id': 'joint_cd', 'type': 'revolute', "
        "'parent': 'coupler', 'child': 'rocker', "
        "'axis_world': (0.0, 0.0, 1.0), 'anchor_world_mm': C},\n"
        + "        {'id': 'joint_output', 'type': 'revolute', "
        "'parent': 'ground', 'child': 'rocker', "
        "'axis_world': (0.0, 0.0, 1.0), 'anchor_world_mm': D},\n"
        + "    ]\n"
        + "    ports = {\n"
        + "        'input_port': {'id': 'input_port', "
        "'part': 'joint_input', 'kind': 'revolute_joint', "
        "'pose_local_mm': (0.0, 0.0, 0.0)},\n"
        + "        'output_port': {'id': 'output_port', "
        "'part': 'joint_output', 'kind': 'revolute_joint', "
        "'pose_local_mm': (0.0, 0.0, 0.0)},\n"
        + "        'coupler_point': {'id': 'coupler_point', "
        "'part': 'coupler', 'kind': 'frame', "
        "'pose_local_mm': (OFF_X, OFF_Y, 0.0)},\n"
        + "    }\n"
        + "    return {\n"
        + "        'schema_version': 'design_ir.v2',\n"
        + "        'parts': parts,\n"
        + "        'joints': joints,\n"
        + "        'ports': ports,\n"
        + "        'params': {\n"
        + "            'link_lengths_mm': {\n"
        + "                'ground': GROUND, 'crank': CRANK,\n"
        + "                'coupler': COUPLER, 'rocker': ROCKER,\n"
        + "            }\n"
        + "        },\n"
        + "    }\n"
    )


class FourbarPathGenerator(TaskGenerator):
    family = "fourbar_path"
    tier = "planar_kinematics"

    def generate(self, seed: int, difficulty: int = 2) -> GeneratedTask:
        rng = random.Random(seed + 7777)
        ground, crank, coupler, rocker = _grashof_crank_rocker(rng)
        off_x = round(rng.uniform(20.0, 60.0), 2)
        off_y = round(rng.uniform(-25.0, 30.0), 2)

        target_pts = _trace_coupler(ground, crank, coupler, rocker,
                                     off_x, off_y, n=360)
        # Hidden uses a denser/coarser sample for generalization-gap test.
        hidden_pts = _trace_coupler(ground, crank, coupler, rocker,
                                     off_x, off_y, n=180)

        task_id = make_task_id(self.family, seed)
        prompt = (
            "# Four-bar coupler path\n\n"
            "Design a planar 4-bar mechanism whose coupler point "
            "traces the curve in `fixtures/target_path.csv`.\n\n"
            "Required ports: `input_port` (revolute_joint, "
            "grounded), `output_port` (revolute_joint, grounded), "
            "`coupler_point` (frame on the coupler).\n\n"
            "Mobility must equal 1. The reference comparison is the "
            "symmetric Chamfer distance after centroid+RMS-radius "
            "normalization.\n"
        )

        task_toml: dict[str, Any] = {
            "task": {"id": task_id, "family": self.family,
                     "difficulty": int(difficulty), "units": "mm",
                     "tier": self.tier},
            "requirements": {
                "required_ports": ["input_port", "output_port",
                                    "coupler_point"],
                "expected_mobility": 1,
                "max_envelope_mm": [220, 220, 50],
            },
            "objective": {
                "description": (
                    "Trace the target coupler path within Chamfer 0.05 "
                    "(normalized)."
                ),
                "target_path_csv": "target_path.csv",
                "max_chamfer_normalized": 0.05,
                "allow_massless_links": False,
                "ground_required": True,
            },
            "visibility": {
                "public_split": ["mobility", "coupler_path"],
                "hidden_split": ["coupler_path"],
            },
        }

        def _cfg(target_csv: str, max_ch: float) -> dict[str, Any]:
            return {
                "probes": [
                    {"id": "mobility", "type": "dof_grubler",
                     "space": "planar", "expected": 1,
                     "tolerance": 0,
                     "hard_gate": True, "severity": "critical"},
                    {"id": "coupler_path", "type": "path_trace_chamfer",
                     "moving_frame": "coupler_point",
                     "target_csv": target_csv,
                     "normalize": True,
                     "max_chamfer": float(max_ch),
                     "weight": 1.0, "severity": "major"},
                ],
                "feedback": {
                    "public_metrics": [
                        "mobility.observed", "mobility.expected",
                        "coupler_path.chamfer", "coupler_path.n_observed",
                        "coupler_path.n_target",
                    ],
                    "hidden_metrics": [],
                },
                "hard_gate": {"require": ["mobility"]},
            }

        ref_py = _fourbar_reference_py(ground, crank, coupler, rocker,
                                         off_x, off_y)

        negatives = {
            "wrong_coupler_offset": _negative_overlay(
                f"    ir['ports']['coupler_point']['pose_local_mm'] = "
                f"({round(off_x + 25.0, 2)}, {round(off_y - 25.0, 2)}, 0.0)"
            ),
            "wrong_mobility_extra_fixed": _negative_overlay(
                "    ir['joints'].append({\n"
                "        'id': 'extra_fix',\n"
                "        'type': 'fixed',\n"
                "        'parent': 'ground',\n"
                "        'child': 'rocker',\n"
                "        'axis_world': (0.0, 0.0, 1.0),\n"
                f"        'anchor_world_mm': ({ground}, 0.0, 0.0),\n"
                "    })"
            ),
        }
        expected = {
            "description": "Tier 1 fourbar_path negative controls.",
            "controls": [
                {
                    "id": "wrong_coupler_offset",
                    "submission": "negative_solutions/wrong_coupler_offset",
                    "expected_failure_codes": ["path_error"],
                    "expected_hard_gate_passed": True,
                    "expected_score_below": 0.6,
                },
                {
                    "id": "wrong_mobility_extra_fixed",
                    "submission":
                        "negative_solutions/wrong_mobility_extra_fixed",
                    "expected_failure_codes": ["wrong_mobility"],
                    "expected_hard_gate_passed": False,
                    "expected_score_below": 0.001,
                },
            ],
        }

        return GeneratedTask(
            task_id=task_id, family=self.family,
            difficulty=int(difficulty), prompt_md=prompt,
            task_toml=task_toml,
            eval_config_toml=_cfg("target_path.csv", 0.05),
            eval_config_hidden_toml=_cfg("target_path_hidden.csv", 0.04),
            fixtures={
                "target_path.csv": _csv_text(target_pts),
                "target_path_hidden.csv": _csv_text(hidden_pts),
            },
            reference_solution_py=ref_py,
            negative_solutions=negatives,
            expected_failures=expected,
            metadata=common_metadata(self.family, self.tier, seed,
                                     difficulty,
                                     ground_mm=ground, crank_mm=crank,
                                     coupler_mm=coupler, rocker_mm=rocker,
                                     coupler_offset_mm=[off_x, off_y]),
        )
