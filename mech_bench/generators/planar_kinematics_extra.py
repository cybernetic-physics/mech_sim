"""Tier 1 — planar kinematic task generators.

Ten families that exercise ``dof_grubler``, ``required_ports``,
``analytic_param_check``, and (in the path-trace variants)
``path_trace_chamfer`` against fixtures produced by the same
generator. They run on the existing ``planar_kinematics`` adapter
without needing any contact-dynamics backend.
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
from mech_bench.generators.common_designs import (
    dof_probe,
    fixed_joint,
    frame_port,
    make_basic_design_py,
    make_expected_failures,
    make_ground_part,
    make_negative_overlay,
    make_revolute_part,
    make_slider_part,
    param_check_probe,
    prismatic_joint,
    prismatic_joint_port,
    required_ports_probe,
    revolute_joint,
    revolute_joint_port,
)


# --------------------------------------------------------------------- #
# Shared kinematic helpers                                              #
# --------------------------------------------------------------------- #


def _csv_text(rows: list[tuple[float, float]]) -> str:
    buf = io.StringIO()
    buf.write("x_mm,y_mm\n")
    for x, y in rows:
        buf.write(f"{x:.6f},{y:.6f}\n")
    return buf.getvalue()


def _slider_output_trace(
    crank: float,
    coupler: float,
    *,
    n: int = 360,
) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    previous: float | None = None
    previous_branch = 1.0
    for i in range(n):
        theta = 2.0 * math.pi * i / n
        bx = crank * math.cos(theta)
        by = crank * math.sin(theta)
        disc = coupler * coupler - by * by
        if disc < 0.0:
            continue
        root = math.sqrt(disc)
        s_plus = bx + root
        s_minus = bx - root
        if previous is None:
            slider_x = s_plus
            previous_branch = 1.0
        elif abs(s_plus - previous) <= abs(s_minus - previous):
            slider_x = s_plus if previous_branch >= 0 else s_minus
            previous_branch = 1.0
        else:
            slider_x = s_minus
            previous_branch = -1.0
        previous = slider_x
        out.append((slider_x, 0.0))
    return out


def _fourbar_coupler_trace(
    ground: float, crank: float, coupler: float, rocker: float,
    off_x: float, off_y: float,
    *, n: int = 360,
) -> list[tuple[float, float]]:
    A = (0.0, 0.0)
    D = (ground, 0.0)
    out: list[tuple[float, float]] = []
    prev_C: tuple[float, float] | None = None
    for i in range(n):
        theta = 2.0 * math.pi * i / n
        Bx = A[0] + crank * math.cos(theta)
        By = A[1] + crank * math.sin(theta)
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
        p1 = (Mx + h * (-uy), My + h * ux)
        p2 = (Mx - h * (-uy), My - h * ux)
        if prev_C is None:
            C = p1
        else:
            d1 = ((p1[0] - prev_C[0]) ** 2
                  + (p1[1] - prev_C[1]) ** 2)
            d2 = ((p2[0] - prev_C[0]) ** 2
                  + (p2[1] - prev_C[1]) ** 2)
            C = p1 if d1 <= d2 else p2
        prev_C = C
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


def _grashof_lengths(rng: random.Random
                     ) -> tuple[float, float, float, float]:
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
    return 100.0, 30.0, 90.0, 80.0


def _four_bar_parts(ground: float, crank: float, coupler: float,
                    rocker: float, off_x: float, off_y: float,
                    ) -> tuple[list[dict], list[dict], dict[str, dict]]:
    A = (0.0, 0.0, 0.0)
    D = (ground, 0.0, 0.0)
    B = (crank, 0.0, 0.0)
    BDx = D[0] - B[0]
    BDy = D[1] - B[1]
    d = math.hypot(BDx, BDy)
    a = (coupler ** 2 - rocker ** 2 + d ** 2) / (2.0 * d)
    h = math.sqrt(max(coupler ** 2 - a ** 2, 0.0))
    Mx = B[0] + a * BDx / d
    My = B[1] + a * BDy / d
    Cx = Mx + h * (-BDy / d)
    Cy = My + h * (BDx / d)
    C = (Cx, Cy, 0.0)

    parts = [
        make_ground_part("ground", com_local_mm=(ground / 2, 0.0, 0.0)),
        make_revolute_part(
            "crank", "crank", 0.02, (crank / 2, 0.0, 0.0)),
        make_revolute_part(
            "coupler", "coupler", 0.06, (coupler / 2, 0.0, 0.0)),
        make_revolute_part(
            "rocker", "rocker", 0.05, (rocker / 2, 0.0, 0.0)),
    ]
    joints = [
        revolute_joint("joint_input", "ground", "crank", A),
        revolute_joint("joint_bc", "crank", "coupler", B),
        revolute_joint("joint_cd", "coupler", "rocker", C),
        revolute_joint("joint_output", "ground", "rocker", D),
    ]
    ports = {
        "input_port": revolute_joint_port("input_port", "joint_input"),
        "output_port": revolute_joint_port("output_port", "joint_output"),
        "coupler_point": frame_port(
            "coupler_point", "coupler", (off_x, off_y, 0.0)),
    }
    return parts, joints, ports


# --------------------------------------------------------------------- #
# 11. fourbar_crank_rocker_sweep                                        #
# --------------------------------------------------------------------- #


class FourbarCrankRockerSweepGenerator(TaskGenerator):
    family = "fourbar_crank_rocker_sweep"
    tier = "planar_kinematics"

    def generate(self, seed: int, difficulty: int = 2) -> GeneratedTask:
        rng = random.Random(seed + 2111)
        ground, crank, coupler, rocker = _grashof_lengths(rng)
        off_x = round(rng.uniform(20.0, 40.0), 2)
        off_y = round(rng.uniform(-10.0, 20.0), 2)
        task_id = make_task_id(self.family, seed)

        parts, joints, ports = _four_bar_parts(
            ground, crank, coupler, rocker, off_x, off_y)
        params = {
            "link_lengths_mm": {
                "ground": ground, "crank": crank,
                "coupler": coupler, "rocker": rocker,
            },
            "expected_mobility": 1,
        }
        prompt = (
            "# Four-bar crank-rocker sweep\n\n"
            "Design a planar Grashof crank-rocker with "
            f"link lengths ground={ground}, crank={crank}, "
            f"coupler={coupler}, rocker={rocker} (mm).\n\n"
            "* Required ports: `input_port` (revolute_joint, grounded), "
            "`output_port` (revolute_joint, grounded), "
            "`coupler_point` (frame on coupler).\n"
            "* Mobility = 1.\n"
        )

        def _cfg() -> dict[str, Any]:
            return {
                "probes": [
                    dof_probe(expected=1),
                    required_ports_probe(
                        "ports",
                        ["input_port", "output_port", "coupler_point"],
                        require_grounded=["input_port", "output_port"],
                        require_kinds={
                            "input_port": "revolute_joint",
                            "output_port": "revolute_joint",
                            "coupler_point": "frame"},
                    ),
                ],
                "feedback": {
                    "public_metrics": [
                        "mobility.observed", "mobility.expected",
                        "ports.ports_required",
                    ],
                    "hidden_metrics": [],
                },
                "hard_gate": {"require": ["mobility", "ports"]},
            }

        ref_py = make_basic_design_py(parts, joints, ports, params)
        negatives = {
            "wrong_mobility": make_negative_overlay(
                "    ir['joints'].append({\n"
                "        'id': 'extra_fix', 'type': 'fixed',\n"
                "        'parent': 'ground', 'child': 'rocker',\n"
                "        'axis_world': (0.0, 0.0, 1.0),\n"
                f"        'anchor_world_mm': ({ground}, 0.0, 0.0)}})"
            ),
            "wrong_anchor": make_negative_overlay(
                "    del ir['ports']['input_port']"
            ),
        }
        expected = make_expected_failures(
            f"Tier 1 {self.family} negatives.",
            [
                {"id": "wrong_mobility",
                 "expected_failure_codes": ["wrong_mobility"],
                 "expected_hard_gate_passed": False,
                 "expected_score_below": 0.001},
                {"id": "wrong_anchor",
                 "expected_failure_codes": ["missing_port"],
                 "expected_hard_gate_passed": False,
                 "expected_score_below": 0.001},
            ],
        )
        task_toml = {
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
                "description": "Crank-rocker four-bar; mobility=1.",
                "ground_required": True,
            },
        }
        return GeneratedTask(
            task_id=task_id, family=self.family,
            difficulty=int(difficulty), prompt_md=prompt,
            task_toml=task_toml,
            eval_config_toml=_cfg(),
            eval_config_hidden_toml=_cfg(),
            fixtures={}, reference_solution_py=ref_py,
            negative_solutions=negatives,
            expected_failures=expected,
            metadata=common_metadata(self.family, self.tier, seed,
                                     difficulty, ground_mm=ground,
                                     crank_mm=crank, coupler_mm=coupler,
                                     rocker_mm=rocker),
        )


# --------------------------------------------------------------------- #
# 12. fourbar_wiper_arc                                                 #
# --------------------------------------------------------------------- #


class FourbarWiperArcGenerator(TaskGenerator):
    family = "fourbar_wiper_arc"
    tier = "planar_kinematics"

    def generate(self, seed: int, difficulty: int = 2) -> GeneratedTask:
        rng = random.Random(seed + 2212)
        ground, crank, coupler, rocker = _grashof_lengths(rng)
        # Wiper-like: pick a coupler offset perpendicular to coupler.
        off_x = round(rng.uniform(25.0, 50.0), 2)
        off_y = round(rng.uniform(10.0, 30.0), 2)
        path_pts = _fourbar_coupler_trace(
            ground, crank, coupler, rocker, off_x, off_y, n=360)
        hidden_pts = _fourbar_coupler_trace(
            ground, crank, coupler, rocker, off_x, off_y, n=180)
        task_id = make_task_id(self.family, seed)

        parts, joints, ports = _four_bar_parts(
            ground, crank, coupler, rocker, off_x, off_y)
        params = {
            "link_lengths_mm": {
                "ground": ground, "crank": crank,
                "coupler": coupler, "rocker": rocker,
            },
        }
        prompt = (
            "# Four-bar wiper arc\n\n"
            "Trace a wiper-like arc with a four-bar coupler point.\n\n"
            "Target curve: `fixtures/target_path.csv`.\n"
            "Required ports: input/output revolute joints (grounded), "
            "`coupler_point` (frame on coupler).\n"
        )

        def _cfg(csv: str, max_ch: float) -> dict[str, Any]:
            return {
                "probes": [
                    dof_probe(expected=1),
                    required_ports_probe(
                        "ports",
                        ["input_port", "output_port", "coupler_point"],
                        require_grounded=["input_port", "output_port"],
                    ),
                    {"id": "coupler_path",
                     "type": "path_trace_chamfer",
                     "moving_frame": "coupler_point",
                     "target_csv": csv,
                     "normalize": True,
                     "max_chamfer": float(max_ch),
                     "weight": 1.0, "severity": "major"},
                ],
                "feedback": {
                    "public_metrics": [
                        "mobility.observed",
                        "coupler_path.chamfer",
                        "coupler_path.n_observed",
                        "coupler_path.n_target",
                    ],
                    "hidden_metrics": [],
                },
                "hard_gate": {"require": ["mobility", "ports"]},
            }

        ref_py = make_basic_design_py(parts, joints, ports, params)
        negatives = {
            "shifted_coupler_point": make_negative_overlay(
                f"    ir['ports']['coupler_point']['pose_local_mm'] = "
                f"({round(off_x + 30.0, 2)}, "
                f"{round(off_y - 20.0, 2)}, 0.0)"
            ),
            "missing_coupler_point": make_negative_overlay(
                "    del ir['ports']['coupler_point']"
            ),
        }
        expected = make_expected_failures(
            f"Tier 1 {self.family} negatives.",
            [
                {"id": "shifted_coupler_point",
                 "expected_failure_codes": ["path_error"],
                 "expected_hard_gate_passed": True,
                 "expected_score_below": 0.6},
                {"id": "missing_coupler_point",
                 "expected_failure_codes": ["missing_port"],
                 "expected_hard_gate_passed": False,
                 "expected_score_below": 0.001},
            ],
        )
        task_toml = {
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
                "description": "Wiper-arc coupler trace.",
                "target_path_csv": "target_path.csv",
                "max_chamfer_normalized": 0.05,
                "ground_required": True,
            },
        }
        return GeneratedTask(
            task_id=task_id, family=self.family,
            difficulty=int(difficulty), prompt_md=prompt,
            task_toml=task_toml,
            eval_config_toml=_cfg("target_path.csv", 0.05),
            eval_config_hidden_toml=_cfg(
                "target_path_hidden.csv", 0.04),
            fixtures={
                "target_path.csv": _csv_text(path_pts),
                "target_path_hidden.csv": _csv_text(hidden_pts),
            },
            reference_solution_py=ref_py,
            negative_solutions=negatives,
            expected_failures=expected,
            metadata=common_metadata(self.family, self.tier, seed,
                                     difficulty,
                                     coupler_offset_mm=[off_x, off_y]),
        )


# --------------------------------------------------------------------- #
# 13. fourbar_straight_line_approx                                      #
# --------------------------------------------------------------------- #


class FourbarStraightLineApproxGenerator(TaskGenerator):
    family = "fourbar_straight_line_approx"
    tier = "planar_kinematics"

    def generate(self, seed: int, difficulty: int = 2) -> GeneratedTask:
        rng = random.Random(seed + 2313)
        # Pick lengths and a coupler point so the trace is approx. linear
        # near the top of the cycle.
        ground = round(rng.uniform(95.0, 105.0), 2)
        crank = round(rng.uniform(28.0, 32.0), 2)
        coupler = round(rng.uniform(85.0, 95.0), 2)
        rocker = round(rng.uniform(78.0, 88.0), 2)
        off_x = round(coupler * 0.5, 2)
        off_y = round(rng.uniform(2.0, 8.0), 2)
        path_pts = _fourbar_coupler_trace(
            ground, crank, coupler, rocker, off_x, off_y, n=360)
        task_id = make_task_id(self.family, seed)

        parts, joints, ports = _four_bar_parts(
            ground, crank, coupler, rocker, off_x, off_y)
        params = {
            "link_lengths_mm": {
                "ground": ground, "crank": crank,
                "coupler": coupler, "rocker": rocker,
            },
        }
        prompt = (
            "# Four-bar straight-line approximation\n\n"
            "Trace a short straight-line approximation with a "
            "coupler point.\n"
        )

        def _cfg(csv: str, max_ch: float) -> dict[str, Any]:
            return {
                "probes": [
                    dof_probe(expected=1),
                    required_ports_probe(
                        "ports",
                        ["input_port", "output_port", "coupler_point"],
                        require_grounded=["input_port", "output_port"],
                    ),
                    {"id": "coupler_path",
                     "type": "path_trace_chamfer",
                     "moving_frame": "coupler_point",
                     "target_csv": csv,
                     "normalize": True,
                     "max_chamfer": float(max_ch),
                     "weight": 1.0, "severity": "major"},
                ],
                "feedback": {
                    "public_metrics": [
                        "coupler_path.chamfer",
                    ],
                    "hidden_metrics": [],
                },
                "hard_gate": {"require": ["mobility", "ports"]},
            }

        ref_py = make_basic_design_py(parts, joints, ports, params)
        negatives = {
            "curved_path": make_negative_overlay(
                f"    ir['ports']['coupler_point']['pose_local_mm'] = "
                f"({round(off_x, 2)}, {round(off_y + 30.0, 2)}, 0.0)"
            ),
            "wrong_link_length": make_negative_overlay(
                "    del ir['ports']['output_port']"
            ),
        }
        expected = make_expected_failures(
            f"Tier 1 {self.family} negatives.",
            [
                {"id": "curved_path",
                 "expected_failure_codes": ["path_error"],
                 "expected_hard_gate_passed": True,
                 "expected_score_below": 0.6},
                {"id": "wrong_link_length",
                 "expected_failure_codes": ["missing_port"],
                 "expected_hard_gate_passed": False,
                 "expected_score_below": 0.001},
            ],
        )
        task_toml = {
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
                "description": "Approx-straight-line coupler path.",
                "target_path_csv": "target_path.csv",
                "ground_required": True,
            },
        }
        return GeneratedTask(
            task_id=task_id, family=self.family,
            difficulty=int(difficulty), prompt_md=prompt,
            task_toml=task_toml,
            eval_config_toml=_cfg("target_path.csv", 0.05),
            eval_config_hidden_toml=_cfg("target_path.csv", 0.04),
            fixtures={"target_path.csv": _csv_text(path_pts)},
            reference_solution_py=ref_py,
            negative_solutions=negatives,
            expected_failures=expected,
            metadata=common_metadata(self.family, self.tier, seed,
                                     difficulty,
                                     coupler_offset_mm=[off_x, off_y]),
        )


# --------------------------------------------------------------------- #
# 14. fourbar_dwell_path                                                #
# --------------------------------------------------------------------- #


class FourbarDwellPathGenerator(TaskGenerator):
    family = "fourbar_dwell_path"
    tier = "planar_kinematics"

    def generate(self, seed: int, difficulty: int = 2) -> GeneratedTask:
        rng = random.Random(seed + 2414)
        ground, crank, coupler, rocker = _grashof_lengths(rng)
        # Off-coupler point that gives a quasi-dwell loop.
        off_x = round(coupler * 0.8, 2)
        off_y = round(rng.uniform(15.0, 25.0), 2)
        path_pts = _fourbar_coupler_trace(
            ground, crank, coupler, rocker, off_x, off_y, n=360)
        task_id = make_task_id(self.family, seed)

        parts, joints, ports = _four_bar_parts(
            ground, crank, coupler, rocker, off_x, off_y)
        params = {
            "link_lengths_mm": {
                "ground": ground, "crank": crank,
                "coupler": coupler, "rocker": rocker,
            },
        }
        prompt = (
            "# Four-bar dwell path\n\n"
            "Target coupler-path has a near-dwell region.\n"
        )

        def _cfg(csv: str, max_ch: float) -> dict[str, Any]:
            return {
                "probes": [
                    dof_probe(expected=1),
                    required_ports_probe(
                        "ports",
                        ["input_port", "output_port", "coupler_point"],
                        require_grounded=["input_port", "output_port"],
                    ),
                    {"id": "coupler_path",
                     "type": "path_trace_chamfer",
                     "moving_frame": "coupler_point",
                     "target_csv": csv,
                     "normalize": True,
                     "max_chamfer": float(max_ch),
                     "weight": 1.0, "severity": "major"},
                ],
                "feedback": {
                    "public_metrics": ["coupler_path.chamfer"],
                    "hidden_metrics": [],
                },
                "hard_gate": {"require": ["mobility", "ports"]},
            }

        ref_py = make_basic_design_py(parts, joints, ports, params)
        negatives = {
            "no_dwell": make_negative_overlay(
                f"    ir['ports']['coupler_point']['pose_local_mm'] = "
                f"({round(off_x * 0.4, 2)}, "
                f"{round(off_y - 12.0, 2)}, 0.0)"
            ),
            "wrong_ground_spacing": make_negative_overlay(
                "    del ir['ports']['input_port']"
            ),
        }
        expected = make_expected_failures(
            f"Tier 1 {self.family} negatives.",
            [
                {"id": "no_dwell",
                 "expected_failure_codes": ["path_error"],
                 "expected_hard_gate_passed": True,
                 "expected_score_below": 0.6},
                {"id": "wrong_ground_spacing",
                 "expected_failure_codes": ["missing_port"],
                 "expected_hard_gate_passed": False,
                 "expected_score_below": 0.001},
            ],
        )
        task_toml = {
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
                "description": "Dwell-region coupler path.",
                "target_path_csv": "target_path.csv",
                "ground_required": True,
            },
        }
        return GeneratedTask(
            task_id=task_id, family=self.family,
            difficulty=int(difficulty), prompt_md=prompt,
            task_toml=task_toml,
            eval_config_toml=_cfg("target_path.csv", 0.05),
            eval_config_hidden_toml=_cfg("target_path.csv", 0.04),
            fixtures={"target_path.csv": _csv_text(path_pts)},
            reference_solution_py=ref_py,
            negative_solutions=negatives,
            expected_failures=expected,
            metadata=common_metadata(self.family, self.tier, seed,
                                     difficulty),
        )


# --------------------------------------------------------------------- #
# 15. fourbar_pump_handle                                               #
# --------------------------------------------------------------------- #


class FourbarPumpHandleGenerator(TaskGenerator):
    family = "fourbar_pump_handle"
    tier = "planar_kinematics"

    def generate(self, seed: int, difficulty: int = 2) -> GeneratedTask:
        rng = random.Random(seed + 2515)
        ground, crank, coupler, rocker = _grashof_lengths(rng)
        off_x = round(coupler * 0.5, 2)
        off_y = round(rng.uniform(0.0, 5.0), 2)
        rocker_min = round(-math.pi / 4, 3)
        rocker_max = round(math.pi / 4, 3)
        task_id = make_task_id(self.family, seed)

        parts, joints, ports = _four_bar_parts(
            ground, crank, coupler, rocker, off_x, off_y)
        params = {
            "link_lengths_mm": {
                "ground": ground, "crank": crank,
                "coupler": coupler, "rocker": rocker,
            },
            "output_rocker_min_rad": rocker_min,
            "output_rocker_max_rad": rocker_max,
        }
        prompt = (
            "# Four-bar pump handle\n\n"
            "Output rocker should sweep within the declared range.\n"
            f"* Declare `params.output_rocker_min_rad` = {rocker_min}.\n"
            f"* Declare `params.output_rocker_max_rad` = {rocker_max}.\n"
        )

        def _cfg() -> dict[str, Any]:
            return {
                "probes": [
                    dof_probe(expected=1),
                    required_ports_probe(
                        "ports", ["input_port", "output_port"],
                        require_grounded=["input_port", "output_port"],
                        require_kinds={
                            "input_port": "revolute_joint",
                            "output_port": "revolute_joint"},
                    ),
                    param_check_probe(
                        "min_range", "params.output_rocker_min_rad",
                        rocker_min, comparator="le",
                        tolerance_abs=0.1,
                        failure_code="wrong_ratio", weight=0.5,
                    ),
                    param_check_probe(
                        "max_range", "params.output_rocker_max_rad",
                        rocker_max, comparator="ge",
                        tolerance_abs=0.1,
                        failure_code="wrong_ratio", weight=0.5,
                    ),
                ],
                "feedback": {
                    "public_metrics": [
                        "min_range.observed", "max_range.observed",
                    ],
                    "hidden_metrics": [],
                },
                "hard_gate": {"require": ["mobility", "ports"]},
            }

        ref_py = make_basic_design_py(parts, joints, ports, params)
        negatives = {
            "inverted_output": make_negative_overlay(
                "    ir['params']['output_rocker_min_rad'] = 2.0\n"
                "    ir['params']['output_rocker_max_rad'] = -2.0"
            ),
            "missing_output_port": make_negative_overlay(
                "    del ir['ports']['output_port']"
            ),
        }
        expected = make_expected_failures(
            f"Tier 1 {self.family} negatives.",
            [
                {"id": "inverted_output",
                 "expected_failure_codes": ["wrong_ratio"],
                 "expected_hard_gate_passed": True,
                 "expected_score_below": 0.5},
                {"id": "missing_output_port",
                 "expected_failure_codes": ["missing_port"],
                 "expected_hard_gate_passed": False,
                 "expected_score_below": 0.001},
            ],
        )
        task_toml = {
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
                "description": "Pump-handle four-bar; bounded output.",
                "ground_required": True,
            },
        }
        return GeneratedTask(
            task_id=task_id, family=self.family,
            difficulty=int(difficulty), prompt_md=prompt,
            task_toml=task_toml,
            eval_config_toml=_cfg(),
            eval_config_hidden_toml=_cfg(),
            fixtures={}, reference_solution_py=ref_py,
            negative_solutions=negatives,
            expected_failures=expected,
            metadata=common_metadata(self.family, self.tier, seed,
                                     difficulty,
                                     rocker_min=rocker_min,
                                     rocker_max=rocker_max),
        )


# --------------------------------------------------------------------- #
# Slider-crank helpers                                                  #
# --------------------------------------------------------------------- #


def _slider_crank_parts(crank: float, coupler: float
                        ) -> tuple[list[dict], list[dict], dict[str, dict]]:
    parts = [
        make_ground_part("ground"),
        make_revolute_part("crank", "crank", 0.02, (crank / 2, 0.0, 0.0)),
        make_revolute_part(
            "coupler", "coupler", 0.05, (coupler / 2, 0.0, 0.0)),
        make_slider_part("slider", 0.08, (0.0, 0.0, 0.0)),
    ]
    joints = [
        revolute_joint("joint_input", "ground", "crank", (0.0, 0.0, 0.0)),
        revolute_joint("joint_bc", "crank", "coupler", (crank, 0.0, 0.0)),
        revolute_joint("joint_cs", "coupler", "slider",
                       (crank + coupler, 0.0, 0.0)),
        prismatic_joint("joint_slide", "ground", "slider"),
    ]
    ports = {
        "input_port": revolute_joint_port("input_port", "joint_input"),
        "output_port": prismatic_joint_port("output_port", "joint_slide"),
    }
    return parts, joints, ports


# --------------------------------------------------------------------- #
# 16. slider_crank_stroke_precision                                     #
# --------------------------------------------------------------------- #


class SliderCrankStrokePrecisionGenerator(TaskGenerator):
    family = "slider_crank_stroke_precision"
    tier = "planar_kinematics"

    def generate(self, seed: int, difficulty: int = 2) -> GeneratedTask:
        rng = random.Random(seed + 2616)
        crank = round(rng.uniform(15.0, 30.0), 2)
        coupler = round(rng.uniform(2.4 * crank, 3.2 * crank), 2)
        stroke = round(2.0 * crank, 4)
        task_id = make_task_id(self.family, seed)

        parts, joints, ports = _slider_crank_parts(crank, coupler)
        params = {
            "crank_mm": crank, "coupler_mm": coupler,
            "declared_stroke_mm": stroke,
        }
        prompt = (
            "# Slider-crank stroke precision\n\n"
            f"Design a slider-crank with crank {crank}, coupler {coupler}.\n"
            f"* Declare `params.declared_stroke_mm` = {stroke} mm.\n"
            "* Match the simulated slider trace in "
            "`fixtures/target_slider_path.csv`.\n"
        )

        def _cfg(tol_pct: float, csv: str, max_chamfer: float) -> dict[str, Any]:
            return {
                "probes": [
                    dof_probe(expected=1),
                    required_ports_probe(
                        "ports", ["input_port", "output_port"],
                        require_grounded=["input_port"],
                        require_kinds={
                            "input_port": "revolute_joint",
                            "output_port": "prismatic_joint"},
                    ),
                    param_check_probe(
                        "stroke", "params.declared_stroke_mm",
                        stroke, tolerance_pct=tol_pct,
                        failure_code="wrong_ratio",
                    ),
                    {"id": "slider_path",
                     "type": "path_trace_chamfer",
                     "moving_frame": "output_port",
                     "target_csv": csv,
                     "normalize": False,
                     "max_chamfer": float(max_chamfer),
                     "weight": 1.0, "severity": "major"},
                ],
                "feedback": {
                    "public_metrics": [
                        "mobility.observed", "stroke.observed",
                        "slider_path.chamfer",
                    ],
                    "hidden_metrics": [
                        "stroke.error_pct",
                        "slider_path.chamfer",
                    ],
                },
                "hard_gate": {"require": ["mobility", "ports"]},
            }

        ref_py = make_basic_design_py(parts, joints, ports, params)
        target_path = _slider_output_trace(crank, coupler, n=360)
        hidden_path = _slider_output_trace(crank, coupler, n=240)
        negatives = {
            "wrong_stroke": make_negative_overlay(
                f"    ir['params']['declared_stroke_mm'] = "
                f"{round(stroke * 1.4, 4)}"
            ),
            "wrong_crank_geometry": make_negative_overlay(
                "    for joint in ir['joints']:\n"
                "        if joint['id'] == 'joint_bc':\n"
                f"            joint['anchor_world_mm'] = "
                f"({round(crank * 0.55, 4)}, 0.0, 0.0)"
            ),
            "wrong_joint_type": make_negative_overlay(
                "    ir['ports']['output_port']['kind'] = "
                "'revolute_joint'"
            ),
        }
        expected = make_expected_failures(
            f"Tier 1 {self.family} negatives.",
            [
                {"id": "wrong_stroke",
                 "expected_failure_codes": ["wrong_ratio"],
                 "expected_hard_gate_passed": True,
                 "expected_score_below": 0.5},
                {"id": "wrong_crank_geometry",
                 "expected_failure_codes": ["path_error"],
                 "expected_hard_gate_passed": True,
                 "expected_score_below": 0.8},
                {"id": "wrong_joint_type",
                 "expected_failure_codes": ["wrong_topology"],
                 "expected_hard_gate_passed": False,
                 "expected_score_below": 0.001},
            ],
        )
        task_toml = {
            "task": {"id": task_id, "family": self.family,
                     "difficulty": int(difficulty), "units": "mm",
                     "tier": self.tier},
            "requirements": {
                "required_ports": ["input_port", "output_port"],
                "expected_mobility": 1,
                "max_envelope_mm": [220, 80, 50],
            },
            "objective": {
                "description": (
                    f"Slider-crank stroke = {stroke} mm and output "
                    "slider trace matches the fixture."
                ),
                "target_path_csv": "target_slider_path.csv",
                "max_chamfer_mm": 0.35,
                "ground_required": True,
            },
        }
        return GeneratedTask(
            task_id=task_id, family=self.family,
            difficulty=int(difficulty), prompt_md=prompt,
            task_toml=task_toml,
            eval_config_toml=_cfg(
                tol_pct=2.0,
                csv="target_slider_path.csv",
                max_chamfer=0.35,
            ),
            eval_config_hidden_toml=_cfg(
                tol_pct=1.0,
                csv="target_slider_path_hidden.csv",
                max_chamfer=0.25,
            ),
            fixtures={
                "target_slider_path.csv": _csv_text(target_path),
                "target_slider_path_hidden.csv": _csv_text(hidden_path),
            },
            reference_solution_py=ref_py,
            negative_solutions=negatives,
            expected_failures=expected,
            metadata=common_metadata(self.family, self.tier, seed,
                                     difficulty, crank_mm=crank,
                                     coupler_mm=coupler,
                                     declared_stroke_mm=stroke),
        )


# --------------------------------------------------------------------- #
# 17. slider_crank_quick_return_proxy                                   #
# --------------------------------------------------------------------- #


class SliderCrankQuickReturnProxyGenerator(TaskGenerator):
    family = "slider_crank_quick_return_proxy"
    tier = "planar_kinematics"

    def generate(self, seed: int, difficulty: int = 2) -> GeneratedTask:
        rng = random.Random(seed + 2717)
        crank = round(rng.uniform(20.0, 35.0), 2)
        coupler = round(rng.uniform(2.0 * crank, 3.0 * crank), 2)
        ratio = round(rng.uniform(1.5, 2.4), 4)
        task_id = make_task_id(self.family, seed)

        parts, joints, ports = _slider_crank_parts(crank, coupler)
        params = {
            "crank_mm": crank, "coupler_mm": coupler,
            "declared_quick_return_ratio": ratio,
        }
        prompt = (
            "# Slider-crank quick-return proxy\n\n"
            "Declare a forward/reverse time ratio.\n"
            f"* `params.declared_quick_return_ratio` = {ratio}.\n"
        )

        def _cfg(tol_pct: float) -> dict[str, Any]:
            return {
                "probes": [
                    dof_probe(expected=1),
                    required_ports_probe(
                        "ports", ["input_port", "output_port"],
                        require_grounded=["input_port"],
                        require_kinds={
                            "input_port": "revolute_joint",
                            "output_port": "prismatic_joint"},
                    ),
                    param_check_probe(
                        "ratio",
                        "params.declared_quick_return_ratio",
                        ratio, tolerance_pct=tol_pct,
                        failure_code="wrong_ratio",
                    ),
                ],
                "feedback": {
                    "public_metrics": ["ratio.observed", "ratio.expected"],
                    "hidden_metrics": ["ratio.error_pct"],
                },
                "hard_gate": {"require": ["mobility", "ports"]},
            }

        ref_py = make_basic_design_py(parts, joints, ports, params)
        negatives = {
            "wrong_ratio": make_negative_overlay(
                f"    ir['params']['declared_quick_return_ratio'] = "
                f"{round(ratio * 0.4, 4)}"
            ),
            "missing_slider_port": make_negative_overlay(
                "    del ir['ports']['output_port']"
            ),
        }
        expected = make_expected_failures(
            f"Tier 1 {self.family} negatives.",
            [
                {"id": "wrong_ratio",
                 "expected_failure_codes": ["wrong_ratio"],
                 "expected_hard_gate_passed": True,
                 "expected_score_below": 0.5},
                {"id": "missing_slider_port",
                 "expected_failure_codes": ["missing_port"],
                 "expected_hard_gate_passed": False,
                 "expected_score_below": 0.001},
            ],
        )
        task_toml = {
            "task": {"id": task_id, "family": self.family,
                     "difficulty": int(difficulty), "units": "mm",
                     "tier": self.tier},
            "requirements": {
                "required_ports": ["input_port", "output_port"],
                "expected_mobility": 1,
                "max_envelope_mm": [220, 80, 50],
            },
            "objective": {
                "description": "Quick-return ratio proxy.",
                "ground_required": True,
            },
        }
        return GeneratedTask(
            task_id=task_id, family=self.family,
            difficulty=int(difficulty), prompt_md=prompt,
            task_toml=task_toml,
            eval_config_toml=_cfg(tol_pct=2.0),
            eval_config_hidden_toml=_cfg(tol_pct=1.0),
            fixtures={},
            reference_solution_py=ref_py,
            negative_solutions=negatives,
            expected_failures=expected,
            metadata=common_metadata(self.family, self.tier, seed,
                                     difficulty, ratio=ratio),
        )


# --------------------------------------------------------------------- #
# 18. reciprocating_pump_plunger                                        #
# --------------------------------------------------------------------- #


class ReciprocatingPumpPlungerGenerator(TaskGenerator):
    family = "reciprocating_pump_plunger"
    tier = "planar_kinematics"

    def generate(self, seed: int, difficulty: int = 2) -> GeneratedTask:
        rng = random.Random(seed + 2818)
        crank = round(rng.uniform(15.0, 25.0), 2)
        coupler = round(rng.uniform(2.4 * crank, 3.0 * crank), 2)
        stroke = round(2.0 * crank, 4)
        task_id = make_task_id(self.family, seed)

        parts, joints, ports = _slider_crank_parts(crank, coupler)
        params = {
            "crank_mm": crank, "coupler_mm": coupler,
            "declared_stroke_mm": stroke,
        }
        prompt = (
            "# Reciprocating pump plunger\n\n"
            "Slider-crank-driven plunger; output stroke along one axis.\n"
            "Match `fixtures/target_slider_path.csv` under actuation.\n"
        )

        def _cfg(tol_pct: float, csv: str, max_chamfer: float) -> dict[str, Any]:
            return {
                "probes": [
                    dof_probe(expected=1),
                    required_ports_probe(
                        "ports", ["input_port", "output_port"],
                        require_grounded=["input_port"],
                        require_kinds={
                            "input_port": "revolute_joint",
                            "output_port": "prismatic_joint"},
                    ),
                    param_check_probe(
                        "stroke", "params.declared_stroke_mm",
                        stroke, tolerance_pct=tol_pct,
                        failure_code="wrong_ratio",
                    ),
                    {"id": "slider_path",
                     "type": "path_trace_chamfer",
                     "moving_frame": "output_port",
                     "target_csv": csv,
                     "normalize": False,
                     "max_chamfer": float(max_chamfer),
                     "weight": 1.0, "severity": "major"},
                ],
                "feedback": {
                    "public_metrics": [
                        "mobility.observed", "stroke.observed",
                        "slider_path.chamfer",
                    ],
                    "hidden_metrics": [
                        "stroke.error_pct",
                        "slider_path.chamfer",
                    ],
                },
                "hard_gate": {"require": ["mobility", "ports"]},
            }

        ref_py = make_basic_design_py(parts, joints, ports, params)
        target_path = _slider_output_trace(crank, coupler, n=360)
        hidden_path = _slider_output_trace(crank, coupler, n=240)
        negatives = {
            "off_axis_slider": make_negative_overlay(
                "    for j in ir['joints']:\n"
                "        if j['id'] == 'joint_slide':\n"
                "            j['axis_world'] = (0.7, 0.7, 0.0)"
            ),
            "wrong_crank_geometry": make_negative_overlay(
                "    for joint in ir['joints']:\n"
                "        if joint['id'] == 'joint_bc':\n"
                f"            joint['anchor_world_mm'] = "
                f"({round(crank * 0.5, 4)}, 0.0, 0.0)"
            ),
            "short_stroke": make_negative_overlay(
                f"    ir['params']['declared_stroke_mm'] = "
                f"{round(stroke * 0.3, 4)}"
            ),
        }
        expected = make_expected_failures(
            f"Tier 1 {self.family} negatives.",
            [
                {"id": "off_axis_slider",
                 "expected_failure_codes": ["path_error"],
                 "expected_hard_gate_passed": True,
                 "expected_score_below": 0.8},
                {"id": "wrong_crank_geometry",
                 "expected_failure_codes": ["path_error"],
                 "expected_hard_gate_passed": True,
                 "expected_score_below": 0.8},
                {"id": "short_stroke",
                 "expected_failure_codes": ["wrong_ratio"],
                 "expected_hard_gate_passed": True,
                 "expected_score_below": 0.5},
            ],
        )
        task_toml = {
            "task": {"id": task_id, "family": self.family,
                     "difficulty": int(difficulty), "units": "mm",
                     "tier": self.tier},
            "requirements": {
                "required_ports": ["input_port", "output_port"],
                "expected_mobility": 1,
                "max_envelope_mm": [220, 80, 50],
            },
            "objective": {
                "description": "Pump plunger; stroke and path target.",
                "target_path_csv": "target_slider_path.csv",
                "max_chamfer_mm": 0.35,
                "ground_required": True,
            },
        }
        return GeneratedTask(
            task_id=task_id, family=self.family,
            difficulty=int(difficulty), prompt_md=prompt,
            task_toml=task_toml,
            eval_config_toml=_cfg(
                tol_pct=2.0,
                csv="target_slider_path.csv",
                max_chamfer=0.35,
            ),
            eval_config_hidden_toml=_cfg(
                tol_pct=1.0,
                csv="target_slider_path_hidden.csv",
                max_chamfer=0.25,
            ),
            fixtures={
                "target_slider_path.csv": _csv_text(target_path),
                "target_slider_path_hidden.csv": _csv_text(hidden_path),
            },
            reference_solution_py=ref_py,
            negative_solutions=negatives,
            expected_failures=expected,
            metadata=common_metadata(self.family, self.tier, seed,
                                     difficulty,
                                     declared_stroke_mm=stroke),
        )


# --------------------------------------------------------------------- #
# 19. toggle_overcenter_margin                                          #
# --------------------------------------------------------------------- #


class ToggleOvercenterMarginGenerator(TaskGenerator):
    family = "toggle_overcenter_margin"
    tier = "planar_kinematics"

    def generate(self, seed: int, difficulty: int = 2) -> GeneratedTask:
        rng = random.Random(seed + 2919)
        ground, crank, coupler, rocker = _grashof_lengths(rng)
        margin = round(rng.uniform(1.5, 4.0), 3)
        task_id = make_task_id(self.family, seed)

        parts, joints, ports = _four_bar_parts(
            ground, crank, coupler, rocker, crank / 2, 0.0)
        params = {
            "link_lengths_mm": {
                "ground": ground, "crank": crank,
                "coupler": coupler, "rocker": rocker,
            },
            "overcenter_margin_mm": margin,
        }
        prompt = (
            "# Toggle linkage overcenter margin\n\n"
            f"Planar toggle with declared overcenter margin "
            f"{margin} mm.\n"
        )

        def _cfg(min_margin: float) -> dict[str, Any]:
            return {
                "probes": [
                    dof_probe(expected=1),
                    required_ports_probe(
                        "ports", ["input_port", "output_port"],
                        require_grounded=["input_port", "output_port"],
                    ),
                    param_check_probe(
                        "margin", "params.overcenter_margin_mm",
                        min_margin, comparator="ge",
                        failure_code="insufficient_clearance",
                    ),
                ],
                "feedback": {
                    "public_metrics": ["margin.observed", "margin.expected"],
                    "hidden_metrics": [],
                },
                "hard_gate": {"require": ["mobility", "ports"]},
            }

        ref_py = make_basic_design_py(parts, joints, ports, params)
        negatives = {
            "undercenter": make_negative_overlay(
                "    ir['params']['overcenter_margin_mm'] = 0.1"
            ),
            "wrong_mobility": make_negative_overlay(
                "    ir['joints'].append({\n"
                "        'id': 'extra_fix', 'type': 'fixed',\n"
                "        'parent': 'ground', 'child': 'rocker',\n"
                "        'axis_world': (0.0, 0.0, 1.0),\n"
                f"        'anchor_world_mm': ({ground}, 0.0, 0.0)}})"
            ),
        }
        expected = make_expected_failures(
            f"Tier 1 {self.family} negatives.",
            [
                {"id": "undercenter",
                 "expected_failure_codes": ["insufficient_clearance"],
                 "expected_hard_gate_passed": True,
                 "expected_score_below": 0.5},
                {"id": "wrong_mobility",
                 "expected_failure_codes": ["wrong_mobility"],
                 "expected_hard_gate_passed": False,
                 "expected_score_below": 0.001},
            ],
        )
        task_toml = {
            "task": {"id": task_id, "family": self.family,
                     "difficulty": int(difficulty), "units": "mm",
                     "tier": self.tier},
            "requirements": {
                "required_ports": ["input_port", "output_port"],
                "expected_mobility": 1,
                "max_envelope_mm": [220, 220, 50],
            },
            "objective": {
                "description": "Toggle overcenter margin.",
                "ground_required": True,
            },
        }
        return GeneratedTask(
            task_id=task_id, family=self.family,
            difficulty=int(difficulty), prompt_md=prompt,
            task_toml=task_toml,
            eval_config_toml=_cfg(1.0),
            eval_config_hidden_toml=_cfg(margin * 0.8),
            fixtures={}, reference_solution_py=ref_py,
            negative_solutions=negatives,
            expected_failures=expected,
            metadata=common_metadata(self.family, self.tier, seed,
                                     difficulty,
                                     overcenter_margin_mm=margin),
        )


# --------------------------------------------------------------------- #
# 20. rocker_limit_stop_topology                                        #
# --------------------------------------------------------------------- #


class RockerLimitStopTopologyGenerator(TaskGenerator):
    family = "rocker_limit_stop_topology"
    tier = "planar_kinematics"

    def generate(self, seed: int, difficulty: int = 2) -> GeneratedTask:
        rng = random.Random(seed + 3020)
        ground, crank, coupler, rocker = _grashof_lengths(rng)
        out_min = round(-math.pi / 3, 4)
        out_max = round(math.pi / 3, 4)
        task_id = make_task_id(self.family, seed)

        parts, joints, ports = _four_bar_parts(
            ground, crank, coupler, rocker, crank / 2, 0.0)
        params = {
            "link_lengths_mm": {
                "ground": ground, "crank": crank,
                "coupler": coupler, "rocker": rocker,
            },
            "output_min_angle_rad": out_min,
            "output_max_angle_rad": out_max,
        }
        prompt = (
            "# Rocker limit-stop topology\n\n"
            "Declare the rocker output's min/max angle.\n"
        )

        def _cfg(tol: float) -> dict[str, Any]:
            return {
                "probes": [
                    dof_probe(expected=1),
                    required_ports_probe(
                        "ports", ["input_port", "output_port"],
                        require_grounded=["input_port", "output_port"],
                    ),
                    param_check_probe(
                        "out_min", "params.output_min_angle_rad",
                        out_min, comparator="le",
                        tolerance_abs=tol,
                        failure_code="wrong_ratio", weight=0.5,
                    ),
                    param_check_probe(
                        "out_max", "params.output_max_angle_rad",
                        out_max, comparator="ge",
                        tolerance_abs=tol,
                        failure_code="wrong_ratio", weight=0.5,
                    ),
                ],
                "feedback": {
                    "public_metrics": [
                        "out_min.observed", "out_max.observed",
                    ],
                    "hidden_metrics": [],
                },
                "hard_gate": {"require": ["mobility", "ports"]},
            }

        ref_py = make_basic_design_py(parts, joints, ports, params)
        negatives = {
            "missing_stop_param": make_negative_overlay(
                "    del ir['params']['output_max_angle_rad']"
            ),
            "wrong_output_port": make_negative_overlay(
                "    del ir['ports']['output_port']"
            ),
        }
        expected = make_expected_failures(
            f"Tier 1 {self.family} negatives.",
            [
                {"id": "missing_stop_param",
                 "expected_failure_codes": ["invalid_artifact"],
                 "expected_hard_gate_passed": True,
                 "expected_score_below": 0.7},
                {"id": "wrong_output_port",
                 "expected_failure_codes": ["missing_port"],
                 "expected_hard_gate_passed": False,
                 "expected_score_below": 0.001},
            ],
        )
        task_toml = {
            "task": {"id": task_id, "family": self.family,
                     "difficulty": int(difficulty), "units": "mm",
                     "tier": self.tier},
            "requirements": {
                "required_ports": ["input_port", "output_port"],
                "expected_mobility": 1,
                "max_envelope_mm": [220, 220, 50],
            },
            "objective": {
                "description": "Rocker stop topology.",
                "ground_required": True,
            },
        }
        return GeneratedTask(
            task_id=task_id, family=self.family,
            difficulty=int(difficulty), prompt_md=prompt,
            task_toml=task_toml,
            eval_config_toml=_cfg(0.1),
            eval_config_hidden_toml=_cfg(0.05),
            fixtures={}, reference_solution_py=ref_py,
            negative_solutions=negatives,
            expected_failures=expected,
            metadata=common_metadata(self.family, self.tier, seed,
                                     difficulty),
        )
