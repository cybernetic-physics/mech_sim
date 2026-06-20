"""Tier 1 generator: planar slider-crank stroke tasks.

The reference is a centric slider-crank (the prismatic axis passes
through the crank pivot). The generator constrains stroke = 2 * crank
length, which the agent must reproduce; negative controls perturb
either the link lengths (wrong stroke → analytic param mismatch) or
the topology (drop the prismatic → mobility wrong).
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


def _csv_text(rows: list[tuple[float, float]]) -> str:
    buf = io.StringIO()
    buf.write("x_mm,y_mm\n")
    for x, y in rows:
        buf.write(f"{x:.6f},{y:.6f}\n")
    return buf.getvalue()


def _slider_output_trace(
    crank_mm: float,
    coupler_mm: float,
    *,
    n: int = 360,
) -> list[tuple[float, float]]:
    """Reference slider trace for the centric slider-crank convention.

    This mirrors the planar_kinematics adapter's slider-crank solver for the
    generator's canonical geometry: input pivot at the origin, slider axis on
    +x through the origin, crank-coupler joint initialized at (crank, 0), and
    coupler-slider joint initialized at (crank + coupler, 0).
    """
    out: list[tuple[float, float]] = []
    previous: float | None = None
    previous_branch = 1.0
    for i in range(n):
        theta = 2.0 * math.pi * i / n
        bx = crank_mm * math.cos(theta)
        by = crank_mm * math.sin(theta)
        disc = coupler_mm * coupler_mm - by * by
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


def _slider_crank_reference_py(crank_mm: float, coupler_mm: float,
                                  stroke_mm: float) -> str:
    return (
        _PUBLIC_HEAD
        + "from pathlib import Path\n\n\n"
        + "def build_design(out_dir: Path) -> dict:\n"
        + f"    CRANK = {crank_mm}\n"
        + f"    COUPLER = {coupler_mm}\n"
        + f"    STROKE = {stroke_mm}\n"
        + "    parts = [\n"
        + "        {'id': 'ground', 'role': 'ground', 'mass_kg': 0.0, "
        "'fixed': True, 'com_local_mm': (0.0, 0.0, 0.0)},\n"
        + "        {'id': 'crank', 'role': 'crank', 'mass_kg': 0.02, "
        "'com_local_mm': (CRANK / 2, 0.0, 0.0)},\n"
        + "        {'id': 'coupler', 'role': 'coupler', 'mass_kg': 0.05, "
        "'com_local_mm': (COUPLER / 2, 0.0, 0.0)},\n"
        + "        {'id': 'slider', 'role': 'slider', 'mass_kg': 0.08, "
        "'com_local_mm': (0.0, 0.0, 0.0)},\n"
        + "    ]\n"
        + "    joints = [\n"
        + "        {'id': 'joint_input', 'type': 'revolute', "
        "'parent': 'ground', 'child': 'crank', "
        "'axis_world': (0.0, 0.0, 1.0), "
        "'anchor_world_mm': (0.0, 0.0, 0.0)},\n"
        + "        {'id': 'joint_bc', 'type': 'revolute', "
        "'parent': 'crank', 'child': 'coupler', "
        "'axis_world': (0.0, 0.0, 1.0), "
        "'anchor_world_mm': (CRANK, 0.0, 0.0)},\n"
        + "        {'id': 'joint_cs', 'type': 'revolute', "
        "'parent': 'coupler', 'child': 'slider', "
        "'axis_world': (0.0, 0.0, 1.0), "
        "'anchor_world_mm': (CRANK + COUPLER, 0.0, 0.0)},\n"
        + "        {'id': 'joint_slide', 'type': 'prismatic', "
        "'parent': 'ground', 'child': 'slider', "
        "'axis_world': (1.0, 0.0, 0.0), "
        "'anchor_world_mm': (0.0, 0.0, 0.0)},\n"
        + "    ]\n"
        + "    ports = {\n"
        + "        'input_port': {'id': 'input_port', "
        "'part': 'joint_input', 'kind': 'revolute_joint', "
        "'pose_local_mm': (0.0, 0.0, 0.0)},\n"
        + "        'output_port': {'id': 'output_port', "
        "'part': 'joint_slide', 'kind': 'prismatic_joint', "
        "'pose_local_mm': (0.0, 0.0, 0.0)},\n"
        + "    }\n"
        + "    return {\n"
        + "        'schema_version': 'design_ir.v2',\n"
        + "        'parts': parts,\n"
        + "        'joints': joints,\n"
        + "        'ports': ports,\n"
        + "        'params': {\n"
        + "            'crank_mm': CRANK,\n"
        + "            'coupler_mm': COUPLER,\n"
        + "            'declared_stroke_mm': STROKE,\n"
        + "        },\n"
        + "    }\n"
    )


class SliderCrankStrokeGenerator(TaskGenerator):
    family = "slider_crank_stroke"
    tier = "planar_kinematics"

    def generate(self, seed: int, difficulty: int = 2) -> GeneratedTask:
        rng = random.Random(seed + 3331)
        crank_mm = round(rng.uniform(15.0, 30.0), 2)
        coupler_mm = round(rng.uniform(2.4 * crank_mm,
                                         3.5 * crank_mm), 2)
        stroke_mm = round(2.0 * crank_mm, 4)

        task_id = make_task_id(self.family, seed)
        prompt = (
            "# Slider-crank stroke\n\n"
            f"Design a centric slider-crank with crank length "
            f"{crank_mm} mm and coupler {coupler_mm} mm.\n\n"
            f"* Declare `params.declared_stroke_mm` = {stroke_mm} mm "
            "(twice the crank length).\n"
            "* Match the simulated `output_port` slider trace in "
            "`fixtures/target_slider_path.csv`.\n"
            "* Required ports: `input_port` (revolute_joint, "
            "grounded), `output_port` (prismatic_joint).\n"
            "* Mobility = 1.\n"
        )

        task_toml: dict[str, Any] = {
            "task": {"id": task_id, "family": self.family,
                     "difficulty": int(difficulty), "units": "mm",
                     "tier": self.tier},
            "requirements": {
                "required_ports": ["input_port", "output_port"],
                "expected_mobility": 1,
                "max_envelope_mm": [200, 100, 50],
            },
            "objective": {
                "description": (
                    f"Centric slider-crank; declare stroke = "
                    f"{stroke_mm} mm; match output slider trace; mobility=1."
                ),
                "target_path_csv": "target_slider_path.csv",
                "max_chamfer_mm": 0.35,
                "allow_massless_links": False,
                "ground_required": True,
            },
            "visibility": {
                "public_split": ["mobility", "stroke"],
                "hidden_split": ["stroke"],
            },
        }

        def _cfg(stroke_target: float, tol_pct: float,
                 csv: str, max_chamfer: float) -> dict[str, Any]:
            return {
                "probes": [
                    {"id": "mobility", "type": "dof_grubler",
                     "space": "planar", "expected": 1, "tolerance": 0,
                     "hard_gate": True, "severity": "critical"},
                    {"id": "ports", "type": "required_ports",
                     "ports": ["input_port", "output_port"],
                     "require_grounded": ["input_port"],
                     "require_kinds": {"input_port": "revolute_joint",
                                        "output_port": "prismatic_joint"},
                     "hard_gate": True, "severity": "critical"},
                    {"id": "stroke", "type": "analytic_param_check",
                     "path": "params.declared_stroke_mm",
                     "expected": float(stroke_target),
                     "comparator": "eq",
                     "tolerance_pct": float(tol_pct),
                     "failure_code": "wrong_ratio",
                     "weight": 1.0, "severity": "major"},
                    {"id": "slider_path", "type": "path_trace_chamfer",
                     "moving_frame": "output_port",
                     "target_csv": csv,
                     "normalize": False,
                     "max_chamfer": float(max_chamfer),
                     "weight": 1.0, "severity": "major"},
                ],
                "feedback": {
                    "public_metrics": [
                        "mobility.observed", "mobility.expected",
                        "stroke.observed", "stroke.expected",
                        "slider_path.chamfer",
                        "slider_path.n_observed",
                        "slider_path.n_target",
                    ],
                    "hidden_metrics": [
                        "stroke.error_pct",
                        "slider_path.chamfer",
                    ],
                },
                "hard_gate": {"require": ["mobility", "ports"]},
            }

        ref_py = _slider_crank_reference_py(crank_mm, coupler_mm, stroke_mm)
        target_path = _slider_output_trace(crank_mm, coupler_mm, n=360)
        hidden_path = _slider_output_trace(crank_mm, coupler_mm, n=240)

        negatives = {
            "wrong_stroke": _negative_overlay(
                f"    ir['params']['declared_stroke_mm'] = "
                f"{round(stroke_mm * 1.3, 4)}"
            ),
            "wrong_crank_geometry": _negative_overlay(
                "    for joint in ir['joints']:\n"
                "        if joint['id'] == 'joint_bc':\n"
                f"            joint['anchor_world_mm'] = "
                f"({round(crank_mm * 0.58, 4)}, 0.0, 0.0)"
            ),
            "wrong_mobility_extra_fixed": _negative_overlay(
                "    ir['joints'].append({\n"
                "        'id': 'extra_fix',\n"
                "        'type': 'fixed',\n"
                "        'parent': 'ground',\n"
                "        'child': 'crank',\n"
                "        'axis_world': (0.0, 0.0, 1.0),\n"
                "        'anchor_world_mm': (0.0, 0.0, 0.0),\n"
                "    })"
            ),
        }
        expected = {
            "description": "Tier 1 slider_crank_stroke negative controls.",
            "controls": [
                {
                    "id": "wrong_stroke",
                    "submission": "negative_solutions/wrong_stroke",
                    "expected_failure_codes": ["wrong_ratio"],
                    "expected_hard_gate_passed": True,
                    "expected_score_below": 0.5,
                },
                {
                    "id": "wrong_crank_geometry",
                    "submission": "negative_solutions/wrong_crank_geometry",
                    "expected_failure_codes": ["path_error"],
                    "expected_hard_gate_passed": True,
                    "expected_score_below": 0.8,
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
            eval_config_toml=_cfg(
                stroke_mm, tol_pct=2.0,
                csv="target_slider_path.csv",
                max_chamfer=0.35,
            ),
            eval_config_hidden_toml=_cfg(
                stroke_mm, tol_pct=1.0,
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
                                     crank_mm=crank_mm,
                                     coupler_mm=coupler_mm,
                                     target_stroke_mm=stroke_mm),
        )
