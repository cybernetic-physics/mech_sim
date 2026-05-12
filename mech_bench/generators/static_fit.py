"""Tier 0 (artifact-static) generators.

Three families:

* ``static_fit_bracket``     — a fully-grounded fixture; mobility = 0.
* ``shaft_collar_clearance`` — analytic clearance declared as a param.
* ``simple_hinge_fit``       — a single-DOF planar revolute hinge.

These tasks exercise the parts of the runtime that are purely
topological (``dof_grubler``, ``required_ports``, ``analytic_param_check``)
so they do not depend on any simulator adapter.
"""

from __future__ import annotations

import random
from typing import Any

from mech_bench.generators.base import (
    GeneratedTask,
    TaskGenerator,
    common_metadata,
    make_task_id,
)


# --------------------------------------------------------------------- #
# Helpers shared by Tier 0 generators                                   #
# --------------------------------------------------------------------- #


_PUBLIC_HEAD = (
    "# auto-generated; do not edit by hand. See mech_bench.generators.\n"
)


def _design_reference(parts_src: str, joints_src: str, ports_src: str,
                       params_src: str) -> str:
    return (
        _PUBLIC_HEAD
        + "from pathlib import Path\n\n\n"
        + "def build_design(out_dir: Path) -> dict:\n"
        + f"    parts = {parts_src}\n"
        + f"    joints = {joints_src}\n"
        + f"    ports = {ports_src}\n"
        + f"    params = {params_src}\n"
        + "    return {\n"
        + "        'schema_version': 'design_ir.v2',\n"
        + "        'parts': parts,\n"
        + "        'joints': joints,\n"
        + "        'ports': ports,\n"
        + "        'params': params,\n"
        + "    }\n"
    )


def _negative_overlay(modifier_py: str) -> str:
    """Build a negative-control design.py that reuses the reference.

    *modifier_py* is the body that mutates the in-memory dict ``ir``.
    """
    return (
        _PUBLIC_HEAD
        + "import sys\nfrom pathlib import Path\n\n\n"
        + "def build_design(out_dir: Path) -> dict:\n"
        + "    ref_dir = (\n"
        + "        Path(__file__).resolve().parent.parent.parent\n"
        + "        / 'reference_solution'\n"
        + "    )\n"
        + "    sys.path.insert(0, str(ref_dir))\n"
        + "    try:\n"
        + "        import design as ref  # noqa: I001\n"
        + "    finally:\n"
        + "        sys.path.pop(0)\n"
        + "    ir = ref.build_design(out_dir)\n"
        + modifier_py
        + "\n    return ir\n"
    )


# --------------------------------------------------------------------- #
# static_fit_bracket                                                    #
# --------------------------------------------------------------------- #


class StaticFitBracketGenerator(TaskGenerator):
    family = "static_fit_bracket"
    tier = "artifact_static"

    def generate(self, seed: int, difficulty: int = 1) -> GeneratedTask:
        rng = random.Random(seed)
        hole_diameter_mm = round(rng.uniform(4.0, 12.0), 3)
        pitch_mm = round(rng.uniform(20.0, 40.0), 3)
        min_wall_mm = round(rng.uniform(1.5, 3.0), 3)
        # Declared in params so the analytic probe can verify it.
        declared_min_wall = min_wall_mm  # reference declares the truth

        task_id = make_task_id(self.family, seed)
        prompt = (
            "# Static bracket fit\n\n"
            f"Design a single-piece bracket with two fastener holes "
            f"(diameter {hole_diameter_mm} mm, hole pitch {pitch_mm} mm).\n\n"
            "The bracket must:\n"
            f"* declare `params.declared_min_wall_mm` ≥ {min_wall_mm} mm.\n"
            "* expose ports `mount_a` and `mount_b` attached to the\n"
            "  fixed bracket body so they are grounded.\n"
            "* have mobility 0 (everything fixed to ground).\n"
        )

        parts_src = (
            "[{'id': 'bracket', 'role': 'frame', 'mass_kg': 0.05, "
            "'fixed': True, 'com_local_mm': (0.0, 0.0, 0.0), "
            f"'params': {{'hole_diameter_mm': {hole_diameter_mm}, "
            f"'pitch_mm': {pitch_mm}}}}}]"
        )
        joints_src = "[]"
        ports_src = (
            "{"
            "'mount_a': {'id': 'mount_a', 'part': 'bracket', "
            "'kind': 'frame', 'pose_local_mm': (0.0, 0.0, 0.0)},"
            "'mount_b': {'id': 'mount_b', 'part': 'bracket', "
            f"'kind': 'frame', 'pose_local_mm': ({pitch_mm}, 0.0, 0.0)}}"
            "}"
        )
        params_src = (
            f"{{'declared_min_wall_mm': {declared_min_wall}, "
            f"'hole_diameter_mm': {hole_diameter_mm}, "
            f"'pitch_mm': {pitch_mm}}}"
        )
        ref_py = _design_reference(parts_src, joints_src, ports_src, params_src)

        task_toml: dict[str, Any] = {
            "task": {"id": task_id, "family": self.family,
                     "difficulty": int(difficulty), "units": "mm",
                     "tier": self.tier},
            "objective": {
                "description": (
                    f"Static bracket: mobility=0, ports grounded, wall "
                    f">= {min_wall_mm} mm."
                ),
                "allow_massless_links": False,
                "ground_required": False,
            },
            "requirements": {
                "required_ports": ["mount_a", "mount_b"],
                "expected_mobility": 0,
                "max_envelope_mm": [100, 100, 50],
            },
            "visibility": {"public_split": ["mobility", "ports"],
                            "hidden_split": ["min_wall_declared"]},
        }

        eval_public, eval_hidden = _make_static_fit_eval(min_wall_mm,
                                                          relax_pct=0.0,
                                                          tighten_pct=10.0)

        negatives = {
            "missing_port": _negative_overlay(
                "    del ir['ports']['mount_b']"
            ),
            "underwall_declared": _negative_overlay(
                f"    ir['params']['declared_min_wall_mm'] = "
                f"{round(min_wall_mm * 0.3, 4)}"
            ),
        }
        expected = {
            "description": "Tier 0 static_fit_bracket negative controls.",
            "controls": [
                {
                    "id": "missing_port",
                    "submission": "negative_solutions/missing_port",
                    "expected_failure_codes": ["missing_port"],
                    "expected_hard_gate_passed": False,
                    "expected_score_below": 0.001,
                    "expected_metric_direction": "below",
                },
                {
                    "id": "underwall_declared",
                    "submission": "negative_solutions/underwall_declared",
                    "expected_failure_codes": ["insufficient_clearance"],
                    "expected_hard_gate_passed": True,
                    "expected_score_below": 0.5,
                },
            ],
        }
        return GeneratedTask(
            task_id=task_id, family=self.family,
            difficulty=int(difficulty), prompt_md=prompt,
            task_toml=task_toml,
            eval_config_toml=eval_public,
            eval_config_hidden_toml=eval_hidden,
            fixtures={}, reference_solution_py=ref_py,
            negative_solutions=negatives,
            expected_failures=expected,
            metadata=common_metadata(self.family, self.tier, seed,
                                     difficulty,
                                     min_wall_mm=min_wall_mm,
                                     pitch_mm=pitch_mm,
                                     hole_diameter_mm=hole_diameter_mm),
        )


def _make_static_fit_eval(min_wall_mm: float, *,
                          relax_pct: float, tighten_pct: float
                          ) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build (public, hidden) eval configs for the bracket family.

    Hidden uses a tightened minimum wall to test generalization.
    """
    def _cfg(min_wall: float) -> dict[str, Any]:
        return {
            "probes": [
                {"id": "mobility", "type": "dof_grubler", "space": "planar",
                 "expected": 0, "tolerance": 0,
                 "hard_gate": True, "severity": "critical"},
                {"id": "ports", "type": "required_ports",
                 "ports": ["mount_a", "mount_b"],
                 "require_grounded": ["mount_a", "mount_b"],
                 "require_kinds": {"mount_a": "frame", "mount_b": "frame"},
                 "hard_gate": True, "severity": "critical"},
                {"id": "min_wall_declared", "type": "analytic_param_check",
                 "path": "params.declared_min_wall_mm",
                 "expected": float(min_wall),
                 "comparator": "ge",
                 "tolerance_abs": 0.05,
                 "failure_code": "insufficient_clearance",
                 "weight": 1.0, "severity": "major"},
            ],
            "feedback": {
                "public_metrics": [
                    "mobility.observed", "mobility.expected",
                    "ports.ports_required",
                    "min_wall_declared.observed",
                    "min_wall_declared.expected",
                ],
                "hidden_metrics": [
                    "min_wall_declared.error_pct",
                ],
            },
            "hard_gate": {"require": ["mobility", "ports"]},
        }
    public_min = min_wall_mm * (1.0 - relax_pct / 100.0)
    hidden_min = min_wall_mm * (1.0 + tighten_pct / 100.0)
    return _cfg(public_min), _cfg(hidden_min)


# --------------------------------------------------------------------- #
# shaft_collar_clearance                                                #
# --------------------------------------------------------------------- #


class ShaftCollarClearanceGenerator(TaskGenerator):
    family = "shaft_collar_clearance"
    tier = "artifact_static"

    def generate(self, seed: int, difficulty: int = 1) -> GeneratedTask:
        rng = random.Random(seed + 1003)
        shaft_d_mm = round(rng.uniform(6.0, 12.0), 3)
        # Reference design clears with margin; collar inner diameter is
        # shaft_d + 0.5 mm.
        target_clearance_mm = 0.2  # min required diametrical clearance
        ref_clearance_mm = round(target_clearance_mm + rng.uniform(0.05,
                                                                    0.3), 3)
        collar_id_mm = round(shaft_d_mm + ref_clearance_mm, 3)

        task_id = make_task_id(self.family, seed)
        prompt = (
            "# Shaft collar clearance\n\n"
            f"Design a shaft (Ø{shaft_d_mm} mm) running through a fixed "
            f"collar.\n\n"
            f"* `params.shaft_diameter_mm` must equal {shaft_d_mm}.\n"
            f"* `params.collar_inner_diameter_mm` minus the shaft "
            f"diameter must be ≥ {target_clearance_mm} mm.\n"
            "* Required ports: `shaft_origin`, `collar_face` "
            "(collar_face grounded).\n"
            "* Mobility = 0.\n"
        )

        parts_src = (
            "[{'id': 'collar', 'role': 'frame', 'mass_kg': 0.05, "
            "'fixed': True, 'com_local_mm': (0.0, 0.0, 0.0)},"
            "{'id': 'shaft', 'role': 'shaft', 'mass_kg': 0.02, "
            "'com_local_mm': (0.0, 0.0, 0.0)}]"
        )
        joints_src = (
            "[{'id': 'shaft_clamp', 'type': 'fixed', "
            "'parent': 'collar', 'child': 'shaft', "
            "'axis_world': (0.0, 0.0, 1.0), "
            "'anchor_world_mm': (0.0, 0.0, 0.0)}]"
        )
        ports_src = (
            "{"
            "'collar_face': {'id': 'collar_face', 'part': 'collar', "
            "'kind': 'frame', 'pose_local_mm': (0.0, 0.0, 0.0)},"
            "'shaft_origin': {'id': 'shaft_origin', 'part': 'shaft', "
            "'kind': 'frame', 'pose_local_mm': (0.0, 0.0, 0.0)}"
            "}"
        )
        params_src = (
            f"{{'shaft_diameter_mm': {shaft_d_mm}, "
            f"'collar_inner_diameter_mm': {collar_id_mm}, "
            f"'declared_clearance_mm': {ref_clearance_mm}}}"
        )
        ref_py = _design_reference(parts_src, joints_src, ports_src,
                                    params_src)

        task_toml: dict[str, Any] = {
            "task": {"id": task_id, "family": self.family,
                     "difficulty": int(difficulty), "units": "mm",
                     "tier": self.tier},
            "objective": {
                "description": (
                    f"Shaft Ø{shaft_d_mm} mm in a fixed collar; "
                    f"declared diametrical clearance ≥ "
                    f"{target_clearance_mm} mm."
                ),
                "allow_massless_links": False,
                "ground_required": False,
            },
            "requirements": {
                "required_ports": ["shaft_origin", "collar_face"],
                "expected_mobility": 0,
                "max_envelope_mm": [200, 80, 80],
            },
            "visibility": {"public_split": ["mobility", "ports", "clearance"],
                            "hidden_split": ["shaft_diameter"]},
        }

        eval_public, eval_hidden = _make_collar_eval(
            shaft_d_mm, target_clearance_mm)

        negatives = {
            "missing_port": _negative_overlay(
                "    del ir['ports']['shaft_origin']"
            ),
            "tight_clearance": _negative_overlay(
                "    ir['params']['declared_clearance_mm'] = 0.05"
            ),
        }
        expected = {
            "description": "Tier 0 shaft_collar_clearance negative controls.",
            "controls": [
                {
                    "id": "missing_port",
                    "submission": "negative_solutions/missing_port",
                    "expected_failure_codes": ["missing_port"],
                    "expected_hard_gate_passed": False,
                    "expected_score_below": 0.001,
                },
                {
                    "id": "tight_clearance",
                    "submission": "negative_solutions/tight_clearance",
                    "expected_failure_codes": ["insufficient_clearance"],
                    "expected_hard_gate_passed": True,
                    "expected_score_below": 0.5,
                },
            ],
        }

        return GeneratedTask(
            task_id=task_id, family=self.family,
            difficulty=int(difficulty), prompt_md=prompt,
            task_toml=task_toml,
            eval_config_toml=eval_public,
            eval_config_hidden_toml=eval_hidden,
            fixtures={}, reference_solution_py=ref_py,
            negative_solutions=negatives,
            expected_failures=expected,
            metadata=common_metadata(self.family, self.tier, seed,
                                     difficulty,
                                     shaft_diameter_mm=shaft_d_mm,
                                     target_clearance_mm=target_clearance_mm),
        )


def _make_collar_eval(shaft_d_mm: float, target_clearance_mm: float
                       ) -> tuple[dict[str, Any], dict[str, Any]]:
    def _cfg(target: float, tighten_shaft: float = 0.0) -> dict[str, Any]:
        return {
            "probes": [
                {"id": "mobility", "type": "dof_grubler", "space": "planar",
                 "expected": 0, "tolerance": 0,
                 "hard_gate": True, "severity": "critical"},
                {"id": "ports", "type": "required_ports",
                 "ports": ["shaft_origin", "collar_face"],
                 "require_grounded": ["collar_face"],
                 "hard_gate": True, "severity": "critical"},
                {"id": "shaft_diameter", "type": "analytic_param_check",
                 "path": "params.shaft_diameter_mm",
                 "expected": float(shaft_d_mm + tighten_shaft),
                 "comparator": "eq",
                 "tolerance_abs": 0.01,
                 "failure_code": "invalid_artifact",
                 "weight": 0.3, "severity": "major"},
                {"id": "clearance", "type": "analytic_param_check",
                 "path": "params.declared_clearance_mm",
                 "expected": float(target),
                 "comparator": "ge",
                 "tolerance_abs": 0.0,
                 "failure_code": "insufficient_clearance",
                 "weight": 0.7, "severity": "major"},
            ],
            "feedback": {
                "public_metrics": [
                    "mobility.observed", "mobility.expected",
                    "clearance.observed", "clearance.expected",
                    "shaft_diameter.observed", "shaft_diameter.expected",
                ],
                "hidden_metrics": [
                    "clearance.error_abs",
                ],
            },
            "hard_gate": {"require": ["mobility", "ports"]},
        }
    return _cfg(target_clearance_mm), _cfg(target_clearance_mm * 1.5,
                                              tighten_shaft=0.0)


# --------------------------------------------------------------------- #
# simple_hinge_fit                                                      #
# --------------------------------------------------------------------- #


class SimpleHingeFitGenerator(TaskGenerator):
    family = "simple_hinge_fit"
    tier = "artifact_static"

    def generate(self, seed: int, difficulty: int = 1) -> GeneratedTask:
        rng = random.Random(seed + 4201)
        leaf_length_mm = round(rng.uniform(40.0, 80.0), 2)
        knuckle_d_mm = round(rng.uniform(4.0, 10.0), 2)
        target_pin_clearance_mm = 0.1
        declared_pin_clearance_mm = round(
            target_pin_clearance_mm + rng.uniform(0.02, 0.20), 3)

        task_id = make_task_id(self.family, seed)
        prompt = (
            "# Simple hinge fit\n\n"
            f"Design a planar hinge (leaf length {leaf_length_mm} mm, "
            f"knuckle Ø{knuckle_d_mm} mm).\n\n"
            "* Two leaves connected by a revolute joint.\n"
            "* One leaf is grounded; the other rotates about the\n"
            "  hinge axis.\n"
            f"* Mobility = 1 (planar revolute).\n"
            f"* Declared pin clearance ≥ {target_pin_clearance_mm} mm.\n"
            "* Required ports: `mount_a` (frame on grounded leaf), "
            "`hinge_joint` (revolute joint port), `tip_b` (frame on "
            "moving leaf).\n"
        )

        parts_src = (
            "[{'id': 'leaf_a', 'role': 'leaf', 'mass_kg': 0.06, "
            "'fixed': True, 'com_local_mm': (0.0, 0.0, 0.0)},"
            "{'id': 'leaf_b', 'role': 'leaf', 'mass_kg': 0.06, "
            "'com_local_mm': (0.0, 0.0, 0.0)}]"
        )
        joints_src = (
            "[{'id': 'hinge', 'type': 'revolute', "
            "'parent': 'leaf_a', 'child': 'leaf_b', "
            "'axis_world': (0.0, 0.0, 1.0), "
            "'anchor_world_mm': (0.0, 0.0, 0.0)}]"
        )
        ports_src = (
            "{"
            "'mount_a': {'id': 'mount_a', 'part': 'leaf_a', "
            "'kind': 'frame', 'pose_local_mm': (0.0, 0.0, 0.0)},"
            "'hinge_joint': {'id': 'hinge_joint', 'part': 'hinge', "
            "'kind': 'revolute_joint', 'pose_local_mm': (0.0, 0.0, 0.0)},"
            "'tip_b': {'id': 'tip_b', 'part': 'leaf_b', "
            f"'kind': 'frame', 'pose_local_mm': ({leaf_length_mm}, 0.0, 0.0)}}"
            "}"
        )
        params_src = (
            f"{{'leaf_length_mm': {leaf_length_mm}, "
            f"'knuckle_diameter_mm': {knuckle_d_mm}, "
            f"'declared_pin_clearance_mm': {declared_pin_clearance_mm}}}"
        )
        ref_py = _design_reference(parts_src, joints_src, ports_src,
                                    params_src)

        task_toml: dict[str, Any] = {
            "task": {"id": task_id, "family": self.family,
                     "difficulty": int(difficulty), "units": "mm",
                     "tier": self.tier},
            "objective": {
                "description": (
                    f"Simple planar hinge; mobility=1, declared pin "
                    f"clearance ≥ {target_pin_clearance_mm} mm."
                ),
                "allow_massless_links": False,
                "ground_required": False,
            },
            "requirements": {
                "required_ports": ["mount_a", "hinge_joint", "tip_b"],
                "expected_mobility": 1,
                "max_envelope_mm": [200, 100, 50],
            },
            "visibility": {"public_split": ["mobility", "ports", "clearance"],
                            "hidden_split": []},
        }

        eval_public, eval_hidden = _make_hinge_eval(target_pin_clearance_mm)

        negatives = {
            "wrong_mobility_extra_fixed": _negative_overlay(
                "    ir['joints'].append({\n"
                "        'id': 'extra_fix',\n"
                "        'type': 'fixed',\n"
                "        'parent': 'leaf_a',\n"
                "        'child': 'leaf_b',\n"
                "        'axis_world': (0.0, 0.0, 1.0),\n"
                "        'anchor_world_mm': (0.0, 0.0, 0.0),\n"
                "    })"
            ),
            "tight_clearance": _negative_overlay(
                "    ir['params']['declared_pin_clearance_mm'] = 0.01"
            ),
        }
        expected = {
            "description": "Tier 0 simple_hinge_fit negative controls.",
            "controls": [
                {
                    "id": "wrong_mobility_extra_fixed",
                    "submission":
                        "negative_solutions/wrong_mobility_extra_fixed",
                    "expected_failure_codes": ["wrong_mobility"],
                    "expected_hard_gate_passed": False,
                    "expected_score_below": 0.001,
                },
                {
                    "id": "tight_clearance",
                    "submission": "negative_solutions/tight_clearance",
                    "expected_failure_codes": ["insufficient_clearance"],
                    "expected_hard_gate_passed": True,
                    "expected_score_below": 0.5,
                },
            ],
        }
        return GeneratedTask(
            task_id=task_id, family=self.family,
            difficulty=int(difficulty), prompt_md=prompt,
            task_toml=task_toml,
            eval_config_toml=eval_public,
            eval_config_hidden_toml=eval_hidden,
            fixtures={}, reference_solution_py=ref_py,
            negative_solutions=negatives,
            expected_failures=expected,
            metadata=common_metadata(self.family, self.tier, seed,
                                     difficulty,
                                     leaf_length_mm=leaf_length_mm,
                                     knuckle_diameter_mm=knuckle_d_mm),
        )


def _make_hinge_eval(target_pin_clearance_mm: float
                      ) -> tuple[dict[str, Any], dict[str, Any]]:
    def _cfg(target: float) -> dict[str, Any]:
        return {
            "probes": [
                {"id": "mobility", "type": "dof_grubler",
                 "space": "planar", "expected": 1, "tolerance": 0,
                 "hard_gate": True, "severity": "critical"},
                {"id": "ports", "type": "required_ports",
                 "ports": ["mount_a", "hinge_joint", "tip_b"],
                 "require_grounded": ["mount_a"],
                 "require_kinds": {"mount_a": "frame",
                                    "hinge_joint": "revolute_joint",
                                    "tip_b": "frame"},
                 "hard_gate": True, "severity": "critical"},
                {"id": "clearance", "type": "analytic_param_check",
                 "path": "params.declared_pin_clearance_mm",
                 "expected": float(target),
                 "comparator": "ge",
                 "tolerance_abs": 0.0,
                 "failure_code": "insufficient_clearance",
                 "weight": 1.0, "severity": "major"},
            ],
            "feedback": {
                "public_metrics": [
                    "mobility.observed", "mobility.expected",
                    "ports.ports_required", "ports.ports_present",
                    "clearance.observed", "clearance.expected",
                ],
                "hidden_metrics": [],
            },
            "hard_gate": {"require": ["mobility", "ports"]},
        }
    return _cfg(target_pin_clearance_mm), _cfg(target_pin_clearance_mm * 1.5)
