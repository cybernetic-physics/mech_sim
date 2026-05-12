"""Reusable helpers for procedural task generators.

These helpers shrink the boilerplate of writing a ``design.py`` source
string and an ``expected_failures.json`` block for every new family.

All helpers return plain ``str`` (Python source) or ``dict`` (TOML /
expected-failures fragments). Generators stitch them together; they do
NOT execute design code.
"""

from __future__ import annotations

import json
from typing import Any


_PUBLIC_HEAD = (
    "# auto-generated; do not edit by hand. See mech_bench.generators.\n"
)


# --------------------------------------------------------------------- #
# Small part / joint / port builders                                    #
# --------------------------------------------------------------------- #


def make_ground_part(part_id: str = "ground",
                     com_local_mm: tuple[float, float, float]
                     = (0.0, 0.0, 0.0)) -> dict[str, Any]:
    return {
        "id": part_id,
        "role": "ground",
        "mass_kg": 0.0,
        "fixed": True,
        "com_local_mm": com_local_mm,
    }


def make_revolute_part(part_id: str, role: str = "link",
                       mass_kg: float = 0.05,
                       com_local_mm: tuple[float, float, float]
                       = (0.0, 0.0, 0.0),
                       params: dict[str, Any] | None = None,
                       ) -> dict[str, Any]:
    out: dict[str, Any] = {
        "id": part_id,
        "role": role,
        "mass_kg": float(mass_kg),
        "com_local_mm": com_local_mm,
    }
    if params:
        out["params"] = dict(params)
    return out


def make_slider_part(part_id: str = "slider", mass_kg: float = 0.08,
                     com_local_mm: tuple[float, float, float]
                     = (0.0, 0.0, 0.0)) -> dict[str, Any]:
    return {
        "id": part_id,
        "role": "slider",
        "mass_kg": float(mass_kg),
        "com_local_mm": com_local_mm,
    }


def revolute_joint(joint_id: str, parent: str, child: str,
                   anchor_world_mm: tuple[float, float, float],
                   ) -> dict[str, Any]:
    return {
        "id": joint_id,
        "type": "revolute",
        "parent": parent,
        "child": child,
        "axis_world": (0.0, 0.0, 1.0),
        "anchor_world_mm": tuple(anchor_world_mm),
    }


def prismatic_joint(joint_id: str, parent: str, child: str,
                    axis: tuple[float, float, float] = (1.0, 0.0, 0.0),
                    anchor_world_mm: tuple[float, float, float]
                    = (0.0, 0.0, 0.0)) -> dict[str, Any]:
    return {
        "id": joint_id,
        "type": "prismatic",
        "parent": parent,
        "child": child,
        "axis_world": tuple(axis),
        "anchor_world_mm": tuple(anchor_world_mm),
    }


def fixed_joint(joint_id: str, parent: str, child: str,
                anchor_world_mm: tuple[float, float, float]
                = (0.0, 0.0, 0.0)) -> dict[str, Any]:
    return {
        "id": joint_id,
        "type": "fixed",
        "parent": parent,
        "child": child,
        "axis_world": (0.0, 0.0, 1.0),
        "anchor_world_mm": tuple(anchor_world_mm),
    }


def frame_port(port_id: str, part: str,
               pose_local_mm: tuple[float, float, float]
               = (0.0, 0.0, 0.0)) -> dict[str, Any]:
    return {
        "id": port_id,
        "part": part,
        "kind": "frame",
        "pose_local_mm": tuple(pose_local_mm),
    }


def revolute_joint_port(port_id: str, joint_id: str,
                        pose_local_mm: tuple[float, float, float]
                        = (0.0, 0.0, 0.0)) -> dict[str, Any]:
    return {
        "id": port_id,
        "part": joint_id,
        "kind": "revolute_joint",
        "pose_local_mm": tuple(pose_local_mm),
    }


def prismatic_joint_port(port_id: str, joint_id: str,
                         pose_local_mm: tuple[float, float, float]
                         = (0.0, 0.0, 0.0)) -> dict[str, Any]:
    return {
        "id": port_id,
        "part": joint_id,
        "kind": "prismatic_joint",
        "pose_local_mm": tuple(pose_local_mm),
    }


# --------------------------------------------------------------------- #
# design.py emitters                                                    #
# --------------------------------------------------------------------- #


def _py_repr(value: Any, indent: int = 4, depth: int = 1) -> str:
    """Emit *value* as Python source.

    Tuples stay tuples, dicts use single-quoted keys, booleans use
    ``True``/``False``, ``None`` survives. Floats / ints / strings rely
    on ``repr`` so the output is a valid Python literal in all cases.
    """
    pad = " " * (indent * depth)
    outer = " " * (indent * (depth - 1))
    if isinstance(value, dict):
        if not value:
            return "{}"
        items = []
        for k, v in value.items():
            items.append(
                f"{pad}{_py_repr(k, indent, depth)}: "
                f"{_py_repr(v, indent, depth + 1)}"
            )
        return "{\n" + ",\n".join(items) + ",\n" + outer + "}"
    if isinstance(value, list):
        if not value:
            return "[]"
        items = [
            f"{pad}{_py_repr(v, indent, depth + 1)}"
            for v in value
        ]
        return "[\n" + ",\n".join(items) + ",\n" + outer + "]"
    if isinstance(value, tuple):
        if not value:
            return "()"
        inner = ", ".join(_py_repr(v, indent, depth + 1) for v in value)
        return f"({inner})"
    if isinstance(value, bool):
        return "True" if value else "False"
    if value is None:
        return "None"
    return repr(value)


def make_basic_design_py(
    parts: list[dict[str, Any]],
    joints: list[dict[str, Any]],
    ports: dict[str, dict[str, Any]],
    params: dict[str, Any] | None = None,
    *,
    post_mutation: str = "",
) -> str:
    """Emit a ``design.py`` whose ``build_design`` returns a literal IR.

    The IR fragments are formatted as Python source via :func:`_py_repr`
    so that tuples survive, booleans round-trip as ``True``/``False``,
    and the result executes verbatim under ``exec_module``.
    """
    parts_src = _py_repr(parts)
    joints_src = _py_repr(joints)
    ports_src = _py_repr(ports)
    params_src = _py_repr(params or {})
    return (
        _PUBLIC_HEAD
        + "from pathlib import Path\n\n\n"
        + "def build_design(out_dir: Path) -> dict:\n"
        + f"    parts = {parts_src}\n"
        + f"    joints = {joints_src}\n"
        + f"    ports = {ports_src}\n"
        + f"    params = {params_src}\n"
        + "    ir = {\n"
        + "        'schema_version': 'design_ir.v2',\n"
        + "        'parts': parts,\n"
        + "        'joints': joints,\n"
        + "        'ports': ports,\n"
        + "        'params': params,\n"
        + "    }\n"
        + post_mutation
        + "    return ir\n"
    )


def make_negative_overlay(modifier_py: str) -> str:
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
# Probe-config helpers (used directly inside eval_config)               #
# --------------------------------------------------------------------- #


def required_ports_probe(
    probe_id: str,
    ports: list[str],
    *,
    require_grounded: list[str] | None = None,
    require_kinds: dict[str, str] | None = None,
    hard_gate: bool = True,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "id": probe_id,
        "type": "required_ports",
        "ports": list(ports),
        "hard_gate": bool(hard_gate),
        "severity": "critical",
    }
    if require_grounded:
        out["require_grounded"] = list(require_grounded)
    if require_kinds:
        out["require_kinds"] = dict(require_kinds)
    return out


def dof_probe(
    probe_id: str = "mobility",
    *,
    expected: int = 1,
    space: str = "planar",
    tolerance: int = 0,
    hard_gate: bool = True,
) -> dict[str, Any]:
    return {
        "id": probe_id,
        "type": "dof_grubler",
        "space": space,
        "expected": int(expected),
        "tolerance": int(tolerance),
        "hard_gate": bool(hard_gate),
        "severity": "critical",
    }


def param_check_probe(
    probe_id: str,
    path: str,
    expected: float,
    *,
    comparator: str = "eq",
    tolerance_pct: float = 0.0,
    tolerance_abs: float = 0.0,
    failure_code: str = "wrong_ratio",
    weight: float = 1.0,
    severity: str = "major",
    hard_gate: bool = False,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "id": probe_id,
        "type": "analytic_param_check",
        "path": path,
        "expected": float(expected),
        "comparator": comparator,
        "failure_code": failure_code,
        "weight": float(weight),
        "severity": severity,
        "hard_gate": bool(hard_gate),
    }
    if tolerance_pct > 0:
        out["tolerance_pct"] = float(tolerance_pct)
    if tolerance_abs > 0:
        out["tolerance_abs"] = float(tolerance_abs)
    return out


def fake_contact_probe_config(
    probe_id: str,
    pairs: list[str],
    *,
    min_rms_force_N: float = 0.5,
    min_engagement_fraction: float = 0.2,
    weight: float = 1.0,
    severity: str = "critical",
    hard_gate: bool = False,
    adapter: str | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "id": probe_id,
        "type": "contact_engagement",
        "required_pairs": list(pairs),
        "min_rms_force_N": float(min_rms_force_N),
        "min_engagement_fraction": float(min_engagement_fraction),
        "weight": float(weight),
        "severity": severity,
        "hard_gate": bool(hard_gate),
    }
    if adapter:
        out["adapter"] = adapter
    return out


def fake_oracle_adapters_block(
    pairs: list[str],
    *,
    contact_force_N: float = 5.0,
    penetration_mm: float = 0.005,
    samples: int = 360,
    duration_s: float = 1.0,
    input_speed_rad_s: float = 1.0,
    output_load_Nm: float = 0.0,
    ratio_observed: float | None = None,
    torque_ripple_pct: float = 5.0,
    power_balance_error_pct: float = 1.0,
    lockup: bool = False,
    contact_engagement_fraction: float = 0.95,
) -> dict[str, Any]:
    """Build the ``[adapters.fake_contact_oracle]`` table.

    The block carries ``enabled=true`` so the evaluator's explicit-opt-in
    gate registers the fake oracle. This is the only synthetic oracle
    surface in the runtime; reports tag it as ``oracle_is_synthetic``.
    """
    block: dict[str, Any] = {
        "enabled": True,
        "contact_pairs": list(pairs),
        "contact_force_N": float(contact_force_N),
        "penetration_mm": float(penetration_mm),
        "samples": int(samples),
        "duration_s": float(duration_s),
        "input_speed_rad_s": float(input_speed_rad_s),
        "output_load_Nm": float(output_load_Nm),
        "torque_ripple_pct": float(torque_ripple_pct),
        "power_balance_error_pct": float(power_balance_error_pct),
        "lockup": bool(lockup),
        "contact_engagement_fraction": float(contact_engagement_fraction),
    }
    if ratio_observed is not None:
        block["ratio_observed"] = float(ratio_observed)
    return block


# --------------------------------------------------------------------- #
# Expected-failures helper                                              #
# --------------------------------------------------------------------- #


def make_expected_failures(
    description: str,
    controls: list[dict[str, Any]],
) -> dict[str, Any]:
    out_controls: list[dict[str, Any]] = []
    for c in controls:
        c2 = dict(c)
        c2.setdefault(
            "submission",
            f"negative_solutions/{c2.get('id', '')}",
        )
        out_controls.append(c2)
    return {"description": description, "controls": out_controls}
