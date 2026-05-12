"""DesignIR validation layer.

This layer must be **total**: no malformed field is allowed to raise
out of ``validate_design_ir``. Everything is reported as a structured
``Failure``. Probes downstream may assume the IR has the right shape
once this layer has cleared it of CRITICAL failures.
"""

from __future__ import annotations

import math
import os
import re
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import numpy as np

from mech_bench.feedback import Failure, FailureCode, Severity
from mech_bench.schema import DesignIR, Joint, Part, Port, TaskSpec


SCHEMA_VERSION = "design_ir.v2"

CRITICAL_CODES: frozenset[FailureCode] = frozenset({
    FailureCode.SCHEMA_ERROR,
    FailureCode.INVALID_ARTIFACT,
    FailureCode.INVALID_MASS_PROPERTIES,
    FailureCode.MISSING_PORT,
    FailureCode.WRONG_TOPOLOGY,
})

# Conservative identifier grammar. No slashes, backslashes, whitespace,
# control characters, or pure-numeric ids; max length 128.
_ID_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.:-]{0,127}$")

# Joint and port enum membership. Keep these in sync with schema.py
# but enforce membership eagerly here so structurally bad input cannot
# slip past dataclass construction.
_JOINT_TYPES = frozenset({
    "revolute", "prismatic", "fixed", "contact_pair", "spherical",
})
_PORT_KINDS = frozenset({
    "frame", "revolute_joint", "prismatic_joint",
})

_CONTROL_CHARS = frozenset(range(0, 32)) | {127}


# --------------------------------------------------------------------- #
# Small primitives                                                      #
# --------------------------------------------------------------------- #


def _is_finite_number(x) -> bool:
    return (
        isinstance(x, (int, float))
        and not isinstance(x, bool)
        and math.isfinite(float(x))
    )


def _is_finite_tuple(t, length: int | None = None) -> bool:
    if not isinstance(t, (tuple, list)):
        return False
    if length is not None and len(t) != length:
        return False
    return all(_is_finite_number(x) for x in t)


def _id_is_valid(ident: Any) -> bool:
    return isinstance(ident, str) and bool(_ID_RE.match(ident))


def _make(code: FailureCode, severity: Severity, msg: str,
          where: str | None = None, **extra) -> Failure:
    return Failure(code=code, severity=severity, message=msg,
                   where=where, extra=dict(extra) if extra else {})


# --------------------------------------------------------------------- #
# Schema-version / id checks                                            #
# --------------------------------------------------------------------- #


def _check_schema_version(ir: DesignIR) -> list[Failure]:
    if ir.schema_version != SCHEMA_VERSION:
        return [_make(
            FailureCode.SCHEMA_ERROR, Severity.CRITICAL,
            f"DesignIR schema_version must be {SCHEMA_VERSION!r}, "
            f"got {ir.schema_version!r}.",
            where="schema_version",
        )]
    return []


def _check_id_grammar(ident: Any, kind: str,
                      index_label: str) -> list[Failure]:
    if not isinstance(ident, str) or not ident:
        return [_make(
            FailureCode.SCHEMA_ERROR, Severity.CRITICAL,
            f"{kind} {index_label} has empty or non-string id.",
            where=f"{index_label}.id",
        )]
    if not _id_is_valid(ident):
        return [_make(
            FailureCode.SCHEMA_ERROR, Severity.CRITICAL,
            f"{kind} id {ident!r} does not match the allowed grammar "
            f"^[A-Za-z_][A-Za-z0-9_.:-]{{0,127}}$.",
            where=f"{kind}.{ident!r}.id",
        )]
    return []


def _check_unique_nonempty_ids(items: Iterable, name: str) -> list[Failure]:
    seen: set[str] = set()
    failures: list[Failure] = []
    for i, item in enumerate(items):
        ident = getattr(item, "id", None)
        failures.extend(_check_id_grammar(ident, name, f"{name}[{i}]"))
        if isinstance(ident, str) and ident:
            if ident in seen:
                failures.append(_make(
                    FailureCode.SCHEMA_ERROR, Severity.CRITICAL,
                    f"Duplicate {name} id {ident!r}.",
                    where=f"{name}.{ident}",
                ))
            seen.add(ident)
    return failures


# --------------------------------------------------------------------- #
# Topology / reference checks                                           #
# --------------------------------------------------------------------- #


def _check_joints(ir: DesignIR) -> list[Failure]:
    failures: list[Failure] = []
    part_ids = ir.part_ids()
    for j in ir.joints:
        # Type membership.
        if not isinstance(j.type, str) or j.type not in _JOINT_TYPES:
            failures.append(_make(
                FailureCode.SCHEMA_ERROR, Severity.CRITICAL,
                f"Joint {j.id!r} has unsupported type {j.type!r}. "
                f"Allowed: {sorted(_JOINT_TYPES)}.",
                where=f"joints.{j.id}.type",
            ))
        # parent / child must be non-empty strings (they're checked for
        # existence below; we want strict type first).
        for end_attr in ("parent", "child"):
            val = getattr(j, end_attr, None)
            if not isinstance(val, str) or not val:
                failures.append(_make(
                    FailureCode.SCHEMA_ERROR, Severity.CRITICAL,
                    f"Joint {j.id!r} {end_attr} must be a non-empty "
                    f"string.",
                    where=f"joints.{j.id}.{end_attr}",
                ))
        if isinstance(j.parent, str) and j.parent \
                and j.parent not in part_ids:
            failures.append(_make(
                FailureCode.WRONG_TOPOLOGY, Severity.CRITICAL,
                f"Joint {j.id!r} parent {j.parent!r} is not an existing "
                f"part id.",
                where=f"joints.{j.id}.parent",
            ))
        if isinstance(j.child, str) and j.child \
                and j.child not in part_ids:
            failures.append(_make(
                FailureCode.WRONG_TOPOLOGY, Severity.CRITICAL,
                f"Joint {j.id!r} child {j.child!r} is not an existing "
                f"part id.",
                where=f"joints.{j.id}.child",
            ))
        if isinstance(j.parent, str) and j.parent == j.child \
                and j.parent in part_ids:
            failures.append(_make(
                FailureCode.WRONG_TOPOLOGY, Severity.CRITICAL,
                f"Joint {j.id!r} parent and child reference the same "
                f"part {j.parent!r}.",
                where=f"joints.{j.id}",
            ))
        if not isinstance(j.params, dict):
            failures.append(_make(
                FailureCode.SCHEMA_ERROR, Severity.CRITICAL,
                f"Joint {j.id!r} params must be a dict.",
                where=f"joints.{j.id}.params",
            ))
    return failures


def _check_ports(ir: DesignIR) -> list[Failure]:
    failures: list[Failure] = []
    part_ids = ir.part_ids()
    joint_ids = {j.id for j in ir.joints}
    for pid, port in ir.ports.items():
        if pid != port.id:
            failures.append(_make(
                FailureCode.SCHEMA_ERROR, Severity.CRITICAL,
                f"Port dict key {pid!r} does not match port.id "
                f"{port.id!r}.",
                where=f"ports.{pid}",
            ))
        if not isinstance(port.kind, str) or port.kind not in _PORT_KINDS:
            failures.append(_make(
                FailureCode.SCHEMA_ERROR, Severity.CRITICAL,
                f"Port {pid!r} has unknown kind {port.kind!r}. "
                f"Allowed: {sorted(_PORT_KINDS)}.",
                where=f"ports.{pid}.kind",
            ))
            continue
        if port.kind == "frame":
            if port.part not in part_ids:
                failures.append(_make(
                    FailureCode.MISSING_PORT, Severity.CRITICAL,
                    f"Port {pid!r} (frame) references part {port.part!r} "
                    f"that does not exist.",
                    where=f"ports.{pid}.part",
                ))
        else:
            # revolute_joint / prismatic_joint
            if port.part not in joint_ids:
                failures.append(_make(
                    FailureCode.MISSING_PORT, Severity.CRITICAL,
                    f"Port {pid!r} ({port.kind}) references joint "
                    f"{port.part!r} that does not exist.",
                    where=f"ports.{pid}.part",
                ))
        if not _is_finite_tuple(port.pose_local_mm, length=3):
            failures.append(_make(
                FailureCode.SCHEMA_ERROR, Severity.CRITICAL,
                f"Port {pid!r} pose_local_mm must be a 3-tuple of "
                f"finite floats.",
                where=f"ports.{pid}.pose_local_mm",
            ))
    return failures


def _check_required_ports(ir: DesignIR,
                          task: TaskSpec | None) -> list[Failure]:
    if task is None:
        return []
    failures: list[Failure] = []
    for required in task.required_ports:
        if required not in ir.ports:
            failures.append(_make(
                FailureCode.MISSING_PORT, Severity.CRITICAL,
                f"Required port {required!r} (task {task.id!r}) is "
                f"missing from the submission.",
                where=f"ports.{required}",
            ))
    return failures


def _check_ground(ir: DesignIR, task: TaskSpec | None) -> list[Failure]:
    if task is None:
        return []
    family = (task.family or "").lower()
    if not family.startswith("planar"):
        return []
    if task.objective.get("ground_required", True) is False:
        return []
    n_fixed = sum(1 for p in ir.parts if p.fixed is True)
    if n_fixed == 1:
        return []
    return [_make(
        FailureCode.WRONG_TOPOLOGY, Severity.CRITICAL,
        f"Planar task {task.id!r} requires exactly one fixed (ground) "
        f"part, found {n_fixed}.",
        where="parts",
    )]


# --------------------------------------------------------------------- #
# Part / mass / inertia checks                                          #
# --------------------------------------------------------------------- #


def _check_part_basic(p: Part) -> list[Failure]:
    failures: list[Failure] = []
    if not isinstance(p.fixed, bool):
        failures.append(_make(
            FailureCode.SCHEMA_ERROR, Severity.CRITICAL,
            f"Part {p.id!r} field 'fixed' must be a bool, got "
            f"{type(p.fixed).__name__}.",
            where=f"parts.{p.id}.fixed",
        ))
    if not isinstance(p.params, dict):
        failures.append(_make(
            FailureCode.SCHEMA_ERROR, Severity.CRITICAL,
            f"Part {p.id!r} params must be a dict, got "
            f"{type(p.params).__name__}.",
            where=f"parts.{p.id}.params",
        ))
    if not isinstance(p.geometry, dict):
        failures.append(_make(
            FailureCode.INVALID_ARTIFACT, Severity.CRITICAL,
            f"Part {p.id!r} geometry must be a dict[str, str], got "
            f"{type(p.geometry).__name__}.",
            where=f"parts.{p.id}.geometry",
        ))
    else:
        for k, v in p.geometry.items():
            if not isinstance(k, str):
                failures.append(_make(
                    FailureCode.INVALID_ARTIFACT, Severity.CRITICAL,
                    f"Part {p.id!r} geometry key {k!r} is not a string.",
                    where=f"parts.{p.id}.geometry",
                ))
            if not isinstance(v, str):
                failures.append(_make(
                    FailureCode.INVALID_ARTIFACT, Severity.CRITICAL,
                    f"Part {p.id!r} geometry[{k!r}] value must be a "
                    f"string path, got {type(v).__name__}.",
                    where=f"parts.{p.id}.geometry.{k}",
                ))
    return failures


def _check_mass_properties(ir: DesignIR,
                           task: TaskSpec | None) -> list[Failure]:
    failures: list[Failure] = []
    allow_massless = (
        task is not None
        and bool(task.objective.get("allow_massless_links", False))
    )
    for p in ir.parts:
        if not _is_finite_number(p.mass_kg):
            failures.append(_make(
                FailureCode.INVALID_MASS_PROPERTIES, Severity.CRITICAL,
                f"Part {p.id!r} mass_kg is not a finite number "
                f"({p.mass_kg!r}).",
                where=f"parts.{p.id}.mass_kg",
            ))
        else:
            if float(p.mass_kg) < 0.0:
                failures.append(_make(
                    FailureCode.INVALID_MASS_PROPERTIES, Severity.CRITICAL,
                    f"Part {p.id!r} has negative mass {p.mass_kg}.",
                    where=f"parts.{p.id}.mass_kg",
                ))
            elif (p.fixed is not True
                  and float(p.mass_kg) <= 0.0
                  and not allow_massless):
                failures.append(_make(
                    FailureCode.INVALID_MASS_PROPERTIES, Severity.CRITICAL,
                    f"Moving part {p.id!r} has non-positive mass "
                    f"{p.mass_kg}. Set a positive mass or declare "
                    f"allow_massless_links in task.objective.",
                    where=f"parts.{p.id}.mass_kg",
                ))
        if not _is_finite_tuple(p.com_local_mm, length=3):
            failures.append(_make(
                FailureCode.INVALID_MASS_PROPERTIES, Severity.CRITICAL,
                f"Part {p.id!r} com_local_mm must be a 3-tuple of "
                f"finite floats.",
                where=f"parts.{p.id}.com_local_mm",
            ))
        failures.extend(_check_inertia(p, allow_massless=allow_massless))
    return failures


def _check_inertia(p: Part, *, allow_massless: bool) -> list[Failure]:
    failures: list[Failure] = []
    I = p.inertia_kg_m2
    # Shape.
    if (not isinstance(I, (tuple, list)) or len(I) != 3
            or any(not isinstance(r, (tuple, list)) or len(r) != 3
                   for r in I)):
        failures.append(_make(
            FailureCode.INVALID_MASS_PROPERTIES, Severity.CRITICAL,
            f"Part {p.id!r} inertia_kg_m2 must be a 3x3 matrix.",
            where=f"parts.{p.id}.inertia_kg_m2",
        ))
        return failures
    flat = [v for row in I for v in row]
    if not all(_is_finite_number(v) for v in flat):
        failures.append(_make(
            FailureCode.INVALID_MASS_PROPERTIES, Severity.CRITICAL,
            f"Part {p.id!r} inertia_kg_m2 contains non-finite entries.",
            where=f"parts.{p.id}.inertia_kg_m2",
        ))
        return failures

    M = np.array(I, dtype=float)
    max_abs = float(np.max(np.abs(M)))
    # Symmetry uses a relative tolerance scaled by the matrix norm,
    # with an absolute floor for tiny inertias.
    sym_atol = max(1e-12, 1e-6 * max_abs)
    if not np.allclose(M, M.T, atol=sym_atol, rtol=0.0):
        failures.append(_make(
            FailureCode.INVALID_MASS_PROPERTIES, Severity.CRITICAL,
            f"Part {p.id!r} inertia is not symmetric within tolerance.",
            where=f"parts.{p.id}.inertia_kg_m2",
        ))
        return failures

    sym = 0.5 * (M + M.T)
    try:
        eigs = np.linalg.eigvalsh(sym)
    except np.linalg.LinAlgError:
        failures.append(_make(
            FailureCode.INVALID_MASS_PROPERTIES, Severity.CRITICAL,
            f"Part {p.id!r} inertia eigendecomposition failed.",
            where=f"parts.{p.id}.inertia_kg_m2",
        ))
        return failures
    if not np.all(np.isfinite(eigs)):
        failures.append(_make(
            FailureCode.INVALID_MASS_PROPERTIES, Severity.CRITICAL,
            f"Part {p.id!r} inertia eigenvalues are not finite.",
            where=f"parts.{p.id}.inertia_kg_m2",
        ))
        return failures

    # ``zero_tol`` is the absolute threshold below which we call an
    # eigenvalue "indistinguishable from zero." Physical inertias
    # never go below this (e.g. 1 g·mm² ~ 1e-9 kg·m²). Keep it well
    # below default schema sentinels (1e-6) so untouched defaults are
    # still considered physical.
    zero_tol = 1e-12
    neg_tol = max(zero_tol, 1e-6 * max_abs)
    if float(np.min(eigs)) < -neg_tol:
        failures.append(_make(
            FailureCode.INVALID_MASS_PROPERTIES, Severity.CRITICAL,
            f"Part {p.id!r} inertia has a negative eigenvalue "
            f"({float(np.min(eigs)):.3e}); not a physical rigid body.",
            where=f"parts.{p.id}.inertia_kg_m2",
        ))
        return failures

    is_moving_with_mass = (
        p.fixed is not True
        and _is_finite_number(p.mass_kg)
        and float(p.mass_kg) > 0.0
    )
    if is_moving_with_mass and not allow_massless:
        if float(np.min(eigs)) <= zero_tol:
            failures.append(_make(
                FailureCode.INVALID_MASS_PROPERTIES, Severity.CRITICAL,
                f"Part {p.id!r} has positive mass {p.mass_kg} but a "
                f"near-zero inertia eigenvalue "
                f"({float(np.min(eigs)):.3e}); rigid body must have "
                f"strictly positive principal moments.",
                where=f"parts.{p.id}.inertia_kg_m2",
            ))
            return failures

    # Triangle inequality on eigenvalues. Skip if everything is
    # effectively zero (massless analytic body).
    e = np.sort(np.clip(eigs, 0.0, None))
    if float(e[-1]) > zero_tol:
        ttol = max(zero_tol, 1e-6 * float(e[-1]))
        if (e[0] + e[1] < e[2] - ttol
                or e[1] + e[2] < e[0] - ttol
                or e[0] + e[2] < e[1] - ttol):
            failures.append(_make(
                FailureCode.INVALID_MASS_PROPERTIES, Severity.CRITICAL,
                f"Part {p.id!r} principal moments "
                f"{tuple(float(x) for x in e)} violate the triangle "
                f"inequality for a physical rigid body.",
                where=f"parts.{p.id}.inertia_kg_m2",
            ))
    return failures


# --------------------------------------------------------------------- #
# Joint geometry                                                        #
# --------------------------------------------------------------------- #


def _check_joint_geometry(ir: DesignIR) -> list[Failure]:
    failures: list[Failure] = []
    for j in ir.joints:
        if j.axis_world is not None:
            if not _is_finite_tuple(j.axis_world, length=3):
                failures.append(_make(
                    FailureCode.SCHEMA_ERROR, Severity.CRITICAL,
                    f"Joint {j.id!r} axis_world must be a 3-tuple of "
                    f"finite floats.",
                    where=f"joints.{j.id}.axis_world",
                ))
            else:
                if sum(v * v for v in j.axis_world) <= 0.0:
                    failures.append(_make(
                        FailureCode.SCHEMA_ERROR, Severity.CRITICAL,
                        f"Joint {j.id!r} axis_world is a zero vector.",
                        where=f"joints.{j.id}.axis_world",
                    ))
        if j.anchor_world_mm is not None \
                and not _is_finite_tuple(j.anchor_world_mm, length=3):
            failures.append(_make(
                FailureCode.SCHEMA_ERROR, Severity.CRITICAL,
                f"Joint {j.id!r} anchor_world_mm must be a 3-tuple of "
                f"finite floats.",
                where=f"joints.{j.id}.anchor_world_mm",
            ))
        if j.limits_rad is not None:
            if not _is_finite_tuple(j.limits_rad, length=2):
                failures.append(_make(
                    FailureCode.SCHEMA_ERROR, Severity.CRITICAL,
                    f"Joint {j.id!r} limits_rad must be a 2-tuple of "
                    f"finite floats.",
                    where=f"joints.{j.id}.limits_rad",
                ))
            elif j.limits_rad[0] > j.limits_rad[1]:
                failures.append(_make(
                    FailureCode.SCHEMA_ERROR, Severity.CRITICAL,
                    f"Joint {j.id!r} limits_rad lower bound exceeds "
                    f"upper bound.",
                    where=f"joints.{j.id}.limits_rad",
                ))
    return failures


# --------------------------------------------------------------------- #
# Geometry path safety                                                  #
# --------------------------------------------------------------------- #


def _path_string_diagnosis(s: str) -> str | None:
    """Return a human-readable reason if ``s`` is structurally unsafe.

    Reasons we reject:
      * empty
      * embedded NUL or other control characters
      * parent-directory traversal in either Posix or Windows form
      * raw backslash (never legitimate on Linux/macOS submissions)
    """
    if not s:
        return "path is empty"
    if any(ord(c) in _CONTROL_CHARS for c in s):
        return "path contains a control character or NUL"
    if "\\" in s:
        # Normalize and look for traversal, but also reject the
        # raw backslash unconditionally: cross-platform submissions
        # should use forward slashes.
        norm = s.replace("\\", "/")
        parts = PurePosixPath(norm).parts
        if ".." in parts:
            return "path contains a parent-directory traversal segment"
        return "path contains a backslash separator; use '/' only"
    parts = PurePosixPath(s).parts
    if ".." in parts:
        return "path contains a parent-directory traversal segment"
    return None


def _is_under(child: Path, root: Path) -> bool:
    try:
        child.relative_to(root)
        return True
    except (ValueError, OSError):
        return False


def _resolve_safe(candidate: Path) -> Path | None:
    try:
        return candidate.resolve()
    except (OSError, ValueError, RuntimeError):
        return None


def _check_geometry_paths(ir: DesignIR,
                          build_root: Path | None) -> list[Failure]:
    failures: list[Failure] = []
    root_resolved = build_root.resolve() if build_root is not None else None

    for p in ir.parts:
        if not isinstance(p.geometry, dict) or not p.geometry:
            continue
        for key, raw_path in p.geometry.items():
            if not isinstance(raw_path, str):
                # _check_part_basic already flagged this; skip path logic.
                continue
            reason = _path_string_diagnosis(raw_path)
            if reason is not None:
                failures.append(_make(
                    FailureCode.INVALID_ARTIFACT, Severity.CRITICAL,
                    f"Part {p.id!r} geometry[{key!r}] {raw_path!r} "
                    f"rejected: {reason}.",
                    where=f"parts.{p.id}.geometry.{key}",
                ))
                continue
            try:
                candidate = Path(raw_path)
            except (TypeError, ValueError):
                failures.append(_make(
                    FailureCode.INVALID_ARTIFACT, Severity.CRITICAL,
                    f"Part {p.id!r} geometry[{key!r}] path "
                    f"{raw_path!r} could not be parsed as a filesystem "
                    f"path.",
                    where=f"parts.{p.id}.geometry.{key}",
                ))
                continue
            if candidate.is_absolute():
                if root_resolved is None:
                    failures.append(_make(
                        FailureCode.INVALID_ARTIFACT, Severity.CRITICAL,
                        f"Part {p.id!r} geometry[{key!r}] uses an "
                        f"absolute path {raw_path!r}; only relative "
                        f"paths are allowed.",
                        where=f"parts.{p.id}.geometry.{key}",
                    ))
                    continue
                resolved = _resolve_safe(candidate)
                if resolved is None or not _is_under(resolved, root_resolved):
                    failures.append(_make(
                        FailureCode.INVALID_ARTIFACT, Severity.CRITICAL,
                        f"Part {p.id!r} geometry[{key!r}] absolute "
                        f"path {raw_path!r} resolves outside the "
                        f"build root.",
                        where=f"parts.{p.id}.geometry.{key}",
                    ))
                    continue
                resolved_path: Path | None = resolved
            else:
                if root_resolved is None:
                    resolved_path = None
                else:
                    resolved = _resolve_safe(root_resolved / candidate)
                    if resolved is None or not _is_under(
                            resolved, root_resolved):
                        failures.append(_make(
                            FailureCode.INVALID_ARTIFACT, Severity.CRITICAL,
                            f"Part {p.id!r} geometry[{key!r}] relative "
                            f"path {raw_path!r} resolves outside the "
                            f"build root (likely a symlink escape).",
                            where=f"parts.{p.id}.geometry.{key}",
                        ))
                        continue
                    resolved_path = resolved
            if resolved_path is not None and not resolved_path.exists():
                failures.append(_make(
                    FailureCode.INVALID_ARTIFACT, Severity.CRITICAL,
                    f"Part {p.id!r} geometry[{key!r}] file does not "
                    f"exist at {resolved_path}.",
                    where=f"parts.{p.id}.geometry.{key}",
                ))
    return failures


# --------------------------------------------------------------------- #
# Top-level                                                             #
# --------------------------------------------------------------------- #


def _wrap_total(fn, *args, **kwargs) -> list[Failure]:
    """Run a sub-check defensively. Any exception becomes a single
    SCHEMA_ERROR failure so the validator is total.
    """
    try:
        return list(fn(*args, **kwargs))
    except Exception as e:  # noqa: BLE001 — intentional firewall
        return [_make(
            FailureCode.SCHEMA_ERROR, Severity.CRITICAL,
            f"Internal validation error in {fn.__name__}: "
            f"{type(e).__name__}: {e}",
            where=fn.__name__,
        )]


def validate_design_ir(
    ir: DesignIR,
    *,
    task: TaskSpec | None = None,
    build_root: Path | None = None,
) -> list[Failure]:
    """Return a (possibly empty) list of structured failures.

    This function MUST NOT raise. Any malformed field becomes a
    structured Failure with a CRITICAL FailureCode.
    """
    if not isinstance(ir, DesignIR):
        return [_make(
            FailureCode.SCHEMA_ERROR, Severity.CRITICAL,
            f"validate_design_ir expected a DesignIR, got "
            f"{type(ir).__name__}.",
        )]
    if not isinstance(ir.params, dict):
        # Catch this early — _wrap_total still protects the rest.
        return [_make(
            FailureCode.SCHEMA_ERROR, Severity.CRITICAL,
            f"DesignIR params must be a dict, got "
            f"{type(ir.params).__name__}.",
            where="params",
        )]

    failures: list[Failure] = []
    failures.extend(_wrap_total(_check_schema_version, ir))
    failures.extend(_wrap_total(_check_unique_nonempty_ids, ir.parts,
                                "parts"))
    failures.extend(_wrap_total(_check_unique_nonempty_ids, ir.joints,
                                "joints"))
    failures.extend(_wrap_total(_check_unique_nonempty_ids,
                                list(ir.ports.values()), "ports"))
    for p in ir.parts:
        failures.extend(_wrap_total(_check_part_basic, p))
    failures.extend(_wrap_total(_check_joints, ir))
    failures.extend(_wrap_total(_check_ports, ir))
    failures.extend(_wrap_total(_check_required_ports, ir, task))
    failures.extend(_wrap_total(_check_ground, ir, task))
    failures.extend(_wrap_total(_check_mass_properties, ir, task))
    failures.extend(_wrap_total(_check_joint_geometry, ir))
    failures.extend(_wrap_total(_check_geometry_paths, ir, build_root))
    return failures


def has_critical_failures(failures: list[Failure]) -> bool:
    return any(f.code in CRITICAL_CODES and f.severity == Severity.CRITICAL
               for f in failures)
