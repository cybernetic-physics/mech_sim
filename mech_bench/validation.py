"""DesignIR validation layer.

Runs before any probe touches the IR. The contract is:
  * Anything structurally malformed surfaces as a SCHEMA_ERROR.
  * Anything that would make probes downstream produce noise — bad
    references, NaN mass, non-physical inertia — surfaces with a
    specific FailureCode so the agent can repair it.

Validation never imports adapter or probe code; it is a pure
DesignIR/TaskSpec check.
"""

from __future__ import annotations

import math
import os
from pathlib import Path, PurePosixPath
from typing import Iterable

from mech_bench.feedback import Failure, FailureCode, Severity
from mech_bench.schema import DesignIR, Joint, Part, Port, TaskSpec


SCHEMA_VERSION = "design_ir.v2"

# FailureCodes that the evaluator treats as critical hard-gate stops.
CRITICAL_CODES: frozenset[FailureCode] = frozenset({
    FailureCode.SCHEMA_ERROR,
    FailureCode.INVALID_ARTIFACT,
    FailureCode.INVALID_MASS_PROPERTIES,
    FailureCode.MISSING_PORT,
    FailureCode.WRONG_TOPOLOGY,
})


def _is_finite_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) \
        and math.isfinite(float(x))


def _is_finite_tuple(t, length: int | None = None) -> bool:
    if not isinstance(t, (tuple, list)):
        return False
    if length is not None and len(t) != length:
        return False
    return all(_is_finite_number(x) for x in t)


def _make(code: FailureCode, severity: Severity, msg: str,
          where: str | None = None, **extra) -> Failure:
    return Failure(code=code, severity=severity, message=msg,
                   where=where, extra=dict(extra) if extra else {})


def _check_schema_version(ir: DesignIR) -> list[Failure]:
    if ir.schema_version != SCHEMA_VERSION:
        return [_make(
            FailureCode.SCHEMA_ERROR, Severity.CRITICAL,
            f"DesignIR schema_version must be {SCHEMA_VERSION!r}, "
            f"got {ir.schema_version!r}.",
            where="schema_version",
        )]
    return []


def _check_unique_nonempty_ids(items: Iterable, name: str,
                               id_attr: str = "id") -> list[Failure]:
    seen: set[str] = set()
    failures: list[Failure] = []
    for i, item in enumerate(items):
        ident = getattr(item, id_attr, None) if not isinstance(item, str) \
            else item
        if not isinstance(ident, str) or not ident:
            failures.append(_make(
                FailureCode.SCHEMA_ERROR, Severity.CRITICAL,
                f"{name}[{i}] has empty or non-string id.",
                where=f"{name}[{i}].id",
            ))
            continue
        if ident in seen:
            failures.append(_make(
                FailureCode.SCHEMA_ERROR, Severity.CRITICAL,
                f"Duplicate {name} id {ident!r}.",
                where=f"{name}.{ident}",
            ))
        seen.add(ident)
    return failures


def _check_joint_refs(ir: DesignIR) -> list[Failure]:
    failures: list[Failure] = []
    part_ids = ir.part_ids()
    for j in ir.joints:
        if j.parent not in part_ids:
            failures.append(_make(
                FailureCode.WRONG_TOPOLOGY, Severity.CRITICAL,
                f"Joint {j.id!r} parent {j.parent!r} is not an existing "
                f"part id.",
                where=f"joints.{j.id}.parent",
            ))
        if j.child not in part_ids:
            failures.append(_make(
                FailureCode.WRONG_TOPOLOGY, Severity.CRITICAL,
                f"Joint {j.id!r} child {j.child!r} is not an existing "
                f"part id.",
                where=f"joints.{j.id}.child",
            ))
        if j.parent and j.child and j.parent == j.child:
            failures.append(_make(
                FailureCode.WRONG_TOPOLOGY, Severity.CRITICAL,
                f"Joint {j.id!r} parent and child reference the same "
                f"part {j.parent!r}.",
                where=f"joints.{j.id}",
            ))
    return failures


def _check_port_refs(ir: DesignIR) -> list[Failure]:
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
        if port.kind == "frame":
            if port.part not in part_ids:
                failures.append(_make(
                    FailureCode.MISSING_PORT, Severity.CRITICAL,
                    f"Port {pid!r} (frame) references part {port.part!r} "
                    f"that does not exist.",
                    where=f"ports.{pid}.part",
                ))
        elif port.kind in ("revolute_joint", "prismatic_joint"):
            if port.part not in joint_ids:
                failures.append(_make(
                    FailureCode.MISSING_PORT, Severity.CRITICAL,
                    f"Port {pid!r} ({port.kind}) references joint "
                    f"{port.part!r} that does not exist.",
                    where=f"ports.{pid}.part",
                ))
        else:
            failures.append(_make(
                FailureCode.SCHEMA_ERROR, Severity.CRITICAL,
                f"Port {pid!r} has unknown kind {port.kind!r}.",
                where=f"ports.{pid}.kind",
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
    n_fixed = sum(1 for p in ir.parts if p.fixed)
    if n_fixed == 1:
        return []
    msg = (f"Planar task {task.id!r} requires exactly one fixed "
           f"(ground) part, found {n_fixed}.")
    return [_make(
        FailureCode.WRONG_TOPOLOGY, Severity.CRITICAL, msg,
        where="parts",
    )]


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
            continue
        if float(p.mass_kg) < 0.0:
            failures.append(_make(
                FailureCode.INVALID_MASS_PROPERTIES, Severity.CRITICAL,
                f"Part {p.id!r} has negative mass {p.mass_kg}.",
                where=f"parts.{p.id}.mass_kg",
            ))
            continue
        if not p.fixed and float(p.mass_kg) <= 0.0 and not allow_massless:
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
        failures.extend(_check_inertia(p))
    return failures


def _check_inertia(p: Part) -> list[Failure]:
    failures: list[Failure] = []
    I = p.inertia_kg_m2
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
    # Symmetry within tolerance.
    tol = 1e-9 + 1e-6 * max(abs(v) for v in flat)
    sym_ok = (
        abs(I[0][1] - I[1][0]) <= tol
        and abs(I[0][2] - I[2][0]) <= tol
        and abs(I[1][2] - I[2][1]) <= tol
    )
    if not sym_ok:
        failures.append(_make(
            FailureCode.INVALID_MASS_PROPERTIES, Severity.CRITICAL,
            f"Part {p.id!r} inertia is not symmetric within tolerance.",
            where=f"parts.{p.id}.inertia_kg_m2",
        ))
    # Diagonal positivity (skip the all-zero case for fixed/no-mass).
    diag = (float(I[0][0]), float(I[1][1]), float(I[2][2]))
    if any(d < 0.0 for d in diag):
        failures.append(_make(
            FailureCode.INVALID_MASS_PROPERTIES, Severity.CRITICAL,
            f"Part {p.id!r} principal inertia diagonal has a negative "
            f"entry: {diag}.",
            where=f"parts.{p.id}.inertia_kg_m2",
        ))
    # Triangle inequality on principal moments (only meaningful when
    # the matrix is non-degenerate).
    Ix, Iy, Iz = diag
    if max(diag) > 0.0:
        # tol for triangle inequality is relative to the largest moment.
        ttol = 1e-9 + 1e-6 * max(diag)
        if (Ix + Iy < Iz - ttol
                or Iy + Iz < Ix - ttol
                or Ix + Iz < Iy - ttol):
            failures.append(_make(
                FailureCode.INVALID_MASS_PROPERTIES, Severity.CRITICAL,
                f"Part {p.id!r} principal moments {diag} violate the "
                f"triangle inequality for a physical rigid body.",
                where=f"parts.{p.id}.inertia_kg_m2",
            ))
    return failures


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


def _has_traversal(path_str: str) -> bool:
    pure = PurePosixPath(path_str.replace(os.sep, "/"))
    return ".." in pure.parts


def _check_geometry_paths(ir: DesignIR,
                          build_root: Path | None) -> list[Failure]:
    failures: list[Failure] = []
    root_resolved = build_root.resolve() if build_root is not None else None
    for p in ir.parts:
        if not p.geometry:
            continue
        for key, raw_path in p.geometry.items():
            if not isinstance(raw_path, str) or not raw_path:
                failures.append(_make(
                    FailureCode.INVALID_ARTIFACT, Severity.CRITICAL,
                    f"Part {p.id!r} geometry[{key!r}] is not a non-empty "
                    f"string path.",
                    where=f"parts.{p.id}.geometry.{key}",
                ))
                continue
            if _has_traversal(raw_path):
                failures.append(_make(
                    FailureCode.INVALID_ARTIFACT, Severity.CRITICAL,
                    f"Part {p.id!r} geometry[{key!r}] path "
                    f"{raw_path!r} contains a parent-directory "
                    f"traversal segment.",
                    where=f"parts.{p.id}.geometry.{key}",
                ))
                continue
            candidate = Path(raw_path)
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
                try:
                    resolved = candidate.resolve()
                except OSError:
                    failures.append(_make(
                        FailureCode.INVALID_ARTIFACT, Severity.CRITICAL,
                        f"Part {p.id!r} geometry[{key!r}] could not be "
                        f"resolved: {raw_path!r}.",
                        where=f"parts.{p.id}.geometry.{key}",
                    ))
                    continue
                if not _is_under(resolved, root_resolved):
                    failures.append(_make(
                        FailureCode.INVALID_ARTIFACT, Severity.CRITICAL,
                        f"Part {p.id!r} geometry[{key!r}] absolute path "
                        f"{raw_path!r} resolves outside the build "
                        f"root.",
                        where=f"parts.{p.id}.geometry.{key}",
                    ))
                    continue
                resolved_path = resolved
            else:
                if root_resolved is None:
                    resolved_path = None
                else:
                    try:
                        resolved_path = (root_resolved / candidate).resolve()
                    except OSError:
                        failures.append(_make(
                            FailureCode.INVALID_ARTIFACT, Severity.CRITICAL,
                            f"Part {p.id!r} geometry[{key!r}] could not "
                            f"be resolved against build_root.",
                            where=f"parts.{p.id}.geometry.{key}",
                        ))
                        continue
                    if not _is_under(resolved_path, root_resolved):
                        failures.append(_make(
                            FailureCode.INVALID_ARTIFACT, Severity.CRITICAL,
                            f"Part {p.id!r} geometry[{key!r}] relative "
                            f"path {raw_path!r} escapes the build root.",
                            where=f"parts.{p.id}.geometry.{key}",
                        ))
                        continue
            if resolved_path is not None and not resolved_path.exists():
                failures.append(_make(
                    FailureCode.INVALID_ARTIFACT, Severity.CRITICAL,
                    f"Part {p.id!r} geometry[{key!r}] file does not "
                    f"exist at {resolved_path}.",
                    where=f"parts.{p.id}.geometry.{key}",
                ))
    return failures


def _is_under(child: Path, root: Path) -> bool:
    try:
        child.relative_to(root)
        return True
    except ValueError:
        return False


def validate_design_ir(
    ir: DesignIR,
    *,
    task: TaskSpec | None = None,
    build_root: Path | None = None,
) -> list[Failure]:
    """Return a (possibly empty) list of structured failures.

    The check order is fixed so that callers get a deterministic
    list. Severity is CRITICAL for everything emitted here — by the
    time we are running probes we expect the IR to be structurally
    sound; anything weaker than CRITICAL is reserved for the probes
    themselves.
    """
    failures: list[Failure] = []
    failures.extend(_check_schema_version(ir))
    failures.extend(_check_unique_nonempty_ids(ir.parts, "parts"))
    failures.extend(_check_unique_nonempty_ids(ir.joints, "joints"))
    failures.extend(_check_unique_nonempty_ids(
        list(ir.ports.values()), "ports"))
    failures.extend(_check_joint_refs(ir))
    failures.extend(_check_port_refs(ir))
    failures.extend(_check_required_ports(ir, task))
    failures.extend(_check_ground(ir, task))
    failures.extend(_check_mass_properties(ir, task))
    failures.extend(_check_joint_geometry(ir))
    failures.extend(_check_geometry_paths(ir, build_root))
    return failures


def has_critical_failures(failures: list[Failure]) -> bool:
    return any(f.code in CRITICAL_CODES and f.severity == Severity.CRITICAL
               for f in failures)
