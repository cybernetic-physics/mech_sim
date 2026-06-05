"""Geometry-grounded analytic-mechanics adapter.

This is a *real* (closed-form rigid-body) oracle — the opposite of
``fake_contact_oracle``, which fabricates contact force / ratio / torque from
config constants independent of the agent's geometry. Everything this adapter
emits is computed **from the DesignIR geometry**:

* transmission ratio from gear tooth counts (or pitch radii / pulley radii),
  never from ``params['declared_ratio']``;
* contact engagement & force from the *actual distance* between the paired
  bodies versus the sum of their pitch radii — so two gears placed 100 m apart
  produce **zero** force and fail engagement (the central anti-hack property);
* penetration from geometric overlap;
* torque / power from the ideal rigid lever relation using the geometry-derived
  ratio and the task's applied input speed / output load.

It is therefore unfakeable by construction and ``oracle_is_synthetic = False``.

It is deterministic (pure NumPy, no RNG / wall-clock). It is *not* a mesh
contact solver — true mesh interpenetration and compliant contact remain the
job of the high-fidelity ``chrono_contact`` tier (hence it does NOT advertise
``MESH_OVERLAP``). When the geometry is insufficient to determine a quantity,
the adapter emits an honest zero / absent value rather than a fabricated one,
so a probe cannot earn credit it did not geometrically deserve.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from mech_bench.adapters import SimAdapter, register_adapter
from mech_bench.probes import Capability
from mech_bench.schema import DesignIR, Joint, Part


_DEFAULT_MODULE_MM = 1.0  # gear module fallback when only tooth counts exist.


def _normalize_pair(p: str) -> str:
    a, _, b = str(p).partition(":")
    if not b:
        return str(p)
    return ":".join(sorted([a, b]))


def _part_by_id(ir: DesignIR, pid: str) -> Part | None:
    for p in ir.parts:
        if p.id == pid:
            return p
    return None


def _joint_by_id(ir: DesignIR, jid: str) -> Joint | None:
    for j in ir.joints:
        if j.id == jid:
            return j
    return None


def _moving_part_for_port(ir: DesignIR, port_id: str) -> str | None:
    """Resolve the moving body a port drives.

    ``port.part`` may name either a Part or a Joint (the generators use the
    revolute-joint id). When it is a joint, the moving body is the joint child.
    """
    port = ir.ports.get(port_id)
    if port is None:
        return None
    ref = port.part
    if _part_by_id(ir, ref) is not None:
        return ref
    j = _joint_by_id(ir, ref)
    if j is not None:
        return j.child
    return None


def _part_center_mm(ir: DesignIR, part_id: str | None) -> np.ndarray:
    """World center of a part, preferring the anchor of the revolute joint it
    rotates about (gears spin about their axis), then explicit position params,
    then COM, then origin."""
    if part_id is None:
        return np.zeros(3, dtype=float)
    for j in ir.joints:
        if j.child == part_id and j.anchor_world_mm is not None:
            return np.asarray(j.anchor_world_mm, dtype=float)
    part = _part_by_id(ir, part_id)
    if part is not None:
        for key in ("center_world_mm", "position_mm", "center_mm"):
            v = (part.params or {}).get(key)
            if v is not None and len(v) >= 3:
                return np.asarray(v[:3], dtype=float)
        if part.com_local_mm is not None:
            return np.asarray(part.com_local_mm, dtype=float)
    return np.zeros(3, dtype=float)


def _teeth(part: Part | None) -> float | None:
    if part is None:
        return None
    for key in ("teeth", "n_teeth", "tooth_count"):
        v = (part.params or {}).get(key)
        if v is not None:
            try:
                t = float(v)
            except (TypeError, ValueError):
                return None
            return t if t > 0 else None
    return None


def _pitch_radius_mm(part: Part | None) -> float | None:
    """Pitch / effective radius of a rotating body, from explicit radius, or
    teeth * module / 2, or a declared pulley/gear radius. ``None`` when the
    geometry does not specify it (we never invent one)."""
    if part is None:
        return None
    params = part.params or {}
    for key in ("pitch_radius_mm", "radius_mm", "pulley_radius_mm"):
        v = params.get(key)
        if v is not None:
            try:
                r = float(v)
            except (TypeError, ValueError):
                continue
            if r > 0:
                return r
    teeth = _teeth(part)
    if teeth is not None:
        module = params.get("module_mm")
        try:
            module = float(module) if module is not None else _DEFAULT_MODULE_MM
        except (TypeError, ValueError):
            module = _DEFAULT_MODULE_MM
        if module > 0:
            return teeth * module / 2.0
    return None


def _capability_unavailable_payload(reason: str) -> dict[str, Any]:
    """Honest decline: when the geometry does not let us compute any real
    quantity, we surface capability_unavailable rather than fabricate zeros.
    This keeps geometry-less placeholder tasks on the unavailable path (no
    silent pass) while real-geometry tasks get genuinely graded."""
    return {
        "time_s": np.zeros(0, dtype=float),
        "joint_positions": {},
        "joint_velocities": {},
        "contact_forces": {},
        "penetration": {},
        "body_poses": {},
        "port_traces": {},
        "scalar_metrics": {},
        "metadata": {
            "adapter": "analytic_mechanics",
            "simulator": "analytic_mechanics",
            "oracle_is_synthetic": False,
            "preflight_issues": [reason],
        },
        "__capability_unavailable__": True,
        "adapter": "analytic_mechanics",
    }


def _geometry_ratio(ir: DesignIR, input_part: str | None, output_part: str | None) -> float | None:
    """Transmission ratio (output revolutions per input revolution magnitude)
    computed from geometry. Convention: ``ratio = teeth_out / teeth_in`` (a
    reducer with more output teeth has ratio > 1). Returns ``None`` when the
    geometry does not determine it — never falls back to a declared value."""
    pin = _part_by_id(ir, input_part) if input_part else None
    pout = _part_by_id(ir, output_part) if output_part else None
    t_in, t_out = _teeth(pin), _teeth(pout)
    if t_in and t_out:
        return t_out / t_in
    r_in, r_out = _pitch_radius_mm(pin), _pitch_radius_mm(pout)
    if r_in and r_out and r_in > 0:
        return r_out / r_in
    return None


class AnalyticMechanics(SimAdapter):
    """Real, deterministic, geometry-grounded rigid-body oracle."""

    type_name = "analytic_mechanics"
    capabilities_provided = frozenset({
        Capability.RIGID_BODY_DYNAMICS,
        Capability.CONTACT_FORCES,
        Capability.JOINT_CONSTRAINTS,
        Capability.MOTOR_DRIVES,
        Capability.LOAD_TORQUES,
        Capability.POSE_TRACES,
        Capability.PLANAR_KINEMATICS,
    })
    # Cheaper than fake_contact_oracle (50/1000) so the dispatcher prefers this
    # real oracle whenever a task is not explicitly pinned to the fake one.
    cost_tier = 10

    def run(self, ir: DesignIR, config: dict[str, Any]) -> dict[str, Any]:
        cfg = dict(config or {})
        n_samples = int(cfg.get("samples", 360))
        n_samples = max(n_samples, 2)
        duration_s = float(cfg.get("duration_s", 1.0))
        time_s = np.linspace(0.0, duration_s, n_samples, dtype=float)

        in_speed = float(cfg.get("input_speed_rad_s", 1.0))
        out_load = float(cfg.get("output_load_Nm", 0.0))
        # Clearance fraction: how far beyond perfect mesh still counts as
        # engaged (manufacturing slop). Geometry, not a force constant.
        clearance_frac = float(cfg.get("contact_clearance_frac", 0.10))

        inp_port = "input_port" if "input_port" in ir.ports else ""
        out_port = "output_port" if "output_port" in ir.ports else ""
        input_part = _moving_part_for_port(ir, inp_port) if inp_port else None
        output_part = _moving_part_for_port(ir, out_port) if out_port else None

        ratio = _geometry_ratio(ir, input_part, output_part)
        ratio_known = ratio is not None and math.isfinite(ratio) and ratio != 0.0

        # Resolve contact pairs up front so we can decide whether there is any
        # computable geometry at all.
        contact_pairs: list[str] = []
        for p in (cfg.get("contact_pairs") or []):
            contact_pairs.append(_normalize_pair(p))
        if not contact_pairs:
            for j in ir.joints:
                if str(j.type) == "contact_pair":
                    contact_pairs.append(_normalize_pair(f"{j.parent}:{j.child}"))

        def _pair_has_radii(pair: str) -> bool:
            a, _, b = pair.partition(":")
            return (
                _pitch_radius_mm(_part_by_id(ir, a)) is not None
                and _pitch_radius_mm(_part_by_id(ir, b)) is not None
            )

        any_contact_geometry = any(_pair_has_radii(p) for p in contact_pairs)

        # Honest decline (capability_unavailable, mirroring the real-Chrono
        # unavailable path): this adapter is only dispatched when a probe needs
        # contact / dynamics it cannot get from the cost-0 planar adapter. It
        # can only serve such a probe when the design *explicitly* declares
        # contact pairs (a ``contact_pair`` joint or config ``contact_pairs``)
        # whose bodies carry real pitch radii. Otherwise we decline rather than
        # emit fabricated zeros that would read as a genuine "missing_contact"
        # verdict. Tasks that model real contact (teeth + positions + a
        # contact_pair joint) get genuinely graded; placeholder stubs that only
        # declare revolute joints fall through to capability_unavailable, with
        # no silent pass.
        if not (contact_pairs and any_contact_geometry):
            return _capability_unavailable_payload(
                "analytic_mechanics: no explicit contact pair with gear "
                "geometry (teeth / pitch radii) to evaluate; declare a "
                "contact_pair joint between bodies with tooth counts."
            )

        # ----- joint kinematics from the geometry-derived ratio -----
        input_pos = in_speed * time_s
        input_vel = np.full_like(time_s, in_speed)
        if ratio_known:
            out_speed = in_speed / float(ratio)
        else:
            # Geometry does not determine a ratio: no observable output motion.
            out_speed = 0.0
        output_pos = out_speed * time_s
        output_vel = np.full_like(time_s, out_speed)

        joint_positions: dict[str, np.ndarray] = {}
        joint_velocities: dict[str, np.ndarray] = {}
        if inp_port:
            joint_positions[inp_port] = input_pos
            joint_velocities[inp_port] = input_vel
        if out_port:
            joint_positions[out_port] = output_pos
            joint_velocities[out_port] = output_vel
        if input_part:
            joint_positions[input_part] = input_pos
            joint_velocities[input_part] = input_vel
        if output_part:
            joint_positions[output_part] = output_pos
            joint_velocities[output_part] = output_vel

        # ----- contact engagement / force from real distances -----
        contact_forces: dict[str, np.ndarray] = {}
        penetration: dict[str, np.ndarray] = {}
        engaged_pairs = 0
        for pair in contact_pairs:
            a, _, b = pair.partition(":")
            ca = _part_center_mm(ir, a)
            cb = _part_center_mm(ir, b)
            d = float(np.linalg.norm(ca - cb))
            ra = _pitch_radius_mm(_part_by_id(ir, a))
            rb = _pitch_radius_mm(_part_by_id(ir, b))
            if ra is None or rb is None:
                # Geometry insufficient to judge contact: honest zero force.
                contact_forces[pair] = np.zeros(n_samples, dtype=float)
                penetration[pair] = np.zeros(n_samples, dtype=float)
                continue
            r_sum = ra + rb
            gap = d - r_sum  # >0 separated, <0 interpenetrating, ~0 meshing
            engaged = d <= r_sum * (1.0 + clearance_frac)
            # Engagement quality: 1 at perfect mesh, ->0 at the clearance edge.
            tol = max(clearance_frac * r_sum, 1e-9)
            quality = max(0.0, 1.0 - abs(gap) / tol) if engaged else 0.0
            if engaged and quality > 0.0:
                engaged_pairs += 1
                # Transmitted tangential force = output torque / output radius.
                r_out_mm = _pitch_radius_mm(_part_by_id(ir, output_part)) or rb
                f_mag = abs(out_load) * 1000.0 / max(r_out_mm, 1e-6)
                # If there is no load, fall back to a unit engagement force so
                # the pair is still observably in contact (engagement != load).
                if f_mag <= 0.0:
                    f_mag = 1.0
                f_mag *= quality
                contact_forces[pair] = np.full(n_samples, f_mag, dtype=float)
            else:
                contact_forces[pair] = np.zeros(n_samples, dtype=float)
            penetration[pair] = np.full(
                n_samples, max(0.0, -gap), dtype=float)

        # ----- torque / power (ideal rigid lever) -----
        if ratio_known:
            t_out = abs(out_load)
            t_in = t_out / abs(float(ratio))
            p_in = abs(t_in * in_speed)
            p_out = abs(t_out * out_speed)
        else:
            t_in = t_out = p_in = p_out = 0.0
        power_err_pct = 0.0
        if p_in > 0.0:
            power_err_pct = abs(p_in - p_out) / p_in * 100.0

        max_pen = float(max((float(np.max(v)) for v in penetration.values()),
                            default=0.0))
        scalar_metrics: dict[str, float] = {
            "ratio_observed": float(ratio) if ratio_known else 0.0,
            "ratio_known": 1.0 if ratio_known else 0.0,
            "max_penetration_mm": max_pen,
            "max_constraint_error_mm": 0.0,
            "torque_ripple_pct": 0.0,
            "input_torque_ripple_pct": 0.0,
            "power_balance_error_pct": float(power_err_pct),
            "lockup_detected": 0.0 if ratio_known else 1.0,
            "n_contacts_max": float(engaged_pairs),
            "input_torque_Nm_mean": float(t_in),
            "output_torque_Nm_mean": float(t_out),
            "input_power_W_mean": float(p_in),
            "output_power_W_mean": float(p_out),
        }

        metadata: dict[str, Any] = {
            "adapter": self.type_name,
            "simulator": "analytic_mechanics",
            "trust_level": "analytic_real",
            "is_physical_oracle": True,
            "oracle_is_synthetic": False,
            "solver": "closed_form_rigid_body",
            "contact_method": "geometric_engagement",
            "duration_s": duration_s,
            "dt": float(duration_s / max(n_samples - 1, 1)),
            "ratio_source": "geometry" if ratio_known else "undetermined",
            "preflight_issues": [],
            "build_meta": {
                "n_bodies": len(ir.parts),
                "n_joints": len(ir.joints),
                "n_pairs": len(contact_pairs),
                "n_engaged_pairs": engaged_pairs,
            },
        }

        return {
            "time_s": time_s,
            "joint_positions": joint_positions,
            "joint_velocities": joint_velocities,
            "contact_forces": contact_forces,
            "penetration": penetration,
            "body_poses": {},
            "port_traces": {},
            "scalar_metrics": scalar_metrics,
            "metadata": metadata,
            "adapter": self.type_name,
        }


register_adapter(AnalyticMechanics)
