"""Analytical planar-kinematics adapter.

Handles closed-loop planar four-bar, slider-crank, and simple
lead-screw topologies analytically from DesignIR topology + joint
anchor data. Emits port traces, joint position / velocity time-series,
and a shared time axis. Pure NumPy — no physics solver.

Topology detection is auto by default:
  * 4 parts + 4 revolutes → four-bar.
  * 4 parts + 3 revolutes + 1 prismatic → slider-crank (planar).
  * 3 parts + 1 revolute + 1 prismatic + lead_mm param → lead screw.

Anything outside these returns an empty trace (and probes downstream
surface ``SIMULATOR_DIVERGENCE`` or ``LOCKUP`` as appropriate).

Two correctness improvements over the prior v0:

1. **Branch continuity.** Rather than always picking the "elbow-up"
   intersection of the coupler/rocker circles, the solver picks the
   intersection closest to the previous sample's position. This keeps
   the trace smooth across the entire sweep instead of flipping when
   the Grashof inequality is briefly crossed.

2. **Per-sample invalid flagging.** If a single sample becomes
   geometrically unreachable (the circles don't intersect), the
   adapter records NaN for that sample and continues, instead of
   discarding the whole trace. ``strict_geometry=true`` (config) keeps
   the legacy behavior — return empty traces and let probes report.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mech_bench.adapters import SimAdapter, register_adapter
from mech_bench.probes import Capability
from mech_bench.schema import DesignIR, Joint, Part


def _world_anchor(j: Joint) -> np.ndarray:
    a = j.anchor_world_mm or (0.0, 0.0, 0.0)
    return np.array(a[:2], dtype=float)


def unwrap_angles(theta: np.ndarray) -> np.ndarray:
    """Return *theta* unwrapped so jumps larger than π are removed."""
    arr = np.asarray(theta, dtype=float).reshape(-1)
    if arr.size == 0:
        return arr
    finite = np.isfinite(arr)
    if not np.any(finite):
        return arr
    if np.all(finite):
        return np.unwrap(arr)
    # Unwrap the finite stretch only; preserve NaN positions.
    out = arr.copy()
    idx = np.where(finite)[0]
    out[idx] = np.unwrap(arr[idx])
    return out


def angular_velocity_finite_diff(
    theta: np.ndarray, time_s: np.ndarray,
) -> np.ndarray:
    """Central-difference angular velocity of an angle signal."""
    theta = unwrap_angles(theta)
    t = np.asarray(time_s, dtype=float).reshape(-1)
    if theta.size < 2 or t.size != theta.size:
        return np.zeros_like(theta)
    # ``np.gradient`` propagates NaN cleanly; that's what we want.
    return np.gradient(theta, t, edge_order=1)


def _intersect_circles(
    B: np.ndarray, D: np.ndarray, r1: float, r2: float,
    *, prev: np.ndarray | None,
) -> np.ndarray | None:
    """Return one intersection of circle(B, r1) and circle(D, r2).

    If ``prev`` is provided, the intersection nearer to ``prev`` is
    selected (continuity branch). Otherwise the "elbow-up" branch
    is returned. Returns ``None`` if the circles do not intersect.
    """
    BD = D - B
    d = float(np.linalg.norm(BD))
    if d <= 0.0:
        return None
    if d > r1 + r2 or d < abs(r1 - r2):
        return None
    a = (r1 * r1 - r2 * r2 + d * d) / (2.0 * d)
    h_sq = r1 * r1 - a * a
    if h_sq < 0.0:
        return None
    h = float(np.sqrt(h_sq))
    u = BD / d
    perp = np.array([-u[1], u[0]])
    M = B + a * u
    p_plus = M + h * perp
    p_minus = M - h * perp
    if prev is None:
        return p_plus  # elbow-up default for first sample
    if np.linalg.norm(p_plus - prev) <= np.linalg.norm(p_minus - prev):
        return p_plus
    return p_minus


def _solve_fourbar(
    ir: DesignIR,
    n_samples: int,
    *,
    strict_geometry: bool,
) -> dict[str, Any] | None:
    if len(ir.parts) != 4:
        return None
    revs = [j for j in ir.joints if j.type == "revolute"]
    if len(revs) != 4:
        return None

    fixed = [p for p in ir.parts if p.fixed]
    if len(fixed) != 1:
        return None
    ground = fixed[0].id

    ground_joints = [j for j in revs if ground in (j.parent, j.child)]
    if len(ground_joints) != 2:
        return None

    input_port = ir.ports.get("input_port")
    if input_port is None or input_port.kind != "revolute_joint":
        return None
    input_joint_id = input_port.part
    j_in = next((j for j in ground_joints if j.id == input_joint_id), None)
    if j_in is None:
        j_in = ground_joints[0]
    j_out = next(j for j in ground_joints if j.id != j_in.id)

    crank_id = j_in.child if j_in.parent == ground else j_in.parent
    rocker_id = j_out.child if j_out.parent == ground else j_out.parent

    coupling = [j for j in revs if j.id not in (j_in.id, j_out.id)]
    if len(coupling) != 2:
        return None
    crank_side = next((j for j in coupling
                       if crank_id in (j.parent, j.child)), None)
    rocker_side = next((j for j in coupling
                        if rocker_id in (j.parent, j.child)), None)
    if crank_side is None or rocker_side is None:
        return None
    coupler_id_a = (crank_side.child if crank_side.parent == crank_id
                    else crank_side.parent)
    coupler_id_b = (rocker_side.child if rocker_side.parent == rocker_id
                    else rocker_side.parent)
    if coupler_id_a != coupler_id_b:
        return None
    coupler_id = coupler_id_a

    A = _world_anchor(j_in)
    D = _world_anchor(j_out)
    l_ground = float(np.linalg.norm(D - A))

    BC_anchor = _world_anchor(crank_side)
    CD_anchor = _world_anchor(rocker_side)
    l_crank = float(np.linalg.norm(BC_anchor - A))
    l_coupler = float(np.linalg.norm(CD_anchor - BC_anchor))
    l_rocker = float(np.linalg.norm(CD_anchor - D))

    coupler_pt_port = ir.ports.get("coupler_point")

    thetas = np.linspace(0.0, 2.0 * np.pi, n_samples, endpoint=False)
    out_bc: list[np.ndarray] = []
    out_cp: list[np.ndarray] = []
    out_cd: list[np.ndarray] = []
    input_angles: list[float] = []
    output_angles: list[float] = []
    invalid_count = 0

    nan2 = np.array([np.nan, np.nan])
    initial_theta = float(np.arctan2(BC_anchor[1] - A[1],
                                     BC_anchor[0] - A[0]))

    prev_C: np.ndarray | None = None
    for dtheta in thetas:
        theta_2 = initial_theta + dtheta
        B = A + l_crank * np.array([np.cos(theta_2), np.sin(theta_2)])
        C = _intersect_circles(B, D, l_coupler, l_rocker, prev=prev_C)
        if C is None:
            if strict_geometry:
                return None
            invalid_count += 1
            out_bc.append(B.copy())
            out_cd.append(nan2.copy())
            out_cp.append(nan2.copy())
            input_angles.append(theta_2)
            output_angles.append(float("nan"))
            continue
        prev_C = C
        out_bc.append(B.copy())
        out_cd.append(C.copy())
        input_angles.append(theta_2)
        output_angles.append(float(np.arctan2(C[1] - D[1], C[0] - D[0])))
        if coupler_pt_port is not None:
            v = C - B
            n = float(np.linalg.norm(v))
            if n < 1e-12:
                P = B
            else:
                ex = v / n
                ey = np.array([-ex[1], ex[0]])
                px, py = coupler_pt_port.pose_local_mm[:2]
                P = B + px * ex + py * ey
            out_cp.append(P.copy())

    traces: dict[str, np.ndarray] = {
        "__crank_coupler": np.asarray(out_bc),
        "__coupler_rocker": np.asarray(out_cd),
    }
    if coupler_pt_port is not None:
        traces["coupler_point"] = np.asarray(out_cp)

    output_port = ir.ports.get("output_port")
    if output_port is not None and output_port.kind == "revolute_joint":
        traces["output_port"] = np.asarray(out_cd)

    input_arr = unwrap_angles(np.asarray(input_angles, dtype=float))
    output_arr = unwrap_angles(np.asarray(output_angles, dtype=float))
    time_s = np.linspace(
        0.0, 2.0 * np.pi, n_samples, endpoint=False, dtype=float)

    joint_positions: dict[str, np.ndarray] = {
        "input_port": input_arr,
        "output_port": output_arr,
        j_in.id: input_arr,
        j_out.id: output_arr,
    }
    joint_velocities: dict[str, np.ndarray] = {
        "input_port": angular_velocity_finite_diff(input_arr, time_s),
        "output_port": angular_velocity_finite_diff(output_arr, time_s),
    }
    joint_velocities[j_in.id] = joint_velocities["input_port"]
    joint_velocities[j_out.id] = joint_velocities["output_port"]

    # Estimate scalar ratio over valid samples (median is robust to
    # near-zero-velocity outliers near hard-point reversals).
    v_in = joint_velocities["input_port"]
    v_out = joint_velocities["output_port"]
    mask = np.isfinite(v_in) & np.isfinite(v_out) & (np.abs(v_in) > 1e-6)
    ratio_estimate = float(np.median(v_out[mask] / v_in[mask])) if (
        mask.sum() > 0) else float("nan")

    return {
        "port_traces": traces,
        "joint_positions": joint_positions,
        "joint_velocities": joint_velocities,
        "time_s": time_s,
        "topology": "fourbar",
        "link_lengths_mm": {
            "ground": l_ground,
            "crank": l_crank,
            "coupler": l_coupler,
            "rocker": l_rocker,
        },
        "invalid_samples": invalid_count,
        "ratio_estimate": ratio_estimate,
        "coupler_id": coupler_id,
        "rocker_id": rocker_id,
    }


def _solve_slider_crank(
    ir: DesignIR,
    n_samples: int,
    *,
    strict_geometry: bool,
) -> dict[str, Any] | None:
    """Minimal slider-crank forward kinematics.

    Topology: 4 parts (one fixed ground), 3 revolutes + 1 prismatic
    joint. The driven input is the revolute joint named by
    ``input_port`` (kind="revolute_joint"). The slider position
    becomes the ``output_port`` trace.

    Geometry conventions:
      * Slider runs along the prismatic joint's ``axis_world``
        (projected to xy), passing through its anchor.
      * Crank length = |A − B_anchor|, where A is the input revolute's
        anchor and B_anchor is the crank-coupler revolute's anchor.
      * Coupler length = |B_anchor − C_anchor|, where C_anchor is the
        coupler-slider revolute's anchor.
    """
    if len(ir.parts) != 4:
        return None
    revs = [j for j in ir.joints if j.type == "revolute"]
    prisms = [j for j in ir.joints if j.type == "prismatic"]
    if len(revs) != 3 or len(prisms) != 1:
        return None
    fixed = [p for p in ir.parts if p.fixed]
    if len(fixed) != 1:
        return None
    ground = fixed[0].id
    j_pr = prisms[0]
    if ground not in (j_pr.parent, j_pr.child):
        return None

    input_port = ir.ports.get("input_port")
    if input_port is None or input_port.kind != "revolute_joint":
        return None
    j_in = next((j for j in revs if j.id == input_port.part), None)
    if j_in is None or ground not in (j_in.parent, j_in.child):
        return None

    # The remaining two revolutes are crank-coupler and coupler-slider.
    other_revs = [j for j in revs if j.id != j_in.id]
    if len(other_revs) != 2:
        return None
    crank_id = j_in.child if j_in.parent == ground else j_in.parent
    crank_side = next((j for j in other_revs
                       if crank_id in (j.parent, j.child)), None)
    if crank_side is None:
        return None
    coupler_id = (crank_side.child if crank_side.parent == crank_id
                  else crank_side.parent)
    slider_side = next((j for j in other_revs
                        if j.id != crank_side.id
                        and coupler_id in (j.parent, j.child)), None)
    if slider_side is None:
        return None
    slider_id = (slider_side.child if slider_side.parent == coupler_id
                 else slider_side.parent)
    # Slider must be one of the prismatic joint's endpoints.
    if slider_id not in (j_pr.parent, j_pr.child):
        return None

    A = _world_anchor(j_in)
    B_anchor = _world_anchor(crank_side)
    C_anchor = _world_anchor(slider_side)
    l_crank = float(np.linalg.norm(B_anchor - A))
    l_coupler = float(np.linalg.norm(C_anchor - B_anchor))
    # Slider line: through prismatic anchor, along axis xy.
    pr_anchor = _world_anchor(j_pr)
    axis = j_pr.axis_world or (1.0, 0.0, 0.0)
    ax = np.array([axis[0], axis[1]], dtype=float)
    n_ax = float(np.linalg.norm(ax))
    if n_ax < 1e-9:
        return None
    ax = ax / n_ax
    perp = np.array([-ax[1], ax[0]])

    initial_theta = float(np.arctan2(B_anchor[1] - A[1],
                                     B_anchor[0] - A[0]))

    thetas = np.linspace(0.0, 2.0 * np.pi, n_samples, endpoint=False)
    slider_positions: list[float] = []
    input_angles: list[float] = []
    invalid_count = 0
    prev_branch = +1.0
    for dtheta in thetas:
        theta_2 = initial_theta + dtheta
        B = A + l_crank * np.array([np.cos(theta_2), np.sin(theta_2)])
        # Project B onto slider line.
        rel = B - pr_anchor
        along = float(np.dot(rel, ax))
        offset = float(np.dot(rel, perp))
        # We need a point C = pr_anchor + s*ax such that |B-C| = l_coupler.
        # That means (along - s)^2 + offset^2 = l_coupler^2.
        disc = l_coupler * l_coupler - offset * offset
        if disc < 0.0:
            if strict_geometry:
                return None
            invalid_count += 1
            slider_positions.append(float("nan"))
            input_angles.append(theta_2)
            continue
        root = float(np.sqrt(disc))
        s_plus = along + root
        s_minus = along - root
        # Branch continuity: maintain previous side.
        if abs(s_plus - (slider_positions[-1] if slider_positions
                          and np.isfinite(slider_positions[-1])
                          else s_plus)) <= abs(
            s_minus - (slider_positions[-1] if slider_positions
                         and np.isfinite(slider_positions[-1])
                         else s_plus)):
            s = s_plus if prev_branch >= 0 else s_minus
            prev_branch = +1.0
        else:
            s = s_minus
            prev_branch = -1.0
        slider_positions.append(s)
        input_angles.append(theta_2)

    input_arr = unwrap_angles(np.asarray(input_angles, dtype=float))
    slider_arr = np.asarray(slider_positions, dtype=float)
    time_s = np.linspace(0.0, 2.0 * np.pi, n_samples,
                         endpoint=False, dtype=float)

    # World-space slider trace.
    cp_trace = np.stack([
        pr_anchor[0] + slider_arr * ax[0],
        pr_anchor[1] + slider_arr * ax[1],
    ], axis=-1)

    traces: dict[str, np.ndarray] = {"output_port": cp_trace}
    joint_positions: dict[str, np.ndarray] = {
        "input_port": input_arr,
        "output_port": slider_arr,
        j_in.id: input_arr,
        j_pr.id: slider_arr,
    }
    joint_velocities: dict[str, np.ndarray] = {
        "input_port": np.gradient(input_arr, time_s, edge_order=1),
        "output_port": np.gradient(slider_arr, time_s, edge_order=1),
    }
    return {
        "port_traces": traces,
        "joint_positions": joint_positions,
        "joint_velocities": joint_velocities,
        "time_s": time_s,
        "topology": "slider_crank",
        "link_lengths_mm": {
            "crank": l_crank,
            "coupler": l_coupler,
        },
        "invalid_samples": invalid_count,
        "ratio_estimate": float("nan"),
    }


def _float_param(ir: DesignIR, *keys: str) -> float | None:
    for key in keys:
        if key not in ir.params:
            continue
        try:
            value = float(ir.params[key])
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            return value
    return None


def _solve_lead_screw(
    ir: DesignIR,
    n_samples: int,
    *,
    strict_geometry: bool,
) -> dict[str, Any] | None:
    """Lead-screw kinematics from one revolute input to one prismatic output."""
    del strict_geometry
    if len(ir.parts) != 3:
        return None
    revs = [j for j in ir.joints if j.type == "revolute"]
    prisms = [j for j in ir.joints if j.type == "prismatic"]
    if len(revs) != 1 or len(prisms) != 1:
        return None
    fixed = [p for p in ir.parts if p.fixed]
    if len(fixed) != 1:
        return None
    ground = fixed[0].id
    j_in = revs[0]
    j_out = prisms[0]
    if ground not in (j_in.parent, j_in.child):
        return None
    if ground not in (j_out.parent, j_out.child):
        return None
    input_port = ir.ports.get("input_port")
    output_port = ir.ports.get("output_port")
    if input_port is None or input_port.kind != "revolute_joint":
        return None
    if output_port is None or output_port.kind != "prismatic_joint":
        return None
    if input_port.part != j_in.id or output_port.part != j_out.id:
        return None
    lead_mm = _float_param(
        ir,
        "lead_mm",
        "lead_mm_per_rev",
        "declared_travel_per_rev_mm",
    )
    if lead_mm is None or abs(lead_mm) < 1e-12:
        return None

    axis = j_out.axis_world or (1.0, 0.0, 0.0)
    ax = np.array([axis[0], axis[1]], dtype=float)
    n_ax = float(np.linalg.norm(ax))
    if n_ax < 1e-9:
        return None
    ax = ax / n_ax
    pr_anchor = _world_anchor(j_out)

    time_s = np.linspace(0.0, 2.0 * np.pi, n_samples,
                         endpoint=False, dtype=float)
    input_arr = time_s.copy()
    travel_per_rad = float(lead_mm / (2.0 * np.pi))
    output_arr = input_arr * travel_per_rad
    output_trace = np.stack([
        pr_anchor[0] + output_arr * ax[0],
        pr_anchor[1] + output_arr * ax[1],
    ], axis=-1)

    return {
        "port_traces": {"output_port": output_trace},
        "joint_positions": {
            "input_port": input_arr,
            "output_port": output_arr,
            j_in.id: input_arr,
            j_out.id: output_arr,
        },
        "joint_velocities": {
            "input_port": np.gradient(input_arr, time_s, edge_order=1),
            "output_port": np.gradient(output_arr, time_s, edge_order=1),
        },
        "time_s": time_s,
        "topology": "lead_screw",
        "link_lengths_mm": {"lead_per_rev": float(lead_mm)},
        "invalid_samples": 0,
        "ratio_estimate": travel_per_rad,
    }


@register_adapter
class PlanarKinematics(SimAdapter):
    type_name = "planar_kinematics"
    capabilities_provided = frozenset({
        Capability.PLANAR_KINEMATICS,
        Capability.PATH_TRACE,
        Capability.DOF_DETECTION,
        Capability.POSE_TRACES,
    })
    cost_tier = 0

    def run(
        self,
        ir: DesignIR,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        n_samples = int(config.get("samples", 360))
        topology = config.get("topology", "auto")
        strict_geometry = bool(config.get("strict_geometry", False))

        result: dict[str, Any] = {
            "port_traces": {},
            "joint_positions": {},
            "joint_velocities": {},
            "time_s": np.zeros(0, dtype=float),
            "adapter": self.type_name,
            "scalar_metrics": {"samples": float(n_samples)},
            "metadata": {
                "adapter": self.type_name,
                "samples": n_samples,
                "topology_requested": str(topology),
            },
        }
        candidates = []
        if topology in ("auto", "fourbar"):
            candidates.append(_solve_fourbar)
        if topology in ("auto", "slider_crank"):
            candidates.append(_solve_slider_crank)
        if topology in ("auto", "lead_screw"):
            candidates.append(_solve_lead_screw)
        for solver in candidates:
            solved = solver(ir, n_samples, strict_geometry=strict_geometry)
            if solved is not None:
                result["port_traces"] = solved["port_traces"]
                result["joint_positions"] = solved["joint_positions"]
                result["joint_velocities"] = solved["joint_velocities"]
                result["time_s"] = solved["time_s"]
                result["metadata"]["topology"] = solved["topology"]
                result["metadata"]["invalid_samples"] = solved.get(
                    "invalid_samples", 0)
                ll = solved["link_lengths_mm"]
                result["scalar_metrics"].update({
                    f"link_length_{k}_mm": float(v) for k, v in ll.items()
                })
                ratio = solved.get("ratio_estimate")
                if ratio is not None:
                    result["scalar_metrics"]["ratio_estimate"] = float(
                        ratio)
                result["scalar_metrics"]["invalid_samples"] = float(
                    solved.get("invalid_samples", 0))
                return result
        # No matching topology — adapter returns empty traces; the
        # downstream probes surface SIMULATOR_DIVERGENCE.
        result["unsolved"] = True
        result["metadata"]["topology"] = "unsolved"
        return result
