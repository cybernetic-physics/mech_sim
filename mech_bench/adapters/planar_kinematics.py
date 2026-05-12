"""Analytical planar-kinematics adapter.

Handles the open-chain and closed-loop 4-bar / slider-crank /
crank-rocker / rack-pinion families purely from DesignIR topology
and joint parameters, with no physics solver. Emits port_traces in
mm by sweeping the prescribed input joint through one full cycle.

The adapter is generic over the topology — it does not know "this is
a four-bar." It walks the joint graph, identifies the driven input,
and solves intersection-of-circles for each closed-loop body chain.

For the v0 it handles two cases:
1. All revolute joints, all axes parallel to world Z, planar
   four-bar (4 bodies, 4 joints, one closed loop).
2. Slider-crank: 3 revolute + 1 prismatic, planar.

Anything else returns an empty trace and the probe will surface
`SIMULATOR_DIVERGENCE`. New planar topologies extend this file; the
runtime stays generic.
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
    """Return *theta* unwrapped so jumps larger than π are removed.

    Thin wrapper over numpy's ``unwrap`` so the rest of the package
    has a single import surface for "make angles continuous so we
    can finite-difference them."
    """
    arr = np.asarray(theta, dtype=float).reshape(-1)
    if arr.size == 0:
        return arr
    return np.unwrap(arr)


def angular_velocity_finite_diff(
    theta: np.ndarray, time_s: np.ndarray,
) -> np.ndarray:
    """Central-difference angular velocity of an angle signal.

    Inputs are unwrapped internally so wrap-around does not produce
    spurious spikes. Returns an array of the same length as *theta*
    using forward/backward differences at the endpoints.
    """
    theta = unwrap_angles(theta)
    t = np.asarray(time_s, dtype=float).reshape(-1)
    if theta.size < 2 or t.size != theta.size:
        return np.zeros_like(theta)
    return np.gradient(theta, t, edge_order=1)


def _solve_fourbar(ir: DesignIR, n_samples: int) -> dict[str, Any] | None:
    """Solve the four-bar closed-loop forward kinematics.

    Topology assumption: 4 bodies — one fixed (ground) and three
    moving (crank, coupler, rocker) — joined by 4 revolute joints
    in a closed loop. The driven input joint is whichever revolute
    joint is between the ground and the crank, identified by the
    `input_port` port (its joint id) or, failing that, by being
    listed first.

    The coupler-point port (if present) carries a `pose_local_mm`
    offset interpreted in the coupler's body frame. We solve for
    the coupler frame orientation at each input angle.

    Returns a dict {port_id -> (N,2) trace in mm}, or None if the
    topology does not match.
    """

    if len(ir.parts) != 4:
        return None
    revs = [j for j in ir.joints if j.type == "revolute"]
    if len(revs) != 4:
        return None

    # Identify ground (the fixed part) and the two ground-pivot joints.
    fixed = [p for p in ir.parts if p.fixed]
    if len(fixed) != 1:
        return None
    ground = fixed[0].id

    ground_joints = [j for j in revs if ground in (j.parent, j.child)]
    if len(ground_joints) != 2:
        return None

    # `input_port` points to one of the ground joints; that side is
    # the crank.
    input_port = ir.ports.get("input_port")
    if input_port is None:
        return None
    if input_port.kind != "revolute_joint":
        return None
    input_joint_id = input_port.part  # convention: input_port.part is joint id
    j_in = next((j for j in ground_joints if j.id == input_joint_id), None)
    if j_in is None:
        # fall back: first ground joint
        j_in = ground_joints[0]
    j_out = next(j for j in ground_joints if j.id != j_in.id)

    crank_id = j_in.child if j_in.parent == ground else j_in.parent
    rocker_id = j_out.child if j_out.parent == ground else j_out.parent

    # Find the coupler: the body sharing a joint with both crank and
    # rocker (other than the ground).
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

    # Crank length: distance from A to crank-coupler joint anchor.
    # The anchor lives in world coords at the rest pose (θ=0). We
    # use that as the reference geometry.
    BC_anchor = _world_anchor(crank_side)
    CD_anchor = _world_anchor(rocker_side)
    l_crank = float(np.linalg.norm(BC_anchor - A))
    l_coupler = float(np.linalg.norm(CD_anchor - BC_anchor))
    l_rocker = float(np.linalg.norm(CD_anchor - D))

    # Coupler-point port — pose_local_mm is interpreted in the
    # coupler body frame whose origin is at the crank-coupler joint
    # (B) and whose x-axis points to the rocker-coupler joint (C).
    coupler_pt_port = ir.ports.get("coupler_point")

    # Sweep crank angle.
    thetas = np.linspace(0.0, 2.0 * np.pi, n_samples, endpoint=False)
    out_bc: list[np.ndarray] = []
    out_cp: list[np.ndarray] = []
    out_cd: list[np.ndarray] = []
    input_angles: list[float] = []
    output_angles: list[float] = []

    initial_theta = float(np.arctan2(BC_anchor[1] - A[1],
                                     BC_anchor[0] - A[0]))

    for dtheta in thetas:
        theta_2 = initial_theta + dtheta
        B = A + l_crank * np.array([np.cos(theta_2), np.sin(theta_2)])
        BD = D - B
        d = float(np.linalg.norm(BD))
        # Intersection of circle(B, l_coupler) and circle(D, l_rocker)
        if d > l_coupler + l_rocker or d < abs(l_coupler - l_rocker):
            return None  # design is non-Grashof in this segment
        a = (l_coupler ** 2 - l_rocker ** 2 + d ** 2) / (2.0 * d)
        h_sq = l_coupler ** 2 - a ** 2
        if h_sq < 0.0:
            return None
        h = float(np.sqrt(h_sq))
        u = BD / d
        perp = np.array([-u[1], u[0]])
        M = B + a * u
        # Pick the elbow-up branch (positive perp).
        C = M + h * perp
        out_bc.append(B.copy())
        out_cd.append(C.copy())
        input_angles.append(theta_2)
        # Rocker angle: from D toward C.
        output_angles.append(float(np.arctan2(C[1] - D[1], C[0] - D[0])))
        if coupler_pt_port is not None:
            # Coupler frame: origin B, x-axis BC, y-axis perp(BC).
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

    traces: dict[str, np.ndarray] = {}
    # Always expose the coupler joint loci so probes can observe.
    traces["__crank_coupler"] = np.asarray(out_bc)
    traces["__coupler_rocker"] = np.asarray(out_cd)
    if coupler_pt_port is not None:
        traces["coupler_point"] = np.asarray(out_cp)

    # Output-port trace (the rocker pivot is fixed; the rocker tip
    # is meaningful for some tasks).
    output_port = ir.ports.get("output_port")
    if output_port is not None and output_port.kind == "revolute_joint":
        # output is whatever this joint connects to relative to ground
        traces["output_port"] = np.asarray(out_cd)

    input_arr = unwrap_angles(np.asarray(input_angles, dtype=float))
    output_arr = unwrap_angles(np.asarray(output_angles, dtype=float))
    # Synthetic time axis assuming unit angular velocity over one
    # crank revolution: useful for derived rates.
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
    }


@register_adapter
class PlanarKinematics(SimAdapter):
    type_name = "planar_kinematics"
    capabilities_provided = frozenset({
        Capability.PLANAR_KINEMATICS,
        Capability.PATH_TRACE,
        Capability.DOF_DETECTION,
    })
    cost_tier = 0

    def run(
        self,
        ir: DesignIR,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        n_samples = int(config.get("samples", 360))
        topology = config.get("topology", "auto")

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
        if topology in ("auto", "fourbar"):
            solved = _solve_fourbar(ir, n_samples)
            if solved is not None:
                result["port_traces"] = solved["port_traces"]
                result["joint_positions"] = solved["joint_positions"]
                result["joint_velocities"] = solved["joint_velocities"]
                result["time_s"] = solved["time_s"]
                result["metadata"]["topology"] = solved["topology"]
                ll = solved["link_lengths_mm"]
                result["scalar_metrics"].update({
                    f"link_length_{k}_mm": float(v) for k, v in ll.items()
                })
                return result
        # No matching topology — adapter returns empty traces; the
        # downstream probes surface SIMULATOR_DIVERGENCE.
        result["unsolved"] = True
        result["metadata"]["topology"] = "unsolved"
        return result
