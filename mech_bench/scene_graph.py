"""Generic SceneGraph intermediate representation.

A SceneGraph is the simulator-facing projection of a DesignIR plus the
relevant task / eval-config metadata. It is mechanism-agnostic — the
runtime knows about bodies, joints, drives, loads, contact pairs,
collision filters, materials, initial poses, and named ports, and
nothing about cycloidal or four-bar.

The phys-sim port goal: keep the shape that real contact simulators
(Chrono, Bullet, MuJoCo) need, but strip every mechanism-specific
field. New simulators read this IR and write back the SimOutput dict
the existing probes consume.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mech_bench.feedback import Failure, FailureCode, Severity
from mech_bench.schema import DesignIR, EvalConfig, TaskSpec


# --------------------------------------------------------------------- #
# Dataclasses                                                           #
# --------------------------------------------------------------------- #


@dataclass
class SceneBody:
    id: str
    role: str = ""
    mass_kg: float = 0.0
    com_local_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    inertia_kg_m2: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ] = ((1e-6, 0, 0), (0, 1e-6, 0), (0, 0, 1e-6))
    fixed: bool = False
    geometry: dict[str, str] = field(default_factory=dict)
    material: str = ""
    initial_pose_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    initial_orientation_quat: tuple[float, float, float, float] = (
        1.0, 0.0, 0.0, 0.0,
    )
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class SceneJoint:
    id: str
    type: str
    parent: str
    child: str
    axis_world: tuple[float, float, float] | None = None
    anchor_world_mm: tuple[float, float, float] | None = None
    limits_rad: tuple[float, float] | None = None
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class SceneMotor:
    """A drive applied to a joint or port.

    ``mode`` is one of ``speed`` (rad/s for revolute, mm/s for prismatic),
    ``torque`` (Nm), or ``force`` (N). The actual time-profile is up to
    the adapter — the SceneGraph only encodes the steady-state target.
    """

    id: str
    joint_id: str
    mode: str = "speed"
    value: float = 0.0
    profile: str = "constant"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class SceneLoad:
    """An applied torque or force, typically on the output port.

    ``mode`` is ``torque`` (Nm) or ``force`` (N). For revolute joints
    the load is interpreted as torque about the joint axis.
    """

    id: str
    joint_id: str
    mode: str = "torque"
    value: float = 0.0
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class SceneContactPair:
    """A nominated contact-pair between two bodies.

    ``required`` flips the engagement probe from "monitor if present"
    to "this pair MUST carry RMS force ≥ min_force_N".
    """

    pair_id: str
    body_a: str
    body_b: str
    friction_mu: float = 0.2
    restitution: float = 0.0
    min_force_N: float = 0.0
    required: bool = False
    contact_method: str = "default"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScenePort:
    """Port metadata copied through from DesignIR.

    The simulator uses port frames for two things: knowing where to
    drive (input_port) and which body's velocity to report (output_port).

    A frame port targets a body; a revolute/prismatic port targets a
    joint. ``target_type`` disambiguates; ``target_id`` is the actual
    referenced id, and the legacy ``body_id`` and new ``joint_id``
    attributes are populated for backwards-compat consumers.
    """

    id: str
    body_id: str
    kind: str = "frame"
    pose_local_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    target_type: str = "body"
    target_id: str = ""
    joint_id: str | None = None


@dataclass
class SceneGraph:
    """Capability-tagged scene description handed to a SimAdapter.

    Mechanism-agnostic: every adapter consumes the same shape.
    """

    bodies: list[SceneBody] = field(default_factory=list)
    joints: list[SceneJoint] = field(default_factory=list)
    motors: list[SceneMotor] = field(default_factory=list)
    loads: list[SceneLoad] = field(default_factory=list)
    contact_pairs: list[SceneContactPair] = field(default_factory=list)
    collision_filters: list[tuple[str, str]] = field(default_factory=list)
    materials: dict[str, dict[str, float]] = field(default_factory=dict)
    initial_poses: dict[str, tuple[float, float, float]] = field(
        default_factory=dict)
    ports: dict[str, ScenePort] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def body_ids(self) -> set[str]:
        return {b.id for b in self.bodies}

    def joint_ids(self) -> set[str]:
        return {j.id for j in self.joints}


@dataclass
class SceneGraphBuildResult:
    """Result of mapping a DesignIR + task + eval_config to a scene.

    ``preflight_failures`` carries structural mismatches (missing
    bodies referenced by contact pairs, unknown joints under motors,
    etc.). Callers may treat any non-empty list as a hard-gate fail.
    """

    scene: SceneGraph
    preflight_failures: list[Failure] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(
            f.severity == Severity.CRITICAL for f in self.preflight_failures
        )


# --------------------------------------------------------------------- #
# Builder                                                               #
# --------------------------------------------------------------------- #


def _extract_contact_pairs_from_eval(
    cfg: EvalConfig,
) -> list[dict[str, Any]]:
    """Collect contact-pair specs declared inside eval_config probes.

    Recognized probe types:
      * ``contact_engagement`` — ``required_pairs`` (list of
        ``"a:b"`` strings) and optional ``min_rms_force_N``.
      * ``swept_collision`` — ``allowed_pairs`` (ignored), ``ignored_pairs``
        (ignored). The pair list itself is derived from the design,
        so this probe doesn't contribute new entries.
      * ``torque_load_trial`` — defines a drive + load on input/output
        ports, not contact pairs; consumed below.
    """
    pairs: list[dict[str, Any]] = []
    for spec in cfg.probes:
        if spec.type == "contact_engagement":
            min_force = float(spec.config.get("min_rms_force_N", 0.5))
            for raw in spec.config.get("required_pairs", []) or []:
                a, _, b = str(raw).partition(":")
                if not b:
                    continue
                pairs.append({
                    "pair_id": f"{a}:{b}",
                    "body_a": a,
                    "body_b": b,
                    "min_force_N": min_force,
                    "required": True,
                    "source_probe": spec.id,
                })
    return pairs


def _extract_motors_loads_from_eval(
    cfg: EvalConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    motors: list[dict[str, Any]] = []
    loads: list[dict[str, Any]] = []
    for spec in cfg.probes:
        if spec.type == "torque_load_trial":
            inp = str(spec.config.get("input_port", "input_port"))
            outp = str(spec.config.get("output_port", "output_port"))
            speed = float(spec.config.get("input_speed_rad_s", 1.0))
            load = float(spec.config.get("output_load_Nm", 0.0))
            motors.append({
                "id": f"drive_{spec.id}",
                "joint_id": inp,
                "mode": "speed",
                "value": speed,
                "source_probe": spec.id,
            })
            if load > 0.0:
                loads.append({
                    "id": f"load_{spec.id}",
                    "joint_id": outp,
                    "mode": "torque",
                    "value": load,
                    "source_probe": spec.id,
                })
    return motors, loads


def build_scene_graph_from_design_ir(
    ir: DesignIR,
    task: TaskSpec,
    cfg: EvalConfig,
) -> SceneGraphBuildResult:
    """Map a DesignIR + TaskSpec + EvalConfig to a SceneGraph.

    The mapping is intentionally narrow:

    1. Every Part becomes a SceneBody (mass / inertia / fixed flag).
    2. Every Joint becomes a SceneJoint.
    3. ``required_ports`` are validated against the IR's port table.
    4. ``contact_engagement.required_pairs`` becomes SceneContactPair
       entries; missing bodies emit CRITICAL preflight failures.
    5. ``torque_load_trial`` declarations become Motor / Load entries.
    """
    bodies: list[SceneBody] = []
    initial_poses: dict[str, tuple[float, float, float]] = {}
    materials: dict[str, dict[str, float]] = {}
    for p in ir.parts:
        material = str(p.params.get("material", "")) if p.params else ""
        bodies.append(SceneBody(
            id=p.id,
            role=p.role,
            mass_kg=float(p.mass_kg),
            com_local_mm=tuple(p.com_local_mm),
            inertia_kg_m2=p.inertia_kg_m2,
            fixed=bool(p.fixed),
            geometry=dict(p.geometry or {}),
            material=material,
            params=dict(p.params or {}),
        ))
        initial_poses[p.id] = tuple(p.com_local_mm)
        if material and material not in materials:
            materials[material] = {}

    joints: list[SceneJoint] = []
    for j in ir.joints:
        joints.append(SceneJoint(
            id=j.id,
            type=j.type,
            parent=j.parent,
            child=j.child,
            axis_world=j.axis_world,
            anchor_world_mm=j.anchor_world_mm,
            limits_rad=j.limits_rad,
            params=dict(j.params or {}),
        ))

    ports: dict[str, ScenePort] = {}
    joint_id_set = {j.id for j in ir.joints}
    for pid, port in ir.ports.items():
        is_joint = port.kind in ("revolute_joint", "prismatic_joint")
        if is_joint:
            ports[pid] = ScenePort(
                id=port.id,
                body_id="",
                kind=port.kind,
                pose_local_mm=tuple(port.pose_local_mm),
                target_type="joint",
                target_id=port.part,
                joint_id=port.part,
            )
        else:
            ports[pid] = ScenePort(
                id=port.id,
                body_id=port.part,
                kind=port.kind,
                pose_local_mm=tuple(port.pose_local_mm),
                target_type="body",
                target_id=port.part,
                joint_id=None,
            )

    preflight: list[Failure] = []
    body_ids = {b.id for b in bodies}
    joint_ids = {j.id for j in joints}

    # Required-ports preflight.
    for required in task.required_ports:
        if required not in ports:
            preflight.append(Failure(
                code=FailureCode.MISSING_PORT,
                severity=Severity.CRITICAL,
                message=(f"SceneGraph: required port {required!r} is "
                         f"absent from the DesignIR."),
                where="scene_graph.ports",
            ))

    # Joint-target preflight: revolute/prismatic ports must reference
    # an existing joint id, not a phantom one.
    for pid, port in ports.items():
        if port.target_type != "joint":
            continue
        if (port.joint_id or port.target_id) not in joint_id_set:
            preflight.append(Failure(
                code=FailureCode.MISSING_PORT,
                severity=Severity.CRITICAL,
                message=(
                    f"SceneGraph: port {pid!r} ({port.kind}) targets "
                    f"joint {(port.joint_id or port.target_id)!r} "
                    f"which is not in the DesignIR."
                ),
                where=f"scene_graph.ports.{pid}",
            ))

    # Contact-pair preflight.
    contact_pairs: list[SceneContactPair] = []
    for entry in _extract_contact_pairs_from_eval(cfg):
        missing = [b for b in (entry["body_a"], entry["body_b"])
                   if b not in body_ids]
        if missing:
            preflight.append(Failure(
                code=FailureCode.WRONG_TOPOLOGY,
                severity=Severity.CRITICAL,
                message=(
                    f"SceneGraph: contact pair "
                    f"{entry['pair_id']!r} references bodies that are "
                    f"not in the design: {missing!r}."
                ),
                where="scene_graph.contact_pairs",
                extra={"source_probe": entry.get("source_probe", "")},
            ))
            continue
        contact_pairs.append(SceneContactPair(
            pair_id=str(entry["pair_id"]),
            body_a=str(entry["body_a"]),
            body_b=str(entry["body_b"]),
            min_force_N=float(entry.get("min_force_N", 0.0)),
            required=bool(entry.get("required", False)),
            params={"source_probe": entry.get("source_probe", "")},
        ))

    motors: list[SceneMotor] = []
    loads: list[SceneLoad] = []
    motor_specs, load_specs = _extract_motors_loads_from_eval(cfg)
    for m in motor_specs:
        # Motors reference ports; resolve port → joint when possible.
        ref = m["joint_id"]
        if ref in ports:
            port_obj = ports[ref]
            if port_obj.target_type == "joint":
                ref_joint = port_obj.joint_id or port_obj.target_id
            else:
                ref_joint = port_obj.body_id
        else:
            ref_joint = ref
        if ref_joint not in joint_ids and ref not in ports:
            preflight.append(Failure(
                code=FailureCode.MISSING_PORT,
                severity=Severity.MAJOR,
                message=(f"SceneGraph: motor on {ref!r} but neither a "
                         f"matching port nor joint exists."),
                where="scene_graph.motors",
            ))
            continue
        motors.append(SceneMotor(
            id=str(m["id"]),
            joint_id=str(ref_joint),
            mode=str(m.get("mode", "speed")),
            value=float(m.get("value", 0.0)),
            params={"source_probe": m.get("source_probe", "")},
        ))
    for ld in load_specs:
        ref = ld["joint_id"]
        if ref in ports:
            port_obj = ports[ref]
            if port_obj.target_type == "joint":
                ref_joint = port_obj.joint_id or port_obj.target_id
            else:
                ref_joint = port_obj.body_id
        else:
            ref_joint = ref
        if ref_joint not in joint_ids and ref not in ports:
            preflight.append(Failure(
                code=FailureCode.MISSING_PORT,
                severity=Severity.MAJOR,
                message=(f"SceneGraph: load on {ref!r} but neither a "
                         f"matching port nor joint exists."),
                where="scene_graph.loads",
            ))
            continue
        loads.append(SceneLoad(
            id=str(ld["id"]),
            joint_id=str(ref_joint),
            mode=str(ld.get("mode", "torque")),
            value=float(ld.get("value", 0.0)),
            params={"source_probe": ld.get("source_probe", "")},
        ))

    metadata: dict[str, Any] = {
        "task_id": task.id,
        "task_family": task.family,
        "schema_version": ir.schema_version,
        "n_bodies": len(bodies),
        "n_joints": len(joints),
        "n_contact_pairs": len(contact_pairs),
        "n_motors": len(motors),
        "n_loads": len(loads),
    }

    scene = SceneGraph(
        bodies=bodies,
        joints=joints,
        motors=motors,
        loads=loads,
        contact_pairs=contact_pairs,
        collision_filters=[],
        materials=materials,
        initial_poses=initial_poses,
        ports=ports,
        metadata=metadata,
    )

    notes: list[str] = []
    if not motors:
        notes.append("no motors declared (no torque_load_trial probe)")
    if not contact_pairs:
        notes.append("no contact pairs declared")

    return SceneGraphBuildResult(
        scene=scene,
        preflight_failures=preflight,
        notes=notes,
    )
