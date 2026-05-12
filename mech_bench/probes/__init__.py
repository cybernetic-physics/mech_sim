"""Probes are the verifiable unit of evaluation.

A probe is a configurable check that, given a DesignIR and the
outputs of a simulator adapter, emits a `ProbeResult` (passed,
metrics, failures). New mechanism families add tasks (which select
existing probes via config) or new probe *types* (which are reusable
across mechanisms). They do not add bespoke validators.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar

from mech_bench.schema import DesignIR, ProbeResult


class Capability(str, Enum):
    """Tags a probe requires from a simulator adapter, and that an
    adapter advertises. The evaluator dispatches based on these.
    """

    # No simulation needed — pure topology / metadata
    NONE = "none"

    # Kinematics
    PLANAR_KINEMATICS = "planar_kinematics"
    SPATIAL_KINEMATICS = "spatial_kinematics"
    PATH_TRACE = "path_trace"
    DOF_DETECTION = "dof_detection"

    # Geometry probes
    MESH_OVERLAP = "mesh_overlap"
    MESH = "mesh"

    # Dynamics
    RIGID_BODY_DYNAMICS = "rigid_body_dynamics"
    CONTACT_FORCES = "contact_forces"
    JOINT_CONSTRAINTS = "joint_constraints"
    MOTOR_DRIVES = "motor_drives"
    LOAD_TORQUES = "load_torques"
    POSE_TRACES = "pose_traces"

    # Structural
    FEA_STATIC = "fea_static"
    SAFETY_FACTOR = "safety_factor"


class Probe(ABC):
    """Subclasses register themselves via `register_probe` below.

    The class-level `type_name` is the key in eval_config.toml's
    `type=` field. `capabilities_required` declares what the
    dispatcher needs from a simulator.
    """

    type_name: ClassVar[str] = ""
    capabilities_required: ClassVar[frozenset[Capability]] = frozenset()

    @abstractmethod
    def run(
        self,
        ir: DesignIR,
        sim_outputs: dict[str, Any],
        config: dict[str, Any],
    ) -> ProbeResult:
        ...


_REGISTRY: dict[str, type[Probe]] = {}


def register_probe(cls: type[Probe]) -> type[Probe]:
    """Decorator: register a Probe subclass under its `type_name`."""
    if not cls.type_name:
        raise ValueError(f"Probe {cls.__name__} has no type_name")
    if cls.type_name in _REGISTRY:
        raise ValueError(f"Probe type {cls.type_name!r} already registered")
    _REGISTRY[cls.type_name] = cls
    return cls


def get_probe(type_name: str) -> Probe:
    if type_name not in _REGISTRY:
        raise KeyError(
            f"Unknown probe type {type_name!r}. "
            f"Known: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[type_name]()


def known_probe_types() -> list[str]:
    return sorted(_REGISTRY)


# Trigger registration of built-in probes.
from mech_bench.probes import dof_grubler  # noqa: E402, F401
from mech_bench.probes import path_trace_chamfer  # noqa: E402, F401
from mech_bench.probes import required_ports  # noqa: E402, F401
from mech_bench.probes import port_velocity_ratio  # noqa: E402, F401
from mech_bench.probes import swept_collision  # noqa: E402, F401
from mech_bench.probes import contact_engagement  # noqa: E402, F401
from mech_bench.probes import lockup  # noqa: E402, F401
from mech_bench.probes import torque_load_trial  # noqa: E402, F401
from mech_bench.probes import printability_dfam  # noqa: E402, F401
from mech_bench.probes import safety_factor  # noqa: E402, F401
from mech_bench.probes import analytic_param_check  # noqa: E402, F401
