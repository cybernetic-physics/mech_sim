"""Simulator adapters, capability-tagged.

Each adapter advertises a Capability set. The evaluator picks the
cheapest adapter whose advertised capabilities cover the union of
the active probes' requirements. Adapters never know which probe
will consume their output — they emit a generic outputs dict keyed
on capability category (port_traces, contact_forces, …).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

from mech_bench.probes import Capability
from mech_bench.schema import DesignIR


@dataclass
class SimOutput:
    """Canonical, trace-compatible output from a SimAdapter.run().

    Adapters may return either this dataclass *or* a plain dict with
    the same top-level keys. The evaluator normalizes both into a
    :class:`mech_bench.traces.TraceData` for evidence and into a dict
    for probe consumption — see ``to_dict()``.
    """

    port_traces: dict[str, np.ndarray] = field(default_factory=dict)
    port_velocities: dict[str, np.ndarray] = field(default_factory=dict)
    joint_positions: dict[str, np.ndarray] = field(default_factory=dict)
    joint_velocities: dict[str, np.ndarray] = field(default_factory=dict)
    body_poses: dict[str, np.ndarray] = field(default_factory=dict)
    body_twists: dict[str, np.ndarray] = field(default_factory=dict)
    contact_forces: dict[str, np.ndarray] = field(default_factory=dict)
    penetration: dict[str, np.ndarray] = field(default_factory=dict)
    time_s: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    scalar_metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = dict(self.extras)
        d.update({
            "port_traces": self.port_traces,
            "port_velocities": self.port_velocities,
            "joint_positions": self.joint_positions,
            "joint_velocities": self.joint_velocities,
            "body_poses": self.body_poses,
            "body_twists": self.body_twists,
            "contact_forces": self.contact_forces,
            "penetration": self.penetration,
            "time_s": self.time_s,
            "scalar_metrics": self.scalar_metrics,
            "metadata": self.metadata,
        })
        return d


def normalize_sim_output(out: Any) -> dict[str, Any]:
    """Coerce a SimOutput or dict to a dict shaped for probes.

    Probes only care about a handful of keys (``port_traces``, etc.).
    This helper preserves backward compatibility with adapters that
    historically returned a plain dict.
    """
    if isinstance(out, SimOutput):
        return out.to_dict()
    if isinstance(out, dict):
        return out
    return {}


class SimAdapter(ABC):
    """Subclasses register via `register_adapter`."""

    type_name: ClassVar[str] = ""
    capabilities_provided: ClassVar[frozenset[Capability]] = frozenset()
    cost_tier: ClassVar[int] = 0  # lower = cheaper; dispatcher prefers low

    @abstractmethod
    def run(
        self,
        ir: DesignIR,
        config: dict[str, Any],
    ) -> SimOutput | dict[str, Any]:
        """Return a :class:`SimOutput` or an equivalently-shaped dict.

        Top-level keys:

        - ``port_traces``: dict[port_id, ndarray (N, 2 or 3)] in mm
        - ``port_velocities``: dict[port_id, ndarray]
        - ``joint_positions`` / ``joint_velocities``: dict[joint_id, ndarray]
        - ``body_poses`` / ``body_twists``: dict[body_id, ndarray]
        - ``contact_forces`` / ``penetration``: dict[pair_key, ndarray]
        - ``time_s``: ndarray (N,) — shared time axis
        - ``scalar_metrics``: dict[str, float]
        - ``metadata``: dict[str, str | float | int | bool]

        Probes consume only the keys they need; the evaluator
        normalizes everything else into a TraceData for evidence.
        """


_REGISTRY: dict[str, type[SimAdapter]] = {}


def register_adapter(cls: type[SimAdapter]) -> type[SimAdapter]:
    if not cls.type_name:
        raise ValueError(f"{cls.__name__} has no type_name")
    if cls.type_name in _REGISTRY:
        raise ValueError(f"Adapter {cls.type_name!r} already registered")
    _REGISTRY[cls.type_name] = cls
    return cls


def all_adapters() -> list[type[SimAdapter]]:
    return list(_REGISTRY.values())


def get_adapter(type_name: str) -> SimAdapter:
    return _REGISTRY[type_name]()


# Trigger registration.
from mech_bench.adapters import planar_kinematics  # noqa: E402, F401
