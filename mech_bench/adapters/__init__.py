"""Simulator adapters, capability-tagged.

Each adapter advertises a Capability set. The evaluator picks the
cheapest adapter whose advertised capabilities cover the union of
the active probes' requirements. Adapters never know which probe
will consume their output — they emit a generic outputs dict keyed
on capability category (port_traces, contact_forces, …).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from mech_bench.probes import Capability
from mech_bench.schema import DesignIR


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
    ) -> dict[str, Any]:
        """Return a dict whose top-level keys are output categories:

        - `port_traces`: dict[port_id, ndarray (N, 2)] in mm
        - `port_velocities`: dict[port_id, ndarray (N, ...)]
        - `contact_forces`: dict[pair_key, ndarray]
        - `lockup`: bool
        - `n_contacts_max`: int
        - ...

        Probes consume only the keys they need.
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
