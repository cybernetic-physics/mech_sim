"""Structured failure grammar.

Every probe failure is one of a closed set of codes. A generation
agent reading a report can pattern-match on `code` to decide how to
repair, without parsing prose. The text in `message` and
`public_hint` is for humans.

Ported in spirit from phys-sim/mech_harness/standards/sarif.py: the
SARIF rule registry maps roughly onto FailureCode here, but flattened
into a single enum that probes share.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Severity(str, Enum):
    INFO = "info"
    MINOR = "minor"
    MAJOR = "major"
    CRITICAL = "critical"


class FailureCode(str, Enum):
    # Artifact / submission integrity
    INVALID_ARTIFACT = "invalid_artifact"
    INVALID_MASS_PROPERTIES = "invalid_mass_properties"
    MISSING_PORT = "missing_port"
    SCHEMA_ERROR = "schema_error"

    # Kinematic topology
    WRONG_MOBILITY = "wrong_mobility"
    WRONG_TOPOLOGY = "wrong_topology"
    WRONG_RATIO = "wrong_ratio"

    # Motion / geometric correctness
    PATH_ERROR = "path_error"
    COLLISION = "collision"
    INSUFFICIENT_CLEARANCE = "insufficient_clearance"

    # Dynamic / contact
    MISSING_CONTACT = "missing_contact"
    LOCKUP = "lockup"
    EXCESSIVE_PENETRATION = "excessive_penetration"
    EXCESSIVE_TORQUE_RIPPLE = "excessive_torque_ripple"
    POWER_BALANCE_ERROR = "power_balance_error"

    # Structural / manufacturing
    INSUFFICIENT_SAFETY_FACTOR = "insufficient_safety_factor"
    UNPRINTABLE = "unprintable"

    # Pipeline-level
    SIMULATOR_DIVERGENCE = "simulator_divergence"
    CAPABILITY_UNAVAILABLE = "capability_unavailable"


@dataclass
class Failure:
    """A single structured failure emitted by a probe or pipeline stage.

    Fields mirror the structure recommended in the design notes: code,
    severity, machine-readable metric/observed/target, plus human
    hint and an optional pointer to a hidden trace for trusted-side
    inspection.
    """

    code: FailureCode
    severity: Severity
    message: str
    metric: str | None = None
    observed: float | None = None
    target: float | None = None
    where: str | None = None
    confidence: float = 1.0
    public_hint: str | None = None
    private_trace: str | None = None
    # ``extra`` is private by default — it never appears in public().
    # Anything that is safe to expose to the agent must go in
    # ``extra_public``.
    extra: dict = field(default_factory=dict)
    extra_public: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {
            "code": self.code.value,
            "severity": self.severity.value,
            "message": self.message,
            "confidence": self.confidence,
        }
        for k in ("metric", "observed", "target", "where",
                  "public_hint", "private_trace"):
            v = getattr(self, k)
            if v is not None:
                d[k] = v
        if self.extra:
            d["extra"] = self.extra
        if self.extra_public:
            d["extra_public"] = self.extra_public
        return d

    def public(self) -> dict:
        """Agent-visible projection.

        Strips ``private_trace`` and the private ``extra`` bag. Only
        ``extra_public`` survives.
        """
        d = {
            "code": self.code.value,
            "severity": self.severity.value,
            "message": self.message,
            "confidence": self.confidence,
        }
        for k in ("metric", "observed", "target", "where", "public_hint"):
            v = getattr(self, k)
            if v is not None:
                d[k] = v
        if self.extra_public:
            d["extra_public"] = self.extra_public
        return d
