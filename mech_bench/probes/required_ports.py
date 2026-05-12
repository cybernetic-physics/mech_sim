"""Required-ports probe.

Pure-topology check that the submission exposes a set of named ports,
optionally with constraints on each port's `kind` and on whether the
referenced part is grounded (rigidly attached to the fixed body).

This probe is intentionally mechanism-agnostic: a four-bar task may
require ``input_port``/``output_port``/``coupler_point``, a planetary
gearbox may require ``carrier_port``/``ring_port``/``sun_port`` — both
share the same probe with different config.

Failure codes:
  * ``MISSING_PORT`` — a required port is absent.
  * ``WRONG_TOPOLOGY`` — port exists but has the wrong kind, or its
    grounded-attachment requirement is not satisfied.
"""

from __future__ import annotations

from typing import Any

from mech_bench.feedback import Failure, FailureCode, Severity
from mech_bench.probes import Capability, Probe, register_probe
from mech_bench.schema import DesignIR, ProbeResult


def _joint_lookup(ir: DesignIR) -> dict[str, Any]:
    return {j.id: j for j in ir.joints}


def _part_lookup(ir: DesignIR) -> dict[str, Any]:
    return {p.id: p for p in ir.parts}


def _port_is_grounded(ir: DesignIR, port_id: str) -> bool:
    """Return True if the port's referenced entity is rigidly attached
    to a fixed (ground) part.

    For frame ports we check the directly-referenced part. For joint
    ports we check both joint endpoints — being grounded means at
    least one endpoint is the ground body.
    """
    port = ir.ports.get(port_id)
    if port is None:
        return False
    parts = _part_lookup(ir)
    joints = _joint_lookup(ir)
    if port.kind == "frame":
        part = parts.get(port.part)
        return bool(part is not None and part.fixed)
    joint = joints.get(port.part)
    if joint is None:
        return False
    for end in (joint.parent, joint.child):
        part = parts.get(end)
        if part is not None and part.fixed:
            return True
    return False


@register_probe
class RequiredPorts(Probe):
    type_name = "required_ports"
    capabilities_required = frozenset({Capability.NONE})

    def run(
        self,
        ir: DesignIR,
        sim_outputs: dict[str, Any],
        config: dict[str, Any],
    ) -> ProbeResult:
        required = list(config.get("ports", []))
        require_grounded = set(config.get("require_grounded", []) or [])
        require_kinds = dict(config.get("require_kinds", {}) or {})

        failures: list[Failure] = []
        present_ct = 0
        kind_ok_ct = 0
        grounded_ok_ct = 0
        kind_checks = 0
        grounded_checks = 0

        for port_id in required:
            port = ir.ports.get(port_id)
            if port is None:
                failures.append(Failure(
                    code=FailureCode.MISSING_PORT,
                    severity=Severity.CRITICAL,
                    message=(f"Required port {port_id!r} is missing "
                             f"from the design."),
                    where=f"ports.{port_id}",
                    public_hint=(
                        "Add a port with this id to the DesignIR. The "
                        "task contract requires it."
                    ),
                ))
                continue
            present_ct += 1
            expected_kind = require_kinds.get(port_id)
            if expected_kind is not None:
                kind_checks += 1
                if port.kind != expected_kind:
                    failures.append(Failure(
                        code=FailureCode.WRONG_TOPOLOGY,
                        severity=Severity.CRITICAL,
                        message=(f"Port {port_id!r} has kind "
                                 f"{port.kind!r}, expected "
                                 f"{expected_kind!r}."),
                        metric="kind",
                        where=f"ports.{port_id}.kind",
                    ))
                else:
                    kind_ok_ct += 1
            if port_id in require_grounded:
                grounded_checks += 1
                if not _port_is_grounded(ir, port_id):
                    failures.append(Failure(
                        code=FailureCode.WRONG_TOPOLOGY,
                        severity=Severity.CRITICAL,
                        message=(f"Port {port_id!r} must be grounded "
                                 f"(attached to a fixed part) but its "
                                 f"reference does not touch a fixed "
                                 f"body."),
                        where=f"ports.{port_id}",
                        public_hint=(
                            "Either mark the referenced part fixed "
                            "or route the port through a joint whose "
                            "parent or child is the ground body."
                        ),
                    ))
                else:
                    grounded_ok_ct += 1

        n_required = max(1, len(required))
        metrics = {
            "ports_required": float(len(required)),
            "ports_present": float(present_ct),
            "ports_missing": float(len(required) - present_ct),
            "kind_checks": float(kind_checks),
            "kind_ok": float(kind_ok_ct),
            "grounded_checks": float(grounded_checks),
            "grounded_ok": float(grounded_ok_ct),
        }
        passed = not failures
        score = 1.0 if passed else max(
            0.0, present_ct / n_required - 0.5
        )
        # Clamp dense score so a partial pass cannot earn full credit.
        score = min(score, 0.5 if not passed else 1.0)
        return ProbeResult(
            probe_id="",
            probe_type=self.type_name,
            passed=passed,
            score=float(score),
            metrics=metrics,
            failures=failures,
        )
