"""DOF probe: Grübler-Kutzbach mobility.

Pure topology, no simulator required. Ported from
phys-sim/mech_harness/validators/assembly.py with the same formulas
but generalized to declare its `space` (planar / spatial) per
config.

  M_planar  = 3(n − 1) − 2 j_lower
  M_spatial = 6(n − 1) − 5 j_lower

where `n` is the body count (parts; "world" / fixed parts count too,
following standard convention) and `j_lower` is the count of lower-
pair (1-DOF) joints: revolute, prismatic, fixed (contributes 6 in
spatial, 3 in planar — but we treat as constraint count).
"""

from __future__ import annotations

from typing import Any

from mech_bench.feedback import Failure, FailureCode, Severity
from mech_bench.probes import Capability, Probe, register_probe
from mech_bench.schema import DesignIR, ProbeResult


def _grubler(ir: DesignIR, space: str) -> int:
    n = len(ir.parts)
    revolute = sum(1 for j in ir.joints if j.type == "revolute")
    prismatic = sum(1 for j in ir.joints if j.type == "prismatic")
    fixed = sum(1 for j in ir.joints if j.type == "fixed")
    spherical = sum(1 for j in ir.joints if j.type == "spherical")
    if space == "planar":
        # 1-DOF joints (revolute, prismatic) subtract 2 each; fixed
        # subtracts 3 (full constraint in plane).
        return 3 * (n - 1) - 2 * (revolute + prismatic) - 3 * fixed
    # spatial
    return 6 * (n - 1) - 5 * (revolute + prismatic) - 6 * fixed - 3 * spherical


@register_probe
class DOFGrubler(Probe):
    type_name = "dof_grubler"
    capabilities_required = frozenset({Capability.NONE})

    def run(
        self,
        ir: DesignIR,
        sim_outputs: dict[str, Any],
        config: dict[str, Any],
    ) -> ProbeResult:
        space = str(config.get("space", "planar"))
        expected = config.get("expected")
        tolerance = int(config.get("tolerance", 0))
        observed = _grubler(ir, space)
        passed = expected is None or abs(observed - int(expected)) <= tolerance
        metrics = {
            "observed": float(observed),
            "expected": float(expected) if expected is not None else float("nan"),
            "space": 0.0 if space == "planar" else 1.0,
        }
        failures: list[Failure] = []
        if not passed:
            failures.append(Failure(
                code=FailureCode.WRONG_MOBILITY,
                severity=Severity.CRITICAL,
                message=(f"Mobility (Grübler, {space}) is {observed}, "
                         f"task expects {expected}."),
                metric="observed",
                observed=float(observed),
                target=float(expected),
                public_hint=(
                    "Check joint counts and types. For a planar "
                    "single-DOF mechanism with all revolute joints: "
                    "M = 3(n-1) - 2j = 1 implies the correct "
                    "topology."
                ),
            ))
        score = 1.0 if passed else 0.0
        return ProbeResult(
            probe_id="",
            probe_type=self.type_name,
            passed=passed,
            score=score,
            metrics=metrics,
            failures=failures,
        )
