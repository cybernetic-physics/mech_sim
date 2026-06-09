"""DOF probe: Grübler-Kutzbach mobility.

Pure topology, no simulator required. Ported from
phys-sim/mech_harness/validators/assembly.py with the same formulas
but generalized to declare its `space` (planar / spatial) per
config.

  M_planar  = 3(n − 1) − 2 j_lower
  M_spatial = 6(n − 1) − 5 j_lower

where `n` is the link count after collapsing grounded/fixed links and
fixed joints, and `j_lower` is the count of lower-pair moving joints.
``Part.fixed`` denotes membership in the single ground link; it is not
an additional free body.
"""

from __future__ import annotations

from typing import Any

from mech_bench.feedback import Failure, FailureCode, Severity
from mech_bench.probes import Capability, Probe, register_probe
from mech_bench.schema import DesignIR, ProbeResult


def _grubler(ir: DesignIR, space: str) -> int:
    parent = {p.id: p.id for p in ir.parts}
    fixed_ids = [p.id for p in ir.parts if p.fixed]

    def find(x: str) -> str:
        root = parent[x]
        if root != x:
            parent[x] = find(root)
        return parent[x]

    def union(a: str, b: str) -> None:
        if a not in parent or b not in parent:
            return
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    if fixed_ids:
        ground = fixed_ids[0]
        for part_id in fixed_ids[1:]:
            union(ground, part_id)
    for joint in ir.joints:
        if joint.type == "fixed":
            union(joint.parent, joint.child)

    components = {find(p.id) for p in ir.parts}
    n = len(components)
    revolute = 0
    prismatic = 0
    spherical = 0
    for joint in ir.joints:
        if joint.type in {"fixed", "contact_pair"}:
            continue
        if joint.parent not in parent or joint.child not in parent:
            continue
        if find(joint.parent) == find(joint.child):
            continue
        if joint.type == "revolute":
            revolute += 1
        elif joint.type == "prismatic":
            prismatic += 1
        elif joint.type == "spherical":
            spherical += 1
    if space == "planar":
        # 1-DOF joints (revolute, prismatic) subtract 2 each after
        # fixed joints and fixed=True parts have been collapsed.
        return 3 * (n - 1) - 2 * (revolute + prismatic)
    # spatial
    return 6 * (n - 1) - 5 * (revolute + prismatic) - 3 * spherical


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
