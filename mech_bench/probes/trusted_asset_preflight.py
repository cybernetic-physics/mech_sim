"""Trusted-asset preflight probe.

This probe is a high-fidelity drift guard. It lets CAD-backed tasks
hard-gate on explicit geometry/material/provenance metadata before any
expensive simulator runs. It does not perform CAD recomputation itself;
until that exists, ``require_trusted_mass_properties=true`` deliberately
fails rather than treating declared mass/inertia as trusted physics.
"""

from __future__ import annotations

from typing import Any

from mech_bench.feedback import Failure, FailureCode, Severity
from mech_bench.probes import Capability, Probe, register_probe
from mech_bench.schema import DesignIR, ProbeResult
from mech_bench.trusted_assets import build_trusted_asset_manifest


def _parts_to_check(ir: DesignIR, selector: str) -> list:
    selector = selector.lower()
    if selector == "moving":
        return [p for p in ir.parts if p.fixed is not True]
    if selector == "non_ground":
        return [p for p in ir.parts if p.role.lower() != "ground"]
    return list(ir.parts)


@register_probe
class TrustedAssetPreflight(Probe):
    type_name = "trusted_asset_preflight"
    capabilities_required = frozenset({Capability.NONE})

    def run(
        self,
        ir: DesignIR,
        sim_outputs: dict[str, Any],
        config: dict[str, Any],
    ) -> ProbeResult:
        parts = _parts_to_check(
            ir, str(config.get("require_for_parts", "all")))
        required_roles = [
            str(x) for x in (config.get("require_geometry_roles") or [])
        ]
        require_materials = bool(config.get("require_materials", False))
        required_material_props = [
            str(x) for x in (
                config.get("require_material_properties")
                or ["density_kg_m3"]
            )
        ]
        require_provenance = bool(config.get("require_provenance", False))
        require_trusted_mass = bool(
            config.get("require_trusted_mass_properties", False))

        manifest = build_trusted_asset_manifest(ir)
        failures: list[Failure] = []

        geometry_ok = 0
        for p in parts:
            geom = p.geometry if isinstance(p.geometry, dict) else {}
            missing = [role for role in required_roles if role not in geom]
            if missing:
                failures.append(Failure(
                    code=FailureCode.INVALID_ARTIFACT,
                    severity=Severity.CRITICAL,
                    message=(
                        f"Part {p.id!r} is missing required geometry "
                        f"roles {missing!r}."
                    ),
                    where=f"parts.{p.id}.geometry",
                    public_hint=(
                        "CAD-backed high-fidelity tasks must declare "
                        "the required geometry roles, such as 'cad' or "
                        "'collision'."
                    ),
                ))
            else:
                geometry_ok += 1

            if require_materials:
                material_id = getattr(p, "material", "")
                mat = ir.materials.get(material_id) if material_id else None
                if not material_id or mat is None:
                    failures.append(Failure(
                        code=FailureCode.SCHEMA_ERROR,
                        severity=Severity.CRITICAL,
                        message=(
                            f"Part {p.id!r} does not reference a defined "
                            "material record."
                        ),
                        where=f"parts.{p.id}.material",
                    ))
                    continue
                missing_props = [
                    prop for prop in required_material_props
                    if getattr(mat, prop, None) is None
                ]
                if missing_props:
                    failures.append(Failure(
                        code=FailureCode.SCHEMA_ERROR,
                        severity=Severity.CRITICAL,
                        message=(
                            f"Material {material_id!r} for part {p.id!r} "
                            f"is missing required properties "
                            f"{missing_props!r}."
                        ),
                        where=f"materials.{material_id}",
                    ))
                if require_provenance and not mat.provenance:
                    failures.append(Failure(
                        code=FailureCode.SCHEMA_ERROR,
                        severity=Severity.CRITICAL,
                        message=(
                            f"Material {material_id!r} for part {p.id!r} "
                            "has no provenance string."
                        ),
                        where=f"materials.{material_id}.provenance",
                    ))

        if require_trusted_mass:
            failures.append(Failure(
                code=FailureCode.INVALID_MASS_PROPERTIES,
                severity=Severity.CRITICAL,
                message=(
                    "Trusted CAD mass-property recomputation is required "
                    "by this probe, but no trusted CAD kernel is currently "
                    "integrated."
                ),
                where="trusted_assets.mass_properties",
                public_hint=(
                    "This task cannot claim high-fidelity physical "
                    "validation until mass/inertia are recomputed on the "
                    "trusted side."
                ),
            ))

        n = max(1, len(parts))
        metrics = {
            "parts_checked": float(len(parts)),
            "required_geometry_roles": float(len(required_roles)),
            "parts_with_required_geometry": float(geometry_ok),
            "material_records": float(len(ir.materials or {})),
            "trusted_mass_properties_recomputed": (
                1.0 if manifest.trusted_mass_properties_recomputed else 0.0
            ),
        }
        passed = not failures
        score = 1.0 if passed else min(0.5, geometry_ok / n)
        return ProbeResult(
            probe_id="",
            probe_type=self.type_name,
            passed=passed,
            score=float(score),
            metrics=metrics,
            failures=failures,
        )
