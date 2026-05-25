"""Trusted asset preflight helpers.

This module is the first concrete slice of the high-fidelity asset
layer. It does not claim to recompute CAD mass properties yet. Instead
it builds a provenance manifest from the submitted DesignIR and marks
which pieces are still declared-only. That makes missing trusted
geometry explicit instead of letting future physics tasks blur synthetic
or declared values into validated evidence.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mech_bench.schema import DesignIR


@dataclass
class GeometryArtifactEvidence:
    part_id: str
    role: str
    path: str
    exists: bool = False
    sha256: str | None = None
    size_bytes: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "part_id": self.part_id,
            "role": self.role,
            "path": self.path,
            "exists": self.exists,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass
class MassPropertyEvidence:
    part_id: str
    recomputed: bool = False
    source: str = "declared"
    kernel: str = "unavailable"
    mass_kg: float | None = None
    com_local_mm: tuple[float, float, float] | None = None
    inertia_kg_m2: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ] | None = None
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "part_id": self.part_id,
            "recomputed": self.recomputed,
            "source": self.source,
            "kernel": self.kernel,
            "mass_kg": self.mass_kg,
            "com_local_mm": self.com_local_mm,
            "inertia_kg_m2": self.inertia_kg_m2,
            "reason": self.reason,
        }


@dataclass
class TrustedAssetManifest:
    schema_version: str = "trusted_asset_manifest.v1"
    design_units: str = "mm"
    geometry: list[GeometryArtifactEvidence] = field(default_factory=list)
    mass_properties: list[MassPropertyEvidence] = field(default_factory=list)
    materials: dict[str, dict[str, Any]] = field(default_factory=dict)
    trusted_mass_properties_recomputed: bool = False
    trusted_geometry_kernel: str = "unavailable"
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "design_units": self.design_units,
            "geometry": [g.to_dict() for g in self.geometry],
            "mass_properties": [m.to_dict() for m in self.mass_properties],
            "materials": self.materials,
            "trusted_mass_properties_recomputed": (
                self.trusted_mass_properties_recomputed
            ),
            "trusted_geometry_kernel": self.trusted_geometry_kernel,
            "notes": list(self.notes),
        }


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _trusted_kernel(ir: DesignIR) -> str:
    source = (ir.params or {}).get("cad_source")
    if isinstance(source, dict):
        kernel = str(source.get("kernel", "")).strip()
        if kernel:
            return kernel
    return "unavailable"


def _mass_property_evidence(
    part_id: str,
    params: dict[str, Any],
    kernel: str,
) -> MassPropertyEvidence:
    raw = params.get("cad_mass_properties")
    if not isinstance(raw, dict):
        return MassPropertyEvidence(
            part_id=part_id,
            reason="part has no trusted cad_mass_properties record",
        )
    try:
        mass = float(raw["mass_kg"])
        com_raw = tuple(float(x) for x in raw["com_local_mm"])
        inertia_raw = tuple(
            tuple(float(x) for x in row)
            for row in raw["inertia_kg_m2"]
        )
    except (KeyError, TypeError, ValueError):
        return MassPropertyEvidence(
            part_id=part_id,
            reason="cad_mass_properties record is malformed",
        )
    if len(com_raw) != 3 or len(inertia_raw) != 3 or any(
        len(row) != 3 for row in inertia_raw
    ):
        return MassPropertyEvidence(
            part_id=part_id,
            reason="cad_mass_properties record has wrong dimensions",
        )
    if mass <= 0.0 or any(inertia_raw[i][i] <= 0.0 for i in range(3)):
        return MassPropertyEvidence(
            part_id=part_id,
            reason="cad_mass_properties record has nonphysical values",
        )
    return MassPropertyEvidence(
        part_id=part_id,
        recomputed=True,
        source="trusted_cad_kernel",
        kernel=kernel,
        mass_kg=mass,
        com_local_mm=com_raw,
        inertia_kg_m2=inertia_raw,
    )


def build_trusted_asset_manifest(
    ir: DesignIR,
    *,
    build_root: Path | None = None,
) -> TrustedAssetManifest:
    """Build a deterministic manifest for submitted physical assets.

    This records geometry hashes, material declarations, and any
    trusted-CAD mass properties already attached by the trusted geometry
    bridge. Agent-declared mass and inertia remain declared-only unless a
    trusted bridge has populated ``part.params["cad_mass_properties"]``.
    """
    root = build_root.resolve() if build_root is not None else None
    manifest = TrustedAssetManifest(design_units=ir.units)
    manifest.trusted_geometry_kernel = _trusted_kernel(ir)

    for p in ir.parts:
        manifest.mass_properties.append(
            _mass_property_evidence(
                p.id,
                dict(p.params or {}),
                manifest.trusted_geometry_kernel,
            )
        )
        if not isinstance(p.geometry, dict):
            continue
        for role, raw in sorted(p.geometry.items()):
            evidence = GeometryArtifactEvidence(
                part_id=p.id,
                role=str(role),
                path=str(raw),
            )
            if root is not None and isinstance(raw, str):
                candidate = (root / raw).resolve()
                try:
                    candidate.relative_to(root)
                    inside = True
                except ValueError:
                    inside = False
                if inside and candidate.exists() and candidate.is_file():
                    evidence.exists = True
                    evidence.size_bytes = candidate.stat().st_size
                    evidence.sha256 = _sha256(candidate)
            manifest.geometry.append(evidence)

    for mid, mat in sorted((ir.materials or {}).items()):
        manifest.materials[mid] = {
            "id": mat.id,
            "name": mat.name,
            "density_kg_m3": mat.density_kg_m3,
            "elastic_modulus_pa": mat.elastic_modulus_pa,
            "poisson_ratio": mat.poisson_ratio,
            "yield_strength_pa": mat.yield_strength_pa,
            "process": mat.process,
            "provenance": mat.provenance,
            "uncertainty": dict(mat.uncertainty or {}),
            "properties": dict(mat.properties or {}),
        }

    physical_parts = [
        p.id for p in ir.parts
        if float(getattr(p, "mass_kg", 0.0) or 0.0) > 0.0
    ]
    trusted_parts = {
        evidence.part_id
        for evidence in manifest.mass_properties
        if evidence.recomputed
    }
    manifest.trusted_mass_properties_recomputed = bool(
        physical_parts and set(physical_parts).issubset(trusted_parts)
    )
    if manifest.trusted_mass_properties_recomputed:
        manifest.notes.append(
            "Mass, COM, and inertia were recomputed by a trusted CAD "
            "bridge for every positive-mass part."
        )
    else:
        manifest.notes.append(
            "One or more positive-mass parts still use declared mass or "
            "inertia values; they are not trusted physical evidence."
        )

    return manifest
