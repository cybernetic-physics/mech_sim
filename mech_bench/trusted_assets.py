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
class TrustedAssetManifest:
    schema_version: str = "trusted_asset_manifest.v1"
    design_units: str = "mm"
    geometry: list[GeometryArtifactEvidence] = field(default_factory=list)
    materials: dict[str, dict[str, Any]] = field(default_factory=dict)
    trusted_mass_properties_recomputed: bool = False
    trusted_geometry_kernel: str = "unavailable"
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "design_units": self.design_units,
            "geometry": [g.to_dict() for g in self.geometry],
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


def build_trusted_asset_manifest(
    ir: DesignIR,
    *,
    build_root: Path | None = None,
) -> TrustedAssetManifest:
    """Build a deterministic manifest for submitted physical assets.

    With no CAD kernel integrated yet, this records geometry hashes and
    material declarations but explicitly marks mass-property recompute as
    unavailable. Future CAD ingestion should update only this boundary:
    callers should already be prepared to inspect the manifest.
    """
    root = build_root.resolve() if build_root is not None else None
    manifest = TrustedAssetManifest(design_units=ir.units)
    manifest.notes.append(
        "CAD mass-property recomputation is not implemented in this "
        "runtime slice; part mass/inertia remain declared values."
    )

    for p in ir.parts:
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

    return manifest
