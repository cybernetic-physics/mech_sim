from __future__ import annotations

import hashlib

from mech_bench.schema import DesignIR
from mech_bench.trusted_assets import build_trusted_asset_manifest


def test_trusted_asset_manifest_hashes_geometry(tmp_path):
    cad = tmp_path / "link.step"
    cad.write_text("ISO-10303-21; ENDSEC;")
    raw = {
        "schema_version": "design_ir.v2",
        "units": "mm",
        "materials": {
            "steel": {
                "density_kg_m3": 7850.0,
                "provenance": "test fixture",
            },
        },
        "parts": [
            {"id": "ground", "fixed": True, "mass_kg": 0.0},
            {"id": "link", "mass_kg": 0.1,
             "geometry": {"cad": "link.step"},
             "material": "steel"},
        ],
        "joints": [
            {"id": "j1", "type": "fixed",
             "parent": "ground", "child": "link"},
        ],
        "ports": {},
    }
    manifest = build_trusted_asset_manifest(
        DesignIR.from_dict(raw), build_root=tmp_path)
    blob = manifest.to_dict()
    assert blob["design_units"] == "mm"
    assert blob["trusted_mass_properties_recomputed"] is False
    assert blob["trusted_geometry_kernel"] == "unavailable"
    assert blob["materials"]["steel"]["density_kg_m3"] == 7850.0
    geom = blob["geometry"][0]
    assert geom["exists"] is True
    assert geom["size_bytes"] == cad.stat().st_size
    assert geom["sha256"] == hashlib.sha256(cad.read_bytes()).hexdigest()
