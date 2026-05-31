from mech_bench.schema import DesignIR
from mech_bench.trusted_asset_bridge import augment_with_trusted_assets
from mech_bench.trusted_assets import build_trusted_asset_manifest


def test_bridge_adds_geometry_material_and_trusted_mass(tmp_path):
    ir = DesignIR.from_dict(
        {
            "schema_version": "design_ir.v2",
            "parts": [
                {
                    "id": "frame",
                    "role": "ground",
                    "mass_kg": 0.0,
                    "fixed": True,
                },
                {
                    "id": "link",
                    "role": "output",
                    "mass_kg": 0.05,
                    "com_local_mm": (1.0, 2.0, 3.0),
                },
            ],
            "joints": [
                {
                    "id": "joint",
                    "type": "revolute",
                    "parent": "frame",
                    "child": "link",
                },
            ],
            "ports": {
                "input_port": {
                    "id": "input_port",
                    "part": "joint",
                    "kind": "revolute_joint",
                },
            },
        }
    )

    augment_with_trusted_assets(ir, build_root=tmp_path)
    manifest = build_trusted_asset_manifest(ir, build_root=tmp_path)

    assert all("cad" in part.geometry for part in ir.parts)
    assert "chrono_collision" not in ir.parts[0].params
    assert "chrono_collision" in ir.parts[1].params
    assert ir.parts[1].material == "bridge_al6061"
    assert "cad_source" in ir.params
    assert manifest.trusted_mass_properties_recomputed is True
    assert all(item.exists for item in manifest.geometry)
