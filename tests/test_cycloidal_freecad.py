from __future__ import annotations

from mech_bench.geometry.cycloidal_freecad import _mass_properties_for_body


def test_design_ir_mass_properties_are_canonicalized_for_chrono():
    body = {
        "mass_properties": {
            "mass_kg": 0.123456789012345,
            "com_mm": [
                1.1234567890123,
                -2.1234567890123,
                3.1234567890123,
            ],
            "inertia_kg_m2": [
                [1.234567890123456e-5, 1.234567890123456e-16, 0.0],
                [1.234567890123456e-16, 2.234567890123456e-5, 0.0],
                [0.0, 0.0, 3.234567890123456e-5],
            ],
        }
    }

    mass, com, inertia = _mass_properties_for_body(
        body,
        density_kg_m3=7850.0,
        fallback_mass_kg=0.1,
    )

    assert mass == 0.123456789012
    assert com == (1.123456789, -2.123456789, 3.123456789)
    assert inertia[0][0] == 0.000012345678901
    assert inertia[0][1] == 0.0
