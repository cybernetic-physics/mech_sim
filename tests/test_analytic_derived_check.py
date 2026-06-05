"""Tests for analytic_derived_check (de-self-reference fix).

The defining property: the probe grades a value RECOMPUTED from the agent's
declared primitives (teeth, pitch, radii), never the agent's declared answer.
A correct-teeth / wrong-declared-ratio design must PASS; wrong teeth must FAIL.
"""

from __future__ import annotations

from mech_bench.probes import get_probe
from mech_bench.schema import DesignIR


def _gear_ir(teeth_in: int, teeth_out: int, declared_ratio: float) -> DesignIR:
    return DesignIR.from_dict({
        "schema_version": "design_ir.v2",
        "parts": [
            {"id": "pinion", "params": {"teeth": teeth_in}},
            {"id": "gear", "params": {"teeth": teeth_out}},
        ],
        "joints": [],
        "ports": {},
        # The agent's *claim*; the derived probe must ignore it entirely.
        "params": {"declared_ratio": declared_ratio},
    })


_CFG = {
    "formula": "gear_ratio",
    "inputs": {
        "teeth_in": "parts.pinion.params.teeth",
        "teeth_out": "parts.gear.params.teeth",
    },
    "expected": 4.0,
    "tolerance_pct": 2.0,
    "code": "wrong_ratio",
}


def test_correct_teeth_passes_even_with_wrong_declared_ratio():
    probe = get_probe("analytic_derived_check")
    # Correct geometry (64/16 = 4.0) but a LYING declared_ratio of 999.
    ir = _gear_ir(16, 64, declared_ratio=999.0)
    result = probe.run(ir, {}, _CFG)
    assert result.passed
    assert result.metrics["derived"] == 4.0
    assert result.score > 0.99


def test_wrong_teeth_fails_even_with_correct_declared_ratio():
    probe = get_probe("analytic_derived_check")
    # Geometry gives 50/16 = 3.125 != 4.0, even though declared_ratio is 4.0.
    ir = _gear_ir(16, 50, declared_ratio=4.0)
    result = probe.run(ir, {}, _CFG)
    assert not result.passed
    codes = {f.code.value for f in result.failures}
    assert "wrong_ratio" in codes


def test_missing_input_is_invalid_artifact():
    probe = get_probe("analytic_derived_check")
    ir = DesignIR.from_dict({
        "schema_version": "design_ir.v2",
        "parts": [{"id": "pinion", "params": {"teeth": 16}}],  # no gear
        "joints": [], "ports": {}, "params": {},
    })
    result = probe.run(ir, {}, _CFG)
    assert not result.passed
    codes = {f.code.value for f in result.failures}
    assert "invalid_artifact" in codes


def test_divide_by_zero_teeth_is_invalid():
    probe = get_probe("analytic_derived_check")
    ir = _gear_ir(0, 64, declared_ratio=4.0)
    result = probe.run(ir, {}, _CFG)
    assert not result.passed


def test_compound_product_ratio():
    probe = get_probe("analytic_derived_check")
    ir = DesignIR.from_dict({
        "schema_version": "design_ir.v2",
        "parts": [
            {"id": "a", "params": {"teeth": 10}},
            {"id": "b", "params": {"teeth": 30}},
            {"id": "c", "params": {"teeth": 12}},
            {"id": "d", "params": {"teeth": 48}},
        ],
        "joints": [], "ports": {}, "params": {},
    })
    cfg = {
        "formula": "compound_gear_ratio",
        "inputs": {
            "driver_teeth": ["parts.a.params.teeth", "parts.c.params.teeth"],
            "driven_teeth": ["parts.b.params.teeth", "parts.d.params.teeth"],
        },
        "expected": (30 * 48) / (10 * 12),  # = 12.0
        "tolerance_pct": 1.0,
    }
    result = probe.run(ir, {}, cfg)
    assert result.passed
    assert result.metrics["derived"] == 12.0


def test_lead_screw_travel_formula():
    probe = get_probe("analytic_derived_check")
    ir = DesignIR.from_dict({
        "schema_version": "design_ir.v2",
        "parts": [{"id": "screw", "params": {"lead_mm": 2.0, "turns": 5.0}}],
        "joints": [], "ports": {}, "params": {},
    })
    cfg = {
        "formula": "lead_screw_travel",
        "inputs": {
            "lead_mm": "parts.screw.params.lead_mm",
            "revolutions": "parts.screw.params.turns",
        },
        "expected": 10.0, "tolerance_pct": 1.0,
    }
    result = probe.run(ir, {}, cfg)
    assert result.passed
    assert result.metrics["derived"] == 10.0


def test_unknown_formula_is_schema_error():
    probe = get_probe("analytic_derived_check")
    ir = _gear_ir(16, 64, 4.0)
    result = probe.run(ir, {}, {**_CFG, "formula": "no_such_formula"})
    assert not result.passed
    codes = {f.code.value for f in result.failures}
    assert "schema_error" in codes


def test_dense_score_half_at_tolerance():
    probe = get_probe("analytic_derived_check")
    # derived 4.08 vs expected 4.0 = 2% off, tolerance 2% -> score ~0.5
    ir = _gear_ir(50, 51, declared_ratio=0.0)  # 51/50 = 1.02
    cfg = {**_CFG, "expected": 1.0, "tolerance_pct": 2.0}
    result = probe.run(ir, {}, cfg)
    assert abs(result.score - 0.5) < 1e-6


def test_registered():
    from mech_bench.probes import known_probe_types
    assert "analytic_derived_check" in known_probe_types()
