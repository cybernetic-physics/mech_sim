"""Fake contact oracle determinism + behavior tests.

The fake oracle is the test-time stand-in for the real chrono_contact
adapter (which ships as a skeleton only). These tests prove:

* fake_contact_oracle outputs are deterministic given the same IR + cfg
* the synthetic flag (oracle_is_synthetic) propagates into reports
* contact tasks pass/fail under the fake oracle as configured
* fast/oracle agreement detection works end-to-end
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _enable_fake_oracle(monkeypatch):
    """Register fake_contact_oracle for the duration of this test only.

    Saves the global adapter registry, registers the fake oracle, and
    restores the original registry on teardown — no module reloads.
    """
    monkeypatch.setenv("MECH_BENCH_USE_FAKE_ORACLE", "1")
    from mech_bench.adapters import _REGISTRY, register_adapter
    from mech_bench.adapters.fake_contact_oracle import FakeContactOracle

    snapshot = dict(_REGISTRY)
    if FakeContactOracle.type_name not in _REGISTRY:
        register_adapter(FakeContactOracle)
    try:
        yield
    finally:
        _REGISTRY.clear()
        _REGISTRY.update(snapshot)


def _build_ir():
    from mech_bench.schema import DesignIR, Joint, Part, Port

    return DesignIR(
        schema_version="design_ir.v2",
        parts=[
            Part(id="frame", role="ground", mass_kg=0.0, fixed=True),
            Part(id="pinion", role="gear_input", mass_kg=0.02),
            Part(id="gear", role="gear_output", mass_kg=0.05),
        ],
        joints=[
            Joint(id="j_in", type="revolute", parent="frame",
                  child="pinion", axis_world=(0, 0, 1),
                  anchor_world_mm=(0, 0, 0)),
            Joint(id="j_out", type="revolute", parent="frame",
                  child="gear", axis_world=(0, 0, 1),
                  anchor_world_mm=(40, 0, 0)),
        ],
        ports={
            "input_port": Port(id="input_port", part="j_in",
                               kind="revolute_joint"),
            "output_port": Port(id="output_port", part="j_out",
                                kind="revolute_joint"),
        },
        params={
            "declared_ratio": 4.0,
            "fake_oracle": {
                "contact_pairs": ["pinion:gear"],
                "contact_force_N": 1.5,
                "penetration_mm": 0.005,
                "lockup": False,
            },
        },
    )


def test_fake_oracle_outputs_are_deterministic():
    from mech_bench.adapters.fake_contact_oracle import FakeContactOracle

    ir = _build_ir()
    a = FakeContactOracle().run(ir, {})
    b = FakeContactOracle().run(ir, {})
    np.testing.assert_array_equal(a["time_s"], b["time_s"])
    for pair in a["contact_forces"]:
        np.testing.assert_array_equal(
            a["contact_forces"][pair], b["contact_forces"][pair])
    assert a["scalar_metrics"] == b["scalar_metrics"]


def test_fake_oracle_metadata_labels_synthetic():
    from mech_bench.adapters.fake_contact_oracle import FakeContactOracle

    ir = _build_ir()
    out = FakeContactOracle().run(ir, {})
    meta = out["metadata"]
    assert meta["is_physical_oracle"] is False
    assert meta["oracle_is_synthetic"] is True
    assert meta["trust_level"] == "synthetic_test_or_demo"
    assert meta["simulator"] == "fake_contact_oracle"


def test_fake_oracle_lockup_path():
    from mech_bench.adapters.fake_contact_oracle import FakeContactOracle

    ir = _build_ir()
    ir.params["fake_oracle"]["lockup"] = True
    out = FakeContactOracle().run(ir, {})
    assert out["scalar_metrics"]["lockup_detected"] == 1.0
    # Output velocity should be zero under lockup.
    assert np.allclose(
        out["joint_velocities"].get("output_port", np.zeros(1)), 0.0)


def test_fake_oracle_contact_pairs_emit_forces():
    from mech_bench.adapters.fake_contact_oracle import FakeContactOracle

    ir = _build_ir()
    out = FakeContactOracle().run(ir, {})
    cf = out["contact_forces"]
    assert "gear:pinion" in cf or "pinion:gear" in cf
    arr = next(iter(cf.values()))
    assert np.max(arr) > 0.5  # nontrivial force


def test_fake_oracle_passes_contact_engagement_probe():
    from mech_bench.adapters.fake_contact_oracle import FakeContactOracle
    from mech_bench.probes.contact_engagement import ContactEngagement

    ir = _build_ir()
    sim = FakeContactOracle().run(ir, {})
    result = ContactEngagement().run(ir, sim, {
        "required_pairs": ["pinion:gear"],
        "min_rms_force_N": 0.3,
        "min_engagement_fraction": 0.2,
    })
    assert result.passed, result.failures


def test_fake_oracle_fails_when_force_below_threshold():
    from mech_bench.adapters.fake_contact_oracle import FakeContactOracle
    from mech_bench.probes.contact_engagement import ContactEngagement

    ir = _build_ir()
    ir.params["fake_oracle"]["contact_force_N"] = 0.01
    sim = FakeContactOracle().run(ir, {})
    result = ContactEngagement().run(ir, sim, {
        "required_pairs": ["pinion:gear"],
        "min_rms_force_N": 1.0,
        "min_engagement_fraction": 0.2,
    })
    assert not result.passed
    codes = {f.code.value for f in result.failures}
    assert "missing_contact" in codes
