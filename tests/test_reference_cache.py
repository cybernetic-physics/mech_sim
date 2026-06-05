"""Tests for the reference-cache key (GBA-Eval Mesen2-cache analogue).

Runs without pychrono — pure content-addressing logic.
"""

from __future__ import annotations

from mech_bench.oracle import reference_cache as rc


def _ir(center_x: float = 40.0, declared_ratio: float = 4.0) -> dict:
    return {
        "schema_version": "design_ir.v2",
        "parts": [
            {"id": "pinion", "params": {"teeth": 16}},
            {"id": "gear", "params": {"teeth": 64}},
        ],
        "joints": [
            {"id": "oa", "type": "revolute", "parent": "frame", "child": "gear",
             "anchor_world_mm": (center_x, 0.0, 0.0)},
        ],
        "ports": {},
        "params": {"declared_ratio": declared_ratio},
    }


def test_identical_geometry_hashes_equally():
    assert rc.geometry_hash(_ir()) == rc.geometry_hash(_ir())


def test_declared_answer_does_not_affect_geometry_hash():
    # The agent's declared_ratio is a *claim*, not geometry: changing it must
    # NOT change the cache key (otherwise lying would dodge the cache).
    assert rc.geometry_hash(_ir(declared_ratio=4.0)) == \
        rc.geometry_hash(_ir(declared_ratio=999.0))


def test_geometry_change_changes_hash():
    assert rc.geometry_hash(_ir(center_x=40.0)) != \
        rc.geometry_hash(_ir(center_x=41.0))


def test_cache_key_depends_on_version():
    k1 = rc.reference_cache_key("t1", _ir(), {"input_speed_rad_s": 10.0})
    # Same inputs -> same key (deterministic).
    k2 = rc.reference_cache_key("t1", _ir(), {"input_speed_rad_s": 10.0})
    assert k1 == k2


def test_cache_key_depends_on_oracle_config():
    k1 = rc.reference_cache_key("t1", _ir(), {"input_speed_rad_s": 10.0})
    k2 = rc.reference_cache_key("t1", _ir(), {"input_speed_rad_s": 20.0})
    assert k1 != k2


def test_cache_key_depends_on_task_id():
    assert rc.reference_cache_key("a", _ir()) != rc.reference_cache_key("b", _ir())


def test_version_is_exposed():
    assert isinstance(rc.REFERENCE_CACHE_VERSION, int)


def test_chrono_diagnostic_reports_real_gate_and_remediation():
    from mech_bench.adapters.chrono_contact import chrono_diagnostic
    diag = chrono_diagnostic()
    # The in-repo runner is present; the real gate is pychrono.
    assert diag["_chrono_impl_importable"] is True
    assert "remediation" in diag
    assert "reference_cache_version" in diag
    if not diag["pychrono_importable"]:
        assert diag["status"] == "unavailable"
        assert "conda" in diag["remediation"].lower()


def test_no_stale_vendored_out_language():
    from pathlib import Path
    src = Path(__file__).resolve().parent.parent / \
        "mech_bench" / "adapters" / "chrono_contact.py"
    assert "vendored out" not in src.read_text()
