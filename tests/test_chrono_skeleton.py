"""chrono_contact adapter is a skeleton; tests prove the skeleton path.

When PyChrono is importable but the ``_chrono_impl`` runner module is
not, the adapter must:

* not register itself with the adapter registry
* expose a structured diagnostic with ``runner_status == "skeleton_only"``
* let the evaluator surface CAPABILITY_UNAVAILABLE for tasks that
  require contact_forces
"""

from __future__ import annotations

import types

import pytest


def test_chrono_diagnostic_has_structured_shape():
    from mech_bench.adapters.chrono_contact import chrono_diagnostic

    diag = chrono_diagnostic()
    assert diag["adapter"] == "chrono_contact"
    assert diag["status"] in ("available", "unavailable")
    assert isinstance(diag["pychrono_importable"], bool)
    assert isinstance(diag["pychrono_project_chrono"], bool)
    assert isinstance(diag["_chrono_impl_importable"], bool)
    assert diag["runner_status"] in (
        "ready", "skeleton_only", "missing_dependency",
    )


def test_chrono_not_registered_when_impl_missing():
    """Even if pychrono imports, _chrono_impl must also be present."""
    from mech_bench.adapters import all_adapters
    from mech_bench.adapters.chrono_contact import chrono_diagnostic

    diag = chrono_diagnostic()
    if diag["pychrono_importable"] and not diag["_chrono_impl_importable"]:
        type_names = {a.type_name for a in all_adapters()}
        assert "chrono_contact" not in type_names, (
            "chrono_contact must not register when _chrono_impl missing"
        )
        assert diag["runner_status"] == "skeleton_only"


def test_probe_rejects_non_project_chrono_pychrono(monkeypatch):
    """The PyPI timing package named pychrono must not register Chrono."""
    import mech_bench.adapters.chrono_contact as chrono_contact

    fake = types.SimpleNamespace(__file__="/tmp/pychrono/__init__.py")

    def fake_import(name, *args, **kwargs):
        if name == "pychrono":
            return fake
        return real_import(name, *args, **kwargs)

    real_import = __import__
    monkeypatch.setattr("builtins.__import__", fake_import)

    available, reason = chrono_contact._probe_pychrono()

    assert available is False
    assert "not projectchrono::pychrono" in reason


def test_chrono_unavailable_yields_capability_unavailable(tmp_path):
    """Tasks requiring contact_forces must surface capability_unavailable
    when neither the real chrono nor the fake oracle is registered."""
    from mech_bench.adapters import _REGISTRY

    # Temporarily remove the fake oracle from the registry, then restore.
    snapshot = dict(_REGISTRY)
    _REGISTRY.pop("fake_contact_oracle", None)
    try:
        # Build a tiny task whose only probe requires contact_forces.
        from mech_bench.generators.benchmark_suite import (
            ContactGearPairStubGenerator,
        )
        from mech_bench.generators.base import write_task_directory
        from mech_bench.evaluator import evaluate

        gen = ContactGearPairStubGenerator()
        task = gen.generate(seed=0)
        task_dir = write_task_directory(task, tmp_path)
        report = evaluate(task_dir, task_dir / "reference_solution")
        assert report.score == 0.0
        assert not report.hard_gate_passed
        codes = {f.code.value if hasattr(f.code, "value") else str(f.code)
                 for f in report.feedback}
        assert "capability_unavailable" in codes
    finally:
        _REGISTRY.clear()
        _REGISTRY.update(snapshot)
