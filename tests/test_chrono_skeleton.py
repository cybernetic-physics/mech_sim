"""chrono_contact adapter is a skeleton; tests prove the skeleton path.

When PyChrono is importable but the ``_chrono_impl`` runner module is
not, the adapter must:

* not register itself with the adapter registry
* expose a structured diagnostic with ``runner_status == "skeleton_only"``
* let the evaluator surface CAPABILITY_UNAVAILABLE for tasks that
  require contact_forces
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import pytest


def test_chrono_diagnostic_has_structured_shape():
    from mech_bench.adapters.chrono_contact import chrono_diagnostic

    diag = chrono_diagnostic()
    assert diag["adapter"] == "chrono_contact"
    assert diag["status"] in ("available", "unavailable")
    assert isinstance(diag["pychrono_importable"], bool)
    assert isinstance(diag["pychrono_project_chrono"], bool)
    assert isinstance(diag["pychrono_inprocess_importable"], bool)
    assert isinstance(diag["pychrono_subprocess_importable"], bool)
    assert diag["pychrono_import_mode"] in ("in_process", "subprocess", "missing")
    assert diag["chrono_python"] is None or isinstance(diag["chrono_python"], str)
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

    monkeypatch.delenv("MECH_BENCH_CHRONO_PYTHON", raising=False)
    monkeypatch.setenv("MECH_BENCH_DISABLE_CHRONO_AUTO", "1")
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


def test_chrono_python_probe_accepts_project_chrono_subprocess(tmp_path, monkeypatch):
    from mech_bench.adapters.chrono_env import probe_chrono_python

    fake_pkg = tmp_path / "fake_site"
    fake_pkg.mkdir()
    (fake_pkg / "pychrono.py").write_text(
        "class ChSystemSMC:\n"
        "    pass\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("PYTHONPATH", str(fake_pkg))

    available, reason = probe_chrono_python(Path(sys.executable))

    assert available is True
    assert "pychrono + _chrono_impl available" in reason


def test_chrono_python_candidates_include_abi_specific_env(tmp_path, monkeypatch):
    import mech_bench.adapters.chrono_env as chrono_env

    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr(chrono_env, "REPO_ROOT", root)
    monkeypatch.chdir(root)
    monkeypatch.delenv("MECH_BENCH_CHRONO_ENV", raising=False)

    py_tag = f"py{sys.version_info.major}{sys.version_info.minor}"
    expected_external = (
        root / ".external" / f"chrono_env_{py_tag}" / "bin" / "python"
    )
    expected_sibling = root / f"chrono_env_{py_tag}" / "bin" / "python"

    candidates = chrono_env._candidate_chrono_pythons()
    assert expected_external in candidates
    assert expected_sibling in candidates


def test_chrono_child_env_adds_chrono_lib_path(tmp_path, monkeypatch):
    import mech_bench.adapters.chrono_env as chrono_env

    chrono_python = tmp_path / "chrono_env_py312" / "bin" / "python"
    chrono_lib = chrono_python.parent.parent / "lib"
    chrono_lib.mkdir(parents=True)
    monkeypatch.setenv("LD_LIBRARY_PATH", f"/already{os.pathsep}{chrono_lib}")

    env = chrono_env.chrono_child_env(chrono_python)

    assert env["LD_LIBRARY_PATH"].split(os.pathsep)[0] == str(chrono_lib)
    assert env["LD_LIBRARY_PATH"].split(os.pathsep).count(str(chrono_lib)) == 1


def test_chrono_unavailable_yields_capability_unavailable(tmp_path):
    """Tasks requiring contact_forces must surface capability_unavailable
    when neither the real chrono nor the fake oracle is registered."""
    from mech_bench.adapters.chrono_contact import chrono_diagnostic

    diag = chrono_diagnostic()
    if diag["status"] == "available":
        pytest.skip("real chrono_contact is available in this environment")

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
