"""Docker execution path for the Chrono adapter (mocked — no image needed)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

from mech_bench.adapters import chrono_contact as cc
from mech_bench.schema import DesignIR


def _ir() -> DesignIR:
    return DesignIR.from_dict({
        "schema_version": "design_ir.v2",
        "parts": [{"id": "a"}], "joints": [], "ports": {}, "params": {},
    })


def test_docker_enabled_modes(monkeypatch):
    monkeypatch.delenv("MECH_BENCH_CHRONO_DOCKER", raising=False)
    assert cc._docker_enabled({"docker": True}) is True
    assert cc._docker_enabled({}) is False
    monkeypatch.setenv("MECH_BENCH_CHRONO_DOCKER", "1")
    assert cc._docker_enabled({}) is True
    monkeypatch.setenv("MECH_BENCH_CHRONO_DOCKER", "auto")
    # auto == "use docker only when native pychrono is absent"
    assert cc._docker_enabled({}) is (not cc.CHRONO_AVAILABLE)


def test_docker_run_builds_expected_command(monkeypatch):
    captured = {}

    def fake_which(name):
        return "/usr/bin/docker" if name == "docker" else None

    def fake_run(cmd, **kw):
        captured["cmd"] = cmd
        # Emulate the container writing output.json into the mounted /io dir.
        # Find the host dir from the -v mount and write a minimal result.
        mount = next(c for c in cmd if ":/io" in c)
        host_io = mount.split(":/io")[0]
        result = {
            "time_s": [0.0, 1.0],
            "joint_positions": {}, "joint_velocities": {},
            "contact_forces": {"a:b": [1.0, 2.0]}, "penetration": {},
            "scalar_metrics": {"ratio_observed": 4.0},
            "metadata": {"adapter": "chrono_contact", "oracle_is_synthetic": False},
        }
        (Path(host_io) / "output.json").write_text(json.dumps(result))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(cc.shutil, "which", fake_which)
    monkeypatch.setattr(cc.subprocess, "run", fake_run)
    monkeypatch.setenv("MECH_BENCH_CHRONO_DOCKER_IMAGE", "mech-bench-solver:test")
    monkeypatch.setenv("MECH_BENCH_SOLVER_PLATFORM", "linux/amd64")

    out = cc._run_chrono_docker(_ir(), {"samples": 10})

    cmd = captured["cmd"]
    assert cmd[0] == "/usr/bin/docker"
    assert "run" in cmd and "--rm" in cmd
    assert "--platform" in cmd and "linux/amd64" in cmd
    assert "mech-bench-solver:test" in cmd
    assert any(c.endswith(":/io") for c in cmd)
    assert "__capability_unavailable__" not in out
    assert out["metadata"]["execution"] == "docker"
    assert out["metadata"]["docker_image"] == "mech-bench-solver:test"
    # numpy coercion happened.
    assert isinstance(out["contact_forces"]["a:b"], np.ndarray)


def test_docker_missing_cli_is_capability_unavailable(monkeypatch):
    monkeypatch.setattr(cc.shutil, "which", lambda name: None)
    out = cc._run_chrono_docker(_ir(), {})
    assert out["__capability_unavailable__"] is True


def test_docker_nonzero_exit_is_capability_unavailable(monkeypatch):
    monkeypatch.setattr(cc.shutil, "which", lambda name: "/usr/bin/docker")

    def fake_run(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 125, stdout="",
                                           stderr="Unable to find image")

    monkeypatch.setattr(cc.subprocess, "run", fake_run)
    out = cc._run_chrono_docker(_ir(), {})
    assert out["__capability_unavailable__"] is True
    assert "125" in out["metadata"]["preflight_issues"][0]


def test_diagnostic_reports_docker_fields(monkeypatch):
    monkeypatch.setattr(cc.shutil, "which", lambda name: "/usr/bin/docker")
    monkeypatch.setenv("MECH_BENCH_CHRONO_DOCKER", "1")
    diag = cc.chrono_diagnostic()
    assert diag["docker_mode"] is True
    assert diag["docker_cli_available"] is True
    if not cc.CHRONO_AVAILABLE:
        assert diag["status"] == "available"
        assert diag["runner_status"] == "docker"
