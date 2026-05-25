from __future__ import annotations

import sys
import types
import subprocess


def test_oracle_smoke_report_shape():
    from mech_bench.oracle.smoke import run_oracle_smoke

    report = run_oracle_smoke(require_real=False)
    assert report["status"] in {"ok", "partial"}
    assert report["require_real"] is False
    assert report["all_required_ok"] is True
    names = {check["name"] for check in report["checks"]}
    assert {"numpy", "hdf5", "opencascade", "gmsh", "chrono"} <= names


def test_oracle_smoke_require_real_exit_policy():
    from mech_bench.oracle.smoke import oracle_smoke_exit_code

    assert oracle_smoke_exit_code({
        "require_real": True,
        "all_required_ok": False,
    }) == 1
    assert oracle_smoke_exit_code({
        "require_real": False,
        "all_required_ok": False,
    }) == 0


def test_component_json_marker_extraction():
    from mech_bench.oracle import smoke

    stdout = (
        "native prelude\n"
        f"{smoke._JSON_BEGIN}\n"
        "{\"name\":\"chrono\",\"status\":\"ok\"}\n"
        f"{smoke._JSON_END}\n"
        "native epilogue\n"
    )
    parsed = smoke._extract_component_json(stdout)
    assert parsed == {"name": "chrono", "status": "ok"}


def test_isolated_check_reports_native_process_crash(monkeypatch):
    from mech_bench.oracle import smoke

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=139,
            stdout="",
            stderr="Fatal Python error: Segmentation fault",
        )

    monkeypatch.setattr(smoke.subprocess, "run", fake_run)
    result = smoke._isolated_check("chrono")
    assert result.name == "chrono"
    assert result.status == "failed"
    assert "139" in result.detail
    assert "Segmentation fault" in result.metrics["stderr_tail"]


def test_chrono_smoke_executes_fake_kernel(monkeypatch):
    from mech_bench.oracle import smoke

    class FakeBody:
        def SetFixed(self, fixed: bool) -> None:
            self.fixed = fixed

    class FakeSystem:
        def __init__(self) -> None:
            self.time = 0.0
            self.bodies = []

        def Add(self, body: FakeBody) -> None:
            self.bodies.append(body)

        def DoStepDynamics(self, step: float) -> None:
            self.time += step

        def GetChTime(self) -> float:
            return self.time

    pychrono = types.ModuleType("pychrono")
    core = types.ModuleType("pychrono.core")
    core.ChSystemNSC = FakeSystem
    core.ChBody = FakeBody
    core.CHRONO_VERSION = "fake-chrono"
    pychrono.core = core
    monkeypatch.setitem(sys.modules, "pychrono", pychrono)
    monkeypatch.setitem(sys.modules, "pychrono.core", core)

    result = smoke._check_chrono()
    assert result.status == "ok"
    assert result.module == "pychrono.core"
    assert result.metrics["stepped_time_s"] > 0


def test_opencascade_smoke_accepts_ocp_binding(monkeypatch):
    from mech_bench.oracle import smoke

    class FakeMaker:
        def __init__(self, x: float, y: float, z: float) -> None:
            self.dims = (x, y, z)

        def Shape(self) -> object:
            return object()

    ocp = types.ModuleType("OCP")
    brep = types.ModuleType("OCP.BRepPrimAPI")
    brep.BRepPrimAPI_MakeBox = FakeMaker
    monkeypatch.setitem(sys.modules, "OCP", ocp)
    monkeypatch.setitem(sys.modules, "OCP.BRepPrimAPI", brep)

    result = smoke._check_opencascade()
    assert result.status == "ok"
    assert result.module == "OCP.BRepPrimAPI"
    assert result.metrics["kernel"] == "OCP"
