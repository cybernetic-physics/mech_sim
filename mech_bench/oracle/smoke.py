"""Native solver stack smoke checks.

These checks are intentionally small but real: they import the native
bindings and execute one kernel-level operation for each backend. This
keeps dependency availability from being a vague README promise.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import importlib
import json
import subprocess
import sys
import tempfile
import traceback
from typing import Any


REQUIRED_COMPONENTS = (
    "numpy",
    "hdf5",
    "opencascade",
    "gmsh",
    "chrono",
)

STATUS_OK = "ok"
STATUS_UNAVAILABLE = "unavailable"
STATUS_FAILED = "failed"
_JSON_BEGIN = "MECH_BENCH_COMPONENT_JSON_BEGIN"
_JSON_END = "MECH_BENCH_COMPONENT_JSON_END"


@dataclass
class ComponentCheck:
    """Structured result for one native dependency check."""

    name: str
    status: str
    module: str | None = None
    version: str | None = None
    detail: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        return {k: v for k, v in out.items() if v not in (None, {}, [])}


def run_oracle_smoke(
    require_real: bool = False,
    isolate_native: bool = True,
) -> dict[str, Any]:
    """Run native solver checks and return a JSON-serializable report."""

    checks = _run_isolated_checks() if isolate_native else [
        _check_numpy(),
        _check_hdf5(),
        _check_opencascade(),
        _check_gmsh(),
        _check_chrono(),
    ]
    check_map = {c.name: c for c in checks}
    required = list(REQUIRED_COMPONENTS if require_real else ())
    required_bad = [
        name for name in required
        if check_map.get(name) is None
        or check_map[name].status != STATUS_OK
    ]
    all_required_ok = not required_bad

    if required_bad:
        status = STATUS_FAILED
    elif all(c.status == STATUS_OK for c in checks):
        status = STATUS_OK
    else:
        status = "partial"

    return {
        "status": status,
        "require_real": require_real,
        "isolate_native": isolate_native,
        "required": required,
        "all_required_ok": all_required_ok,
        "required_bad": required_bad,
        "checks": [c.to_dict() for c in checks],
        "notes": [
            "This is a dependency and kernel-operation smoke test, not a "
            "validation benchmark.",
            "Run with --require-real in CI/Docker to fail when any native "
            "solver dependency is missing.",
        ],
    }


def oracle_smoke_exit_code(report: dict[str, Any]) -> int:
    """Exit-code policy for the CLI wrapper."""

    if report.get("require_real") and not report.get("all_required_ok"):
        return 1
    return 0


def _run_isolated_checks() -> list[ComponentCheck]:
    return [
        _check_numpy(),
        _isolated_check("hdf5"),
        _isolated_check("opencascade"),
        _isolated_check("gmsh"),
        _isolated_check("chrono"),
    ]


def _isolated_check(name: str) -> ComponentCheck:
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "mech_bench.oracle.smoke",
             "--component", name],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return ComponentCheck(
            name=name,
            status=STATUS_FAILED,
            detail="component subprocess timed out after 120s",
            metrics={
                "stdout_tail": (exc.stdout or "")[-1000:],
                "stderr_tail": (exc.stderr or "")[-1000:],
            },
        )
    except OSError as exc:
        return ComponentCheck(
            name=name,
            status=STATUS_FAILED,
            detail=_short_exc(exc),
        )
    if proc.returncode != 0:
        return ComponentCheck(
            name=name,
            status=STATUS_FAILED,
            detail=f"component subprocess exited {proc.returncode}",
            metrics={
                "stdout_tail": (proc.stdout or "")[-1000:],
                "stderr_tail": (proc.stderr or "")[-1000:],
            },
        )

    parsed = _extract_component_json(proc.stdout)
    if parsed is None:
        return ComponentCheck(
            name=name,
            status=STATUS_FAILED,
            detail="component subprocess produced no JSON marker",
            metrics={
                "stdout_tail": (proc.stdout or "")[-1000:],
                "stderr_tail": (proc.stderr or "")[-1000:],
            },
        )
    return ComponentCheck(**parsed)


def _extract_component_json(stdout: str) -> dict[str, Any] | None:
    start = stdout.find(_JSON_BEGIN)
    end = stdout.find(_JSON_END)
    if start < 0 or end < 0 or end <= start:
        return None
    payload = stdout[start + len(_JSON_BEGIN):end].strip()
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _check_numpy() -> ComponentCheck:
    try:
        import numpy as np

        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
        det = float(np.linalg.det(a))
        return ComponentCheck(
            name="numpy",
            status=STATUS_OK,
            module="numpy",
            version=getattr(np, "__version__", None),
            metrics={"determinant": det},
        )
    except Exception as exc:  # pragma: no cover - numpy is a hard dep
        return _failed("numpy", "numpy", exc)


def _check_hdf5() -> ComponentCheck:
    module_name = "h5py"
    try:
        h5py = importlib.import_module(module_name)
    except Exception as exc:
        return _unavailable("hdf5", module_name, exc)

    try:
        with tempfile.NamedTemporaryFile(suffix=".h5") as tmp:
            with h5py.File(tmp.name, "w") as h5:
                h5.create_dataset("time_s", data=[0.0, 0.001])
            with h5py.File(tmp.name, "r") as h5:
                n = int(h5["time_s"].shape[0])
        return ComponentCheck(
            name="hdf5",
            status=STATUS_OK,
            module=module_name,
            version=getattr(h5py, "__version__", None),
            metrics={"roundtrip_samples": n},
        )
    except Exception as exc:
        return _failed("hdf5", module_name, exc)


def _check_opencascade() -> ComponentCheck:
    candidates = (
        ("OCP.BRepPrimAPI", "OCP", "BRepPrimAPI_MakeBox"),
        ("OCC.Core.BRepPrimAPI", "pythonocc-core", "BRepPrimAPI_MakeBox"),
    )
    import_errors: dict[str, str] = {}

    for module_name, label, maker_name in candidates:
        try:
            brep = importlib.import_module(module_name)
        except Exception as exc:
            import_errors[module_name] = _short_exc(exc)
            continue

        try:
            maker = getattr(brep, maker_name)
            shape = maker(0.01, 0.02, 0.03).Shape()
            return ComponentCheck(
                name="opencascade",
                status=STATUS_OK,
                module=module_name,
                version=_module_version(label),
                metrics={
                    "kernel": label,
                    "box_shape_type": type(shape).__name__,
                },
            )
        except Exception as exc:
            return _failed("opencascade", module_name, exc)

    return ComponentCheck(
        name="opencascade",
        status=STATUS_UNAVAILABLE,
        detail="No OpenCascade Python binding importable.",
        metrics={"import_errors": import_errors},
    )


def _check_gmsh() -> ComponentCheck:
    module_name = "gmsh"
    try:
        gmsh = importlib.import_module(module_name)
    except Exception as exc:
        return _unavailable("gmsh", module_name, exc)

    initialized = False
    try:
        try:
            gmsh.initialize(["-nopopup"])
        except TypeError:
            gmsh.initialize()
        initialized = True
        gmsh.model.add("mech_bench_oracle_smoke")
        try:
            gmsh.option.setNumber("General.Terminal", 0)
        except Exception:
            pass
        gmsh.model.occ.addBox(0.0, 0.0, 0.0, 0.01, 0.01, 0.01)
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.generate(3)
        node_tags, _, _ = gmsh.model.mesh.getNodes()
        element_types, element_tags, _ = gmsh.model.mesh.getElements(3)
        element_count = sum(len(tags) for tags in element_tags)
        return ComponentCheck(
            name="gmsh",
            status=STATUS_OK,
            module=module_name,
            version=getattr(gmsh, "__version__", None),
            metrics={
                "node_count": int(len(node_tags)),
                "element_type_count": int(len(element_types)),
                "volume_element_count": int(element_count),
            },
        )
    except Exception as exc:
        return _failed("gmsh", module_name, exc)
    finally:
        if initialized:
            try:
                gmsh.finalize()
            except Exception:
                pass


def _check_chrono() -> ComponentCheck:
    import_errors: dict[str, str] = {}
    chrono = None
    module_name = None
    for candidate in ("pychrono.core", "pychrono"):
        try:
            chrono = importlib.import_module(candidate)
            module_name = candidate
            break
        except Exception as exc:
            import_errors[candidate] = _short_exc(exc)

    if chrono is None or module_name is None:
        return ComponentCheck(
            name="chrono",
            status=STATUS_UNAVAILABLE,
            detail="PyChrono is not importable.",
            metrics={"import_errors": import_errors},
        )

    try:
        system_cls = (
            getattr(chrono, "ChSystemNSC", None)
            or getattr(chrono, "ChSystemSMC", None)
        )
        if system_cls is None:
            raise AttributeError("missing ChSystemNSC/ChSystemSMC")
        system = system_cls()

        body_cls = getattr(chrono, "ChBody", None)
        if body_cls is not None:
            body = body_cls()
            if hasattr(body, "SetFixed"):
                body.SetFixed(True)
            _call_first(system, ("Add", "AddBody"), body)

        step = getattr(system, "DoStepDynamics", None)
        if step is None:
            raise AttributeError("missing DoStepDynamics")
        step(1.0e-4)
        time_s = _call_first(system, ("GetChTime", "GetTime"), default=0.0)

        return ComponentCheck(
            name="chrono",
            status=STATUS_OK,
            module=module_name,
            version=_chrono_version(chrono),
            metrics={"stepped_time_s": float(time_s or 0.0)},
        )
    except Exception as exc:
        return _failed("chrono", module_name, exc)


def _call_first(obj: Any, names: tuple[str, ...], *args: Any,
                default: Any = None) -> Any:
    for name in names:
        fn = getattr(obj, name, None)
        if callable(fn):
            return fn(*args)
    if default is not None:
        return default
    raise AttributeError(f"missing any of {names!r}")


def _chrono_version(module: Any) -> str | None:
    for attr in (
        "CHRONO_VERSION",
        "ChronoVersion",
        "__version__",
    ):
        value = getattr(module, attr, None)
        if callable(value):
            try:
                value = value()
            except Exception:
                continue
        if value is not None:
            return str(value)
    return None


def _module_version(module_name: str) -> str | None:
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return None
    return getattr(module, "__version__", None)


def _unavailable(name: str, module: str, exc: BaseException) -> ComponentCheck:
    return ComponentCheck(
        name=name,
        status=STATUS_UNAVAILABLE,
        module=module,
        detail=_short_exc(exc),
    )


def _failed(name: str, module: str, exc: BaseException) -> ComponentCheck:
    return ComponentCheck(
        name=name,
        status=STATUS_FAILED,
        module=module,
        detail=_short_exc(exc),
        metrics={"traceback_tail": traceback.format_exc().splitlines()[-3:]},
    )


def _short_exc(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _component_check(name: str) -> ComponentCheck:
    checkers = {
        "numpy": _check_numpy,
        "hdf5": _check_hdf5,
        "opencascade": _check_opencascade,
        "gmsh": _check_gmsh,
        "chrono": _check_chrono,
    }
    try:
        checker = checkers[name]
    except KeyError as exc:
        raise SystemExit(f"unknown component {name!r}") from exc
    return checker()


def _main() -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="python -m mech_bench.oracle.smoke")
    parser.add_argument("--component", choices=REQUIRED_COMPONENTS)
    args = parser.parse_args()
    if args.component:
        result = _component_check(args.component)
        print(_JSON_BEGIN)
        print(json.dumps(result.to_dict(), indent=2, default=str))
        print(_JSON_END)
        return 0
    print(json.dumps(run_oracle_smoke(), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
