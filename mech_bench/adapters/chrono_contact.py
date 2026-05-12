"""PyChrono contact-dynamics adapter (skeleton).

Real Chrono integration is gated on the optional ``pychrono`` package
being importable. When it's absent, this module still defines the
adapter class so its existence is discoverable, but the adapter is NOT
registered and the dispatcher will surface CAPABILITY_UNAVAILABLE for
any probe that depends on its capabilities.

The adapter supports two execution modes:

1. **In-process** (default): Chrono runs inside the evaluator process.
2. **Subprocess** (``subprocess=True``): Chrono runs in a separate
   Python process, allowing builds that need a distinct conda env.

Outputs follow the canonical SimOutput shape — same keys as
``fake_contact_oracle`` so probes are simulator-agnostic.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from mech_bench.adapters import SimAdapter, register_adapter
from mech_bench.probes import Capability
from mech_bench.schema import DesignIR

_LOG = logging.getLogger("mech_bench.adapters.chrono_contact")


# --------------------------------------------------------------------- #
# Availability detection                                                #
# --------------------------------------------------------------------- #


def _probe_pychrono() -> tuple[bool, str]:
    """Return (available, diagnostic).

    Tries an in-process import. Honors ``MECH_BENCH_CHRONO_PYTHON`` which
    points at a separate python interpreter — if set, we declare
    available iff that interpreter can ``import pychrono``.

    The adapter additionally requires ``mech_bench.adapters._chrono_impl``
    to be present, which is the vendor-out runner module. Without it,
    we register nothing and the dispatcher correctly surfaces
    CAPABILITY_UNAVAILABLE.
    """
    alt_python = os.environ.get("MECH_BENCH_CHRONO_PYTHON")
    if alt_python:
        if not Path(alt_python).exists():
            return False, (
                f"MECH_BENCH_CHRONO_PYTHON is set to {alt_python!r}, "
                f"but that path does not exist."
            )
        try:
            proc = subprocess.run(
                [alt_python, "-c", "import pychrono"],
                capture_output=True, text=True, timeout=10,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as e:
            return False, f"subprocess probe failed: {e}"
        if proc.returncode != 0:
            return False, (
                f"{alt_python} cannot import pychrono: "
                f"{(proc.stderr or '').strip()[-200:]}"
            )
        return True, f"pychrono available via {alt_python} (subprocess)"
    try:
        import pychrono  # type: ignore[import-not-found]  # noqa: F401
    except ImportError as e:
        return False, f"pychrono not importable: {e}"
    try:
        from mech_bench.adapters import _chrono_impl  # noqa: F401
    except ImportError:
        return False, (
            "pychrono is importable but the chrono runner shim "
            "mech_bench.adapters._chrono_impl is not provided; the "
            "Chrono runner is intentionally vendored out to keep the "
            "base install lean."
        )
    return True, "pychrono available (in-process)"


CHRONO_AVAILABLE, _DIAGNOSTIC = _probe_pychrono()


# --------------------------------------------------------------------- #
# Adapter class                                                         #
# --------------------------------------------------------------------- #


class ChronoContactAdapter(SimAdapter):
    """Capability-tagged contact-dynamics adapter backed by PyChrono.

    Registered only when ``pychrono`` is importable. When absent, the
    dispatcher cannot find this adapter and probes requiring
    CONTACT_FORCES surface CAPABILITY_UNAVAILABLE.
    """

    type_name = "chrono_contact"
    capabilities_provided = frozenset({
        Capability.RIGID_BODY_DYNAMICS,
        Capability.CONTACT_FORCES,
        Capability.JOINT_CONSTRAINTS,
        Capability.MOTOR_DRIVES,
        Capability.LOAD_TORQUES,
        Capability.POSE_TRACES,
        Capability.MESH_OVERLAP,
    })
    cost_tier = 100

    def run(
        self,
        ir: DesignIR,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        if not CHRONO_AVAILABLE:
            # This path is impossible during normal operation because
            # the adapter only registers when CHRONO_AVAILABLE; we keep
            # the guard so direct callers see a clear error.
            return _capability_unavailable_payload(_DIAGNOSTIC)

        use_subprocess = bool(
            config.get("subprocess", False)
            or os.environ.get("MECH_BENCH_CHRONO_PYTHON")
        )
        if use_subprocess:
            return _run_chrono_subprocess(ir, config)
        return _run_chrono_inproc(ir, config)


def _capability_unavailable_payload(reason: str) -> dict[str, Any]:
    return {
        "time_s": np.zeros(0, dtype=float),
        "joint_positions": {},
        "joint_velocities": {},
        "contact_forces": {},
        "penetration": {},
        "body_poses": {},
        "scalar_metrics": {},
        "metadata": {
            "adapter": "chrono_contact",
            "simulator": "pychrono",
            "preflight_issues": [reason],
        },
        "__capability_unavailable__": True,
    }


def _run_chrono_inproc(
    ir: DesignIR,
    config: dict[str, Any],
) -> dict[str, Any]:
    """In-process Chrono execution.

    The real implementation builds the scene, ticks the solver, and
    records traces. The skeleton stub here delegates to the canonical
    chrono runner module which is provided externally by the operator
    when wiring up the optional dependency.
    """
    try:
        from mech_bench.adapters import _chrono_impl  # type: ignore[import-not-found]
    except ImportError:
        return _capability_unavailable_payload(
            "pychrono is importable but mech_bench.adapters._chrono_impl "
            "is not provided. The Chrono runner is intentionally vendored "
            "out of this repo to keep the base install lean — copy your "
            "phys-sim _chrono_mesh_runner equivalent into this package "
            "to enable in-process execution."
        )
    return _chrono_impl.run(ir, config)  # type: ignore[no-any-return]


def _run_chrono_subprocess(
    ir: DesignIR,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Run Chrono in a subprocess (alternative interpreter)."""
    alt_python = os.environ.get(
        "MECH_BENCH_CHRONO_PYTHON", sys.executable)
    max_wall_s = float(config.get("max_wall_s", 600.0))

    with tempfile.TemporaryDirectory() as td:
        ir_path = Path(td) / "ir.json"
        cfg_path = Path(td) / "config.json"
        out_path = Path(td) / "output.json"
        ir_path.write_text(json.dumps(_ir_to_dict(ir), default=str))
        cfg_path.write_text(json.dumps(config, default=str))

        runner = (
            "import json, sys\n"
            "from mech_bench.adapters import _chrono_impl\n"
            "from mech_bench.schema import DesignIR\n"
            "ir_path, cfg_path, out_path = sys.argv[1:4]\n"
            "ir = DesignIR.from_dict(json.loads(open(ir_path).read()))\n"
            "cfg = json.loads(open(cfg_path).read())\n"
            "result = _chrono_impl.run(ir, cfg)\n"
            "# numpy arrays are not JSON-serializable; flatten to lists\n"
            "def _enc(x):\n"
            "    if hasattr(x, 'tolist'):\n"
            "        return x.tolist()\n"
            "    raise TypeError(type(x))\n"
            "open(out_path, 'w').write(json.dumps(result, default=_enc))\n"
        )
        try:
            proc = subprocess.run(
                [alt_python, "-c", runner,
                 str(ir_path), str(cfg_path), str(out_path)],
                capture_output=True, text=True,
                timeout=max_wall_s, check=False,
            )
        except subprocess.TimeoutExpired:
            return _capability_unavailable_payload(
                f"chrono subprocess exceeded max_wall_s={max_wall_s}"
            )
        if proc.returncode != 0:
            return _capability_unavailable_payload(
                f"chrono subprocess exited {proc.returncode}: "
                f"{(proc.stderr or '').strip()[-400:]}"
            )
        if not out_path.exists():
            return _capability_unavailable_payload(
                "chrono subprocess produced no output JSON")
        try:
            raw = json.loads(out_path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            return _capability_unavailable_payload(
                f"chrono subprocess JSON unparseable: {e}")
        return _coerce_numpy(raw)


def _ir_to_dict(ir: DesignIR) -> dict[str, Any]:
    return {
        "schema_version": ir.schema_version,
        "parts": [vars(p) for p in ir.parts],
        "joints": [vars(j) for j in ir.joints],
        "ports": {k: vars(v) for k, v in ir.ports.items()},
        "params": dict(ir.params or {}),
    }


def _coerce_numpy(blob: dict[str, Any]) -> dict[str, Any]:
    def _to_arr(d: Any) -> Any:
        if isinstance(d, list):
            try:
                return np.asarray(d, dtype=float)
            except (TypeError, ValueError):
                return d
        if isinstance(d, dict):
            return {k: _to_arr(v) for k, v in d.items()}
        return d
    out: dict[str, Any] = {}
    for k, v in blob.items():
        if k in {"time_s"}:
            out[k] = np.asarray(v, dtype=float) if isinstance(
                v, list) else v
        elif k in {"joint_positions", "joint_velocities", "contact_forces",
                   "penetration", "body_poses", "port_traces"}:
            out[k] = (
                {kk: np.asarray(vv, dtype=float) for kk, vv in v.items()}
                if isinstance(v, dict) else v
            )
        else:
            out[k] = v
    return out


# --------------------------------------------------------------------- #
# Registration (only when pychrono is importable)                       #
# --------------------------------------------------------------------- #


if CHRONO_AVAILABLE:
    register_adapter(ChronoContactAdapter)
else:  # pragma: no cover - exercised by absence of pychrono
    _LOG.info(
        "chrono_contact adapter not registered: %s. Tasks that require "
        "CONTACT_FORCES will surface CAPABILITY_UNAVAILABLE until "
        "pychrono is installed or MECH_BENCH_CHRONO_PYTHON points at a "
        "compatible interpreter.",
        _DIAGNOSTIC,
    )


def chrono_diagnostic() -> dict[str, Any]:
    """Structured diagnostic for the chrono backend state.

    Returns a dict shaped for the dashboard / CLI list-adapters and for
    tests that want to skip when chrono is unavailable. The
    ``runner_status`` is one of ``"skeleton_only"`` (PyChrono is
    importable but the real runner isn't ported) or ``"ready"`` (both
    PyChrono and ``_chrono_impl`` are importable).
    """
    pychrono_ok = False
    try:
        import pychrono  # type: ignore[import-not-found]  # noqa: F401
        pychrono_ok = True
    except ImportError:
        pychrono_ok = False
    impl_ok = False
    try:
        from mech_bench.adapters import _chrono_impl  # noqa: F401
        impl_ok = True
    except ImportError:
        impl_ok = False

    if CHRONO_AVAILABLE:
        runner_status = "ready"
        status = "available"
    elif pychrono_ok and not impl_ok:
        runner_status = "skeleton_only"
        status = "unavailable"
    else:
        runner_status = "missing_dependency"
        status = "unavailable"

    return {
        "adapter": "chrono_contact",
        "status": status,
        "reason": _DIAGNOSTIC,
        "pychrono_importable": pychrono_ok,
        "_chrono_impl_importable": impl_ok,
        "runner_status": runner_status,
    }


# Expose a stable predicate that tests can use to skip integration cases.
def is_chrono_available() -> bool:
    return CHRONO_AVAILABLE


# Silence shutil import warning when unused.
_ = shutil
