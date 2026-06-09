"""Project Chrono interpreter discovery helpers."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DISABLE_AUTO_ENV = "MECH_BENCH_DISABLE_CHRONO_AUTO"
CHRONO_PYTHON_ENV = "MECH_BENCH_CHRONO_PYTHON"
CHRONO_ENV_PREFIX_ENV = "MECH_BENCH_CHRONO_ENV"


def chrono_child_env() -> dict[str, str]:
    """Environment for subprocesses that need to import this checkout."""
    env = os.environ.copy()
    env.pop(CHRONO_PYTHON_ENV, None)
    existing = env.get("PYTHONPATH")
    repo = str(REPO_ROOT)
    env["PYTHONPATH"] = repo if not existing else f"{repo}{os.pathsep}{existing}"
    return env


def find_chrono_python() -> Path | None:
    """Return a Python interpreter that can host Project Chrono, if known."""
    explicit = os.environ.get(CHRONO_PYTHON_ENV)
    if explicit:
        return Path(explicit).expanduser()
    if os.environ.get(DISABLE_AUTO_ENV):
        return None
    seen: set[Path] = set()
    for candidate in _candidate_chrono_pythons():
        path = candidate.expanduser()
        if path in seen:
            continue
        seen.add(path)
        if path.exists():
            return path
    return None


def probe_chrono_python(path: Path, *, timeout_s: float = 30.0) -> tuple[bool, str]:
    """Check whether ``path`` imports Project Chrono and this runner."""
    if not path.exists():
        return False, f"chrono python path does not exist: {path}"
    try:
        proc = subprocess.run(
            [
                str(path),
                "-c",
                (
                    "import pychrono; "
                    "assert any(hasattr(pychrono, n) for n in "
                    "('ChSystem','ChSystemSMC','ChSystemNSC')), "
                    "'pychrono is not Project Chrono'; "
                    "import mech_bench.adapters._chrono_impl"
                ),
            ],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
            env=chrono_child_env(),
            cwd=str(REPO_ROOT),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"chrono python probe failed: {exc}"
    if proc.returncode != 0:
        return False, (
            f"{path} cannot import pychrono + mech_bench.adapters._chrono_impl: "
            f"{(proc.stderr or '').strip()[-200:]}"
        )
    return True, f"pychrono + _chrono_impl available via {path} (subprocess)"


def _candidate_chrono_pythons() -> list[Path]:
    prefix = os.environ.get(CHRONO_ENV_PREFIX_ENV)
    candidates: list[Path] = []
    if prefix:
        candidates.extend(_python_bins(Path(prefix).expanduser()))
    for root in _candidate_roots():
        candidates.extend(_python_bins(root / ".external" / "chrono_env"))
        candidates.extend(
            _python_bins(root / ".external" / "micromamba" / "envs" / "mech-chrono")
        )
        candidates.extend(_python_bins(root / "chrono_env"))
    return candidates


def _candidate_roots() -> list[Path]:
    cwd = Path.cwd()
    roots = [
        REPO_ROOT,
        REPO_ROOT.parent,
        cwd,
        cwd.parent,
    ]
    return roots


def _python_bins(prefix: Path) -> list[Path]:
    return [prefix / "bin" / "python", prefix / "bin" / "python3"]
