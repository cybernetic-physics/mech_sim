#!/usr/bin/env python3
"""Create and link an ABI-matched Project Chrono env for this checkout."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import site
import subprocess
import sys
from pathlib import Path


REQUIRED_SYMBOLS = ("ChSystem", "ChSystemSMC", "ChSystemNSC")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prefix",
        default=None,
        help="Chrono env prefix; defaults to .external/chrono_env_pyXY",
    )
    parser.add_argument(
        "--python-version",
        default=f"{sys.version_info.major}.{sys.version_info.minor}",
        help="Python ABI version for the Chrono env",
    )
    parser.add_argument(
        "--skip-create",
        action="store_true",
        help="do not create the conda env; only verify/link an existing prefix",
    )
    parser.add_argument(
        "--link-current-venv",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="write a .pth file so the current interpreter can import pychrono",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    py_tag = "py" + args.python_version.replace(".", "")
    prefix = (
        Path(args.prefix).expanduser().resolve()
        if args.prefix
        else root / ".external" / f"chrono_env_{py_tag}"
    )
    chrono_python = prefix / "bin" / "python"

    if not args.skip_create and not chrono_python.exists():
        micromamba = find_micromamba(root)
        if micromamba is None:
            raise SystemExit(
                "micromamba not found; install it or pass --skip-create with "
                "an existing --prefix"
            )
        subprocess.run(
            [
                str(micromamba),
                "create",
                "-y",
                "-p",
                str(prefix),
                "--override-channels",
                "-c",
                "projectchrono",
                "-c",
                "conda-forge",
                f"python={args.python_version}",
                "pychrono",
                "numpy",
            ],
            check=True,
            cwd=str(root),
        )

    verify_pychrono(chrono_python)
    chrono_site = site_packages_for(prefix, args.python_version)
    linked_path = None
    if args.link_current_venv:
        if args.python_version != f"{sys.version_info.major}.{sys.version_info.minor}":
            raise SystemExit(
                "--link-current-venv requires --python-version to match the "
                "running interpreter"
            )
        linked_path = link_current_interpreter(chrono_site)
        verify_pychrono(Path(sys.executable))

    print(json.dumps({
        "chrono_python": str(chrono_python),
        "chrono_site_packages": str(chrono_site),
        "linked_pth": str(linked_path) if linked_path else None,
        "pychrono_importable": True,
    }, indent=2, sort_keys=True))
    return 0


def find_micromamba(root: Path) -> Path | None:
    env_path = os.environ.get("MICROMAMBA_EXE") or os.environ.get("MAMBA_EXE")
    candidates = [
        Path(env_path).expanduser() if env_path else None,
        root / ".external" / "bin" / "micromamba",
        root / ".external" / "micromamba" / "bin" / "micromamba",
        Path(shutil.which("micromamba") or ""),
        Path(shutil.which("mamba") or ""),
    ]
    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate
    return None


def site_packages_for(prefix: Path, python_version: str) -> Path:
    path = prefix / "lib" / f"python{python_version}" / "site-packages"
    if not path.exists():
        raise SystemExit(f"Chrono site-packages path does not exist: {path}")
    return path


def link_current_interpreter(chrono_site: Path) -> Path:
    site_paths = site.getsitepackages()
    if not site_paths:
        raise SystemExit("current interpreter has no writable site-packages")
    pth = Path(site_paths[0]) / "projectchrono_external.pth"
    text = f"{chrono_site.resolve()}\n"
    if pth.exists() and pth.read_text(encoding="utf-8") == text:
        return pth
    pth.write_text(text, encoding="utf-8")
    return pth


def verify_pychrono(python: Path) -> None:
    if not python.exists():
        raise SystemExit(f"python executable does not exist: {python}")
    probe = (
        "import pychrono; "
        "missing=[n for n in "
        f"{REQUIRED_SYMBOLS!r} if not hasattr(pychrono, n)]; "
        "assert not missing, missing"
    )
    subprocess.run([str(python), "-c", probe], check=True)


if __name__ == "__main__":
    raise SystemExit(main())
