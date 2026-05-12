"""Untrusted submission runner.

This module exists so that an agent-supplied ``design.py`` is NEVER
imported in the trusted evaluator process. The evaluator spawns this
file as a subprocess (preferably with ``python -I``); the subprocess
loads the design module, calls ``build_design(out_dir)``, JSON-serializes
the result, and writes it to ``--result-json``.

The worker intentionally has a tiny import surface — it must not pull
in ``mech_bench.evaluator``, probes, or adapters, because those would
broaden the attack surface of a process that runs untrusted code.

Exit codes:
    0  — wrote a JSON-serializable dict to result-json.
    2  — build_design returned a non-dict.
    3  — build_design returned something that is not JSON-serializable.
    4  — design.py is missing or fails to import.
    5  — build_design raised.
    1  — argument / IO failure (covers everything else).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import traceback
from pathlib import Path


# When launched by absolute file path under ``python -I``, sys.path[0]
# is the script's directory (``mech_bench/``). That puts the package's
# *inside* on sys.path instead of the repo root, which means
# ``import mech_bench`` would resolve to the script's package init only
# by accident, and submodule imports break. Replace sys.path[0] with the
# repo root so ``mech_bench`` is importable as a real package — which
# matters for raw checkouts that did not run ``pip install -e``.
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
if sys.path and sys.path[0] == str(_HERE):
    sys.path[0] = str(_REPO_ROOT)
elif str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _die(code: int, msg: str) -> int:
    print(msg, file=sys.stderr)
    return code


def _load_module(design_py: Path):
    spec = importlib.util.spec_from_file_location(
        "_mech_submission_inproc", design_py)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not build a module spec for {design_py}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="mech_bench.submission_worker")
    p.add_argument("--design-py", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--result-json", required=True)
    args = p.parse_args(argv)

    design_py = Path(args.design_py)
    out_dir = Path(args.out_dir)
    result_json = Path(args.result_json)

    if not design_py.is_file():
        return _die(4, f"design.py not found: {design_py}")
    out_dir.mkdir(parents=True, exist_ok=True)
    result_json.parent.mkdir(parents=True, exist_ok=True)

    try:
        mod = _load_module(design_py)
    except Exception:
        traceback.print_exc(file=sys.stderr)
        return _die(4, "failed to import design.py")

    if not hasattr(mod, "build_design"):
        return _die(4, "design.py does not define build_design(out_dir)")

    try:
        raw = mod.build_design(out_dir)
    except Exception:
        traceback.print_exc(file=sys.stderr)
        return _die(5, "build_design raised")

    if not isinstance(raw, dict):
        return _die(
            2,
            f"build_design must return a dict, got {type(raw).__name__}",
        )

    try:
        # ``allow_nan=False`` keeps the trust boundary strict: NaN/Inf
        # in the submission IR are surfaced here, not silently passed
        # to the evaluator.
        payload = json.dumps(raw, allow_nan=False, default=_json_default)
    except (TypeError, ValueError) as e:
        return _die(
            3,
            f"build_design result is not JSON-serializable: {e}",
        )

    try:
        result_json.write_text(payload)
    except OSError as e:
        return _die(1, f"failed to write result json: {e}")

    return 0


def _json_default(obj):
    # Honor a few common shapes that designers may emit but that json
    # can't natively serialize. Tuples already become lists.
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"unsupported type: {type(obj).__name__}")


if __name__ == "__main__":
    sys.exit(main())
