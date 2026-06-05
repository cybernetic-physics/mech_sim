#!/usr/bin/env bash
# Enable the real high-fidelity Chrono oracle on capable hardware.
#
# PyChrono ships via conda (channel `projectchrono`); there is NO macOS-ARM
# PyPI wheel, so `pip install pychrono` will not work. This script provisions
# the native solver stack the same way docker/solver/environment.yml does, then
# verifies the runner end-to-end. Run it on Linux x86_64 (or inside the solver
# container); it is expected to be unavailable on an Apple-Silicon dev box.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${MECH_BENCH_CHRONO_ENV:-mech-chrono}"
ENV_YML="${REPO_ROOT}/docker/solver/environment.yml"

# Pick a conda-family front-end.
CONDA_BIN=""
for c in micromamba mamba conda; do
  if command -v "$c" >/dev/null 2>&1; then CONDA_BIN="$c"; break; fi
done

if [[ -z "${CONDA_BIN}" ]]; then
  cat >&2 <<EOF
ERROR: no conda/mamba/micromamba found on PATH.
PyChrono is only distributed through conda (channel 'projectchrono').
Install Miniforge (https://github.com/conda-forge/miniforge) or use the
Docker path instead:

    docker build -f "${REPO_ROOT}/docker/solver/Dockerfile" "${REPO_ROOT}"
    # or:  scripts/solver_smoke.sh
EOF
  exit 2
fi

echo ">> Provisioning '${ENV_NAME}' from ${ENV_YML} via ${CONDA_BIN}"
"${CONDA_BIN}" env create -n "${ENV_NAME}" -f "${ENV_YML}" 2>/dev/null \
  || "${CONDA_BIN}" env update -n "${ENV_NAME}" -f "${ENV_YML}"

CHRONO_PY="$("${CONDA_BIN}" run -n "${ENV_NAME}" which python)"
echo ">> Chrono python: ${CHRONO_PY}"

echo ">> Verifying pychrono + the in-repo runner import"
"${CONDA_BIN}" run -n "${ENV_NAME}" python - <<'PY'
import importlib, sys
ok = True
try:
    import pychrono  # noqa: F401
    print("pychrono: OK")
except Exception as e:  # pragma: no cover - environment-dependent
    print("pychrono: FAIL", e); ok = False
try:
    importlib.import_module("mech_bench.adapters._chrono_impl")
    print("_chrono_impl: OK")
except Exception as e:  # pragma: no cover
    print("_chrono_impl: FAIL", e); ok = False
sys.exit(0 if ok else 1)
PY

cat <<EOF

>> Real oracle enabled. Point mech-bench at this interpreter for contact tasks:

    export MECH_BENCH_CHRONO_PYTHON="${CHRONO_PY}"
    python -m mech_bench oracle-smoke --require-real
    python -m mech_bench evaluate --task <contact_task> --submission <sub> --mode oracle

EOF
