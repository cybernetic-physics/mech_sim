#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-sc}"
REMOTE_ROOT="${REMOTE_ROOT:-/matx/u/knatalia/corl_mechanism_repair_physics}"
JOB_NAME="${JOB_NAME:-corl_mech_phys_pf}"
ACCOUNT="${ACCOUNT:-matx}"
PARTITION="${PARTITION:-matx}"
QOS="${QOS:-normal}"
GRES="${GRES:-}"
CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
MEM="${MEM:-64G}"
TIME="${TIME:-12:00:00}"
OUT_DIR="${OUT_DIR:-runs/mechanism_repair_physics_final}"
TASKS_PER_FAMILY="${TASKS_PER_FAMILY:-10}"
BASE_SEED="${BASE_SEED:-20260610}"
SPLIT_SEED="${SPLIT_SEED:-20260610}"
OVERWRITE="${OVERWRITE:-1}"
VALIDATE_LEVEL3="${VALIDATE_LEVEL3:-1}"
ENABLE_CHRONO_ENV="${ENABLE_CHRONO_ENV:-1}"
CHRONO_CONDA_EXE="${CHRONO_CONDA_EXE:-/matx/u/knatalia/miniconda3/bin/conda}"
CHRONO_ENV_PREFIX="${CHRONO_ENV_PREFIX:-$REMOTE_ROOT/chrono_env}"
CHRONO_CONDA_PKGS_DIR="${CHRONO_CONDA_PKGS_DIR:-$REMOTE_ROOT/conda_pkgs_chrono}"

usage() {
  cat <<EOF
Usage: $0 [--submit]

Stages the current worktree to MATX and writes a Slurm job that runs the
MechanismRepair-Physics benchmark preflight from goals.md.

Useful overrides:
  REMOTE_HOST=$REMOTE_HOST
  REMOTE_ROOT=$REMOTE_ROOT
  OUT_DIR=$OUT_DIR
  TASKS_PER_FAMILY=$TASKS_PER_FAMILY
  VALIDATE_LEVEL3=$VALIDATE_LEVEL3
  ENABLE_CHRONO_ENV=$ENABLE_CHRONO_ENV
  CHRONO_CONDA_EXE=$CHRONO_CONDA_EXE
  CHRONO_ENV_PREFIX=$CHRONO_ENV_PREFIX

By default this stages files and writes the Slurm script. Pass --submit to
call sbatch on the remote host.
EOF
}

submit=0
case "${1:-}" in
  --submit)
    submit=1
    ;;
  -h|--help)
    usage
    exit 0
    ;;
  "")
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
remote_repo="$REMOTE_ROOT/repo"
remote_logs="$REMOTE_ROOT/logs"
remote_sbatch="$REMOTE_ROOT/run_mechanism_repair_physics_preflight.sbatch"

rsync_excludes=(
  --exclude .git/
  --exclude .venv/
  --exclude .mypy_cache/
  --exclude .pytest_cache/
  --exclude __pycache__/
  --exclude runs/
  --exclude .external/
)

ssh "$REMOTE_HOST" "mkdir -p '$remote_repo' '$remote_logs'"
rsync -az --delete "${rsync_excludes[@]}" "$repo_root/" "$REMOTE_HOST:$remote_repo/"

tmp_sbatch="$(mktemp)"
{
  cat <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --account=$ACCOUNT
#SBATCH --partition=$PARTITION
#SBATCH --qos=$QOS
#SBATCH --cpus-per-task=$CPUS_PER_TASK
#SBATCH --mem=$MEM
#SBATCH --time=$TIME
#SBATCH --output=$remote_logs/%x-%j.out
#SBATCH --error=$remote_logs/%x-%j.err
EOF
  if [[ -n "$GRES" ]]; then
    printf '#SBATCH --gres=%s\n' "$GRES"
  fi
  cat <<EOF

set -euo pipefail

cd "$remote_repo"
export PYTHONPATH="$remote_repo:\${PYTHONPATH:-}"
export HF_HOME="\${HF_HOME:-$REMOTE_ROOT/hf_home}"
export UV_CACHE_DIR="\${UV_CACHE_DIR:-$REMOTE_ROOT/uv_cache}"
export XDG_CACHE_HOME="\${XDG_CACHE_HOME:-$REMOTE_ROOT/xdg_cache}"
export HOME="\${JOB_HOME:-$REMOTE_ROOT/home}"
mkdir -p "\$HF_HOME" "\$UV_CACHE_DIR" "\$XDG_CACHE_HOME" "\$HOME"

if ! command -v uv >/dev/null 2>&1; then
  python3 -m venv "$REMOTE_ROOT/uv_bootstrap"
  "$REMOTE_ROOT/uv_bootstrap/bin/python" -m pip install --upgrade pip uv
  export PATH="$REMOTE_ROOT/uv_bootstrap/bin:\$PATH"
fi

uv sync
repo_python="$remote_repo/.venv/bin/python"

if [[ "$ENABLE_CHRONO_ENV" == "1" ]]; then
  if [[ ! -x "$CHRONO_CONDA_EXE" ]]; then
    echo "Chrono conda executable not found: $CHRONO_CONDA_EXE" >&2
    exit 1
  fi
  if [[ ! -x "$CHRONO_ENV_PREFIX/bin/python" ]]; then
    env \\
      CONDA_PKGS_DIRS="$CHRONO_CONDA_PKGS_DIR" \\
      XDG_CACHE_HOME="$REMOTE_ROOT/xdg_cache" \\
      "$CHRONO_CONDA_EXE" create -y -p "$CHRONO_ENV_PREFIX" \\
      --override-channels \\
      -c projectchrono -c conda-forge python=3.13 pychrono numpy
  fi
  export MECH_BENCH_CHRONO_PYTHON="$CHRONO_ENV_PREFIX/bin/python"
fi

"\$repo_python" - <<'PY'
from mech_bench.adapters.chrono_contact import chrono_diagnostic
diag = chrono_diagnostic()
print("chrono diagnostic", diag)
if diag["status"] != "available":
    raise SystemExit(f"chrono_contact unavailable: {diag}")
PY

prepare_args=(
  scripts/prepare_mechanism_repair_physics_benchmark.py
  --out-dir "$OUT_DIR"
  --tasks-per-family "$TASKS_PER_FAMILY"
  --base-seed "$BASE_SEED"
  --split-seed "$SPLIT_SEED"
)
if [[ "$OVERWRITE" == "1" ]]; then
  prepare_args+=(--overwrite)
fi
if [[ "$VALIDATE_LEVEL3" != "1" ]]; then
  prepare_args+=(--skip-level3-validation)
fi

uv run python "\${prepare_args[@]}"
uv run python -m json.tool "$OUT_DIR/claim_audit.json"
EOF
} >"$tmp_sbatch"

scp "$tmp_sbatch" "$REMOTE_HOST:$remote_sbatch"
rm -f "$tmp_sbatch"

echo "Wrote $REMOTE_HOST:$remote_sbatch"
if [[ "$submit" == "1" ]]; then
  ssh "$REMOTE_HOST" "sbatch '$remote_sbatch'"
else
  echo "Not submitted. Re-run with --submit to call sbatch."
fi
