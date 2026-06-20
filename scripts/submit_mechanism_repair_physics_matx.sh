#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-sc}"
REMOTE_USER="${REMOTE_USER:-knatalia}"
REMOTE_ROOT="${REMOTE_ROOT:-/matx/u/knatalia/corl_mechanism_repair_physics}"
JOB_NAME="${JOB_NAME:-corl_mech_phys}"
MERGE_JOB_NAME="${MERGE_JOB_NAME:-corl_mech_phys_merge}"
ANALYSIS_JOB_NAME="${ANALYSIS_JOB_NAME:-corl_mech_phys_analysis}"
JOB_RUNTIME_ROOT="${JOB_RUNTIME_ROOT:-auto}"
ACCOUNT="${ACCOUNT:-matx}"
PARTITION="${PARTITION:-matx}"
QOS="${QOS:-normal}"
GRES="${GRES:-gpu:l40s:1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
MEM="${MEM:-160G}"
TIME="${TIME:-4-00:00:00}"
MERGE_CPUS_PER_TASK="${MERGE_CPUS_PER_TASK:-8}"
MERGE_MEM="${MERGE_MEM:-48G}"
MERGE_TIME="${MERGE_TIME:-04:00:00}"
ANALYSIS_CPUS_PER_TASK="${ANALYSIS_CPUS_PER_TASK:-8}"
ANALYSIS_MEM="${ANALYSIS_MEM:-48G}"
ANALYSIS_TIME="${ANALYSIS_TIME:-04:00:00}"
NUM_SHARDS="${NUM_SHARDS:-24}"
ARRAY_CONCURRENCY="${ARRAY_CONCURRENCY:-1}"
ALLOW_HIGH_CLUSTER_USAGE="${ALLOW_HIGH_CLUSTER_USAGE:-0}"
MAX_ARRAY_TASKS="${MAX_ARRAY_TASKS:-1}"
SHARD_INDICES="${SHARD_INDICES:-}"
SUBMIT_DEPENDENTS="${SUBMIT_DEPENDENTS:-auto}"
RESTAGE_REMOTE_REPO="${RESTAGE_REMOTE_REPO:-auto}"
REFRESH_REMOTE_CODE="${REFRESH_REMOTE_CODE:-auto}"
ALLOW_DESTRUCTIVE_RESTAGE="${ALLOW_DESTRUCTIVE_RESTAGE:-0}"
OUT_DIR="${OUT_DIR:-runs/mechanism_repair_physics_final}"
METHODS="${METHODS-}"
SPLITS="${SPLITS-}"
ANTI_SHORTCUT_SPLITS="${ANTI_SHORTCUT_SPLITS-__default__}"
EVAL_SEEDS="${EVAL_SEEDS-}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-Coder-1.5B-Instruct}"
ROLLOUT_BACKEND="${ROLLOUT_BACKEND:-transformers_local}"
LOCAL_DEVICE="${LOCAL_DEVICE:-cuda}"
LOCAL_TORCH_DTYPE="${LOCAL_TORCH_DTYPE:-bfloat16}"
LOCAL_TRUST_REMOTE_CODE="${LOCAL_TRUST_REMOTE_CODE:-0}"
SGLANG_MODEL="${SGLANG_MODEL:-$BASE_MODEL}"
SGLANG_PORT="${SGLANG_PORT:-30000}"
SGLANG_PIP_SPEC="${SGLANG_PIP_SPEC:-sglang==0.5.9}"
SGLANG_PIP_EXTRA="${SGLANG_PIP_EXTRA:-ninja}"
SGLANG_VENV="${SGLANG_VENV:-auto}"
SGLANG_TP="${SGLANG_TP:-1}"
SGLANG_MEM_FRAC="${SGLANG_MEM_FRAC:-0.82}"
SGLANG_CTX="${SGLANG_CTX:-16384}"
SGLANG_MAX_REQS="${SGLANG_MAX_REQS:-4}"
SGLANG_CUDA_VISIBLE_DEVICES="${SGLANG_CUDA_VISIBLE_DEVICES:-}"
TRAIN_CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-}"
SFT_DEVICE_MAP="${SFT_DEVICE_MAP:-none}"
TTRL_DEVICE_MAP="${TTRL_DEVICE_MAP:-none}"
SFT_MAX_STEPS="${SFT_MAX_STEPS:-64}"
SFT_MAX_SEQ_LENGTH="${SFT_MAX_SEQ_LENGTH:-512}"
SFT_LOAD_IN_4BIT="${SFT_LOAD_IN_4BIT:-0}"
SFT_TORCH_DTYPE="${SFT_TORCH_DTYPE:-float32}"
SFT_ATTN_IMPLEMENTATION="${SFT_ATTN_IMPLEMENTATION:-eager}"
SFT_LEARNING_RATE="${SFT_LEARNING_RATE:-1.0e-6}"
SFT_MAX_GRAD_NORM="${SFT_MAX_GRAD_NORM:-0.0}"
SHARED_SFT_ROOT="${SHARED_SFT_ROOT:-$OUT_DIR/shared_sft}"
TTRL_MAX_CONTEXT_TOKENS="${TTRL_MAX_CONTEXT_TOKENS:-4096}"
TTRL_MAX_TOKENS="${TTRL_MAX_TOKENS:-1536}"
TTRL_LOAD_IN_4BIT="${TTRL_LOAD_IN_4BIT:-0}"
TTRL_TORCH_DTYPE="${TTRL_TORCH_DTYPE:-bfloat16}"
TTRL_ATTN_IMPLEMENTATION="${TTRL_ATTN_IMPLEMENTATION:-eager}"
TTRL_KBIT_PREPARE_MODE="${TTRL_KBIT_PREPARE_MODE:-none}"
TTRL_LEARNING_RATE="${TTRL_LEARNING_RATE:-1.0e-6}"
TTRL_MAX_GRAD_NORM="${TTRL_MAX_GRAD_NORM:-0.0}"
TTRL_LORA_RANK="${TTRL_LORA_RANK:-8}"
TTRL_SAVE_ADAPTER_DTYPE="${TTRL_SAVE_ADAPTER_DTYPE:-bfloat16}"
TTRL_ADAPTER_RETENTION="${TTRL_ADAPTER_RETENTION:-full}"
TTRL_GRADIENT_CHECKPOINTING="${TTRL_GRADIENT_CHECKPOINTING:-1}"
TTRL_BF16="${TTRL_BF16:-1}"
AUDIT_RETRIES="${AUDIT_RETRIES:-0}"
EVIDENCE_LAYOUT="${EVIDENCE_LAYOUT:-bundled}"
LIMIT_TASKS="${LIMIT_TASKS:-0}"
RESUME_EXISTING="${RESUME_EXISTING:-1}"
DRY_RUN="${DRY_RUN:-0}"
ENABLE_CHRONO_ENV="${ENABLE_CHRONO_ENV:-1}"
CHRONO_CONDA_EXE="${CHRONO_CONDA_EXE:-/matx/u/knatalia/miniconda3/bin/conda}"
CHRONO_PYTHON_VERSION="${CHRONO_PYTHON_VERSION:-auto}"
CHRONO_ENV_PREFIX="${CHRONO_ENV_PREFIX:-auto}"
CHRONO_CONDA_PKGS_DIR="${CHRONO_CONDA_PKGS_DIR:-auto}"
CHRONO_LINK_CURRENT_VENV="${CHRONO_LINK_CURRENT_VENV:-1}"
SLURM_OUTPUT="${SLURM_OUTPUT:-auto}"
SLURM_ERROR="${SLURM_ERROR:-auto}"
MERGE_SLURM_OUTPUT="${MERGE_SLURM_OUTPUT:-auto}"
MERGE_SLURM_ERROR="${MERGE_SLURM_ERROR:-auto}"
ANALYSIS_SLURM_OUTPUT="${ANALYSIS_SLURM_OUTPUT:-auto}"
ANALYSIS_SLURM_ERROR="${ANALYSIS_SLURM_ERROR:-auto}"
FINALIZE_AUDIT_JSON="${FINALIZE_AUDIT_JSON:-}"
LOCAL_PYTHON="${LOCAL_PYTHON:-python3}"

usage() {
  cat <<EOF
Usage: $0 [--submit] [--finalize-only]

Stages the requested git ref, including the frozen MechanismRepair-Physics
benchmark, to MATX, writes a Slurm array over shard_0000..shard_N, and writes a
dependent merge/audit job.

Use --finalize-only after all required shard outputs are present. It writes and
optionally submits only CPU merge/analysis jobs; it does not submit a GPU array.

Useful overrides:
  REMOTE_HOST=$REMOTE_HOST
  REMOTE_USER=$REMOTE_USER
  REMOTE_ROOT=$REMOTE_ROOT
  JOB_RUNTIME_ROOT=$JOB_RUNTIME_ROOT
  SOURCE_REF=HEAD
  OUT_DIR=$OUT_DIR
  METHODS=$METHODS
  SPLITS=$SPLITS
  ANTI_SHORTCUT_SPLITS=$ANTI_SHORTCUT_SPLITS
  EVAL_SEEDS=$EVAL_SEEDS
  NUM_SHARDS=$NUM_SHARDS
  ARRAY_CONCURRENCY=$ARRAY_CONCURRENCY
  ALLOW_HIGH_CLUSTER_USAGE=$ALLOW_HIGH_CLUSTER_USAGE
  MAX_ARRAY_TASKS=$MAX_ARRAY_TASKS
  SHARD_INDICES=$SHARD_INDICES
  SUBMIT_DEPENDENTS=$SUBMIT_DEPENDENTS
  RESTAGE_REMOTE_REPO=$RESTAGE_REMOTE_REPO
  REFRESH_REMOTE_CODE=$REFRESH_REMOTE_CODE
  ALLOW_DESTRUCTIVE_RESTAGE=$ALLOW_DESTRUCTIVE_RESTAGE
  GRES=$GRES
  BASE_MODEL=$BASE_MODEL
  ROLLOUT_BACKEND=$ROLLOUT_BACKEND
  LOCAL_DEVICE=$LOCAL_DEVICE
  LOCAL_TORCH_DTYPE=$LOCAL_TORCH_DTYPE
  LOCAL_TRUST_REMOTE_CODE=$LOCAL_TRUST_REMOTE_CODE
  SGLANG_MODEL=$SGLANG_MODEL
  SGLANG_VENV=$SGLANG_VENV
  SHARED_SFT_ROOT=$SHARED_SFT_ROOT
  TTRL_ADAPTER_RETENTION=$TTRL_ADAPTER_RETENTION
  AUDIT_RETRIES=$AUDIT_RETRIES
  EVIDENCE_LAYOUT=$EVIDENCE_LAYOUT
  LIMIT_TASKS=$LIMIT_TASKS
  RESUME_EXISTING=$RESUME_EXISTING
  DRY_RUN=$DRY_RUN
  CHRONO_PYTHON_VERSION=$CHRONO_PYTHON_VERSION
  CHRONO_ENV_PREFIX=$CHRONO_ENV_PREFIX
  CHRONO_LINK_CURRENT_VENV=$CHRONO_LINK_CURRENT_VENV
  SLURM_OUTPUT=$SLURM_OUTPUT
  SLURM_ERROR=$SLURM_ERROR
  MERGE_SLURM_OUTPUT=$MERGE_SLURM_OUTPUT
  MERGE_SLURM_ERROR=$MERGE_SLURM_ERROR
  ANALYSIS_SLURM_OUTPUT=$ANALYSIS_SLURM_OUTPUT
  ANALYSIS_SLURM_ERROR=$ANALYSIS_SLURM_ERROR
  FINALIZE_AUDIT_JSON=$FINALIZE_AUDIT_JSON
EOF
}

submit=0
finalize_only=0
while (($#)); do
  case "$1" in
    --submit)
      submit=1
      ;;
    --finalize-only)
      finalize_only=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage >&2
      exit 2
      ;;
  esac
  shift
done

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
remote_repo="$REMOTE_ROOT/repo"
remote_logs="$REMOTE_ROOT/logs"
remote_sbatch="$REMOTE_ROOT/run_mechanism_repair_physics_array.sbatch"
remote_merge_sbatch="$REMOTE_ROOT/run_mechanism_repair_physics_merge.sbatch"
remote_analysis_sbatch="$REMOTE_ROOT/run_mechanism_repair_physics_analysis.sbatch"
remote_out_dir="$remote_repo/$OUT_DIR"
if [[ "$OUT_DIR" == /* ]]; then
  remote_out_dir="$OUT_DIR"
fi
source_ref="${SOURCE_REF:-HEAD}"
source_commit="$(git -C "$repo_root" rev-parse "$source_ref")"
source_commit_short="${source_commit:0:8}"
if [[ "$JOB_RUNTIME_ROOT" == "auto" ]]; then
  JOB_RUNTIME_ROOT="/tmp/$REMOTE_USER/corl_mech_phys_$source_commit_short"
fi
if [[ "$SGLANG_VENV" == "auto" ]]; then
  SGLANG_VENV="$JOB_RUNTIME_ROOT/sglang_venv"
fi
if [[ "$CHRONO_CONDA_PKGS_DIR" == "auto" ]]; then
  CHRONO_CONDA_PKGS_DIR="$REMOTE_ROOT/conda_pkgs_chrono"
fi
if [[ "$SLURM_OUTPUT" == "auto" ]]; then
  SLURM_OUTPUT="$remote_logs/%x-%A_%a.out"
fi
if [[ "$SLURM_ERROR" == "auto" ]]; then
  SLURM_ERROR="$remote_logs/%x-%A_%a.err"
fi
if [[ "$MERGE_SLURM_OUTPUT" == "auto" ]]; then
  MERGE_SLURM_OUTPUT="$remote_logs/%x-%j.out"
fi
if [[ "$MERGE_SLURM_ERROR" == "auto" ]]; then
  MERGE_SLURM_ERROR="$remote_logs/%x-%j.err"
fi
if [[ "$ANALYSIS_SLURM_OUTPUT" == "auto" ]]; then
  ANALYSIS_SLURM_OUTPUT="$remote_logs/%x-%j.out"
fi
if [[ "$ANALYSIS_SLURM_ERROR" == "auto" ]]; then
  ANALYSIS_SLURM_ERROR="$remote_logs/%x-%j.err"
fi
array_end=$((NUM_SHARDS - 1))
if (( NUM_SHARDS < 1 )); then
  echo "NUM_SHARDS must be >= 1" >&2
  exit 2
fi
if [[ ! "$ARRAY_CONCURRENCY" =~ ^[1-9][0-9]*$ ]]; then
  echo "ARRAY_CONCURRENCY must be a positive integer" >&2
  exit 2
fi
if [[ ! "$MAX_ARRAY_TASKS" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_ARRAY_TASKS must be a positive integer" >&2
  exit 2
fi
array_range="0-$array_end"
array_task_count="$NUM_SHARDS"
if [[ -n "$SHARD_INDICES" ]]; then
  if [[ ! "$SHARD_INDICES" =~ ^[0-9]+(-[0-9]+)?(,[0-9]+(-[0-9]+)?)*$ ]]; then
    echo "SHARD_INDICES must use Slurm index/range syntax like 0,3,5-7" >&2
    exit 2
  fi
  array_range="$SHARD_INDICES"
  array_task_count=0
  IFS=',' read -r -a shard_parts <<< "$SHARD_INDICES"
  for shard_part in "${shard_parts[@]}"; do
    if [[ "$shard_part" == *-* ]]; then
      shard_start="${shard_part%-*}"
      shard_stop="${shard_part#*-}"
    else
      shard_start="$shard_part"
      shard_stop="$shard_part"
    fi
    shard_start_num=$((10#$shard_start))
    shard_stop_num=$((10#$shard_stop))
    if (( shard_start_num > shard_stop_num )); then
      echo "SHARD_INDICES range starts after it ends: $shard_part" >&2
      exit 2
    fi
    if (( shard_start_num < 0 || shard_stop_num > array_end )); then
      echo "SHARD_INDICES out of range for NUM_SHARDS=$NUM_SHARDS: $shard_part" >&2
      exit 2
    fi
    array_task_count=$((array_task_count + shard_stop_num - shard_start_num + 1))
  done
fi
if [[ "$SUBMIT_DEPENDENTS" == "auto" ]]; then
  SUBMIT_DEPENDENTS=0
fi
if [[ "$SUBMIT_DEPENDENTS" != "0" && "$SUBMIT_DEPENDENTS" != "1" ]]; then
  echo "SUBMIT_DEPENDENTS must be 0, 1, or auto" >&2
  exit 2
fi
if (( ! finalize_only )) && [[ "$SUBMIT_DEPENDENTS" == "1" ]]; then
  cat >&2 <<EOF
Refusing dependent merge/analysis submission before shard audit.

GPU shard runs must not submit final merge/analysis jobs directly. Run
scripts/plan_mechanism_repair_shard_resume.py locally after shard outputs are
present, then use --finalize-only with FINALIZE_AUDIT_JSON once merge_ready is
true.
EOF
  exit 2
fi
if [[ "$RESTAGE_REMOTE_REPO" == "auto" ]]; then
  if (( finalize_only )); then
    RESTAGE_REMOTE_REPO=0
  elif [[ -n "$SHARD_INDICES" ]]; then
    RESTAGE_REMOTE_REPO=0
  else
    RESTAGE_REMOTE_REPO=1
  fi
fi
if [[ "$RESTAGE_REMOTE_REPO" != "0" && "$RESTAGE_REMOTE_REPO" != "1" ]]; then
  echo "RESTAGE_REMOTE_REPO must be 0, 1, or auto" >&2
  exit 2
fi
if [[ "$REFRESH_REMOTE_CODE" == "auto" ]]; then
  if [[ "$RESTAGE_REMOTE_REPO" == "0" && ! finalize_only ]]; then
    REFRESH_REMOTE_CODE=1
  else
    REFRESH_REMOTE_CODE=0
  fi
fi
if [[ "$REFRESH_REMOTE_CODE" != "0" && "$REFRESH_REMOTE_CODE" != "1" ]]; then
  echo "REFRESH_REMOTE_CODE must be 0, 1, or auto" >&2
  exit 2
fi
if [[ "$ALLOW_DESTRUCTIVE_RESTAGE" != "0" && "$ALLOW_DESTRUCTIVE_RESTAGE" != "1" ]]; then
  echo "ALLOW_DESTRUCTIVE_RESTAGE must be 0 or 1" >&2
  exit 2
fi
if (( finalize_only )) && [[ "$RESTAGE_REMOTE_REPO" == "1" ]]; then
  cat >&2 <<EOF
Refusing finalize-only restage.

--finalize-only is only valid after shard outputs already exist in the remote
run tree. Restaging would delete or replace the evidence that merge/analysis
must consume. Use RESTAGE_REMOTE_REPO=0, or run a normal selected-shard submit
for deliberate GPU reruns.
EOF
  exit 2
fi
if (( finalize_only && submit )); then
  if [[ -z "$FINALIZE_AUDIT_JSON" ]]; then
    cat >&2 <<EOF
Refusing unaudited finalize-only submit.

Run scripts/plan_mechanism_repair_shard_resume.py against the final run tree
and set FINALIZE_AUDIT_JSON to the resulting local JSON report. The report must
show merge_ready=true before CPU merge/analysis jobs may be submitted.
EOF
    exit 2
  fi
  if [[ ! -f "$FINALIZE_AUDIT_JSON" ]]; then
    echo "FINALIZE_AUDIT_JSON does not exist: $FINALIZE_AUDIT_JSON" >&2
    exit 2
  fi
  if ! finalize_audit_check="$("$LOCAL_PYTHON" - "$FINALIZE_AUDIT_JSON" "$NUM_SHARDS" <<'PY'
import json
import sys

path = sys.argv[1]
expected_shards = int(sys.argv[2])
with open(path, "r", encoding="utf-8") as handle:
    report = json.load(handle)

errors = []
if report.get("schema") != "mechanism_repair_physics.shard_resume_plan.v1":
    errors.append("schema is not mechanism_repair_physics.shard_resume_plan.v1")
if report.get("merge_ready") is not True:
    errors.append("merge_ready is not true")
if int(report.get("shard_count", -1)) != expected_shards:
    errors.append(
        f"shard_count={report.get('shard_count')} does not match NUM_SHARDS={expected_shards}"
    )
for field in ("incomplete_shard_count", "missing_rows", "duplicate_rows", "unexpected_rows"):
    if int(report.get(field, -1)) != 0:
        errors.append(f"{field}={report.get(field)}")
if errors:
    print("; ".join(errors))
    raise SystemExit(1)
print("ok")
PY
  )"; then
    cat >&2 <<EOF
Refusing finalize-only submit because the local shard-resume audit is not clean.

FINALIZE_AUDIT_JSON=$FINALIZE_AUDIT_JSON
$finalize_audit_check
EOF
    exit 2
  fi
fi
requested_gpu_count="$("$LOCAL_PYTHON" - "$GRES" <<'PY'
import re
import sys

gres = sys.argv[1]
total = 0
for item in (part.strip() for part in gres.split(",")):
    if not item or "gpu" not in item:
        continue
    fields = item.split(":")
    if fields[0] != "gpu" and not fields[0].endswith("/gpu"):
        continue
    if fields[-1].isdigit():
        total += int(fields[-1])
    else:
        total += 1
print(total)
PY
)"
if (( ! finalize_only )) && (( requested_gpu_count > 1 )); then
  cat >&2 <<EOF
Refusing multi-GPU MATX physics run.

Requested:
  GRES=$GRES
  requested_gpu_count=$requested_gpu_count

The current goals.md contract allows at most one L40 GPU total for this
project at any time. Use GRES=gpu:l40s:1, submit only one selected shard, and
do not queue additional GPU jobs.
EOF
  exit 2
fi
if (( ! finalize_only )) && [[ "$GRES" == *gpu* && "$ARRAY_CONCURRENCY" != "1" && "$ALLOW_HIGH_CLUSTER_USAGE" != "1" ]]; then
  cat >&2 <<EOF
Refusing to submit multiple concurrent GPU shards.

Requested:
  GRES=$GRES
  ARRAY_CONCURRENCY=$ARRAY_CONCURRENCY

Default policy is one GPU shard at a time to avoid monopolizing shared MATX
resources. Coordinate with the lab and set ALLOW_HIGH_CLUSTER_USAGE=1 only when
that higher concurrency is explicitly acceptable.
EOF
  exit 2
fi
if (( ! finalize_only )) && [[ -n "$SHARD_INDICES" && "$RESTAGE_REMOTE_REPO" == "1" && "$ALLOW_DESTRUCTIVE_RESTAGE" != "1" ]]; then
  cat >&2 <<EOF
Refusing destructive selected-shard resume.

Requested:
  SHARD_INDICES=$SHARD_INDICES
  RESTAGE_REMOTE_REPO=$RESTAGE_REMOTE_REPO

Selected-shard resume is meant to preserve existing shard outputs in
$remote_repo. Restaging deletes that tree before recreating it. Set
RESTAGE_REMOTE_REPO=0 to preserve the existing run, or set
ALLOW_DESTRUCTIVE_RESTAGE=1 only for a deliberate fresh root.
EOF
  exit 2
fi
if (( ! finalize_only )) && [[ "$GRES" == *gpu* ]] && (( array_task_count > MAX_ARRAY_TASKS )) && [[ "$ALLOW_HIGH_CLUSTER_USAGE" != "1" ]]; then
  cat >&2 <<EOF
Refusing to submit a broad GPU array.

Requested:
  GRES=$GRES
  SHARD_INDICES=${SHARD_INDICES:-0-$array_end}
  array_task_count=$array_task_count
  MAX_ARRAY_TASKS=$MAX_ARRAY_TASKS

Default policy is surgical GPU submission: one selected shard at a time, with
no large pending array. Set SHARD_INDICES to a single shard such as
SHARD_INDICES=7. Coordinate with the lab and set ALLOW_HIGH_CLUSTER_USAGE=1
only when a broader array is explicitly acceptable.
EOF
  exit 2
fi
array_spec="$array_range%$ARRAY_CONCURRENCY"

if [[ "$RESTAGE_REMOTE_REPO" == "1" ]]; then
  ssh "$REMOTE_HOST" "rm -rf '$remote_repo' && mkdir -p '$remote_repo' '$remote_logs' '$REMOTE_ROOT/locks' '$REMOTE_ROOT/venvs'"
  git -C "$repo_root" archive --format=tar "$source_commit" \
    | ssh "$REMOTE_HOST" "tar -xf - -C '$remote_repo'"
  ssh "$REMOTE_HOST" "rm -rf '$remote_out_dir/experiment_shards'"
else
  ssh "$REMOTE_HOST" "test -d '$remote_repo' && mkdir -p '$remote_logs' '$REMOTE_ROOT/locks' '$REMOTE_ROOT/venvs'"
  if [[ "$REFRESH_REMOTE_CODE" == "1" ]]; then
    git -C "$repo_root" archive --format=tar "$source_commit" \
      | ssh "$REMOTE_HOST" "tar -xf - -C '$remote_repo'"
  fi
fi

if (( ! finalize_only )); then
  tmp_sbatch="$(mktemp)"
  cat >"$tmp_sbatch" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --account=$ACCOUNT
#SBATCH --partition=$PARTITION
#SBATCH --qos=$QOS
#SBATCH --gres=$GRES
#SBATCH --cpus-per-task=$CPUS_PER_TASK
#SBATCH --mem=$MEM
#SBATCH --time=$TIME
#SBATCH --array=$array_spec
#SBATCH --output=$SLURM_OUTPUT
#SBATCH --error=$SLURM_ERROR

set -euo pipefail

cd "$remote_repo"
export PYTHONPATH="$remote_repo:\${PYTHONPATH:-}"
export HF_HOME="\${HF_HOME:-$JOB_RUNTIME_ROOT/hf_home}"
export TRANSFORMERS_CACHE="\${TRANSFORMERS_CACHE:-$JOB_RUNTIME_ROOT/hf_home/transformers}"
export HF_HUB_CACHE="\${HF_HUB_CACHE:-$JOB_RUNTIME_ROOT/hf_home/hub}"
export UV_CACHE_DIR="\${UV_CACHE_DIR:-$JOB_RUNTIME_ROOT/uv_cache}"
export UV_PROJECT_ENVIRONMENT="\${UV_PROJECT_ENVIRONMENT:-$JOB_RUNTIME_ROOT/venvs/mechanism_repair_$source_commit}"
export XDG_CACHE_HOME="\${XDG_CACHE_HOME:-$JOB_RUNTIME_ROOT/xdg_cache}"
export TRITON_CACHE_DIR="\${TRITON_CACHE_DIR:-$JOB_RUNTIME_ROOT/triton_cache}"
export HOME="\${JOB_HOME:-$JOB_RUNTIME_ROOT/home}"
export SGLANG_DISABLE_CUDNN_CHECK="\${SGLANG_DISABLE_CUDNN_CHECK:-1}"
export USE_HUB_KERNELS="\${USE_HUB_KERNELS:-0}"
export PYTORCH_CUDA_ALLOC_CONF="\${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "\$HF_HOME" "\$TRANSFORMERS_CACHE" "\$HF_HUB_CACHE" \\
  "\$UV_CACHE_DIR" "\$XDG_CACHE_HOME" "\$TRITON_CACHE_DIR" "\$HOME" \\
  "$REMOTE_ROOT/locks" "$REMOTE_ROOT/venvs" "$JOB_RUNTIME_ROOT/venvs"

if ! command -v uv >/dev/null 2>&1; then
  (
    flock 9
    if [[ ! -x "$REMOTE_ROOT/uv_bootstrap/bin/uv" ]]; then
      rm -rf "$REMOTE_ROOT/uv_bootstrap"
      python3 -m venv "$REMOTE_ROOT/uv_bootstrap"
      "$REMOTE_ROOT/uv_bootstrap/bin/python" -m pip install --upgrade pip uv
    fi
  ) 9>"$REMOTE_ROOT/locks/uv_bootstrap.lock"
  export PATH="$REMOTE_ROOT/uv_bootstrap/bin:\$PATH"
fi
(
  flock 9
  uv sync --extra training-grpo
) 9>"$REMOTE_ROOT/locks/uv_sync_$source_commit.lock"
repo_python="\$UV_PROJECT_ENVIRONMENT/bin/python"

if [[ "$ENABLE_CHRONO_ENV" == "1" ]]; then
  if [[ ! -x "$CHRONO_CONDA_EXE" ]]; then
    echo "Chrono conda executable not found: $CHRONO_CONDA_EXE" >&2
    exit 1
  fi
  repo_python_version="\$("\$repo_python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  chrono_python_version="$CHRONO_PYTHON_VERSION"
  if [[ "\$chrono_python_version" == "auto" ]]; then
    chrono_python_version="\$repo_python_version"
  fi
  chrono_env_prefix="$CHRONO_ENV_PREFIX"
  if [[ "\$chrono_env_prefix" == "auto" ]]; then
    chrono_env_prefix="$REMOTE_ROOT/chrono_env_py\${chrono_python_version//./}"
  fi
  chrono_bootstrap_args=(
    scripts/bootstrap_chrono_env.py
    --conda-exe "$CHRONO_CONDA_EXE"
    --prefix "\$chrono_env_prefix"
    --python-version "\$chrono_python_version"
  )
  if [[ "$CHRONO_LINK_CURRENT_VENV" == "1" ]]; then
    if [[ "\$chrono_python_version" != "\$repo_python_version" ]]; then
      echo "CHRONO_LINK_CURRENT_VENV=1 requires Chrono Python \$chrono_python_version to match repo Python \$repo_python_version" >&2
      exit 1
    fi
    chrono_bootstrap_args+=(--link-current-venv)
  else
    chrono_bootstrap_args+=(--no-link-current-venv)
  fi
  export LD_LIBRARY_PATH="\$chrono_env_prefix/lib:\${LD_LIBRARY_PATH:-}"
  (
    flock 9
    env \\
      CONDA_PKGS_DIRS="$CHRONO_CONDA_PKGS_DIR" \\
      XDG_CACHE_HOME="$REMOTE_ROOT/xdg_cache" \\
      "\$repo_python" "\${chrono_bootstrap_args[@]}"
  ) 9>"$REMOTE_ROOT/locks/chrono_env_\${chrono_python_version//./}.lock"
  export MECH_BENCH_CHRONO_PYTHON="\$chrono_env_prefix/bin/python"
  export MECH_BENCH_CHRONO_ENV="\$chrono_env_prefix"
fi

"\$repo_python" - <<'PY'
from mech_bench.adapters.chrono_contact import chrono_diagnostic
diag = chrono_diagnostic()
print("chrono diagnostic", diag)
if diag["status"] != "available":
    raise SystemExit(f"chrono_contact unavailable: {diag}")
PY

sglang_cuda_visible_devices="$SGLANG_CUDA_VISIBLE_DEVICES"
train_cuda_visible_devices="$TRAIN_CUDA_VISIBLE_DEVICES"
slurm_cuda_visible_devices="\${CUDA_VISIBLE_DEVICES:-}"
if [[ -n "\$slurm_cuda_visible_devices" && "\$slurm_cuda_visible_devices" == *,* ]]; then
  first_cuda="\${slurm_cuda_visible_devices%%,*}"
  rest_cuda="\${slurm_cuda_visible_devices#*,}"
  first_train_cuda="\${rest_cuda%%,*}"
  sglang_cuda_visible_devices="\${sglang_cuda_visible_devices:-\$first_cuda}"
  train_cuda_visible_devices="\${train_cuda_visible_devices:-\$first_train_cuda}"
fi
sglang_cuda_visible_devices="\${sglang_cuda_visible_devices:-0}"
train_cuda_visible_devices="\${train_cuda_visible_devices:-0}"
echo "CUDA split: sglang=\$sglang_cuda_visible_devices train=\$train_cuda_visible_devices"
echo "Rollout backend: $ROLLOUT_BACKEND local_device=$LOCAL_DEVICE local_torch_dtype=$LOCAL_TORCH_DTYPE"
sglang_port="$SGLANG_PORT"
if [[ "\${SLURM_ARRAY_TASK_ID:-}" =~ ^[0-9]+$ ]]; then
  sglang_port="\$(( $SGLANG_PORT + SLURM_ARRAY_TASK_ID ))"
fi
export SGLANG_EFFECTIVE_PORT="\$sglang_port"
echo "SGLang port: \$sglang_port"

if [[ "$ROLLOUT_BACKEND" == "sglang_chat" ]]; then
  sglang_venv="$SGLANG_VENV"
  (
    flock 9
    sglang_ready_marker="\$sglang_venv/.corl_sglang_ready"
    if [[ ! -x "\$sglang_venv/bin/python" ]]; then
      rm -rf "\$sglang_venv"
      python3 -m venv "\$sglang_venv"
      "\$sglang_venv/bin/python" -m pip install --upgrade pip
      rm -f "\$sglang_ready_marker"
    fi
    if [[ ! -f "\$sglang_ready_marker" ]]; then
      "\$sglang_venv/bin/python" -m pip install "$SGLANG_PIP_SPEC" $SGLANG_PIP_EXTRA
      touch "\$sglang_ready_marker"
    fi
  ) 9>"$REMOTE_ROOT/locks/sglang_venv.lock"
  export PATH="\$sglang_venv/bin:\$PATH"
  sglang_log="$remote_logs/sglang-\${SLURM_ARRAY_JOB_ID:-manual}_\${SLURM_ARRAY_TASK_ID:-0}.log"
  if ! ss -tln 2>/dev/null | grep -q ":\$sglang_port "; then
    nohup env CUDA_VISIBLE_DEVICES="\$sglang_cuda_visible_devices" \\
      "\$sglang_venv/bin/python" -m sglang.launch_server \\
      --model-path "$SGLANG_MODEL" \\
      --host 127.0.0.1 \\
      --port "\$sglang_port" \\
      --dtype bfloat16 \\
      --tp "$SGLANG_TP" \\
      --context-length "$SGLANG_CTX" \\
      --max-running-requests "$SGLANG_MAX_REQS" \\
      --mem-fraction-static "$SGLANG_MEM_FRAC" \\
      --trust-remote-code \\
      --served-model-name "$BASE_MODEL" \\
      --enable-lora \\
      --max-lora-rank 16 \\
      --lora-target-modules q_proj k_proj v_proj o_proj \\
      --attention-backend triton \\
      --sampling-backend pytorch \\
      >"\$sglang_log" 2>&1 &
  fi

  for i in \$(seq 1 120); do
    if python3 - <<'PY' >/dev/null 2>&1
import urllib.request
import os
urllib.request.urlopen(
    f"http://127.0.0.1:{os.environ['SGLANG_EFFECTIVE_PORT']}/v1/models",
    timeout=2,
).read()
PY
    then
      echo "SGLang is ready"
      break
    fi
    if [[ "\$i" == 120 ]]; then
      echo "SGLang did not become ready; tailing log" >&2
      tail -200 "\$sglang_log" >&2 || true
      exit 1
    fi
    sleep 10
  done
else
  echo "Skipping SGLang server startup for rollout backend $ROLLOUT_BACKEND"
fi

shard_index="\${SLURM_ARRAY_TASK_ID:-0}"
shard_name="\$(printf 'shard_%04d' "\$shard_index")"
shard_file="$OUT_DIR/experiment_shards/\$shard_name.json"
shard_out="$OUT_DIR/shard_runs/\$shard_name"
limit_task_args=()
if [[ "$LIMIT_TASKS" != "0" ]]; then
  limit_task_args=(--limit-tasks "$LIMIT_TASKS")
fi
method_args=()
if [[ -n "$METHODS" ]]; then
  method_args=(--methods "$METHODS")
fi
split_args=()
if [[ -n "$SPLITS" ]]; then
  split_args=(--splits "$SPLITS")
fi
anti_shortcut_args=()
if [[ "$ANTI_SHORTCUT_SPLITS" != "__default__" ]]; then
  anti_shortcut_args=(--anti-shortcut-splits "$ANTI_SHORTCUT_SPLITS")
fi
seed_args=()
if [[ -n "$EVAL_SEEDS" ]]; then
  seed_args=(--eval-seeds "$EVAL_SEEDS")
fi
shard_signature="grouping=split_task_seed_budget_v2;num_shards=$NUM_SHARDS;limit_tasks=$LIMIT_TASKS;methods=$METHODS;splits=$SPLITS;anti_shortcut_splits=$ANTI_SHORTCUT_SPLITS;eval_seeds=$EVAL_SEEDS"
shard_plan_marker="$OUT_DIR/experiment_shards/.submission_signature"
(
  flock 9
  current_shard_signature=""
  if [[ -f "\$shard_plan_marker" ]]; then
    current_shard_signature="\$(<"\$shard_plan_marker")"
  fi
  if [[ "\$current_shard_signature" != "\$shard_signature" || ! -f "\$shard_file" ]]; then
    rm -rf "$OUT_DIR/experiment_shards"
    mkdir -p "$OUT_DIR/experiment_shards"
    echo "Regenerating experiment shards for \$shard_signature"
    "\$repo_python" scripts/run_mechanism_repair_physics_experiment.py \\
      --benchmark-dir "$OUT_DIR" \\
      --out-dir "$OUT_DIR" \\
      --write-shard-files "$NUM_SHARDS" \\
      "\${limit_task_args[@]}" \\
      "\${method_args[@]}" \\
      "\${split_args[@]}" \\
      "\${anti_shortcut_args[@]}" \\
      "\${seed_args[@]}" \\
      --dry-run
    printf '%s\n' "\$shard_signature" > "\$shard_plan_marker"
  fi
) 9>"$REMOTE_ROOT/locks/experiment_shards_$source_commit_short.lock"
if [[ ! -f "\$shard_file" ]]; then
  "\$repo_python" scripts/run_mechanism_repair_physics_experiment.py \\
    --benchmark-dir "$OUT_DIR" \\
    --out-dir "$OUT_DIR" \\
    --write-shard-files "$NUM_SHARDS" \\
    "\${limit_task_args[@]}" \\
    "\${method_args[@]}" \\
    "\${split_args[@]}" \\
    "\${anti_shortcut_args[@]}" \\
    "\${seed_args[@]}" \\
    --dry-run
fi
if [[ ! -f "\$shard_file" ]]; then
  echo "missing shard file: \$shard_file" >&2
  exit 1
fi

resume_args=()
if [[ "$RESUME_EXISTING" == "1" ]]; then
  resume_args=(--resume-existing)
fi
dry_run_args=()
if [[ "$DRY_RUN" == "1" ]]; then
  dry_run_args=(--dry-run)
fi
sft_quant_args=()
if [[ "$SFT_LOAD_IN_4BIT" == "1" ]]; then
  sft_quant_args+=(--sft-load-in-4bit)
fi
ttrl_quant_args=()
if [[ "$TTRL_LOAD_IN_4BIT" == "1" ]]; then
  ttrl_quant_args+=(--ttrl-load-in-4bit)
fi
ttrl_precision_args=()
if [[ "$TTRL_BF16" == "1" ]]; then
  ttrl_precision_args+=(--ttrl-bf16)
fi
ttrl_gradient_args=()
if [[ "$TTRL_GRADIENT_CHECKPOINTING" == "1" ]]; then
  ttrl_gradient_args+=(--ttrl-gradient-checkpointing)
fi
local_trust_args=()
if [[ "$LOCAL_TRUST_REMOTE_CODE" == "1" ]]; then
  local_trust_args+=(--local-trust-remote-code)
fi

exec env CUDA_VISIBLE_DEVICES="\$train_cuda_visible_devices" \\
  "\$repo_python" scripts/run_mechanism_repair_online_experiment.py \\
  --benchmark-dir "$OUT_DIR" \\
  --out-dir "\$shard_out" \\
  --cell-shard-file "\$shard_file" \\
  --runner-python "\$repo_python" \\
  --base-model "$BASE_MODEL" \\
  --rollout-backend "$ROLLOUT_BACKEND" \\
  --local-device "$LOCAL_DEVICE" \\
  --local-torch-dtype "$LOCAL_TORCH_DTYPE" \\
  "\${local_trust_args[@]}" \\
  --sglang-base-url "http://127.0.0.1:\$sglang_port" \\
  --audit-retries "$AUDIT_RETRIES" \\
  --evidence-layout "$EVIDENCE_LAYOUT" \\
  --skip-analysis \\
  "\${resume_args[@]}" \\
  "\${limit_task_args[@]}" \\
  "\${method_args[@]}" \\
  "\${split_args[@]}" \\
  "\${seed_args[@]}" \\
  "\${sft_quant_args[@]}" \\
  --sft-max-steps "$SFT_MAX_STEPS" \\
  --shared-sft-root "$SHARED_SFT_ROOT" \\
  --sft-learning-rate "$SFT_LEARNING_RATE" \\
  --sft-max-grad-norm "$SFT_MAX_GRAD_NORM" \\
  --sft-max-seq-length "$SFT_MAX_SEQ_LENGTH" \\
  --sft-torch-dtype "$SFT_TORCH_DTYPE" \\
  --sft-attn-implementation "$SFT_ATTN_IMPLEMENTATION" \\
  --sft-device-map "$SFT_DEVICE_MAP" \\
  --sft-trust-remote-code \\
  "\${ttrl_quant_args[@]}" \\
  --ttrl-torch-dtype "$TTRL_TORCH_DTYPE" \\
  --ttrl-attn-implementation "$TTRL_ATTN_IMPLEMENTATION" \\
  "\${ttrl_precision_args[@]}" \\
  --ttrl-lora-rank "$TTRL_LORA_RANK" \\
  --ttrl-save-adapter-dtype "$TTRL_SAVE_ADAPTER_DTYPE" \\
  --ttrl-adapter-retention "$TTRL_ADAPTER_RETENTION" \\
  --ttrl-learning-rate "$TTRL_LEARNING_RATE" \\
  --ttrl-max-grad-norm "$TTRL_MAX_GRAD_NORM" \\
  --ttrl-kbit-prepare-mode "$TTRL_KBIT_PREPARE_MODE" \\
  --ttrl-device-map "$TTRL_DEVICE_MAP" \\
  "\${ttrl_gradient_args[@]}" \\
  --ttrl-trust-remote-code \\
  --max-context-tokens "$TTRL_MAX_CONTEXT_TOKENS" \\
  --max-tokens "$TTRL_MAX_TOKENS" \\
  "\${dry_run_args[@]}"
EOF

  scp "$tmp_sbatch" "$REMOTE_HOST:$remote_sbatch"
  rm -f "$tmp_sbatch"
fi

tmp_merge="$(mktemp)"
cat >"$tmp_merge" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=$MERGE_JOB_NAME
#SBATCH --account=$ACCOUNT
#SBATCH --partition=$PARTITION
#SBATCH --qos=$QOS
#SBATCH --cpus-per-task=$MERGE_CPUS_PER_TASK
#SBATCH --mem=$MERGE_MEM
#SBATCH --time=$MERGE_TIME
#SBATCH --output=$MERGE_SLURM_OUTPUT
#SBATCH --error=$MERGE_SLURM_ERROR

set -euo pipefail
cd "$remote_repo"
export PYTHONPATH="$remote_repo:\${PYTHONPATH:-}"
export UV_CACHE_DIR="\${UV_CACHE_DIR:-$JOB_RUNTIME_ROOT/uv_cache}"
export UV_PROJECT_ENVIRONMENT="\${UV_PROJECT_ENVIRONMENT:-$JOB_RUNTIME_ROOT/venvs/mechanism_repair_$source_commit}"
export XDG_CACHE_HOME="\${XDG_CACHE_HOME:-$JOB_RUNTIME_ROOT/xdg_cache}"
export HOME="\${JOB_HOME:-$JOB_RUNTIME_ROOT/home}"
mkdir -p "$REMOTE_ROOT/locks" "$REMOTE_ROOT/venvs" "$JOB_RUNTIME_ROOT/venvs" "\$UV_CACHE_DIR" "\$XDG_CACHE_HOME" "\$HOME"
if ! command -v uv >/dev/null 2>&1; then
  (
    flock 9
    if [[ ! -x "$REMOTE_ROOT/uv_bootstrap/bin/uv" ]]; then
      rm -rf "$REMOTE_ROOT/uv_bootstrap"
      python3 -m venv "$REMOTE_ROOT/uv_bootstrap"
      "$REMOTE_ROOT/uv_bootstrap/bin/python" -m pip install --upgrade pip uv
    fi
  ) 9>"$REMOTE_ROOT/locks/uv_bootstrap.lock"
  export PATH="$REMOTE_ROOT/uv_bootstrap/bin:\$PATH"
fi
(
  flock 9
  uv sync --extra training-grpo
) 9>"$REMOTE_ROOT/locks/uv_sync_$source_commit.lock"
repo_python="\$UV_PROJECT_ENVIRONMENT/bin/python"

if [[ "$ENABLE_CHRONO_ENV" == "1" ]]; then
  if [[ ! -x "$CHRONO_CONDA_EXE" ]]; then
    echo "Chrono conda executable not found: $CHRONO_CONDA_EXE" >&2
    exit 1
  fi
  repo_python_version="\$("\$repo_python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  chrono_python_version="$CHRONO_PYTHON_VERSION"
  if [[ "\$chrono_python_version" == "auto" ]]; then
    chrono_python_version="\$repo_python_version"
  fi
  chrono_env_prefix="$CHRONO_ENV_PREFIX"
  if [[ "\$chrono_env_prefix" == "auto" ]]; then
    chrono_env_prefix="$REMOTE_ROOT/chrono_env_py\${chrono_python_version//./}"
  fi
  chrono_bootstrap_args=(
    scripts/bootstrap_chrono_env.py
    --conda-exe "$CHRONO_CONDA_EXE"
    --prefix "\$chrono_env_prefix"
    --python-version "\$chrono_python_version"
  )
  if [[ "$CHRONO_LINK_CURRENT_VENV" == "1" ]]; then
    if [[ "\$chrono_python_version" != "\$repo_python_version" ]]; then
      echo "CHRONO_LINK_CURRENT_VENV=1 requires Chrono Python \$chrono_python_version to match repo Python \$repo_python_version" >&2
      exit 1
    fi
    chrono_bootstrap_args+=(--link-current-venv)
  else
    chrono_bootstrap_args+=(--no-link-current-venv)
  fi
  export LD_LIBRARY_PATH="\$chrono_env_prefix/lib:\${LD_LIBRARY_PATH:-}"
  (
    flock 9
    env \\
      CONDA_PKGS_DIRS="$CHRONO_CONDA_PKGS_DIR" \\
      XDG_CACHE_HOME="$REMOTE_ROOT/xdg_cache" \\
      "\$repo_python" "\${chrono_bootstrap_args[@]}"
  ) 9>"$REMOTE_ROOT/locks/chrono_env_\${chrono_python_version//./}.lock"
  export MECH_BENCH_CHRONO_PYTHON="\$chrono_env_prefix/bin/python"
  export MECH_BENCH_CHRONO_ENV="\$chrono_env_prefix"
fi

"\$repo_python" - <<'PY'
from mech_bench.adapters.chrono_contact import chrono_diagnostic
diag = chrono_diagnostic()
print("chrono diagnostic", diag)
if diag["status"] != "available":
    raise SystemExit(f"chrono_contact unavailable: {diag}")
PY

"\$repo_python" scripts/merge_mechanism_repair_shards.py \\
  --benchmark-dir "$OUT_DIR" \\
  --out-dir "$OUT_DIR" \\
  --require-all-shards "$NUM_SHARDS"
EOF

scp "$tmp_merge" "$REMOTE_HOST:$remote_merge_sbatch"
rm -f "$tmp_merge"

tmp_analysis="$(mktemp)"
cat >"$tmp_analysis" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=$ANALYSIS_JOB_NAME
#SBATCH --account=$ACCOUNT
#SBATCH --partition=$PARTITION
#SBATCH --qos=$QOS
#SBATCH --cpus-per-task=$ANALYSIS_CPUS_PER_TASK
#SBATCH --mem=$ANALYSIS_MEM
#SBATCH --time=$ANALYSIS_TIME
#SBATCH --output=$ANALYSIS_SLURM_OUTPUT
#SBATCH --error=$ANALYSIS_SLURM_ERROR

set -euo pipefail
cd "$remote_repo"
export PYTHONPATH="$remote_repo:\${PYTHONPATH:-}"
export UV_CACHE_DIR="\${UV_CACHE_DIR:-$JOB_RUNTIME_ROOT/uv_cache}"
export UV_PROJECT_ENVIRONMENT="\${UV_PROJECT_ENVIRONMENT:-$JOB_RUNTIME_ROOT/venvs/mechanism_repair_$source_commit}"
export XDG_CACHE_HOME="\${XDG_CACHE_HOME:-$JOB_RUNTIME_ROOT/xdg_cache}"
export HOME="\${JOB_HOME:-$JOB_RUNTIME_ROOT/home}"
mkdir -p "$REMOTE_ROOT/locks" "$REMOTE_ROOT/venvs" "$JOB_RUNTIME_ROOT/venvs" "\$UV_CACHE_DIR" "\$XDG_CACHE_HOME" "\$HOME"
if ! command -v uv >/dev/null 2>&1; then
  (
    flock 9
    if [[ ! -x "$REMOTE_ROOT/uv_bootstrap/bin/uv" ]]; then
      rm -rf "$REMOTE_ROOT/uv_bootstrap"
      python3 -m venv "$REMOTE_ROOT/uv_bootstrap"
      "$REMOTE_ROOT/uv_bootstrap/bin/python" -m pip install --upgrade pip uv
    fi
  ) 9>"$REMOTE_ROOT/locks/uv_bootstrap.lock"
  export PATH="$REMOTE_ROOT/uv_bootstrap/bin:\$PATH"
fi
(
  flock 9
  uv sync --extra training-grpo
) 9>"$REMOTE_ROOT/locks/uv_sync_$source_commit.lock"
repo_python="\$UV_PROJECT_ENVIRONMENT/bin/python"

if [[ "$ENABLE_CHRONO_ENV" == "1" ]]; then
  if [[ ! -x "$CHRONO_CONDA_EXE" ]]; then
    echo "Chrono conda executable not found: $CHRONO_CONDA_EXE" >&2
    exit 1
  fi
  repo_python_version="\$("\$repo_python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  chrono_python_version="$CHRONO_PYTHON_VERSION"
  if [[ "\$chrono_python_version" == "auto" ]]; then
    chrono_python_version="\$repo_python_version"
  fi
  chrono_env_prefix="$CHRONO_ENV_PREFIX"
  if [[ "\$chrono_env_prefix" == "auto" ]]; then
    chrono_env_prefix="$REMOTE_ROOT/chrono_env_py\${chrono_python_version//./}"
  fi
  chrono_bootstrap_args=(
    scripts/bootstrap_chrono_env.py
    --conda-exe "$CHRONO_CONDA_EXE"
    --prefix "\$chrono_env_prefix"
    --python-version "\$chrono_python_version"
  )
  if [[ "$CHRONO_LINK_CURRENT_VENV" == "1" ]]; then
    if [[ "\$chrono_python_version" != "\$repo_python_version" ]]; then
      echo "CHRONO_LINK_CURRENT_VENV=1 requires Chrono Python \$chrono_python_version to match repo Python \$repo_python_version" >&2
      exit 1
    fi
    chrono_bootstrap_args+=(--link-current-venv)
  else
    chrono_bootstrap_args+=(--no-link-current-venv)
  fi
  export LD_LIBRARY_PATH="\$chrono_env_prefix/lib:\${LD_LIBRARY_PATH:-}"
  (
    flock 9
    env \\
      CONDA_PKGS_DIRS="$CHRONO_CONDA_PKGS_DIR" \\
      XDG_CACHE_HOME="$REMOTE_ROOT/xdg_cache" \\
      "\$repo_python" "\${chrono_bootstrap_args[@]}"
  ) 9>"$REMOTE_ROOT/locks/chrono_env_\${chrono_python_version//./}.lock"
  export MECH_BENCH_CHRONO_PYTHON="\$chrono_env_prefix/bin/python"
  export MECH_BENCH_CHRONO_ENV="\$chrono_env_prefix"
fi

"\$repo_python" - <<'PY'
from mech_bench.adapters.chrono_contact import chrono_diagnostic
diag = chrono_diagnostic()
print("chrono diagnostic", diag)
if diag["status"] != "available":
    raise SystemExit(f"chrono_contact unavailable: {diag}")
PY

analysis_rc=0
"\$repo_python" scripts/analyze_mechanism_repair_results.py \\
  --results "$OUT_DIR/cell_results.jsonl" \\
  --out-dir "$OUT_DIR" \\
  --benchmark-dir "$OUT_DIR" \\
  --bootstrap-samples 5000 \\
  --seed 20260607 || analysis_rc=\$?
echo "\$analysis_rc" > "$OUT_DIR/analysis_exit_code.txt"
"\$repo_python" scripts/run_mechanism_repair_physics_experiment.py \\
  --benchmark-dir "$OUT_DIR" \\
  --out-dir "$OUT_DIR" \\
  --require-complete
echo "analysis_exit_code=\$analysis_rc"
if [[ "\$analysis_rc" != "0" ]]; then
  exit "\$analysis_rc"
fi
EOF

scp "$tmp_analysis" "$REMOTE_HOST:$remote_analysis_sbatch"
rm -f "$tmp_analysis"

if [[ "$RESTAGE_REMOTE_REPO" == "1" ]]; then
  echo "Staged $source_ref ($source_commit) at $REMOTE_HOST:$remote_repo"
else
  echo "Preserved existing remote repo at $REMOTE_HOST:$remote_repo"
fi
if (( ! finalize_only )); then
  echo "Wrote array Slurm script at $REMOTE_HOST:$remote_sbatch"
else
  echo "Finalize-only mode: no GPU array Slurm script was written"
fi
echo "Wrote merge Slurm script at $REMOTE_HOST:$remote_merge_sbatch"
echo "Wrote analysis Slurm script at $REMOTE_HOST:$remote_analysis_sbatch"
if (( submit )); then
  if (( finalize_only )); then
    merge_job_raw="$(ssh "$REMOTE_HOST" "sbatch --parsable '$remote_merge_sbatch'")"
    merge_job="${merge_job_raw%%;*}"
    echo "Submitted finalize merge job: $merge_job_raw"
    analysis_job_raw="$(ssh "$REMOTE_HOST" "sbatch --parsable --dependency=afterok:$merge_job '$remote_analysis_sbatch'")"
    echo "Submitted finalize analysis/audit job: $analysis_job_raw"
  else
    array_job_raw="$(ssh "$REMOTE_HOST" "sbatch --parsable '$remote_sbatch'")"
    array_job="${array_job_raw%%;*}"
    echo "Submitted array job: $array_job_raw"
    if [[ "$SUBMIT_DEPENDENTS" == "1" ]]; then
      merge_job_raw="$(ssh "$REMOTE_HOST" "sbatch --parsable --dependency=afterok:$array_job '$remote_merge_sbatch'")"
      merge_job="${merge_job_raw%%;*}"
      echo "Submitted dependent merge job: $merge_job_raw"
      analysis_job_raw="$(ssh "$REMOTE_HOST" "sbatch --parsable --dependency=afterok:$merge_job '$remote_analysis_sbatch'")"
      echo "Submitted dependent analysis/audit job: $analysis_job_raw"
    else
      echo "Skipped dependent merge/analysis submission because SUBMIT_DEPENDENTS=$SUBMIT_DEPENDENTS"
    fi
  fi
else
  echo "Submit with:"
  if (( finalize_only )); then
    echo "  merge_job=\\\$(ssh $REMOTE_HOST sbatch --parsable '$remote_merge_sbatch')"
    echo "  ssh $REMOTE_HOST sbatch --dependency=afterok:\\\${merge_job%%;*} '$remote_analysis_sbatch'"
  else
    echo "  array_job=\\\$(ssh $REMOTE_HOST sbatch --parsable '$remote_sbatch')"
    if [[ "$SUBMIT_DEPENDENTS" == "1" ]]; then
      echo "  merge_job=\\\$(ssh $REMOTE_HOST sbatch --parsable --dependency=afterok:\\\${array_job%%;*} '$remote_merge_sbatch')"
      echo "  ssh $REMOTE_HOST sbatch --dependency=afterok:\\\${merge_job%%;*} '$remote_analysis_sbatch'"
    else
      echo "  # SUBMIT_DEPENDENTS=$SUBMIT_DEPENDENTS; submit merge/analysis only after all shards are complete"
    fi
  fi
fi
