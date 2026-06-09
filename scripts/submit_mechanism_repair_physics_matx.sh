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
GRES="${GRES:-gpu:l40s:2}"
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
ARRAY_CONCURRENCY="${ARRAY_CONCURRENCY:-4}"
OUT_DIR="${OUT_DIR:-runs/mechanism_repair_physics_final}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-Coder-1.5B-Instruct}"
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
TTRL_ADAPTER_RETENTION="${TTRL_ADAPTER_RETENTION:-metadata}"
TTRL_GRADIENT_CHECKPOINTING="${TTRL_GRADIENT_CHECKPOINTING:-1}"
TTRL_BF16="${TTRL_BF16:-1}"
AUDIT_RETRIES="${AUDIT_RETRIES:-1}"
EVIDENCE_LAYOUT="${EVIDENCE_LAYOUT:-bundled}"
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

usage() {
  cat <<EOF
Usage: $0 [--submit]

Stages the requested git ref, including the frozen MechanismRepair-Physics
benchmark, to MATX, writes a Slurm array over shard_0000..shard_N, and writes a
dependent merge/audit job.

Useful overrides:
  REMOTE_HOST=$REMOTE_HOST
  REMOTE_USER=$REMOTE_USER
  REMOTE_ROOT=$REMOTE_ROOT
  JOB_RUNTIME_ROOT=$JOB_RUNTIME_ROOT
  SOURCE_REF=HEAD
  OUT_DIR=$OUT_DIR
  NUM_SHARDS=$NUM_SHARDS
  ARRAY_CONCURRENCY=$ARRAY_CONCURRENCY
  GRES=$GRES
  BASE_MODEL=$BASE_MODEL
  SGLANG_MODEL=$SGLANG_MODEL
  SGLANG_VENV=$SGLANG_VENV
  SHARED_SFT_ROOT=$SHARED_SFT_ROOT
  TTRL_ADAPTER_RETENTION=$TTRL_ADAPTER_RETENTION
  EVIDENCE_LAYOUT=$EVIDENCE_LAYOUT
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
remote_sbatch="$REMOTE_ROOT/run_mechanism_repair_physics_array.sbatch"
remote_merge_sbatch="$REMOTE_ROOT/run_mechanism_repair_physics_merge.sbatch"
remote_analysis_sbatch="$REMOTE_ROOT/run_mechanism_repair_physics_analysis.sbatch"
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
  CHRONO_CONDA_PKGS_DIR="$JOB_RUNTIME_ROOT/conda_pkgs_chrono"
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

ssh "$REMOTE_HOST" "rm -rf '$remote_repo' && mkdir -p '$remote_repo' '$remote_logs' '$REMOTE_ROOT/locks' '$REMOTE_ROOT/venvs'"
git -C "$repo_root" archive --format=tar "$source_commit" \
  | ssh "$REMOTE_HOST" "tar -xf - -C '$remote_repo'"

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
#SBATCH --array=0-$array_end%$ARRAY_CONCURRENCY
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
    chrono_env_prefix="$JOB_RUNTIME_ROOT/chrono_env_py\${chrono_python_version//./}"
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
sglang_port="$SGLANG_PORT"
if [[ "\${SLURM_ARRAY_TASK_ID:-}" =~ ^[0-9]+$ ]]; then
  sglang_port="\$(( $SGLANG_PORT + SLURM_ARRAY_TASK_ID ))"
fi
export SGLANG_EFFECTIVE_PORT="\$sglang_port"
echo "SGLang port: \$sglang_port"

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

shard_index="\${SLURM_ARRAY_TASK_ID:-0}"
shard_name="\$(printf 'shard_%04d' "\$shard_index")"
shard_file="$OUT_DIR/experiment_shards/\$shard_name.json"
shard_out="$OUT_DIR/shard_runs/\$shard_name"
if [[ ! -f "\$shard_file" ]]; then
  uv run python scripts/run_mechanism_repair_physics_experiment.py \\
    --benchmark-dir "$OUT_DIR" \\
    --out-dir "$OUT_DIR" \\
    --write-shard-files "$NUM_SHARDS" \\
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

exec env CUDA_VISIBLE_DEVICES="\$train_cuda_visible_devices" \\
  "\$repo_python" scripts/run_mechanism_repair_online_experiment.py \\
  --benchmark-dir "$OUT_DIR" \\
  --out-dir "\$shard_out" \\
  --cell-shard-file "\$shard_file" \\
  --runner-python "\$repo_python" \\
  --base-model "$BASE_MODEL" \\
  --sglang-base-url "http://127.0.0.1:\$sglang_port" \\
  --audit-retries "$AUDIT_RETRIES" \\
  --evidence-layout "$EVIDENCE_LAYOUT" \\
  --skip-analysis \\
  "\${resume_args[@]}" \\
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
uv run python scripts/merge_mechanism_repair_shards.py \\
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

analysis_rc=0
uv run python scripts/analyze_mechanism_repair_results.py \\
  --results "$OUT_DIR/cell_results.jsonl" \\
  --out-dir "$OUT_DIR" \\
  --benchmark-dir "$OUT_DIR" \\
  --bootstrap-samples 5000 \\
  --seed 20260607 || analysis_rc=\$?
echo "\$analysis_rc" > "$OUT_DIR/analysis_exit_code.txt"
uv run python scripts/run_mechanism_repair_physics_experiment.py \\
  --benchmark-dir "$OUT_DIR" \\
  --out-dir "$OUT_DIR" \\
  --require-complete
echo "analysis_exit_code=\$analysis_rc"
EOF

scp "$tmp_analysis" "$REMOTE_HOST:$remote_analysis_sbatch"
rm -f "$tmp_analysis"

echo "Staged $source_ref ($source_commit) at $REMOTE_HOST:$remote_repo"
echo "Wrote array Slurm script at $REMOTE_HOST:$remote_sbatch"
echo "Wrote merge Slurm script at $REMOTE_HOST:$remote_merge_sbatch"
echo "Wrote analysis Slurm script at $REMOTE_HOST:$remote_analysis_sbatch"
if (( submit )); then
  array_job_raw="$(ssh "$REMOTE_HOST" "sbatch --parsable '$remote_sbatch'")"
  array_job="${array_job_raw%%;*}"
  echo "Submitted array job: $array_job_raw"
  merge_job_raw="$(ssh "$REMOTE_HOST" "sbatch --parsable --dependency=afterok:$array_job '$remote_merge_sbatch'")"
  merge_job="${merge_job_raw%%;*}"
  echo "Submitted dependent merge job: $merge_job_raw"
  analysis_job_raw="$(ssh "$REMOTE_HOST" "sbatch --parsable --dependency=afterok:$merge_job '$remote_analysis_sbatch'")"
  echo "Submitted dependent analysis/audit job: $analysis_job_raw"
else
  echo "Submit with:"
  echo "  array_job=\\\$(ssh $REMOTE_HOST sbatch --parsable '$remote_sbatch')"
  echo "  merge_job=\\\$(ssh $REMOTE_HOST sbatch --parsable --dependency=afterok:\\\${array_job%%;*} '$remote_merge_sbatch')"
  echo "  ssh $REMOTE_HOST sbatch --dependency=afterok:\\\${merge_job%%;*} '$remote_analysis_sbatch'"
fi
