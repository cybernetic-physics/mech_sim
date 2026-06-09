#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-sc}"
REMOTE_ROOT="${REMOTE_ROOT:-/matx/u/knatalia/corl_mechanism_repair_ttrl}"
JOB_NAME="${JOB_NAME:-corl_mech_ttrl}"
ACCOUNT="${ACCOUNT:-matx}"
PARTITION="${PARTITION:-matx}"
QOS="${QOS:-normal}"
GRES="${GRES:-gpu:l40s:4}"
CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
MEM="${MEM:-160G}"
TIME="${TIME:-4-00:00:00}"
OUT_DIR="${OUT_DIR:-runs/mechanism_repair_ttrl_final}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-Coder-1.5B-Instruct}"
SGLANG_MODEL="${SGLANG_MODEL:-$BASE_MODEL}"
SGLANG_PORT="${SGLANG_PORT:-30000}"
SGLANG_PIP_SPEC="${SGLANG_PIP_SPEC:-sglang==0.5.9}"
SGLANG_PIP_EXTRA="${SGLANG_PIP_EXTRA:-ninja}"
SGLANG_TP="${SGLANG_TP:-1}"
SGLANG_MEM_FRAC="${SGLANG_MEM_FRAC:-0.82}"
SGLANG_CTX="${SGLANG_CTX:-16384}"
SGLANG_MAX_REQS="${SGLANG_MAX_REQS:-4}"
SGLANG_CUDA_VISIBLE_DEVICES="${SGLANG_CUDA_VISIBLE_DEVICES:-}"
BENCH_CUDA_VISIBLE_DEVICES="${BENCH_CUDA_VISIBLE_DEVICES:-}"
TRAIN_CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-}"
TRAIN_DEVICE_MAP="${TRAIN_DEVICE_MAP:-}"
SFT_DEVICE_MAP="${SFT_DEVICE_MAP:-}"
TTRL_DEVICE_MAP="${TTRL_DEVICE_MAP:-}"
SFT_MAX_STEPS="${SFT_MAX_STEPS:-64}"
SFT_MAX_SEQ_LENGTH="${SFT_MAX_SEQ_LENGTH:-512}"
SFT_LOAD_IN_4BIT="${SFT_LOAD_IN_4BIT:-0}"
SFT_LOAD_IN_8BIT="${SFT_LOAD_IN_8BIT:-0}"
SFT_PREPARE_KBIT_TRAINING="${SFT_PREPARE_KBIT_TRAINING:-0}"
SFT_PREPARE_KBIT_TRAINING_MODE="${SFT_PREPARE_KBIT_TRAINING_MODE:-lightweight}"
SFT_TORCH_DTYPE="${SFT_TORCH_DTYPE:-float32}"
SFT_ATTN_IMPLEMENTATION="${SFT_ATTN_IMPLEMENTATION:-eager}"
SFT_GRADIENT_CHECKPOINTING="${SFT_GRADIENT_CHECKPOINTING:-0}"
SFT_LEARNING_RATE="${SFT_LEARNING_RATE:-1.0e-6}"
SFT_MAX_GRAD_NORM="${SFT_MAX_GRAD_NORM:-0.0}"
TTRL_MAX_CONTEXT_TOKENS="${TTRL_MAX_CONTEXT_TOKENS:-4096}"
TTRL_MAX_TOKENS="${TTRL_MAX_TOKENS:-1536}"
TTRL_REWARD_CHANNEL="${TTRL_REWARD_CHANNEL:-artifact_progress}"
TTRL_LOAD_IN_4BIT="${TTRL_LOAD_IN_4BIT:-0}"
TTRL_LOAD_IN_8BIT="${TTRL_LOAD_IN_8BIT:-0}"
TTRL_TORCH_DTYPE="${TTRL_TORCH_DTYPE:-bfloat16}"
TTRL_ATTN_IMPLEMENTATION="${TTRL_ATTN_IMPLEMENTATION:-eager}"
TTRL_KBIT_PREPARE_MODE="${TTRL_KBIT_PREPARE_MODE:-none}"
TTRL_LEARNING_RATE="${TTRL_LEARNING_RATE:-1.0e-6}"
TTRL_MAX_GRAD_NORM="${TTRL_MAX_GRAD_NORM:-0.0}"
TTRL_MAX_MEMORY="${TTRL_MAX_MEMORY:-}"
TTRL_GRADIENT_CHECKPOINTING="${TTRL_GRADIENT_CHECKPOINTING:-1}"
TTRL_BF16="${TTRL_BF16:-1}"
TTRL_FP16="${TTRL_FP16:-0}"
AUDIT_RETRIES="${AUDIT_RETRIES:-0}"
LIMIT_TASKS="${LIMIT_TASKS:-0}"
RESUME_EXISTING="${RESUME_EXISTING:-0}"
METHODS="${METHODS:-frozen_model,verifier_gated,no_update_search,llm_evolve_no_update,sft_model,mechanical_evolve_ttrl}"
SPLITS="${SPLITS:-A,B}"
EVAL_SEEDS="${EVAL_SEEDS:-20260607,20260608,20260609}"
DRY_RUN="${DRY_RUN:-0}"
if [[ -z "${SGLANG_JSON_MODEL_OVERRIDE_ARGS+x}" ]]; then
  if [[ "$SGLANG_MODEL" == *"Qwen3.6-35B-A3B"* ]]; then
    SGLANG_JSON_MODEL_OVERRIDE_ARGS='{"num_hidden_layers":40,"hidden_size":2048,"num_attention_heads":16,"num_key_value_heads":2,"head_dim":256,"attn_output_gate":true}'
  else
    SGLANG_JSON_MODEL_OVERRIDE_ARGS=""
  fi
fi
SGLANG_EXTRA_ARGS="${SGLANG_EXTRA_ARGS:---trust-remote-code --served-model-name $BASE_MODEL --enable-lora --max-lora-rank 16 --lora-target-modules q_proj k_proj v_proj o_proj --attention-backend triton --sampling-backend pytorch}"

usage() {
  cat <<EOF
Usage: $0 [--submit]

Stages the current worktree to MATX and writes a Slurm job for the
MechanismRepair-TTRL online repair experiment.

Useful overrides:
  REMOTE_HOST=$REMOTE_HOST
  REMOTE_ROOT=$REMOTE_ROOT
  OUT_DIR=$OUT_DIR
  BASE_MODEL=$BASE_MODEL
  SGLANG_MODEL=$SGLANG_MODEL
  SGLANG_PORT=$SGLANG_PORT
  TRAIN_CUDA_VISIBLE_DEVICES=$TRAIN_CUDA_VISIBLE_DEVICES
  SFT_DEVICE_MAP=$SFT_DEVICE_MAP
  TTRL_DEVICE_MAP=$TTRL_DEVICE_MAP
  TTRL_REWARD_CHANNEL=$TTRL_REWARD_CHANNEL
  TTRL_BF16=$TTRL_BF16
  TTRL_FP16=$TTRL_FP16
  METHODS=$METHODS
  SPLITS=$SPLITS
  EVAL_SEEDS=$EVAL_SEEDS
  LIMIT_TASKS=$LIMIT_TASKS
  RESUME_EXISTING=$RESUME_EXISTING
  DRY_RUN=$DRY_RUN
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
remote_sbatch="$REMOTE_ROOT/run_mechanism_repair_ttrl.sbatch"

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
#SBATCH --output=$remote_logs/%x-%j.out
#SBATCH --error=$remote_logs/%x-%j.err

set -euo pipefail

cd "$remote_repo"
export PYTHONPATH="$remote_repo:\${PYTHONPATH:-}"
export HF_HOME="\${HF_HOME:-$REMOTE_ROOT/hf_home}"
export TRANSFORMERS_CACHE="\${TRANSFORMERS_CACHE:-$REMOTE_ROOT/hf_home/transformers}"
export HF_HUB_CACHE="\${HF_HUB_CACHE:-$REMOTE_ROOT/hf_home/hub}"
export UV_CACHE_DIR="\${UV_CACHE_DIR:-$REMOTE_ROOT/uv_cache}"
export XDG_CACHE_HOME="\${XDG_CACHE_HOME:-$REMOTE_ROOT/xdg_cache}"
export TRITON_CACHE_DIR="\${TRITON_CACHE_DIR:-$REMOTE_ROOT/triton_cache}"
export HOME="\${JOB_HOME:-$REMOTE_ROOT/home}"
export SGLANG_DISABLE_CUDNN_CHECK="\${SGLANG_DISABLE_CUDNN_CHECK:-1}"
export USE_HUB_KERNELS="\${USE_HUB_KERNELS:-0}"
export PYTORCH_CUDA_ALLOC_CONF="\${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "\$HF_HOME" "\$TRANSFORMERS_CACHE" "\$HF_HUB_CACHE" \\
  "\$UV_CACHE_DIR" "\$XDG_CACHE_HOME" "\$TRITON_CACHE_DIR" "\$HOME"

slurm_cuda_visible_devices="\${CUDA_VISIBLE_DEVICES:-}"
sglang_cuda_visible_devices="$SGLANG_CUDA_VISIBLE_DEVICES"
bench_cuda_visible_devices="$BENCH_CUDA_VISIBLE_DEVICES"
if [[ -n "\$slurm_cuda_visible_devices" && "\$slurm_cuda_visible_devices" == *,* ]]; then
  first_cuda="\${slurm_cuda_visible_devices%%,*}"
  rest_cuda="\${slurm_cuda_visible_devices#*,}"
  sglang_cuda_visible_devices="\${sglang_cuda_visible_devices:-\$first_cuda}"
  bench_cuda_visible_devices="\${bench_cuda_visible_devices:-\$rest_cuda}"
fi
sglang_cuda_visible_devices="\${sglang_cuda_visible_devices:-0}"
bench_cuda_visible_devices="\${bench_cuda_visible_devices:-1}"
train_cuda_visible_devices="$TRAIN_CUDA_VISIBLE_DEVICES"
if [[ -z "\$train_cuda_visible_devices" ]]; then
  train_cuda_visible_devices="\${bench_cuda_visible_devices%%,*}"
fi
sft_device_map="$SFT_DEVICE_MAP"
ttrl_device_map="$TTRL_DEVICE_MAP"
if [[ -n "$TRAIN_DEVICE_MAP" ]]; then
  sft_device_map="\${sft_device_map:-$TRAIN_DEVICE_MAP}"
  ttrl_device_map="\${ttrl_device_map:-$TRAIN_DEVICE_MAP}"
fi
if [[ -z "\$sft_device_map" ]]; then
  sft_device_map="none"
fi
ttrl_device_map="\${ttrl_device_map:-none}"
echo "SGLang CUDA_VISIBLE_DEVICES=\$sglang_cuda_visible_devices"
echo "Benchmark CUDA_VISIBLE_DEVICES=\$bench_cuda_visible_devices"
echo "Training CUDA_VISIBLE_DEVICES=\$train_cuda_visible_devices"
echo "SFT device_map=\$sft_device_map"
echo "TTRL device_map=\$ttrl_device_map"
echo "TTRL reward_channel=$TTRL_REWARD_CHANNEL"
echo "TTRL max_context_tokens=$TTRL_MAX_CONTEXT_TOKENS max_tokens=$TTRL_MAX_TOKENS"
echo "TTRL torch_dtype=$TTRL_TORCH_DTYPE bf16=$TTRL_BF16 fp16=$TTRL_FP16 gradient_checkpointing=$TTRL_GRADIENT_CHECKPOINTING"

if ! command -v uv >/dev/null 2>&1; then
  python3 -m venv "$REMOTE_ROOT/uv_bootstrap"
  "$REMOTE_ROOT/uv_bootstrap/bin/python" -m pip install --upgrade pip uv
  export PATH="$REMOTE_ROOT/uv_bootstrap/bin:\$PATH"
fi
uv sync --extra training-grpo
repo_python="$remote_repo/.venv/bin/python"

sglang_venv="$REMOTE_ROOT/sglang_venv"
if [[ ! -x "\$sglang_venv/bin/python" ]]; then
  python3 -m venv "\$sglang_venv"
  "\$sglang_venv/bin/python" -m pip install --upgrade pip
fi
"\$sglang_venv/bin/python" -m pip install "$SGLANG_PIP_SPEC" $SGLANG_PIP_EXTRA
export PATH="\$sglang_venv/bin:\$PATH"
sglang_log="$remote_logs/sglang-\${SLURM_JOB_ID:-manual}.log"
if ! ss -tln 2>/dev/null | grep -q ":$SGLANG_PORT "; then
  sglang_json_model_override_args=()
  if [[ -n '$SGLANG_JSON_MODEL_OVERRIDE_ARGS' ]]; then
    sglang_json_model_override_args=(
      --json-model-override-args
      '$SGLANG_JSON_MODEL_OVERRIDE_ARGS'
    )
  fi
  nohup env CUDA_VISIBLE_DEVICES="\$sglang_cuda_visible_devices" \\
    "\$sglang_venv/bin/python" -m sglang.launch_server \\
    --model-path "$SGLANG_MODEL" \\
    --host 127.0.0.1 \\
    --port "$SGLANG_PORT" \\
    --dtype bfloat16 \\
    --tp "$SGLANG_TP" \\
    --context-length "$SGLANG_CTX" \\
    --max-running-requests "$SGLANG_MAX_REQS" \\
    --mem-fraction-static "$SGLANG_MEM_FRAC" \\
    "\${sglang_json_model_override_args[@]}" \\
    $SGLANG_EXTRA_ARGS \\
    >"\$sglang_log" 2>&1 &
fi

for i in \$(seq 1 120); do
  if python3 - <<'PY' >/dev/null 2>&1
import urllib.request
urllib.request.urlopen("http://127.0.0.1:$SGLANG_PORT/v1/models", timeout=2).read()
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

if [[ "$RESUME_EXISTING" == "1" && -f "$OUT_DIR/benchmark_manifest.json" ]]; then
  echo "RESUME_EXISTING=1: reusing prepared benchmark at $OUT_DIR"
else
  uv run python scripts/prepare_mechanism_repair_benchmark.py \\
    --out-dir "$OUT_DIR" \\
    --tasks-per-family 5 \\
    --overwrite
fi

dry_run_args=()
if [[ "$DRY_RUN" == "1" ]]; then
  dry_run_args=(--dry-run)
fi
limit_args=()
if [[ "$LIMIT_TASKS" != "0" ]]; then
  limit_args=(--limit-tasks "$LIMIT_TASKS")
fi
resume_existing_args=()
if [[ "$RESUME_EXISTING" == "1" ]]; then
  resume_existing_args=(--resume-existing)
fi
ttrl_max_memory_args=()
if [[ -n "$TTRL_MAX_MEMORY" ]]; then
  ttrl_max_memory_args=(--ttrl-max-memory "$TTRL_MAX_MEMORY")
fi
sft_gradient_checkpointing_args=()
if [[ "$SFT_GRADIENT_CHECKPOINTING" == "1" ]]; then
  sft_gradient_checkpointing_args=(--sft-gradient-checkpointing)
fi
sft_quant_args=()
if [[ "$SFT_LOAD_IN_4BIT" == "1" ]]; then
  sft_quant_args+=(--sft-load-in-4bit)
fi
if [[ "$SFT_LOAD_IN_8BIT" == "1" ]]; then
  sft_quant_args+=(--sft-load-in-8bit)
fi
if [[ "$SFT_PREPARE_KBIT_TRAINING" == "1" ]]; then
  sft_quant_args+=(--sft-prepare-kbit-training)
  sft_quant_args+=(--sft-prepare-kbit-training-mode "$SFT_PREPARE_KBIT_TRAINING_MODE")
fi
ttrl_gradient_checkpointing_args=()
if [[ "$TTRL_GRADIENT_CHECKPOINTING" == "1" ]]; then
  ttrl_gradient_checkpointing_args=(--ttrl-gradient-checkpointing)
fi
ttrl_precision_args=()
if [[ "$TTRL_BF16" == "1" ]]; then
  ttrl_precision_args+=(--ttrl-bf16)
fi
if [[ "$TTRL_FP16" == "1" ]]; then
  ttrl_precision_args+=(--ttrl-fp16)
fi
ttrl_quant_args=()
if [[ "$TTRL_LOAD_IN_4BIT" == "1" ]]; then
  ttrl_quant_args+=(--ttrl-load-in-4bit)
fi
if [[ "$TTRL_LOAD_IN_8BIT" == "1" ]]; then
  ttrl_quant_args+=(--ttrl-load-in-8bit)
fi

exec env CUDA_VISIBLE_DEVICES="\$train_cuda_visible_devices" \\
  "\$repo_python" scripts/run_mechanism_repair_online_experiment.py \\
  --benchmark-dir "$OUT_DIR" \\
  --out-dir "$OUT_DIR" \\
  --runner-python "\$repo_python" \\
  --base-model "$BASE_MODEL" \\
  --sglang-base-url "http://127.0.0.1:$SGLANG_PORT" \\
  --methods "$METHODS" \\
  --splits "$SPLITS" \\
  --eval-seeds "$EVAL_SEEDS" \\
  --audit-retries "$AUDIT_RETRIES" \\
  "\${limit_args[@]}" \\
  "\${resume_existing_args[@]}" \\
  "\${sft_quant_args[@]}" \\
  --sft-max-steps "$SFT_MAX_STEPS" \\
  --sft-learning-rate "$SFT_LEARNING_RATE" \\
  --sft-max-grad-norm "$SFT_MAX_GRAD_NORM" \\
  --sft-max-seq-length "$SFT_MAX_SEQ_LENGTH" \\
  --sft-torch-dtype "$SFT_TORCH_DTYPE" \\
  --sft-attn-implementation "$SFT_ATTN_IMPLEMENTATION" \\
  --sft-device-map "\$sft_device_map" \\
  --sft-trust-remote-code \\
  "\${sft_gradient_checkpointing_args[@]}" \\
  "\${ttrl_quant_args[@]}" \\
  --ttrl-torch-dtype "$TTRL_TORCH_DTYPE" \\
  --ttrl-attn-implementation "$TTRL_ATTN_IMPLEMENTATION" \\
  "\${ttrl_precision_args[@]}" \\
  --ttrl-learning-rate "$TTRL_LEARNING_RATE" \\
  --ttrl-max-grad-norm "$TTRL_MAX_GRAD_NORM" \\
  --ttrl-reward-channel "$TTRL_REWARD_CHANNEL" \\
  --ttrl-kbit-prepare-mode "$TTRL_KBIT_PREPARE_MODE" \\
  --ttrl-device-map "\$ttrl_device_map" \\
  "\${ttrl_max_memory_args[@]}" \\
  "\${ttrl_gradient_checkpointing_args[@]}" \\
  --ttrl-trust-remote-code \\
  --max-context-tokens "$TTRL_MAX_CONTEXT_TOKENS" \\
  --max-tokens "$TTRL_MAX_TOKENS" \\
  "\${dry_run_args[@]}"
EOF

scp "$tmp_sbatch" "$REMOTE_HOST:$remote_sbatch"
rm -f "$tmp_sbatch"

echo "Staged current worktree at $REMOTE_HOST:$remote_repo"
echo "Wrote Slurm script at $REMOTE_HOST:$remote_sbatch"
if (( submit )); then
  ssh "$REMOTE_HOST" "sbatch '$remote_sbatch'"
else
  echo "Submit with: ssh $REMOTE_HOST sbatch '$remote_sbatch'"
fi
