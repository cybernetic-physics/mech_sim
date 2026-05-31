#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-sc}"
REMOTE_ROOT="${REMOTE_ROOT:-/matx/u/knatalia/corl_family_generalization_goal}"
JOB_NAME="${JOB_NAME:-corl_family_goal}"
ACCOUNT="${ACCOUNT:-matx}"
PARTITION="${PARTITION:-matx}"
QOS="${QOS:-normal}"
GRES="${GRES:-gpu:l40s:4}"
CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
MEM="${MEM:-120G}"
TIME="${TIME:-3-00:00:00}"
OUT_DIR="${OUT_DIR:-runs/family_generalization_paper}"
DOCS_DIR="${DOCS_DIR:-docs}"
PREFLIGHT_DOCS_DIR="${PREFLIGHT_DOCS_DIR:-$DOCS_DIR}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.6-35B-A3B}"
SGLANG_PORT="${SGLANG_PORT:-30000}"
SGLANG_MODEL="${SGLANG_MODEL:-Qwen/Qwen3.6-35B-A3B-FP8}"
SGLANG_PIP_SPEC="${SGLANG_PIP_SPEC:-sglang==0.5.9}"
SGLANG_RECREATE_VENV="${SGLANG_RECREATE_VENV:-auto}"
SGLANG_CUDA_VISIBLE_DEVICES="${SGLANG_CUDA_VISIBLE_DEVICES:-}"
BENCH_CUDA_VISIBLE_DEVICES="${BENCH_CUDA_VISIBLE_DEVICES:-}"
SGLANG_TP="${SGLANG_TP:-1}"
TRAIN_DEVICE_MAP="${TRAIN_DEVICE_MAP:-}"
ENABLE_CHRONO_ENV="${ENABLE_CHRONO_ENV:-1}"
CHRONO_CONDA_EXE="${CHRONO_CONDA_EXE:-/matx/u/knatalia/miniconda3/bin/conda}"
CHRONO_ENV_PREFIX="${CHRONO_ENV_PREFIX:-$REMOTE_ROOT/chrono_env}"
SFT_MAX_SEQ_LENGTH="${SFT_MAX_SEQ_LENGTH:-512}"
SFT_TORCH_DTYPE="${SFT_TORCH_DTYPE:-bfloat16}"
SFT_ATTN_IMPLEMENTATION="${SFT_ATTN_IMPLEMENTATION:-eager}"
TTRL_GRPO_MAX_PROMPT_LENGTH="${TTRL_GRPO_MAX_PROMPT_LENGTH:-1024}"
TTRL_GRPO_MAX_COMPLETION_LENGTH="${TTRL_GRPO_MAX_COMPLETION_LENGTH:-512}"
TTRL_GRPO_TORCH_DTYPE="${TTRL_GRPO_TORCH_DTYPE:-bfloat16}"
TTRL_GRPO_ATTN_IMPLEMENTATION="${TTRL_GRPO_ATTN_IMPLEMENTATION:-eager}"
TTRL_GRPO_MAX_MEMORY="${TTRL_GRPO_MAX_MEMORY:-}"
SGLANG_MEM_FRAC="${SGLANG_MEM_FRAC:-0.82}"
SGLANG_CTX="${SGLANG_CTX:-16384}"
SGLANG_MAX_REQS="${SGLANG_MAX_REQS:-4}"
if [[ -z "${SGLANG_JSON_MODEL_OVERRIDE_ARGS+x}" ]]; then
  SGLANG_JSON_MODEL_OVERRIDE_ARGS='{"num_hidden_layers":40,"hidden_size":2048,"num_attention_heads":16,"num_key_value_heads":2,"head_dim":256}'
fi
SGLANG_EXTRA_ARGS="${SGLANG_EXTRA_ARGS:---trust-remote-code --served-model-name $BASE_MODEL --enable-lora --max-lora-rank 16 --lora-target-modules q_proj k_proj v_proj o_proj --attention-backend triton --sampling-backend pytorch}"
REFRESH_EVALS="${REFRESH_EVALS:-1}"

usage() {
  cat <<EOF
Usage: $0 [--submit]

Stages the current worktree to matx and writes a Slurm script that runs the
family-held-out paper benchmark through its preflight-generated resume command.

Environment overrides:
  REMOTE_HOST=$REMOTE_HOST
  REMOTE_ROOT=$REMOTE_ROOT
  ACCOUNT=$ACCOUNT
  PARTITION=$PARTITION
  QOS=$QOS
  GRES=$GRES
  CPUS_PER_TASK=$CPUS_PER_TASK
  MEM=$MEM
  TIME=$TIME
  OUT_DIR=$OUT_DIR
  DOCS_DIR=$DOCS_DIR
  BASE_MODEL=$BASE_MODEL
  SGLANG_MODEL=$SGLANG_MODEL
  SGLANG_PIP_SPEC=$SGLANG_PIP_SPEC
  SGLANG_RECREATE_VENV=$SGLANG_RECREATE_VENV
  SGLANG_CUDA_VISIBLE_DEVICES=$SGLANG_CUDA_VISIBLE_DEVICES
  BENCH_CUDA_VISIBLE_DEVICES=$BENCH_CUDA_VISIBLE_DEVICES
  SGLANG_TP=$SGLANG_TP
  TRAIN_DEVICE_MAP=$TRAIN_DEVICE_MAP
  ENABLE_CHRONO_ENV=$ENABLE_CHRONO_ENV
  CHRONO_CONDA_EXE=$CHRONO_CONDA_EXE
  CHRONO_ENV_PREFIX=$CHRONO_ENV_PREFIX
  SFT_MAX_SEQ_LENGTH=$SFT_MAX_SEQ_LENGTH
  SFT_TORCH_DTYPE=$SFT_TORCH_DTYPE
  SFT_ATTN_IMPLEMENTATION=$SFT_ATTN_IMPLEMENTATION
  TTRL_GRPO_MAX_PROMPT_LENGTH=$TTRL_GRPO_MAX_PROMPT_LENGTH
  TTRL_GRPO_MAX_COMPLETION_LENGTH=$TTRL_GRPO_MAX_COMPLETION_LENGTH
  TTRL_GRPO_TORCH_DTYPE=$TTRL_GRPO_TORCH_DTYPE
  TTRL_GRPO_ATTN_IMPLEMENTATION=$TTRL_GRPO_ATTN_IMPLEMENTATION
  TTRL_GRPO_MAX_MEMORY=$TTRL_GRPO_MAX_MEMORY
  SGLANG_JSON_MODEL_OVERRIDE_ARGS=$SGLANG_JSON_MODEL_OVERRIDE_ARGS
    Set this to an empty string to launch SGLang with the model's full
    architecture, which is required for full-size LoRA adapter compatibility.
  SGLANG_EXTRA_ARGS=$SGLANG_EXTRA_ARGS
  REFRESH_EVALS=$REFRESH_EVALS

By default the script stages files and writes the Slurm script. Pass --submit
to call sbatch on the remote host.
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
remote_sbatch="$REMOTE_ROOT/run_family_generalization.sbatch"

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
export HF_HOME="\${HF_HOME:-$REMOTE_ROOT/hf_home}"
export TRANSFORMERS_CACHE="\${TRANSFORMERS_CACHE:-$REMOTE_ROOT/hf_home/transformers}"
export HF_HUB_CACHE="\${HF_HUB_CACHE:-$REMOTE_ROOT/hf_home/hub}"
export UV_CACHE_DIR="\${UV_CACHE_DIR:-$REMOTE_ROOT/uv_cache}"
export XDG_CACHE_HOME="\${XDG_CACHE_HOME:-$REMOTE_ROOT/xdg_cache}"
export TRITON_CACHE_DIR="\${TRITON_CACHE_DIR:-$REMOTE_ROOT/triton_cache}"
export FLASHINFER_WORKSPACE_DIR="\${FLASHINFER_WORKSPACE_DIR:-$REMOTE_ROOT/flashinfer_cache}"
export FLASHINFER_WORKSPACE_BASE="\${FLASHINFER_WORKSPACE_BASE:-$REMOTE_ROOT/flashinfer_home}"
export HOME="\${JOB_HOME:-$REMOTE_ROOT/home}"
export SGLANG_DISABLE_CUDNN_CHECK="\${SGLANG_DISABLE_CUDNN_CHECK:-1}"
export USE_HUB_KERNELS="\${USE_HUB_KERNELS:-0}"
export PYTORCH_ALLOC_CONF="\${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_CUDA_ALLOC_CONF="\${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
TTRL_GRPO_MAX_MEMORY="$TTRL_GRPO_MAX_MEMORY"
mkdir -p "\$HF_HOME" "\$TRANSFORMERS_CACHE" "\$HF_HUB_CACHE" \\
  "\$UV_CACHE_DIR" "\$XDG_CACHE_HOME" "\$TRITON_CACHE_DIR" \\
  "\$FLASHINFER_WORKSPACE_DIR" "\$FLASHINFER_WORKSPACE_BASE" "\$HOME"

slurm_cuda_visible_devices="\${CUDA_VISIBLE_DEVICES:-}"
sglang_cuda_visible_devices="$SGLANG_CUDA_VISIBLE_DEVICES"
bench_cuda_visible_devices="$BENCH_CUDA_VISIBLE_DEVICES"
if [[ -n "\$slurm_cuda_visible_devices" && "\$slurm_cuda_visible_devices" == *,* ]]; then
  first_cuda="\${slurm_cuda_visible_devices%%,*}"
  rest_cuda="\${slurm_cuda_visible_devices#*,}"
  second_cuda="\${rest_cuda%%,*}"
  sglang_cuda_visible_devices="\${sglang_cuda_visible_devices:-\$first_cuda}"
  bench_cuda_visible_devices="\${bench_cuda_visible_devices:-\$rest_cuda}"
fi
sglang_cuda_visible_devices="\${sglang_cuda_visible_devices:-0}"
bench_cuda_visible_devices="\${bench_cuda_visible_devices:-1}"
train_device_map="$TRAIN_DEVICE_MAP"
if [[ -z "\$train_device_map" ]]; then
  if [[ "\$bench_cuda_visible_devices" == *,* ]]; then
    train_device_map="balanced"
  else
    train_device_map="single"
  fi
fi
echo "Slurm CUDA_VISIBLE_DEVICES=\${slurm_cuda_visible_devices:-<unset>}; SGLang CUDA_VISIBLE_DEVICES=\$sglang_cuda_visible_devices; benchmark CUDA_VISIBLE_DEVICES=\$bench_cuda_visible_devices; train device_map=\$train_device_map"

if ! command -v uv >/dev/null 2>&1; then
  python3 -m venv "$REMOTE_ROOT/uv_bootstrap"
  "$REMOTE_ROOT/uv_bootstrap/bin/python" -m pip install --upgrade pip
  "$REMOTE_ROOT/uv_bootstrap/bin/python" -m pip install uv
  export PATH="$REMOTE_ROOT/uv_bootstrap/bin:\$PATH"
fi

uv sync --extra training-grpo

repo_python="$remote_repo/.venv/bin/python"
if [[ "$ENABLE_CHRONO_ENV" == "1" ]]; then
  if [[ ! -x "$CHRONO_CONDA_EXE" ]]; then
    echo "Chrono conda executable not found: $CHRONO_CONDA_EXE" >&2
    exit 1
  fi
  if [[ ! -x "$CHRONO_ENV_PREFIX/bin/python" ]]; then
    env \\
      CONDA_PKGS_DIRS="$REMOTE_ROOT/conda_pkgs" \\
      XDG_CACHE_HOME="$REMOTE_ROOT/xdg_cache" \\
      "$CHRONO_CONDA_EXE" create -y -p "$CHRONO_ENV_PREFIX" \\
      --override-channels \\
      -c projectchrono -c conda-forge python=3.13 pychrono numpy
  fi
  export MECH_BENCH_CHRONO_PYTHON="$CHRONO_ENV_PREFIX/bin/python"
  export PYTHONPATH="$remote_repo:\${PYTHONPATH:-}"
  "\$repo_python" - <<'PY'
from mech_bench.adapters.chrono_contact import chrono_diagnostic
diag = chrono_diagnostic()
print("chrono diagnostic", diag)
if diag["status"] != "available":
    raise SystemExit(f"chrono_contact unavailable: {diag}")
PY
fi
refresh_repo_cuda_lib_path() {
  repo_cuda_lib_path="\$("\$repo_python" - <<'PY'
from pathlib import Path
import site

paths = []
for site_dir in site.getsitepackages():
    nvidia_dir = Path(site_dir) / "nvidia"
    if not nvidia_dir.exists():
        continue
    for lib_dir in sorted(nvidia_dir.glob("*/lib")):
        if lib_dir.is_dir():
            paths.append(str(lib_dir))
    cu13_lib = nvidia_dir / "cu13" / "lib"
    if cu13_lib.is_dir():
        paths.append(str(cu13_lib))
print(":".join(dict.fromkeys(paths)))
PY
)"
  if [[ -n "\$repo_cuda_lib_path" ]]; then
    export LD_LIBRARY_PATH="\$repo_cuda_lib_path:\${LD_LIBRARY_PATH:-}"
  fi
}
refresh_repo_cuda_lib_path
if ! "\$repo_python" - <<'PY' >/dev/null 2>&1
import torch
PY
then
  uv pip install --python "\$repo_python" --reinstall \\
    torch==2.9.1 nvidia-cudnn-cu12 nvidia-cusparselt-cu12 nvidia-nccl-cu12 nvidia-nvshmem-cu12
  refresh_repo_cuda_lib_path
  "\$repo_python" - <<'PY'
import torch
print("repo torch import ok", torch.__version__)
PY
fi

sglang_venv="$REMOTE_ROOT/sglang_venv"
if [[ "$SGLANG_RECREATE_VENV" == "1" ]]; then
  rm -rf "\$sglang_venv"
elif [[ "$SGLANG_RECREATE_VENV" == "auto" && -x "\$sglang_venv/bin/python" ]]; then
  if ! "\$sglang_venv/bin/python" - <<'PY' >/dev/null 2>&1
import sglang
import sglang.launch_server
PY
  then
    rm -rf "\$sglang_venv"
  fi
fi
if [[ ! -x "\$sglang_venv/bin/python" ]]; then
  python3 -m venv "\$sglang_venv"
  "\$sglang_venv/bin/python" -m pip install --upgrade pip
fi
"\$sglang_venv/bin/python" -m pip install "$SGLANG_PIP_SPEC"
export PATH="\$sglang_venv/bin:\$PATH"
"\$sglang_venv/bin/python" - <<'PY'
import sglang
import sglang.launch_server
print("SGLang import ok", getattr(sglang, "__version__", "unknown"))
PY

sglang_log="$REMOTE_ROOT/logs/sglang-\${SLURM_JOB_ID:-manual}.log"
if ! ss -tln 2>/dev/null | grep -q ":$SGLANG_PORT "; then
  echo "starting SGLang on :$SGLANG_PORT model=$SGLANG_MODEL"
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
  echo \$! > "$REMOTE_ROOT/logs/sglang-\${SLURM_JOB_ID:-manual}.pid"
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

for i in \$(seq 1 60); do
  if python3 - <<'PY' >/dev/null 2>&1
import json
import urllib.request

body = {
    "model": "$BASE_MODEL",
    "messages": [{"role": "user", "content": "Reply with exactly: ready"}],
    "max_tokens": 4,
    "temperature": 0.0,
    "stream": False,
}
req = urllib.request.Request(
    "http://127.0.0.1:$SGLANG_PORT/v1/chat/completions",
    data=json.dumps(body).encode(),
    headers={
        "Content-Type": "application/json",
        "Authorization": "Bearer dummy",
    },
    method="POST",
)
urllib.request.urlopen(req, timeout=30).read()
PY
  then
    echo "SGLang chat completions are ready"
    break
  fi
  if [[ "\$i" == 60 ]]; then
    echo "SGLang chat completions did not become ready; tailing log" >&2
    tail -200 "\$sglang_log" >&2 || true
    exit 1
  fi
  sleep 10
done

uv run python scripts/run_family_generalization_benchmark.py \\
  --out-dir "$OUT_DIR" \\
  --docs-dir "$PREFLIGHT_DOCS_DIR" \\
  --keep-out-dir \\
  --materialize-paper-tasks \\
  --paper-task-overwrite \\
  --runner-python "$remote_repo/.venv/bin/python" \\
  --base-model "$BASE_MODEL" \\
  --sft-load-in-4bit \\
  --sft-max-seq-length "$SFT_MAX_SEQ_LENGTH" \\
  --sft-torch-dtype "$SFT_TORCH_DTYPE" \\
  --sft-attn-implementation "$SFT_ATTN_IMPLEMENTATION" \\
  --sft-device-map "\$train_device_map" \\
  --sft-trust-remote-code \\
  --ttrl-grpo-load-in-4bit \\
  --ttrl-grpo-max-prompt-length "$TTRL_GRPO_MAX_PROMPT_LENGTH" \\
  --ttrl-grpo-max-completion-length "$TTRL_GRPO_MAX_COMPLETION_LENGTH" \\
  --ttrl-grpo-torch-dtype "$TTRL_GRPO_TORCH_DTYPE" \\
  --ttrl-grpo-attn-implementation "$TTRL_GRPO_ATTN_IMPLEMENTATION" \\
  --ttrl-grpo-device-map "\$train_device_map" \\
  \${TTRL_GRPO_MAX_MEMORY:+--ttrl-grpo-max-memory "\$TTRL_GRPO_MAX_MEMORY"} \\
  --ttrl-grpo-trust-remote-code \\
  --sglang-base-url "http://127.0.0.1:$SGLANG_PORT" \\
  --preflight-only

full_cmd="\$(python3 - <<'PY'
import json
from pathlib import Path
payload = json.loads(Path("$PREFLIGHT_DOCS_DIR/family_generalization_preflight.json").read_text())
print(payload["recommended_full_run_command"])
PY
)"

echo "=== family generalization full command ==="
echo "\$full_cmd"
echo "=========================================="
if [[ "$REFRESH_EVALS" == "1" ]]; then
  echo "Refreshing eval summaries while preserving trained adapters"
  rm -rf "$OUT_DIR"/eval_frozen_model \\
    "$OUT_DIR"/eval_verifier_gated \\
    "$OUT_DIR"/eval_no_update_search \\
    "$OUT_DIR"/eval_llm_evolve_no_update \\
    "$OUT_DIR"/eval_sft_model \\
    "$OUT_DIR"/eval_mechanical_evolve_ttrl
fi
exec env CUDA_VISIBLE_DEVICES="\$bench_cuda_visible_devices" bash -lc "\$full_cmd"
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
