#!/usr/bin/env bash
# Single-file orchestrator for the mech_sim RL stack on a 2× RTX 3090
# host. Mirrors rl-spark/run.sh but adapted for our smaller box:
#   GPU 0 — Worldlines PEFT LoRA trainer (:18100)
#   GPU 1 — SGLang OpenAI-compatible inference server (:30000)
#   tmpfs /dev/shm hosts the venv, uv cache, HF model cache, and
#   the worldlines artifact root because the host's / is at 100%.
#
# Usage:
#   ./rl/run.sh up                start both servers
#   ./rl/run.sh up sglang         start sglang only
#   ./rl/run.sh up worldlines     start worldlines only
#   ./rl/run.sh status            one-line health for both
#   ./rl/run.sh logs sglang       tail sglang.log
#   ./rl/run.sh logs worldlines   tail worldlines.log
#   ./rl/run.sh stop              kill both
#   ./rl/run.sh smoke             3-round multi-turn GRPO smoke
#   ./rl/run.sh train             full training run (override env)

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/freiza/mech_sim}"
VENV="${WLD_VENV:-/dev/shm/wld-venv}"
RUN_DIR="${RUN_DIR:-/tmp/mech-rl}"
mkdir -p "$RUN_DIR"

# Worldlines (training)
export WLD_VENV="$VENV"
export PORT="${WLD_PORT:-18100}"
export WLD_ARTIFACTS="${WLD_ARTIFACTS:-/dev/shm/wld-artifacts}"
export WORLDLINES_API_KEY="${WORLDLINES_API_KEY:-wld-local}"
export WORLDLINES_BASE_URL="${WORLDLINES_BASE_URL:-http://127.0.0.1:$PORT}"
export WORLDLINES_DEVICE_MAP="${WORLDLINES_DEVICE_MAP:-cuda:0}"

# SGLang (inference)
SGLANG_PORT="${SGLANG_PORT:-30000}"
SGLANG_CTX="${SGLANG_CTX:-16384}"
SGLANG_MEM_FRAC="${SGLANG_MEM_FRAC:-0.5}"
SGLANG_MAX_REQS="${SGLANG_MAX_REQS:-8}"
SGLANG_BASE_URL="http://127.0.0.1:$SGLANG_PORT"

# Shared
export HF_HOME="${HF_HOME:-/dev/shm/hf-cache}"
BASE_MODEL="${BASE_MODEL:-NousResearch/DeepHermes-3-Llama-3-3B-Preview}"

_log() { echo "[run.sh] $*"; }
_die() { echo "[run.sh] error: $*" >&2; exit 1; }

start_sglang() {
  if ss -tlnp 2>/dev/null | grep -q ":$SGLANG_PORT "; then
    _log "sglang already listening on :$SGLANG_PORT"
    return 0
  fi
  _log "starting sglang on cuda:1 :$SGLANG_PORT model=$BASE_MODEL"
  nohup setsid env CUDA_VISIBLE_DEVICES=1 HF_HOME="$HF_HOME" \
    "$VENV/bin/python" -m sglang.launch_server \
    --model-path "$BASE_MODEL" \
    --host 127.0.0.1 --port "$SGLANG_PORT" \
    --dtype bfloat16 --tp 1 \
    --context-length "$SGLANG_CTX" \
    --max-running-requests "$SGLANG_MAX_REQS" \
    --mem-fraction-static "$SGLANG_MEM_FRAC" \
    --tool-call-parser llama3 \
    </dev/null >"$RUN_DIR/sglang.log" 2>&1 &
  echo $! >"$RUN_DIR/sglang.pid"
  _log "sglang pid=$(cat "$RUN_DIR/sglang.pid"), tail -f $RUN_DIR/sglang.log"
}

start_worldlines() {
  if ss -tlnp 2>/dev/null | grep -q ":$PORT "; then
    _log "worldlines already listening on :$PORT"
    return 0
  fi
  _log "starting worldlines on cuda:0 :$PORT model=$BASE_MODEL"
  nohup setsid env BASE_MODEL="$BASE_MODEL" PORT="$PORT" \
    HF_HOME="$HF_HOME" CUDA_VISIBLE_DEVICES=0 \
    WORLDLINES_DEVICE_MAP="$WORLDLINES_DEVICE_MAP" \
    bash "$REPO_ROOT/rl/launch_worldlines.sh" \
    </dev/null >"$RUN_DIR/worldlines.log" 2>&1 &
  echo $! >"$RUN_DIR/worldlines.pid"
  _log "worldlines pid=$(cat "$RUN_DIR/worldlines.pid"), tail -f $RUN_DIR/worldlines.log"
}

stop_one() {
  local name="$1"
  local pidfile="$RUN_DIR/$name.pid"
  if [[ -f "$pidfile" ]]; then
    local pid; pid="$(cat "$pidfile")"
    if kill "$pid" 2>/dev/null; then
      _log "killed $name pid=$pid"
    fi
    rm -f "$pidfile"
  fi
}

cmd_up() {
  local what="${1:-both}"
  case "$what" in
    both|all)
      start_sglang
      start_worldlines
      ;;
    sglang) start_sglang ;;
    worldlines|wld) start_worldlines ;;
    *) _die "unknown target: $what" ;;
  esac
}

cmd_stop() {
  pkill -f "sglang.launch_server" 2>/dev/null || true
  pkill -f "launch_trainer.py" 2>/dev/null || true
  stop_one sglang
  stop_one worldlines
  _log "stopped"
}

cmd_status() {
  local s w
  if curl -sS http://127.0.0.1:$SGLANG_PORT/v1/models >/dev/null 2>&1; then
    s=ready
  else
    s=DOWN
  fi
  if curl -sS http://127.0.0.1:$PORT/api/v1/get_scheduler_state \
      -H "X-API-Key: $WORLDLINES_API_KEY" >/dev/null 2>&1; then
    w=ready
  else
    w=DOWN
  fi
  echo "sglang(:$SGLANG_PORT)=$s  worldlines(:$PORT)=$w"
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
}

cmd_logs() {
  local name="${1:-sglang}"
  exec tail -F "$RUN_DIR/$name.log"
}

cmd_smoke() {
  local run_name="${RUN_NAME:-smoke_$(date +%s)}"
  _log "smoke run=$run_name"
  exec "$VENV/bin/python" "$REPO_ROOT/rl/train_grpo.py" \
    --backend-url "http://127.0.0.1:$PORT" \
    --api-key "$WORLDLINES_API_KEY" \
    --sglang-url "$SGLANG_BASE_URL" \
    --base-model "$BASE_MODEL" \
    --run-name "$run_name" \
    --rounds "${ROUNDS:-3}" \
    --tasks-per-round "${TASKS_PER_ROUND:-2}" \
    --samples-per-task "${SAMPLES_PER_TASK:-4}" \
    --max-turns "${MAX_TURNS:-3}" \
    --max-tokens-per-turn "${MAX_TOKENS:-3000}" \
    --max-context-tokens "${MAX_CTX:-15000}" \
    --rollout-temperature "${TEMP:-0.8}" \
    --top-p "${TOP_P:-0.95}" \
    --lora-rank "${LORA_RANK:-16}" \
    --lr "${LR:-1e-4}" \
    --tiers "${TIERS:-artifact_static}" \
    --checkpoint-every "${CKPT_EVERY:-0}" \
    "$@"
}

cmd_train() {
  local run_name="${RUN_NAME:-train_$(date +%s)}"
  _log "train run=$run_name"
  exec "$VENV/bin/python" "$REPO_ROOT/rl/train_grpo.py" \
    --backend-url "http://127.0.0.1:$PORT" \
    --api-key "$WORLDLINES_API_KEY" \
    --sglang-url "$SGLANG_BASE_URL" \
    --base-model "$BASE_MODEL" \
    --run-name "$run_name" \
    --rounds "${ROUNDS:-50}" \
    --tasks-per-round "${TASKS_PER_ROUND:-4}" \
    --samples-per-task "${SAMPLES_PER_TASK:-4}" \
    --max-turns "${MAX_TURNS:-4}" \
    --max-tokens-per-turn "${MAX_TOKENS:-4096}" \
    --max-context-tokens "${MAX_CTX:-15000}" \
    --rollout-temperature "${TEMP:-0.8}" \
    --top-p "${TOP_P:-0.95}" \
    --lora-rank "${LORA_RANK:-32}" \
    --lr "${LR:-1e-4}" \
    --checkpoint-every "${CKPT_EVERY:-5}" \
    "$@"
}

case "${1:-help}" in
  up)        shift; cmd_up "$@" ;;
  stop|down) cmd_stop ;;
  status)    cmd_status ;;
  logs)      shift; cmd_logs "$@" ;;
  smoke)     shift; cmd_smoke "$@" ;;
  train)     shift; cmd_train "$@" ;;
  *)
    echo "usage: $0 {up [sglang|worldlines|both]|stop|status|logs <name>|smoke|train} [args]"
    exit 1
    ;;
esac
