#!/usr/bin/env bash
# Launch the Worldlines backend with PEFT LoRA training + Qwen3-0.6B.
#
# Drives `scripts/launch_trainer.py` inside the worldlines repo,
# pointing artifacts at /dev/shm so the trainer doesn't fight the
# host's filled-up root partition.
#
# Prerequisite: `uv sync --group dev --group training` inside the
# worldlines repo, with UV_CACHE_DIR + UV_PROJECT_ENVIRONMENT
# pointed at /dev/shm (see rl/README.md).
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/freiza/worldlines}"
VENV="${WLD_VENV:-/dev/shm/wld-venv}"
ARTIFACTS="${WLD_ARTIFACTS:-/dev/shm/wld-artifacts}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-0.6B}"
PORT="${PORT:-8000}"
HOST="${HOST:-127.0.0.1}"

if [[ ! -d "$VENV" ]]; then
  echo "error: venv $VENV not found. Run uv sync first." >&2
  exit 1
fi

mkdir -p "$ARTIFACTS"

export WORLDLINES_API_KEY="${WORLDLINES_API_KEY:-wld-local}"
export WORLDLINES_BASE_URL="${WORLDLINES_BASE_URL:-http://${HOST}:${PORT}}"
# When we add SGLang for accelerated rollouts, set:
#   export WORLDLINES_SGLANG_BASE_URL=http://127.0.0.1:30000
# For now the local LoRA trainer + HF sampler is enough.

cd "$REPO_ROOT"
# Use our in-process monkey-patched entrypoint (see
# /home/freiza/mech_sim/rl/launch_trainer_patched.py) so the
# PEFT trainer's tensor-device hygiene bug is fixed without
# touching the worldlines submodule itself.
PATCHED="${PATCHED_ENTRY:-/home/freiza/mech_sim/rl/launch_trainer_patched.py}"
WORLDLINES_ROOT="$REPO_ROOT" \
exec "$VENV/bin/python" "$PATCHED" \
  --host "$HOST" \
  --port "$PORT" \
  --artifact-root "$ARTIFACTS" \
  --base-model "$BASE_MODEL" \
  "$@"
