#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-sc}"
REMOTE_ROOT="${REMOTE_ROOT:-/matx/u/knatalia/corl_mechanism_repair_physics}"
OUT_DIR="${OUT_DIR:-runs/mechanism_repair_physics_final}"
LOCAL_RUN_DIR="${LOCAL_RUN_DIR:-$OUT_DIR}"
SHARD_INDEX="${SHARD_INDEX:-0}"
PYTHON="${PYTHON:-.venv/bin/python}"
ROLLOUT_BACKEND="${ROLLOUT_BACKEND:-transformers_local}"
LOCAL_DEVICE="${LOCAL_DEVICE:-cuda}"
LOCAL_TORCH_DTYPE="${LOCAL_TORCH_DTYPE:-bfloat16}"
WRITE_AUDIT_JSON="${WRITE_AUDIT_JSON:-1}"
AUDIT_JSON="${AUDIT_JSON:-$LOCAL_RUN_DIR/shard_resume_audit.json}"

usage() {
  cat <<EOF
Usage: $0

Sync one MechanismRepair-Physics shard output from MATX to the local run tree
and run the local shard-resume audit. This script does not query Slurm and is
not a monitoring loop.

Useful overrides:
  REMOTE_HOST=$REMOTE_HOST
  REMOTE_ROOT=$REMOTE_ROOT
  OUT_DIR=$OUT_DIR
  LOCAL_RUN_DIR=$LOCAL_RUN_DIR
  SHARD_INDEX=$SHARD_INDEX
  PYTHON=$PYTHON
  ROLLOUT_BACKEND=$ROLLOUT_BACKEND
  LOCAL_DEVICE=$LOCAL_DEVICE
  LOCAL_TORCH_DTYPE=$LOCAL_TORCH_DTYPE
  WRITE_AUDIT_JSON=$WRITE_AUDIT_JSON
  AUDIT_JSON=$AUDIT_JSON
EOF
}

case "${1:-}" in
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

if [[ ! "$SHARD_INDEX" =~ ^[0-9]+$ ]]; then
  echo "SHARD_INDEX must be a non-negative integer" >&2
  exit 2
fi
if [[ "$WRITE_AUDIT_JSON" != "0" && "$WRITE_AUDIT_JSON" != "1" ]]; then
  echo "WRITE_AUDIT_JSON must be 0 or 1" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

shard_name="$(printf 'shard_%04d' "$((10#$SHARD_INDEX))")"
remote_out_dir="$REMOTE_ROOT/repo/$OUT_DIR"
if [[ "$OUT_DIR" == /* ]]; then
  remote_out_dir="$OUT_DIR"
fi
remote_shard_dir="$remote_out_dir/shard_runs/$shard_name"
local_shard_dir="$LOCAL_RUN_DIR/shard_runs/$shard_name"

mkdir -p "$(dirname "$local_shard_dir")"
rsync -az --delete "$REMOTE_HOST:$remote_shard_dir/" "$local_shard_dir/"

audit_cmd=(
  "$PYTHON"
  scripts/plan_mechanism_repair_shard_resume.py
  --run-dir
  "$LOCAL_RUN_DIR"
  --rollout-backend
  "$ROLLOUT_BACKEND"
  --local-device
  "$LOCAL_DEVICE"
  --local-torch-dtype
  "$LOCAL_TORCH_DTYPE"
)
if [[ "$WRITE_AUDIT_JSON" == "1" ]]; then
  mkdir -p "$(dirname "$AUDIT_JSON")"
  "${audit_cmd[@]}" --out-json "$AUDIT_JSON"
else
  "${audit_cmd[@]}"
fi
