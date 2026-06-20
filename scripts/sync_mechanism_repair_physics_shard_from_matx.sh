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
PRINT_FULL_AUDIT="${PRINT_FULL_AUDIT:-0}"
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
  PRINT_FULL_AUDIT=$PRINT_FULL_AUDIT
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
if [[ "$PRINT_FULL_AUDIT" != "0" && "$PRINT_FULL_AUDIT" != "1" ]]; then
  echo "PRINT_FULL_AUDIT must be 0 or 1" >&2
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
remote_shared_sft_dir="$remote_out_dir/shared_sft"
local_shared_sft_dir="$LOCAL_RUN_DIR/shared_sft"

mkdir -p "$(dirname "$local_shard_dir")"
rsync -az --delete "$REMOTE_HOST:$remote_shard_dir/" "$local_shard_dir/"
if ssh "$REMOTE_HOST" "test -d '$remote_shared_sft_dir'"; then
  mkdir -p "$local_shared_sft_dir"
  rsync -az --delete "$REMOTE_HOST:$remote_shared_sft_dir/" "$local_shared_sft_dir/"
fi

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
  audit_stdout="$(mktemp)"
  cleanup() {
    rm -f "$audit_stdout"
  }
  trap cleanup EXIT
  "${audit_cmd[@]}" --out-json "$AUDIT_JSON" > "$audit_stdout"
  if [[ "$PRINT_FULL_AUDIT" == "1" ]]; then
    cat "$audit_stdout"
  else
    "$PYTHON" - "$AUDIT_JSON" "$SHARD_INDEX" <<'PY'
import json
import sys
from pathlib import Path

audit_path = Path(sys.argv[1])
shard_index = int(sys.argv[2])
report = json.loads(audit_path.read_text(encoding="utf-8"))
shard = next(
    (
        item
        for item in report.get("shards", [])
        if int(item.get("shard_index", -1)) == shard_index
    ),
    {},
)
summary = {
    "expected_rows": report.get("expected_rows"),
    "observed_rows": report.get("observed_rows"),
    "missing_rows": report.get("missing_rows"),
    "complete_shard_count": report.get("complete_shard_count"),
    "incomplete_shard_count": report.get("incomplete_shard_count"),
    "next_shard_index": report.get("next_shard_index"),
    "merge_ready": report.get("merge_ready"),
    "synced_shard": shard.get("shard"),
    "synced_shard_status": shard.get("status"),
    "synced_shard_observed_rows": shard.get("observed_rows"),
    "synced_shard_missing_rows": shard.get("missing_rows"),
    "synced_shard_blockers": shard.get("blockers", []),
}
print(json.dumps(summary, indent=2, sort_keys=True))
PY
  fi
else
  "${audit_cmd[@]}"
fi
