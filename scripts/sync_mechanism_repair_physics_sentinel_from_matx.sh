#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-sc}"
REMOTE_ROOT="${REMOTE_ROOT:-/matx/u/knatalia/corl_mechanism_repair_physics_sentinel}"
OUT_DIR="${OUT_DIR:-runs/mechanism_repair_physics_sentinel}"
LOCAL_RUN_DIR="${LOCAL_RUN_DIR:-$OUT_DIR}"
SHARD_INDEX="${SHARD_INDEX:-0}"
PYTHON="${PYTHON:-.venv/bin/python}"
PRINT_FULL_AUDIT="${PRINT_FULL_AUDIT:-0}"

usage() {
  cat <<EOF
Usage: $0

Sync one MechanismRepair-Physics sentinel shard output from MATX to the local
sentinel run tree and refresh sentinel_audit.json. This script does not query
Slurm and is not a monitoring loop.

Useful overrides:
  REMOTE_HOST=$REMOTE_HOST
  REMOTE_ROOT=$REMOTE_ROOT
  OUT_DIR=$OUT_DIR
  LOCAL_RUN_DIR=$LOCAL_RUN_DIR
  SHARD_INDEX=$SHARD_INDEX
  PYTHON=$PYTHON
  PRINT_FULL_AUDIT=$PRINT_FULL_AUDIT
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

audit_stdout="$(mktemp)"
cleanup() {
  rm -f "$audit_stdout"
}
trap cleanup EXIT

"$PYTHON" scripts/plan_mechanism_repair_physics_sentinel.py \
  --benchmark-dir "$LOCAL_RUN_DIR" \
  --out-dir "$LOCAL_RUN_DIR" \
  --audit-only > "$audit_stdout"

if [[ "$PRINT_FULL_AUDIT" == "1" ]]; then
  cat "$audit_stdout"
else
  "$PYTHON" - "$LOCAL_RUN_DIR/sentinel_audit.json" "$SHARD_INDEX" <<'PY'
import json
import sys
from pathlib import Path

audit = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
summary = {
    "planned_cell_count": audit.get("planned_cell_count"),
    "observed_cell_count": audit.get("observed_cell_count"),
    "missing_cell_count": audit.get("missing_cell_count"),
    "duplicate_cell_count": audit.get("duplicate_cell_count"),
    "primary_pair_summary": audit.get("primary_pair_summary"),
    "decision": audit.get("decision"),
    "synced_shard_index": int(sys.argv[2]),
}
print(json.dumps(summary, indent=2, sort_keys=True))
PY
fi
