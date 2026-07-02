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
# terminal_recovery*.json files are local-only forensic artifacts generated
# from terminal completion.txt files. Keep --delete for the remote shard mirror,
# but protect those local recovery records from receiver-side deletion.
rsync -az --delete \
  --filter='P terminal_recovery.json' \
  --filter='P terminal_recovery_summary.json' \
  "$REMOTE_HOST:$remote_shard_dir/" "$local_shard_dir/"
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
  "$PYTHON" - "$LOCAL_RUN_DIR/sentinel_audit.json" "$SHARD_INDEX" "$local_shard_dir" <<'PY'
import json
import sys
from pathlib import Path

audit = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
shard_dir = Path(sys.argv[3])
sample_summary = {
    "summary_count": 0,
    "complete_count": 0,
    "incomplete_count": 0,
    "invalid_count": 0,
    "sample_count": 0,
    "task_count": 0,
}
for path in shard_dir.rglob("smoke_summary.json"):
    sample_summary["summary_count"] += 1
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        sample_summary["invalid_count"] += 1
        continue
    if payload.get("complete", True):
        sample_summary["complete_count"] += 1
    else:
        sample_summary["incomplete_count"] += 1
    sample_summary["sample_count"] += len(payload.get("all_samples", []) or [])
    sample_summary["task_count"] += len({
        str(row.get("task_id") or "")
        for row in payload.get("all_samples", []) or []
        if row.get("task_id")
    })
sample_checkpoints = {
    "checkpoint_count": 0,
    "invalid_count": 0,
    "task_count": 0,
}
checkpoint_tasks = set()
for path in shard_dir.rglob("sample_outcome.json"):
    sample_checkpoints["checkpoint_count"] += 1
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        sample_checkpoints["invalid_count"] += 1
        continue
    outcome = payload.get("outcome") or {}
    if isinstance(outcome, dict) and outcome.get("task_id"):
        checkpoint_tasks.add(str(outcome["task_id"]))
sample_checkpoints["task_count"] = len(checkpoint_tasks)
terminal_completions = {
    "completion_count": 0,
    "task_count": 0,
}
completion_tasks = set()
for path in shard_dir.rglob("completion.txt"):
    parts = path.parts
    if len(parts) >= 3 and parts[-3].startswith("sample_"):
        terminal_completions["completion_count"] += 1
        completion_tasks.add(parts[-2])
terminal_completions["task_count"] = len(completion_tasks)
summary = {
    "planned_cell_count": audit.get("planned_cell_count"),
    "observed_cell_count": audit.get("observed_cell_count"),
    "missing_cell_count": audit.get("missing_cell_count"),
    "duplicate_cell_count": audit.get("duplicate_cell_count"),
    "primary_pair_summary": audit.get("primary_pair_summary"),
    "decision": audit.get("decision"),
    "synced_sample_checkpoints": sample_checkpoints,
    "synced_sample_summaries": sample_summary,
    "synced_terminal_completions": terminal_completions,
    "synced_shard_index": int(sys.argv[2]),
}
print(json.dumps(summary, indent=2, sort_keys=True))
PY
fi
