#!/usr/bin/env python3
"""Freeze a family-held-out train/val/test split for mechanism tasks.

The split is mechanism-family based, not just task-id based.  It keeps the
paper claim honest by making the seen/unseen boundary explicit before any
training or evaluation run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mech_bench.family_splits import (
    build_family_split_manifest,
    write_family_split_files,
)


DEFAULT_SEEN = "cycloidal,belt,chain,rack_pinion,fourbar"
DEFAULT_UNSEEN = "planetary,lead_screw,cam_follower,slider_crank"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-root", default="tasks")
    parser.add_argument("--out-dir", default="runs/mechbench_family_splits")
    parser.add_argument("--seen-families", default=DEFAULT_SEEN)
    parser.add_argument("--unseen-families", default=DEFAULT_UNSEEN)
    parser.add_argument("--seed", type=int, default=20260528)
    parser.add_argument("--seen-train-ratio", type=float, default=0.8)
    parser.add_argument("--seen-val-ratio", type=float, default=0.2)
    parser.add_argument("--manifest-json", default=None)
    args = parser.parse_args()

    tasks_root = Path(args.tasks_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    manifest = build_family_split_manifest(
        tasks_root=tasks_root,
        seen_families=[s.strip() for s in args.seen_families.split(",") if s.strip()],
        unseen_families=[s.strip() for s in args.unseen_families.split(",") if s.strip()],
        seed=int(args.seed),
        seen_train_ratio=float(args.seen_train_ratio),
        seen_val_ratio=float(args.seen_val_ratio),
    )
    write_family_split_files(manifest, out_dir)
    if args.manifest_json:
        Path(args.manifest_json).expanduser().resolve().write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
