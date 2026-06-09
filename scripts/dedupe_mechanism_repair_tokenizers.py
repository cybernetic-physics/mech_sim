from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Hardlink duplicate tokenizer.json files in mechanism-repair "
            "metadata checkpoint directories to one shared copy."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="mechanism repair run directory containing shard_runs/",
    )
    parser.add_argument(
        "--interval-s",
        type=float,
        default=0.0,
        help="repeat every N seconds; default runs once",
    )
    parser.add_argument(
        "--while-slurm-jobs",
        default="",
        help="comma-separated Slurm job IDs; stop looping once none are queued",
    )
    parser.add_argument(
        "--squeue",
        default="squeue",
        help="squeue executable for --while-slurm-jobs",
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def same_file(a: Path, b: Path) -> bool:
    try:
        return os.path.samefile(a, b)
    except FileNotFoundError:
        return False


def live_slurm_jobs(job_ids: list[str], *, squeue: str) -> bool:
    if not job_ids:
        return True
    try:
        proc = subprocess.run(
            [squeue, "-h", "-j", ",".join(job_ids)],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except FileNotFoundError:
        return True
    return bool(proc.stdout.strip())


def dedupe_run(run_dir: Path) -> dict[str, int]:
    shard_root = run_dir / "shard_runs"
    shared_root = run_dir / "shared_tokenizers"
    shared_root.mkdir(parents=True, exist_ok=True)

    considered = 0
    deduped = 0
    created = 0
    skipped_incomplete = 0
    failed = 0
    bytes_saved = 0

    for tokenizer in sorted(
        shard_root.glob("shard_*/adapter_checkpoints/**/tokenizer.json")
    ):
        if not (tokenizer.parent / "checkpoint_manifest.json").is_file():
            skipped_incomplete += 1
            continue
        considered += 1
        try:
            size = tokenizer.stat().st_size
            digest = file_sha256(tokenizer)
            target = shared_root / f"tokenizer.{digest}.json"
            if not target.exists():
                tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
                shutil.copy2(tokenizer, tmp)
                os.replace(tmp, target)
                created += 1
            if same_file(tokenizer, target):
                continue
            tmp_link = tokenizer.with_name(f".tokenizer.json.dedupe.{os.getpid()}")
            if tmp_link.exists():
                tmp_link.unlink()
            os.link(target, tmp_link)
            os.replace(tmp_link, tokenizer)
            deduped += 1
            bytes_saved += size
        except OSError as exc:
            failed += 1
            print(f"dedupe_failed path={tokenizer} error={exc}", file=sys.stderr)

    return {
        "considered": considered,
        "deduped": deduped,
        "shared_created": created,
        "skipped_incomplete": skipped_incomplete,
        "failed": failed,
        "approx_bytes_saved": bytes_saved,
    }


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    job_ids = [item.strip() for item in args.while_slurm_jobs.split(",") if item.strip()]

    while True:
        if not live_slurm_jobs(job_ids, squeue=str(args.squeue)):
            print(
                time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                + " no_live_slurm_jobs exiting",
                flush=True,
            )
            return 0
        result = dedupe_run(run_dir)
        stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        print(stamp, result, flush=True)
        if args.interval_s <= 0:
            return 0
        time.sleep(float(args.interval_s))


if __name__ == "__main__":
    raise SystemExit(main())
