"""Local safety checks for the MATX physics launcher."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "submit_mechanism_repair_physics_matx.sh"


def run_launcher(*args: str, **env_overrides: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(env_overrides)
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_launcher_is_valid_bash() -> None:
    proc = subprocess.run(
        ["bash", "-n", str(SCRIPT)],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr


def test_help_documents_finalize_only() -> None:
    proc = run_launcher("--help")

    assert proc.returncode == 0
    assert "--finalize-only" in proc.stdout
    assert "does not submit a GPU array" in proc.stdout


def test_default_refuses_broad_gpu_array_before_remote_contact() -> None:
    proc = run_launcher()

    assert proc.returncode == 2
    assert "Refusing to submit a broad GPU array" in proc.stderr
    assert "array_task_count=24" in proc.stderr


def test_finalize_only_refuses_restaging_before_remote_contact() -> None:
    proc = run_launcher("--finalize-only", RESTAGE_REMOTE_REPO="1")

    assert proc.returncode == 2
    assert "Refusing finalize-only restage" in proc.stderr
    assert "Restaging would delete or replace the evidence" in proc.stderr


def test_finalize_only_submit_requires_local_shard_audit() -> None:
    proc = run_launcher("--finalize-only", "--submit")

    assert proc.returncode == 2
    assert "Refusing unaudited finalize-only submit" in proc.stderr
    assert "FINALIZE_AUDIT_JSON" in proc.stderr


def test_finalize_only_submit_rejects_unclean_local_shard_audit(
    tmp_path: Path,
) -> None:
    audit = tmp_path / "resume_audit.json"
    audit.write_text(
        (
            '{"schema":"mechanism_repair_physics.shard_resume_plan.v1",'
            '"merge_ready":false,"shard_count":24,'
            '"incomplete_shard_count":1,"missing_rows":1,'
            '"duplicate_rows":0,"unexpected_rows":0}\n'
        ),
        encoding="utf-8",
    )

    proc = run_launcher(
        "--finalize-only",
        "--submit",
        FINALIZE_AUDIT_JSON=str(audit),
    )

    assert proc.returncode == 2
    assert "local shard-resume audit is not clean" in proc.stderr
    assert "merge_ready is not true" in proc.stderr
