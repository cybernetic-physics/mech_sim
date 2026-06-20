"""Local safety checks for the MATX physics launcher."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "submit_mechanism_repair_physics_matx.sh"
SYNC_SCRIPT = (
    PROJECT_ROOT / "scripts" / "sync_mechanism_repair_physics_shard_from_matx.sh"
)


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


def test_sync_helper_is_valid_bash() -> None:
    proc = subprocess.run(
        ["bash", "-n", str(SYNC_SCRIPT)],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr


def test_sync_helper_documents_single_shard_audit() -> None:
    text = SYNC_SCRIPT.read_text(encoding="utf-8")

    assert "This script does not query Slurm" in text
    assert 'SHARD_INDEX="${SHARD_INDEX:-0}"' in text
    assert "rsync -az --delete" in text
    assert "scripts/plan_mechanism_repair_shard_resume.py" in text
    assert "--out-json" in text


def test_help_documents_finalize_only() -> None:
    proc = run_launcher("--help")

    assert proc.returncode == 0
    assert "--finalize-only" in proc.stdout
    assert "does not submit a GPU array" in proc.stdout
    assert "AUDIT_RETRIES=0" in proc.stdout


def test_launcher_defaults_to_no_replacement_audit_retries() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert 'AUDIT_RETRIES="${AUDIT_RETRIES:-0}"' in text
    assert "--audit-retries \"$AUDIT_RETRIES\"" in text


def test_launcher_defaults_to_one_l40() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert 'GRES="${GRES:-gpu:l40s:1}"' in text


def test_launcher_defaults_to_single_local_sampler_concurrency() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert 'CONCURRENCY="${CONCURRENCY:-1}"' in text
    assert "--concurrency \"$CONCURRENCY\"" in text


def test_launcher_defaults_to_no_server_local_cuda_rollout() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert 'ROLLOUT_BACKEND="${ROLLOUT_BACKEND:-transformers_local}"' in text
    assert 'LOCAL_DEVICE="${LOCAL_DEVICE:-cuda}"' in text
    assert 'LOCAL_TORCH_DTYPE="${LOCAL_TORCH_DTYPE:-bfloat16}"' in text
    assert 'if [[ "$ROLLOUT_BACKEND" == "sglang_chat" ]]; then' in text
    assert (
        'echo "Skipping SGLang server startup for rollout backend '
        '$ROLLOUT_BACKEND"'
    ) in text
    assert '--rollout-backend "$ROLLOUT_BACKEND"' in text
    assert '--local-device "$LOCAL_DEVICE"' in text
    assert '--local-torch-dtype "$LOCAL_TORCH_DTYPE"' in text


def test_launcher_refuses_multi_gpu_gres() -> None:
    proc = run_launcher(SHARD_INDICES="0", GRES="gpu:l40s:2")

    assert proc.returncode == 2
    assert "Refusing multi-GPU MATX physics run" in proc.stderr
    assert "requested_gpu_count=2" in proc.stderr


def test_selected_shard_refreshes_code_without_restaging_outputs() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert 'REFRESH_REMOTE_CODE="${REFRESH_REMOTE_CODE:-auto}"' in text
    assert 'if [[ "$RESTAGE_REMOTE_REPO" == "0" && ! finalize_only ]]' in text
    assert "tar -xf - -C '$remote_repo'" in text


def test_launcher_can_sync_local_benchmark_without_shard_outputs() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert 'SYNC_LOCAL_BENCHMARK="${SYNC_LOCAL_BENCHMARK:-auto}"' in text
    assert 'RESET_SELECTED_SHARDS="${RESET_SELECTED_SHARDS:-0}"' in text
    assert 'rsync -az --delete \\' in text
    assert "--exclude shard_runs/" in text
    assert "--exclude shared_sft/" in text
    assert "local benchmark scaffold is incomplete" in text


def test_selected_shard_reset_requires_explicit_shard_indices() -> None:
    proc = run_launcher(RESET_SELECTED_SHARDS="1")

    assert proc.returncode == 2
    assert "RESET_SELECTED_SHARDS=1 requires SHARD_INDICES" in proc.stderr


def test_default_refuses_broad_gpu_array_before_remote_contact() -> None:
    proc = run_launcher()

    assert proc.returncode == 2
    assert "Refusing to submit a broad GPU array" in proc.stderr
    assert "array_task_count=24" in proc.stderr


def test_auto_dependents_are_disabled_until_finalize_only() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    auto_block = text.split(
        'if [[ "$SUBMIT_DEPENDENTS" == "auto" ]]',
        1,
    )[1].split("fi", 1)[0]

    assert 'SUBMIT_DEPENDENTS=0' in text
    assert "SUBMIT_DEPENDENTS=1" not in auto_block


def test_non_finalize_refuses_dependent_merge_before_remote_contact() -> None:
    proc = run_launcher(SHARD_INDICES="0", SUBMIT_DEPENDENTS="1")

    assert proc.returncode == 2
    assert "Refusing dependent merge/analysis submission" in proc.stderr
    assert "--finalize-only" in proc.stderr


def test_analysis_job_propagates_analysis_failure() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert 'echo "\\$analysis_rc" > "$OUT_DIR/analysis_exit_code.txt"' in text
    assert 'if [[ "\\$analysis_rc" != "0" ]]; then' in text
    assert 'exit "\\$analysis_rc"' in text


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
