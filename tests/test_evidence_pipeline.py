"""End-to-end tests of the run evidence pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mech_bench.dashboard_payload import build_dashboard_payload
from mech_bench.evaluator import evaluate_with_evidence, write_run_bundle
from mech_bench.media import package_run
from mech_bench.traces import HAS_H5PY, TraceData


TASK_DIR = Path(__file__).resolve().parent.parent / "tasks" / "fourbar_path_t001"
REF = TASK_DIR / "reference_solution"


def _run_bundle(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    evidence = evaluate_with_evidence(
        TASK_DIR, REF, scratch_dir=tmp_path / "scr")
    out_dir = tmp_path / "report"
    paths = write_run_bundle(evidence, out_dir)
    return out_dir, paths


def test_run_bundle_writes_core_artifacts(tmp_path: Path) -> None:
    out_dir, _ = _run_bundle(tmp_path)
    assert (out_dir / "scorecard.json").exists()
    assert (out_dir / "scorecard.public.json").exists()
    assert (out_dir / "metrics.json").exists()
    assert (out_dir / "feedback.public.json").exists()
    assert (out_dir / "dashboard_payload.json").exists()
    assert (out_dir / "media_manifest.json").exists()


@pytest.mark.skipif(not HAS_H5PY, reason="h5py not installed")
def test_run_bundle_writes_traces_h5(tmp_path: Path) -> None:
    out_dir, paths = _run_bundle(tmp_path)
    assert "trace" in paths
    assert (out_dir / "traces.h5").exists()
    from mech_bench.traces import read_trace_hdf5
    td = read_trace_hdf5(out_dir / "traces.h5")
    assert td.adapter == "planar_kinematics"
    assert td.time_s.size > 0
    assert "coupler_point" in td.port_traces
    assert "input_port" in td.joint_positions


def test_dashboard_payload_contents(tmp_path: Path) -> None:
    evidence = evaluate_with_evidence(
        TASK_DIR, REF, scratch_dir=tmp_path / "scr")
    sim = next(iter(evidence.sim_outputs_by_adapter.values()), {})
    trace = TraceData.from_sim_output(
        sim,
        run_id=evidence.report.run_id,
        task_id=evidence.report.task_id,
        adapter="planar_kinematics",
    )
    payload = build_dashboard_payload(
        evidence.report, trace, task=evidence.task)

    assert payload["score"]["dense"] > 0.99
    assert payload["score"]["hard_gate_passed"] is True
    assert "coupler_path.chamfer" in payload["metrics"]
    # Feedback is a list (empty on the reference, but the key exists).
    assert isinstance(payload["feedback"], list)
    # Coupler trace must be present and look 2D.
    coupler = payload["traces"]["coupler_path"]
    assert len(coupler) > 10
    assert len(coupler[0]) == 2
    # Target path is loaded from fixtures and should also appear.
    assert "target_path" in payload["traces"]
    assert "input_angle" in payload["traces"]
    assert "output_angle" in payload["traces"]


def test_static_dashboard_writes_html(tmp_path: Path) -> None:
    plotly = pytest.importorskip("plotly")  # noqa: F841
    from mech_bench.dashboard import write_static_dashboard

    out_dir, _ = _run_bundle(tmp_path)
    payload = json.loads((out_dir / "dashboard_payload.json").read_text())
    html_path = write_static_dashboard(payload, out_dir / "dash.html")
    assert html_path.exists()
    text = html_path.read_text()
    assert "mech-bench" in text
    # Plotly leaves a div/script per figure; just confirm we got HTML.
    assert "<html" in text.lower()


def test_dashboard_html_in_bundle_when_plotly(tmp_path: Path) -> None:
    pytest.importorskip("plotly")
    out_dir, paths = _run_bundle(tmp_path)
    assert "dashboard" in paths
    assert (out_dir / "dashboard.html").exists()


def test_package_run_writes_media_manifest(tmp_path: Path) -> None:
    out_dir = tmp_path / "rd"
    out_dir.mkdir()
    # Minimal scorecard so package_run can recover identity.
    (out_dir / "scorecard.json").write_text(
        json.dumps({"run_id": "abc", "task_id": "t"}))
    files = package_run(out_dir)
    assert "media_manifest.json" in files
    manifest = json.loads(files["media_manifest.json"].read_text())
    assert manifest["run_id"] == "abc"
    assert manifest["task_id"] == "t"
    assert manifest["version"].startswith("mech_bench.media_manifest.")


def test_video_cli_reports_unavailable(tmp_path: Path) -> None:
    """Without a registered backend, `mech-bench video` should exit
    1 and emit a structured capability_unavailable warning."""
    import subprocess
    import sys

    out_dir, _ = _run_bundle(tmp_path)
    cmd = [
        sys.executable, "-m", "mech_bench", "video",
        "--report-dir", str(out_dir),
        "--out", str(tmp_path / "preview.mp4"),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert proc.returncode == 1
    warn = json.loads(proc.stderr)
    assert warn["status"] == "capability_unavailable"
