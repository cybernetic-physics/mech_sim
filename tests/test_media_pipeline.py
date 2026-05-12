"""Media pipeline: planar renderer + ffmpeg + dashboard wiring.

Most assertions are conditional on the optional dependency being
installed — the base contract is "missing dep does not crash the
evaluation," so when matplotlib/ffmpeg are absent the renderer must
report a structured fallback instead of raising.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest


# --------------------------------------------------------------------- #
# Planar renderer                                                       #
# --------------------------------------------------------------------- #


@pytest.fixture
def planar_payload() -> dict:
    import numpy as np
    # Synthetic coupler / target paths for a fourbar-like trace.
    theta = np.linspace(0, 2 * np.pi, 60)
    coupler = np.stack([50 + 30 * np.cos(theta),
                         50 + 20 * np.sin(theta)], axis=-1)
    target = np.stack([50 + 30 * np.cos(theta + 0.1),
                       50 + 22 * np.sin(theta + 0.1)], axis=-1)
    return {
        "version": "mech_bench.dashboard_payload.v1",
        "run": {
            "run_id": "test_run", "task_id": "fourbar_test",
            "task_family": "planar_4bar", "difficulty": 2,
        },
        "score": {"dense": 0.74, "hard_gate_passed": True,
                   "evaluation_valid": True},
        "traces": {
            "coupler_path": coupler.tolist(),
            "target_path": target.tolist(),
        },
        "metrics": {"coupler_path.chamfer": 1.42,
                    "mobility.observed": 1},
    }


def test_planar_renderer_missing_matplotlib_returns_structured(
    planar_payload, tmp_path, monkeypatch,
):
    """When matplotlib is absent, the renderer must not raise."""
    # Force the missing case by patching the HAS_MATPLOTLIB flag.
    import mech_bench.rendering.planar_renderer as pr
    monkeypatch.setattr(pr, "HAS_MATPLOTLIB", False)
    res = pr.PlanarRenderer().render(planar_payload, tmp_path)
    assert res.ok is False
    assert "matplotlib" in res.reason


def test_planar_renderer_with_matplotlib_creates_frames(
    planar_payload, tmp_path,
):
    import mech_bench.rendering.planar_renderer as pr
    if not pr.HAS_MATPLOTLIB:
        pytest.skip("matplotlib not installed")
    res = pr.PlanarRenderer().render(
        planar_payload, tmp_path, n_frames=8, produce_mp4=False)
    assert res.ok
    assert res.thumbnail_png and res.thumbnail_png.exists()
    assert res.frames_dir and res.frames_dir.is_dir()
    pngs = list(res.frames_dir.glob("frame_*.png"))
    assert len(pngs) >= 4


def test_ffmpeg_missing_does_not_crash(tmp_path, monkeypatch):
    import mech_bench.rendering.ffmpeg as ff
    monkeypatch.setattr(ff, "HAS_FFMPEG", False)
    frames = tmp_path / "frames"
    frames.mkdir()
    res = ff.encode_mp4(frames, tmp_path / "out.mp4")
    assert res.ok is False
    assert "ffmpeg" in res.reason.lower()


def test_evaluate_does_not_crash_when_matplotlib_missing(monkeypatch, tmp_path):
    """The full evaluation flow must tolerate missing matplotlib."""
    import mech_bench.rendering.planar_renderer as pr
    monkeypatch.setattr(pr, "HAS_MATPLOTLIB", False)
    from mech_bench.evaluator import evaluate_with_evidence, write_run_bundle
    from mech_bench.generators.benchmark_suite import FourbarPathGenerator
    from mech_bench.generators.base import write_task_directory

    gen = FourbarPathGenerator()
    task = gen.generate(seed=2)
    task_dir = write_task_directory(task, tmp_path)
    evidence = evaluate_with_evidence(
        task_dir, task_dir / "reference_solution")
    out = tmp_path / "bundle"
    paths = write_run_bundle(evidence, out)
    # Manifest must exist regardless.
    assert "media_manifest" in paths
    manifest = json.loads(
        (out / "media_manifest.json").read_text())
    # preview_mp4 should be None when matplotlib is missing.
    assert manifest.get("preview_mp4") is None


# --------------------------------------------------------------------- #
# Dashboard payload                                                     #
# --------------------------------------------------------------------- #


def test_dashboard_payload_includes_media_block_when_rendered(
    planar_payload, tmp_path,
):
    """When the run bundle is written and matplotlib is installed, the
    payload's media block should reference preview/thumbnail/frames."""
    import mech_bench.rendering.planar_renderer as pr
    if not pr.HAS_MATPLOTLIB:
        pytest.skip("matplotlib not installed")
    from mech_bench.evaluator import evaluate_with_evidence, write_run_bundle
    from mech_bench.generators.benchmark_suite import FourbarPathGenerator
    from mech_bench.generators.base import write_task_directory

    gen = FourbarPathGenerator()
    task = gen.generate(seed=3)
    task_dir = write_task_directory(task, tmp_path)
    evidence = evaluate_with_evidence(
        task_dir, task_dir / "reference_solution")
    out = tmp_path / "bundle"
    write_run_bundle(evidence, out)
    payload = json.loads((out / "dashboard_payload.json").read_text())
    media = payload.get("media") or {}
    assert media.get("thumbnail_png")


def test_ffmpeg_missing_keeps_frames(planar_payload, tmp_path, monkeypatch):
    """When matplotlib is present but ffmpeg is not, the renderer should
    still emit frames + thumbnail and surface a structured warning."""
    import mech_bench.rendering.planar_renderer as pr
    import mech_bench.rendering.ffmpeg as ff
    if not pr.HAS_MATPLOTLIB:
        pytest.skip("matplotlib not installed")
    monkeypatch.setattr(ff, "HAS_FFMPEG", False)
    res = pr.PlanarRenderer().render(planar_payload, tmp_path,
                                       n_frames=6, produce_mp4=True)
    assert res.ok
    assert res.preview_mp4 is None
    assert res.frames_dir and res.frames_dir.is_dir()
    assert res.warnings, "expected a structured warning about ffmpeg"


def test_dashboard_html_embeds_video_path_when_present(tmp_path):
    """Static dashboard HTML must include a <video> tag referencing
    preview_mp4 when the payload's media block names it."""
    from mech_bench.dashboard import (
        HAS_PLOTLY,
        write_static_dashboard,
    )
    if not HAS_PLOTLY:
        pytest.skip("plotly not installed")
    payload = {
        "run": {"run_id": "r1", "task_id": "t1",
                 "task_family": "f", "difficulty": 1},
        "score": {"dense": 0.5, "hard_gate_passed": True,
                   "evaluation_valid": True},
        "tier_results": {}, "metrics": {}, "feedback": [],
        "traces": {},
        "media": {"preview_mp4": "preview.mp4",
                  "thumbnail_png": "thumbnail.png"},
        "probe_results": [],
    }
    out = tmp_path / "dash.html"
    write_static_dashboard(payload, out)
    html_text = out.read_text()
    assert "preview.mp4" in html_text
    assert "<video" in html_text


def test_benchmark_summary_includes_tier_and_class_metrics(tmp_path):
    """Per-task scorecards (written by run_suite) must include the new
    tier_results / class_metrics / general_metrics fields. No need to
    enable the fake oracle here — the fourbar family is contact-free.
    """
    from mech_bench.benchmark import run_suite
    from mech_bench.generators.benchmark_suite import generate_suite

    suite = tmp_path / "suite"
    generate_suite(suite, count_per_family=1, base_seed=5,
                    families=["fourbar_path"])
    rd = tmp_path / "reports"
    run_suite(suite, report_dir=rd, eval_mode="public")
    one = next(p for p in rd.iterdir()
                if p.is_dir() and p.name.startswith("fourbar_path_"))
    sc = json.loads((one / "scorecard.json").read_text())
    assert "tier_results" in sc
    assert "class_metrics" in sc
    assert "general_metrics" in sc
    assert "linkage_path_score" in sc["class_metrics"]
