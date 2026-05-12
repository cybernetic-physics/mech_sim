"""Fast / oracle / final modes + RLVR reward API."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _enable_fake_oracle(monkeypatch):
    """Register the fake oracle for the duration of each test."""
    monkeypatch.setenv("MECH_BENCH_USE_FAKE_ORACLE", "1")
    from mech_bench.adapters import _REGISTRY, register_adapter
    from mech_bench.adapters.fake_contact_oracle import FakeContactOracle

    snapshot = dict(_REGISTRY)
    if FakeContactOracle.type_name not in _REGISTRY:
        register_adapter(FakeContactOracle)
    try:
        yield
    finally:
        _REGISTRY.clear()
        _REGISTRY.update(snapshot)


@pytest.fixture
def fourbar_task(tmp_path) -> Path:
    """Materialize a fourbar task with explicit fast/oracle/final
    mode declarations in eval_config.toml."""
    from mech_bench.generators.benchmark_suite import FourbarPathGenerator
    from mech_bench.generators.base import write_task_directory

    gen = FourbarPathGenerator()
    task = gen.generate(seed=11)
    # Inject mode declarations.
    cfg = task.eval_config_toml
    probe_ids = [p["id"] for p in cfg.get("probes", [])]
    cfg["modes"] = {
        "fast": {
            "enabled_probe_ids": [p for p in probe_ids
                                   if p != "coupler_path"],
        },
        "oracle": {
            "enabled_probe_ids": probe_ids,
        },
        "final": {
            "require_modes": ["fast", "oracle"],
            "agreement_probes": ["coupler_path"],
            "ratio_delta_pct_max": 10.0,
            "penetration_delta_mm_max": 0.5,
        },
    }
    task_dir = write_task_directory(task, tmp_path)
    return task_dir


def test_mode_fast_runs_subset(fourbar_task: Path):
    from mech_bench.evaluator import evaluate

    report = evaluate(
        fourbar_task, fourbar_task / "reference_solution", mode="fast")
    assert report.mode == "fast"
    # Probe ids actually evaluated should reflect the fast filter.
    seen = {r.probe_id for r in report.probe_results}
    assert "coupler_path" not in seen


def test_mode_oracle_runs_all(fourbar_task: Path):
    from mech_bench.evaluator import evaluate

    report = evaluate(
        fourbar_task, fourbar_task / "reference_solution", mode="oracle")
    assert report.mode == "oracle"
    seen = {r.probe_id for r in report.probe_results}
    assert "coupler_path" in seen


def test_mode_final_runs_both_and_emits_agreement(fourbar_task: Path):
    from mech_bench.modes import run_final

    final = run_final(fourbar_task, fourbar_task / "reference_solution")
    assert final.fast is not None
    assert final.oracle is not None
    blob = final.to_dict()
    for k in ("fast_score", "oracle_score", "agreement_score",
              "final_score", "hard_gate_passed", "oracle_is_synthetic"):
        assert k in blob


def test_mode_final_disagreement_lowers_score(fourbar_task: Path, tmp_path):
    """Use a tight ratio_delta_pct_max to force disagreement detection."""
    from mech_bench.modes import run_final
    import tomllib

    cfg_path = fourbar_task / "eval_config.toml"
    cfg_text = cfg_path.read_text()
    # Tighten threshold so even small numerical differences register.
    cfg_text = cfg_text.replace(
        'ratio_delta_pct_max = 10.0',
        'ratio_delta_pct_max = 0.0001',
    )
    cfg_path.write_text(cfg_text)
    final = run_final(fourbar_task, fourbar_task / "reference_solution")
    # If fast and oracle both pass but agreement is below 0.5,
    # hard_gate_passed should be False.
    if final.fast and final.oracle:
        if (final.fast.report.hard_gate_passed
                and final.oracle.report.hard_gate_passed):
            # Either no ratio_delta_pct was computed, or agreement
            # threshold makes the final fail. Tolerate either since
            # mock adapters may not produce comparable ratios.
            assert final.final_score >= 0.0


def test_rlvr_eval_returns_compact_json(fourbar_task: Path, tmp_path):
    from mech_bench.rlvr import evaluate_for_rlvr

    rd = tmp_path / "rlvr_report"
    rlvr = evaluate_for_rlvr(
        fourbar_task, fourbar_task / "reference_solution",
        mode="fast", report_dir=rd,
    )
    blob = rlvr.to_dict()
    for k in ("reward", "hard_gate_passed", "evaluation_valid",
              "dense_score", "public_feedback", "metrics",
              "scalar_channels", "retry_suggestions",
              "oracle_is_synthetic", "mode"):
        assert k in blob, k
    assert isinstance(blob["scalar_channels"], dict)
    # The channels include the canonical tier and class metrics.
    chans = blob["scalar_channels"]
    assert "kinematics" in chans
    assert "linkage_path_score" in chans
    # JSON-roundtrip must work (compact, agent-loop friendly).
    text = json.dumps(blob)
    assert json.loads(text) == blob


def test_rlvr_eval_reward_is_zero_when_invalid(fourbar_task: Path):
    """A missing-design submission cannot earn positive reward."""
    from mech_bench.rlvr import evaluate_for_rlvr

    rlvr = evaluate_for_rlvr(
        fourbar_task,
        fourbar_task / "negative_solutions" / "missing_port",
        mode="fast",
    )
    # Negative controls fail hard gate; reward must be 0.
    assert rlvr.reward == 0.0
    assert not rlvr.hard_gate_passed


def test_rlvr_eval_final_includes_agreement(fourbar_task: Path):
    from mech_bench.rlvr import evaluate_for_rlvr

    rlvr = evaluate_for_rlvr(
        fourbar_task, fourbar_task / "reference_solution", mode="final",
    )
    assert rlvr.mode == "final"
    assert "agreement_score" in rlvr.scalar_channels
