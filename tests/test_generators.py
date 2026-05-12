"""Generator-level smoke tests.

These prove that:

* The generator framework writes a valid task-contract directory.
* Reference solutions for static_fit_* and fourbar_* families pass.
* Each generator's negative controls fail with the expected codes.
* Tier 3 stubs surface ``capability_unavailable`` as documented.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest

from mech_bench.evaluator import evaluate
from mech_bench.generators.base import dumps_toml, write_task_directory
from mech_bench.generators.benchmark_suite import (
    SUITE,
    family_names,
    generate_suite,
)


@pytest.fixture(scope="module")
def suite_dir(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("suite")
    generate_suite(out, count_per_family=1, base_seed=42)
    return out


def _codes(report) -> set[str]:
    return {f.code.value if hasattr(f.code, "value") else str(f.code)
            for f in report.feedback}


def test_toml_emitter_roundtrips():
    src = {
        "task": {"id": "t", "family": "f", "difficulty": 1},
        "probes": [
            {"id": "p1", "type": "dof_grubler", "expected": 1,
             "hard_gate": True},
        ],
        "feedback": {"public_metrics": ["a", "b"]},
    }
    text = dumps_toml(src)
    parsed = tomllib.loads(text)
    assert parsed == src


def test_generate_suite_writes_every_family(suite_dir: Path):
    written = sorted(p.name for p in suite_dir.iterdir() if p.is_dir())
    fams = sorted({n.rsplit("_s", 1)[0] for n in written})
    assert fams == sorted(family_names())


def test_task_dirs_have_required_contract(suite_dir: Path):
    for task_dir in suite_dir.iterdir():
        if not task_dir.is_dir():
            continue
        for fname in ("prompt.md", "task.toml", "eval_config.toml",
                      "eval_config.public.toml", "expected_failures.json",
                      "metadata.json"):
            assert (task_dir / fname).exists(), (task_dir, fname)
        assert (task_dir / "fixtures").is_dir()
        assert (task_dir / "reference_solution" / "design.py").exists()
        assert (task_dir / "negative_solutions").is_dir()
        # task.toml round-trip — required fields.
        data = tomllib.loads((task_dir / "task.toml").read_text())
        t = data["task"]
        assert "id" in t and "family" in t and "difficulty" in t
        assert "units" in t and "tier" in t
        assert "requirements" in data
        assert "objective" in data
        cfg = tomllib.loads((task_dir / "eval_config.toml").read_text())
        assert cfg.get("probes")
        assert "feedback" in cfg
        assert "hard_gate" in cfg


@pytest.mark.parametrize("family", [
    "static_fit_bracket",
    "shaft_collar_clearance",
    "simple_hinge_fit",
    "fourbar_path",
    "slider_crank_stroke",
    "spur_gear_ratio_analytic",
    "rack_pinion_conversion",
    "belt_pulley_ratio",
])
def test_reference_solution_passes(suite_dir: Path, family: str, tmp_path):
    matches = [d for d in suite_dir.iterdir()
                if d.is_dir() and d.name.startswith(family + "_s")]
    assert matches, f"no task generated for {family!r}"
    task_dir = matches[0]
    report = evaluate(task_dir, task_dir / "reference_solution",
                       scratch_dir=tmp_path / family)
    assert report.hard_gate_passed, [f.code.value for f in report.feedback]
    assert report.evaluation_valid
    assert report.score > 0.99, report.score


@pytest.mark.parametrize("family,expected_codes", [
    ("contact_gear_pair_stub", {"capability_unavailable"}),
    ("cycloidal_lowN_stub", {"capability_unavailable"}),
])
def test_tier3_stubs_surface_capability_unavailable(
    suite_dir: Path, family: str, expected_codes: set[str], tmp_path,
):
    matches = [d for d in suite_dir.iterdir()
                if d.is_dir() and d.name.startswith(family + "_s")]
    task_dir = matches[0]
    report = evaluate(task_dir, task_dir / "reference_solution",
                       scratch_dir=tmp_path / family)
    assert not report.evaluation_valid
    assert expected_codes.issubset(_codes(report))


def test_negative_controls_match_expected(suite_dir: Path, tmp_path):
    """Every generated negative must trigger the codes its
    expected_failures.json declares (subset semantics)."""
    failures: list[str] = []
    for task_dir in suite_dir.iterdir():
        if not task_dir.is_dir():
            continue
        spec = json.loads(
            (task_dir / "expected_failures.json").read_text()
        )
        for ctrl in spec.get("controls", []) or []:
            sub = task_dir / ctrl["submission"]
            report = evaluate(
                task_dir, sub,
                scratch_dir=tmp_path / f"{task_dir.name}_{ctrl['id']}",
            )
            codes = _codes(report)
            exp_codes = set(ctrl["expected_failure_codes"])
            if not exp_codes.issubset(codes):
                failures.append(
                    f"{task_dir.name}/{ctrl['id']}: expected {exp_codes!r} "
                    f"got {codes!r}"
                )
                continue
            exp_gate = ctrl.get("expected_hard_gate_passed")
            if exp_gate is not None and bool(exp_gate) != bool(
                    report.hard_gate_passed):
                failures.append(
                    f"{task_dir.name}/{ctrl['id']}: gate {exp_gate} "
                    f"got {report.hard_gate_passed}"
                )
                continue
            exp_below = ctrl.get("expected_score_below")
            if exp_below is not None and report.score >= float(exp_below):
                failures.append(
                    f"{task_dir.name}/{ctrl['id']}: score {report.score} "
                    f"not below {exp_below}"
                )
    assert not failures, "\n".join(failures)


def test_suite_covers_at_least_three_tiers_and_five_families():
    tiers = {cls.tier for cls in SUITE}
    families = {cls.family for cls in SUITE}
    assert len(tiers) >= 3, tiers
    assert len(families) >= 5, families
