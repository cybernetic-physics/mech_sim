"""Generator-level smoke tests.

These prove that:

* The generator framework writes a valid task-contract directory.
* Reference solutions for static_fit_* and fourbar_* families pass.
* Each generator's negative controls fail with the expected codes.
* Capability-unavailable Tier 3 stubs surface that status as documented.
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


# --------------------------------------------------------------------- #
# Part B additions                                                       #
# --------------------------------------------------------------------- #


_PART_B_FAMILIES = {
    # Tier 0
    "mounting_plate_hole_pitch", "flange_bolt_circle",
    "bearing_seat_clearance", "press_fit_hub_interference",
    "keyed_shaft_hub_fit", "spacer_stack_height",
    "standoff_pattern_square", "pulley_bore_alignment_static",
    "snap_tab_clearance_static", "box_lid_register_fit",
    # Tier 1
    "fourbar_crank_rocker_sweep", "fourbar_wiper_arc",
    "fourbar_straight_line_approx", "fourbar_dwell_path",
    "fourbar_pump_handle", "slider_crank_stroke_precision",
    "slider_crank_quick_return_proxy", "reciprocating_pump_plunger",
    "toggle_overcenter_margin", "rocker_limit_stop_topology",
    # Tier 2
    "compound_gear_ratio_analytic", "idler_gear_direction_analytic",
    "planetary_fixed_ring_ratio_analytic",
    "planetary_fixed_sun_ratio_analytic",
    "worm_gear_ratio_analytic", "lead_screw_linear_travel",
    "bevel_gear_ratio_analytic", "chain_sprocket_ratio",
    "timing_belt_center_distance", "rack_pinion_force_direction",
    # Tier 3 (synthetic fake-oracle stubs)
    "cam_follower_contact_stub", "ratchet_pawl_engagement_stub",
    "geneva_indexing_stub", "friction_clutch_torque_stub",
    "brake_caliper_contact_stub", "parallel_gripper_retention_stub",
    "latch_release_force_stub", "detent_spring_contact_stub",
    "gear_pair_load_trial_stub", "rack_pinion_contact_stub",
}


def test_part_b_adds_at_least_25_families():
    registered = set(family_names())
    new_families = _PART_B_FAMILIES & registered
    assert len(new_families) >= 25, sorted(new_families)


def test_part_b_reference_solutions_pass(suite_dir: Path, tmp_path):
    """Spot-check at least one reference per new tier."""
    sample_families = [
        "mounting_plate_hole_pitch",       # Tier 0
        "fourbar_crank_rocker_sweep",      # Tier 1
        "compound_gear_ratio_analytic",    # Tier 2
        "cam_follower_contact_stub",       # Tier 3
    ]
    for fam in sample_families:
        matches = [d for d in suite_dir.iterdir()
                   if d.is_dir() and d.name.startswith(fam + "_s")]
        assert matches, f"no task generated for {fam!r}"
        task_dir = matches[0]
        report = evaluate(
            task_dir, task_dir / "reference_solution",
            scratch_dir=tmp_path / fam,
        )
        assert report.evaluation_valid, (fam, [f.code.value for f in report.feedback])
        assert report.hard_gate_passed, (fam, [f.code.value for f in report.feedback])
        assert report.score > 0.5, (fam, report.score)


def test_synthetic_tier3_marks_report_synthetic(suite_dir: Path, tmp_path):
    matches = [d for d in suite_dir.iterdir()
               if d.is_dir()
               and d.name.startswith("cam_follower_contact_stub_s")]
    assert matches
    task_dir = matches[0]
    report = evaluate(
        task_dir, task_dir / "reference_solution",
        scratch_dir=tmp_path / "synthetic_tag",
    )
    assert report.oracle_is_synthetic is True


def test_fake_oracle_not_used_without_explicit_opt_in(tmp_path):
    """If a task doesn't declare ``[adapters.fake_contact_oracle]``,
    the fake oracle must not silently satisfy contact probes — even
    when it has been globally registered by an earlier task in the
    same process."""
    from mech_bench.adapters import fake_contact_oracle as fco
    fco.force_register()  # simulate earlier-task side effect.

    matches = [d for d in tmp_path.glob("**/contact_gear_pair_stub_s*")]
    # build a temporary suite so we can test
    out = tmp_path / "stubsuite"
    generate_suite(out, count_per_family=1, base_seed=1)
    matches = [d for d in out.iterdir()
               if d.is_dir() and d.name.startswith(
                   "contact_gear_pair_stub_s")]
    task_dir = matches[0]
    report = evaluate(
        task_dir, task_dir / "reference_solution",
        scratch_dir=tmp_path / "noimplicit",
    )
    codes = _codes(report)
    # The contact-gear-pair stub has no fake-oracle opt-in, so contact
    # probes must surface capability_unavailable, never missing_contact.
    assert "capability_unavailable" in codes, codes
    assert "missing_contact" not in codes, codes


def test_tier_with_no_probes_is_not_applicable(suite_dir: Path, tmp_path):
    """A four-bar / mounting plate task has no contact probes; the
    contact tier should be marked N/A, not failed."""
    matches = [d for d in suite_dir.iterdir()
               if d.is_dir()
               and d.name.startswith("mounting_plate_hole_pitch_s")]
    assert matches
    task_dir = matches[0]
    report = evaluate(
        task_dir, task_dir / "reference_solution",
        scratch_dir=tmp_path / "tier_na",
    )
    tier = report.tier_results.get("contact")
    if tier is not None:
        assert tier.get("applicable") is False
        assert tier.get("passed") in (None, False)


def test_evaluate_lightweight_bundle_skips_media(tmp_path):
    """``write_run_bundle`` defaults to no frame rendering."""
    from mech_bench.evaluator import evaluate_with_evidence, write_run_bundle
    matches = list((tmp_path / "_x").glob("*"))  # placeholder

    # Build a tiny suite of one task and run.
    suite = tmp_path / "bundle_suite"
    generate_suite(suite, count_per_family=1, base_seed=2)
    one = next(d for d in suite.iterdir()
               if d.is_dir() and d.name.startswith(
                   "mounting_plate_hole_pitch_s"))
    evidence = evaluate_with_evidence(
        one, one / "reference_solution",
        scratch_dir=tmp_path / "scr_bundle",
    )
    out_dir = tmp_path / "report_bundle"
    write_run_bundle(evidence, out_dir)
    assert (out_dir / "scorecard.json").exists()
    assert (out_dir / "dashboard_payload.json").exists()
    # Media is opt-in; no frames / mp4 / thumbnail by default.
    assert not (out_dir / "preview.mp4").exists()
    assert not (out_dir / "frames").exists()


def test_design_ir_try_from_dict_handles_malformed():
    """``DesignIR.try_from_dict`` never raises for malformed roots."""
    from mech_bench.schema import DesignIR
    for raw in (
        None, [1, 2, 3], "string", 42,
        {"schema_version": "design_ir.v2",
         "parts": "not a list", "joints": [], "ports": {}},
        {"schema_version": "design_ir.v2",
         "parts": [], "joints": [], "ports": "not a dict"},
        {"parts": [], "joints": [], "ports": {}},  # missing schema_version
    ):
        ir, errors = DesignIR.try_from_dict(raw)
        assert ir is None, raw
        assert errors, raw


def test_chrono_diagnostic_distinguishes_states():
    from mech_bench.adapters.chrono_contact import chrono_diagnostic
    diag = chrono_diagnostic()
    assert diag["adapter"] == "chrono_contact"
    assert diag["runner_status"] in (
        "ready", "skeleton_only", "missing_dependency")
    assert isinstance(diag["pychrono_importable"], bool)
    assert isinstance(diag["_chrono_impl_importable"], bool)
