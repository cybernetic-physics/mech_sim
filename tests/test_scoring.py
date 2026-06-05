"""Tests for the shared dense-scoring primitives (GBA-Eval scoring shape)."""

from __future__ import annotations

import math

import pytest

from mech_bench import scoring


def test_zero_defect_scores_one():
    assert scoring.quartic_sigmoid(0.0, 0.1) == 1.0


def test_score_half_at_tau():
    assert scoring.quartic_sigmoid(0.1, 0.1) == pytest.approx(0.5)


def test_score_small_past_tau():
    # score(2*tau) ~= 1/(1+16) ~= 0.0588
    assert scoring.quartic_sigmoid(0.2, 0.1) == pytest.approx(1.0 / 17.0)


def test_monotonic_decreasing_in_defect():
    taus = 0.1
    prev = 1.0
    for d in [0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]:
        s = scoring.quartic_sigmoid(d, taus)
        assert s <= prev + 1e-12
        prev = s


def test_nonpositive_tau_requires_exact_match():
    assert scoring.quartic_sigmoid(0.0, 0.0) == 1.0
    assert scoring.quartic_sigmoid(0.01, 0.0) == 0.0
    assert scoring.quartic_sigmoid(0.01, -1.0) == 0.0


def test_non_finite_inputs_score_zero():
    assert scoring.quartic_sigmoid(math.inf, 0.1) == 0.0
    assert scoring.quartic_sigmoid(math.nan, 0.1) == 0.0
    assert scoring.quartic_sigmoid(0.1, math.nan) == 0.0


def test_apply_floor_zeros_subthreshold():
    # floor_frac 0.15 -> defects below 0.15*tau are noise
    assert scoring.apply_floor(0.01, 1.0, floor_frac=0.15) == 0.0
    assert scoring.apply_floor(0.2, 1.0, floor_frac=0.15) == 0.2


def test_score_from_error_pct_half_at_tolerance():
    assert scoring.score_from_error_pct(5.0, 5.0) == pytest.approx(0.5)
    assert scoring.score_from_error_pct(0.0, 5.0) == 1.0
    assert scoring.score_from_error_pct(10.0, 5.0) == pytest.approx(1.0 / 17.0)


def test_score_from_error_uses_absolute_value():
    assert scoring.score_from_error(-5.0, 5.0) == pytest.approx(0.5)


def test_adaptive_tau_clamps_both_ends():
    # rel_frac 0.1 of magnitude 2.0 = 0.2, within [0.01, 1.0]
    assert scoring.adaptive_tau(2.0, 0.1, tau_min=0.01, tau_max=1.0) == pytest.approx(0.2)
    # tiny target clamps up to tau_min
    assert scoring.adaptive_tau(0.0, 0.1, tau_min=0.01, tau_max=1.0) == 0.01
    # huge target clamps down to tau_max
    assert scoring.adaptive_tau(1000.0, 0.1, tau_min=0.01, tau_max=1.0) == 1.0
