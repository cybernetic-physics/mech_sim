"""Shared dense-scoring primitives.

These import the scoring *shape* that makes GBA-Eval's replay grader produce
high-quality learning signal (see ``rl-environment-design-notes.md`` §6 and
``mech-sim-rl-improvement-notes.md`` Fix D+E):

* a **quartic sigmoid** ``score(d, tau) = 1 / (1 + (d/tau)**4)`` that is flat
  near perfect, crosses 0.5 at the threshold ``tau``, and decays to ~0 once the
  defect is clearly past the bar — giving smooth partial credit instead of a
  cliff;
* a **perceptual floor** that zeroes sub-threshold noise so many tiny defects
  do not accumulate into a large one;
* a **content-adaptive threshold** so "close enough" scales with the magnitude
  of the thing being measured (a tight target gets a tight bar).

Every function here is pure and deterministic (no RNG, no wall-clock), so the
reward is reproducible.
"""

from __future__ import annotations

import math

# Defect fraction below which a contribution is treated as imperceptible noise
# and floored to zero. Mirrors GBA-Eval's per-block perceptual floor.
DEFAULT_FLOOR_FRAC = 0.0

# Exponent of the sigmoid. 4 matches GBA-Eval's SHARPNESS — a soft step.
SHARPNESS = 4


def quartic_sigmoid(defect: float, tau: float, *, sharpness: int = SHARPNESS) -> float:
    """Map a non-negative ``defect`` to a score in ``[0, 1]``.

    ``score(0) = 1``, ``score(tau) = 0.5``, ``score(2*tau) ~= 0.06``.
    Robust to ``tau <= 0`` (returns 1.0 only for a zero defect, else 0.0) and
    to non-finite inputs (returns 0.0).
    """
    d = float(defect)
    t = float(tau)
    if not math.isfinite(d) or not math.isfinite(t):
        return 0.0
    if d <= 0.0:
        return 1.0
    if t <= 0.0:
        # No tolerance: only an exact match scores.
        return 0.0
    r = d / t
    return 1.0 / (1.0 + r ** sharpness)


def apply_floor(defect: float, tau: float, *, floor_frac: float = DEFAULT_FLOOR_FRAC) -> float:
    """Zero a defect that is below ``floor_frac * tau`` (sub-threshold noise)."""
    d = float(defect)
    t = float(tau)
    if not math.isfinite(d):
        return 0.0
    if d <= floor_frac * t:
        return 0.0
    return d


def score_from_error(
    error: float,
    tolerance: float,
    *,
    floor_frac: float = DEFAULT_FLOOR_FRAC,
    sharpness: int = SHARPNESS,
) -> float:
    """Dense score for an absolute ``error`` against an absolute ``tolerance``.

    The tolerance is the half-credit point: ``score(error=tolerance) = 0.5``.
    """
    d = abs(float(error))
    d = apply_floor(d, tolerance, floor_frac=floor_frac)
    return quartic_sigmoid(d, tolerance, sharpness=sharpness)


def score_from_error_pct(
    error_pct: float,
    tolerance_pct: float,
    *,
    floor_frac: float = DEFAULT_FLOOR_FRAC,
    sharpness: int = SHARPNESS,
) -> float:
    """Dense score from a percentage error against a percentage tolerance.

    ``score(error_pct=tolerance_pct) = 0.5``. A non-positive tolerance means
    "exact match required".
    """
    return score_from_error(
        error_pct, tolerance_pct, floor_frac=floor_frac, sharpness=sharpness
    )


def adaptive_tau(
    target_magnitude: float,
    rel_frac: float,
    *,
    tau_min: float,
    tau_max: float,
) -> float:
    """A content-adaptive threshold: ``rel_frac`` of the target's magnitude,
    clamped to ``[tau_min, tau_max]``.

    Mirrors GBA-Eval's per-replay tau (scaled to the reference's own activity,
    clamped on both ends). A larger target gets a proportionally larger bar; the
    clamps stop a near-zero target from demanding bit-exactness and a huge
    target from forgiving everything.
    """
    mag = abs(float(target_magnitude))
    tau = rel_frac * mag
    lo = float(tau_min)
    hi = float(tau_max)
    if hi < lo:
        lo, hi = hi, lo
    return max(lo, min(hi, tau))
