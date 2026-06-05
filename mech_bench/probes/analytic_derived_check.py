"""Analytic *derived* check — grade a value the agent must DERIVE, not declare.

This is the de-self-reference fix (``mech-sim-rl-improvement-notes.md`` Fix A).
``analytic_param_check`` reads a value the agent wrote into its own IR (e.g.
``params.declared_ratio``) and compares it to a target — which, when the target
is printed in the prompt, reduces to "copy the answer into a field". That is the
"easily cheatable" anti-pattern GBA-Eval's design notes warn against: there is no
independent oracle, only a self-check.

``analytic_derived_check`` instead reads the *primitives the agent must get
right* (gear tooth counts, pitch, radii — the geometry), **recomputes** the
quantity with a fixed formula, and grades the recomputed value against the task
target. The agent's own ``declared_ratio`` is never consulted, so a design that
declares the correct teeth but a wrong ratio still passes (we grade the
derivation), while wrong teeth fail. The formula is the independent oracle.

Config:
    formula:        one of the FORMULAS keys (e.g. "gear_ratio").
    inputs:         dict mapping each formula variable to an IR path the agent
                    must declare (e.g. {"teeth_in": "parts.pinion.params.teeth"}).
                    A variable may map to a LIST of paths when the formula takes
                    a product (e.g. compound gear trains).
    expected:       the task target the derivation must match.
    tolerance_pct / tolerance_abs:  half-credit point for the dense sigmoid.
    code:           failure code to emit on mismatch (default "wrong_ratio").
"""

from __future__ import annotations

from typing import Any, Callable

from mech_bench import scoring
from mech_bench.feedback import Failure, FailureCode, Severity
from mech_bench.probes import Capability, Probe, register_probe
from mech_bench.probes.analytic_param_check import _CODES, _resolve, _to_float
from mech_bench.schema import DesignIR, ProbeResult


class _DeriveError(Exception):
    """Raised when the formula cannot be evaluated from the declared inputs."""


def _need(values: dict[str, float], *names: str) -> list[float]:
    out: list[float] = []
    for n in names:
        if n not in values:
            raise _DeriveError(f"missing required input {n!r}")
        out.append(values[n])
    return out


def _f_gear_ratio(v: dict[str, float], lists: dict[str, list[float]]) -> float:
    teeth_in, teeth_out = _need(v, "teeth_in", "teeth_out")
    if teeth_in == 0:
        raise _DeriveError("teeth_in is zero")
    return teeth_out / teeth_in


def _f_product_ratio(v: dict[str, float], lists: dict[str, list[float]]) -> float:
    # ratio = product(driven) / product(driver) — compound gear trains.
    driver = lists.get("driver_teeth", [])
    driven = lists.get("driven_teeth", [])
    if not driver or not driven:
        raise _DeriveError("product_ratio needs non-empty driver_teeth/driven_teeth")
    num = 1.0
    den = 1.0
    for d in driven:
        num *= d
    for d in driver:
        den *= d
    if den == 0:
        raise _DeriveError("driver teeth product is zero")
    return num / den


def _f_belt_ratio(v: dict[str, float], lists: dict[str, list[float]]) -> float:
    r_in, r_out = _need(v, "radius_in", "radius_out")
    if r_in == 0:
        raise _DeriveError("radius_in is zero")
    return r_out / r_in


def _f_lead_screw_travel(v: dict[str, float], lists: dict[str, list[float]]) -> float:
    lead_mm, revs = _need(v, "lead_mm", "revolutions")
    return lead_mm * revs


def _f_rack_pinion_travel(v: dict[str, float], lists: dict[str, list[float]]) -> float:
    r_mm, angle_rad = _need(v, "pitch_radius_mm", "angle_rad")
    return r_mm * angle_rad


FORMULAS: dict[str, Callable[[dict[str, float], dict[str, list[float]]], float]] = {
    "gear_ratio": _f_gear_ratio,
    "product_ratio": _f_product_ratio,
    "compound_gear_ratio": _f_product_ratio,
    "belt_ratio": _f_belt_ratio,
    "lead_screw_travel": _f_lead_screw_travel,
    "rack_pinion_travel": _f_rack_pinion_travel,
}


def _fail(code: FailureCode, msg: str, hint: str, **kw: Any) -> ProbeResult:
    return ProbeResult(
        probe_id="", probe_type="analytic_derived_check",
        passed=False, score=0.0, metrics=kw.pop("metrics", {}),
        failures=[Failure(code=code, severity=Severity.CRITICAL,
                          message=msg, public_hint=hint, **kw)],
    )


@register_probe
class AnalyticDerivedCheck(Probe):
    type_name = "analytic_derived_check"
    capabilities_required = frozenset({Capability.NONE})

    def run(
        self,
        ir: DesignIR,
        sim_outputs: dict[str, Any],
        config: dict[str, Any],
    ) -> ProbeResult:
        formula = str(config.get("formula", "")).lower()
        fn = FORMULAS.get(formula)
        if fn is None:
            return _fail(
                FailureCode.SCHEMA_ERROR,
                f"analytic_derived_check: unknown formula {formula!r}.",
                "Use a supported formula key.",
            )
        expected = _to_float(config.get("expected"))
        if expected is None:
            return _fail(
                FailureCode.SCHEMA_ERROR,
                "analytic_derived_check: missing finite 'expected'.",
                "Task config must declare the target value.",
            )
        tol_pct = float(config.get("tolerance_pct", 0.0))
        tol_abs = float(config.get("tolerance_abs", 0.0))
        code = _CODES.get(str(config.get("code", "wrong_ratio")),
                          FailureCode.WRONG_RATIO)

        # Resolve the agent's DECLARED PRIMITIVES (never the declared answer).
        raw_inputs = config.get("inputs", {}) or {}
        scalars: dict[str, float] = {}
        lists: dict[str, list[float]] = {}
        for var, path in raw_inputs.items():
            if isinstance(path, (list, tuple)):
                vals: list[float] = []
                for p in path:
                    f = _to_float(_resolve(ir, str(p)))
                    if f is None:
                        return _fail(
                            FailureCode.INVALID_ARTIFACT,
                            f"analytic_derived_check: input {var!r} path "
                            f"{p!r} did not resolve to a finite number.",
                            "Declare the required geometry primitive "
                            "(numeric, finite) at this path.",
                            metric=str(p),
                        )
                    vals.append(f)
                lists[var] = vals
            else:
                f = _to_float(_resolve(ir, str(path)))
                if f is None:
                    return _fail(
                        FailureCode.INVALID_ARTIFACT,
                        f"analytic_derived_check: input {var!r} path "
                        f"{path!r} did not resolve to a finite number.",
                        "Declare the required geometry primitive "
                        "(numeric, finite) at this path.",
                        metric=str(path),
                    )
                scalars[var] = f

        try:
            derived = float(fn(scalars, lists))
        except _DeriveError as e:
            return _fail(
                FailureCode.INVALID_ARTIFACT,
                f"analytic_derived_check: {e}.",
                "Provide the inputs the formula needs.",
            )

        err_abs = abs(derived - expected)
        err_pct = (err_abs / abs(expected) * 100.0
                   if abs(expected) > 1e-12 else err_abs * 100.0)
        metrics = {
            "derived": float(derived),
            "expected": float(expected),
            "error_abs": float(err_abs),
            "error_pct": float(err_pct),
        }

        within_pct = tol_pct > 0 and err_pct <= tol_pct
        within_abs = tol_abs > 0 and err_abs <= tol_abs
        exact = tol_pct == 0 and tol_abs == 0 and err_abs < 1e-9
        passed = bool(within_pct or within_abs or exact)

        if tol_pct > 0:
            score = scoring.score_from_error_pct(err_pct, tol_pct)
        elif tol_abs > 0:
            score = scoring.score_from_error(err_abs, tol_abs)
        else:
            score = 1.0 if passed else 0.0

        failures = []
        if not passed:
            failures.append(Failure(
                code=code, severity=Severity.MAJOR,
                message=(f"Derived value {derived:.6g} from declared geometry "
                         f"disagrees with target {expected:.6g} "
                         f"({err_pct:.2f}% off)."),
                metric="derived", observed=float(derived), target=float(expected),
                public_hint=("Fix the underlying geometry (teeth / pitch / "
                             "radii) so the recomputed value matches — declaring "
                             "the answer alone does not help."),
            ))

        return ProbeResult(
            probe_id="", probe_type=self.type_name,
            passed=passed, score=float(score), metrics=metrics,
            failures=failures,
        )
