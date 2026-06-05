"""Analytic parameter-check probe.

A mechanism-agnostic, simulator-free probe that pulls a value out of
the submitted ``DesignIR`` (or a nested part / port param) and
compares it to an expected target using one of a handful of
comparators.

Used by Tier 0 (artifact-static) and Tier 2 (transmission-analytic)
generated tasks where the verifier is "did the agent declare the right
number?" not "did the simulation produce the right number?". The probe
deliberately leaves the simulation cost at zero so these tasks scale
out cheaply.

Config::

    [[probes]]
    id = "ratio_declared"
    type = "analytic_param_check"
    path = "params.declared_ratio"      # dotted path resolved below
    expected = 4.0
    tolerance_pct = 1.0                 # optional; default 0.0 (exact)
    comparator = "eq"                   # eq | ge | le
    failure_code = "wrong_ratio"        # optional override
    hard_gate = false

Supported path roots:

* ``params.<key>``                    → ``DesignIR.params[<key>]``
* ``parts.<part_id>.params.<key>``    → that part's params[<key>]
* ``parts.<part_id>.mass_kg``         → scalar attribute
* ``ports.<port_id>.pose_local_mm[<i>]`` → tuple index
"""

from __future__ import annotations

import math
from typing import Any

from mech_bench import scoring
from mech_bench.feedback import Failure, FailureCode, Severity
from mech_bench.probes import Capability, Probe, register_probe
from mech_bench.schema import DesignIR, ProbeResult


_CODES: dict[str, FailureCode] = {
    "wrong_ratio": FailureCode.WRONG_RATIO,
    "wrong_topology": FailureCode.WRONG_TOPOLOGY,
    "insufficient_clearance": FailureCode.INSUFFICIENT_CLEARANCE,
    "insufficient_safety_factor": FailureCode.INSUFFICIENT_SAFETY_FACTOR,
    "unprintable": FailureCode.UNPRINTABLE,
    "path_error": FailureCode.PATH_ERROR,
    "invalid_artifact": FailureCode.INVALID_ARTIFACT,
}


def _split_path(path: str) -> list[str]:
    # Convert "ports.coupler.pose_local_mm[0]" into segments incl. index.
    out: list[str] = []
    buf = ""
    i = 0
    while i < len(path):
        c = path[i]
        if c == ".":
            if buf:
                out.append(buf)
                buf = ""
            i += 1
            continue
        if c == "[":
            if buf:
                out.append(buf)
                buf = ""
            j = path.find("]", i)
            if j < 0:
                buf = path[i:]
                break
            idx = path[i + 1:j]
            out.append(f"[{idx}]")
            i = j + 1
            continue
        buf += c
        i += 1
    if buf:
        out.append(buf)
    return out


def _resolve(ir: DesignIR, path: str) -> Any:
    segs = _split_path(path)
    if not segs:
        return None
    head = segs[0]
    rest = segs[1:]

    if head == "params":
        cur: Any = ir.params
    elif head == "parts":
        if not rest:
            return None
        part_id = rest[0]
        rest = rest[1:]
        cur = next((p for p in ir.parts if p.id == part_id), None)
    elif head == "ports":
        if not rest:
            return None
        port_id = rest[0]
        rest = rest[1:]
        cur = ir.ports.get(port_id)
    elif head == "joints":
        if not rest:
            return None
        joint_id = rest[0]
        rest = rest[1:]
        cur = next((j for j in ir.joints if j.id == joint_id), None)
    else:
        return None

    for seg in rest:
        if cur is None:
            return None
        if seg.startswith("[") and seg.endswith("]"):
            idx_str = seg[1:-1]
            try:
                idx = int(idx_str)
            except ValueError:
                return None
            try:
                cur = cur[idx]
            except (IndexError, KeyError, TypeError):
                return None
            continue
        if isinstance(cur, dict):
            cur = cur.get(seg)
        else:
            cur = getattr(cur, seg, None)
            if cur is None and isinstance(getattr(cur, "params", None), dict):
                pass
    return cur


def _to_float(v: Any) -> float | None:
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return f


@register_probe
class AnalyticParamCheck(Probe):
    type_name = "analytic_param_check"
    capabilities_required = frozenset({Capability.NONE})

    def run(
        self,
        ir: DesignIR,
        sim_outputs: dict[str, Any],
        config: dict[str, Any],
    ) -> ProbeResult:
        path = str(config.get("path", "")).strip()
        expected = config.get("expected")
        comparator = str(config.get("comparator", "eq")).lower()
        tolerance_pct = float(config.get("tolerance_pct", 0.0))
        tolerance_abs = float(config.get("tolerance_abs", 0.0))
        failure_code_name = str(
            config.get("failure_code", "wrong_ratio")
        ).lower()
        code = _CODES.get(failure_code_name, FailureCode.WRONG_RATIO)

        observed_raw = _resolve(ir, path)
        observed = _to_float(observed_raw)
        expected_f = _to_float(expected)

        metrics: dict[str, float] = {
            "expected": float(expected_f) if expected_f is not None else 0.0,
        }
        if observed is None:
            return ProbeResult(
                probe_id="",
                probe_type=self.type_name,
                passed=False,
                score=0.0,
                metrics=metrics,
                failures=[Failure(
                    code=FailureCode.INVALID_ARTIFACT,
                    severity=Severity.CRITICAL,
                    message=(f"analytic_param_check: path {path!r} did "
                             f"not resolve to a finite number "
                             f"({observed_raw!r})."),
                    metric=path,
                    public_hint=(
                        "Declare the requested value at this path "
                        "(numeric, finite)."
                    ),
                )],
            )

        metrics["observed"] = float(observed)
        if expected_f is None:
            return ProbeResult(
                probe_id="",
                probe_type=self.type_name,
                passed=False,
                score=0.0,
                metrics=metrics,
                failures=[Failure(
                    code=FailureCode.SCHEMA_ERROR,
                    severity=Severity.CRITICAL,
                    message=(f"analytic_param_check: probe config "
                             f"missing finite 'expected' for {path!r}."),
                )],
            )

        passed = True
        err_abs = abs(observed - expected_f)
        if abs(expected_f) > 1e-12:
            err_pct = err_abs / abs(expected_f) * 100.0
        else:
            err_pct = err_abs * 100.0
        metrics["error_abs"] = float(err_abs)
        metrics["error_pct"] = float(err_pct)

        if comparator == "ge":
            passed = observed + tolerance_abs >= expected_f
            score = 1.0 if passed else max(0.0, observed / expected_f)
        elif comparator == "le":
            passed = observed - tolerance_abs <= expected_f
            score = 1.0 if passed else max(0.0,
                                            1.0 - (observed - expected_f)
                                            / max(abs(expected_f), 1e-9))
        else:  # eq
            within_pct = (tolerance_pct > 0
                          and err_pct <= tolerance_pct)
            within_abs = (tolerance_abs > 0
                          and err_abs <= tolerance_abs)
            passed = (within_pct or within_abs
                      or (tolerance_pct == 0 and tolerance_abs == 0
                          and err_abs < 1e-9))
            # Dense score (GBA-Eval quartic sigmoid): smooth partial credit
            # that is 1.0 at zero error, 0.5 at the tolerance, and decays
            # toward 0 well past it — instead of a linear cliff to 0.
            if tolerance_pct > 0:
                score = scoring.score_from_error_pct(err_pct, tolerance_pct)
            elif tolerance_abs > 0:
                score = scoring.score_from_error(err_abs, tolerance_abs)
            else:
                score = 1.0 if passed else 0.0

        failures: list[Failure] = []
        if not passed:
            failures.append(Failure(
                code=code,
                severity=Severity.MAJOR,
                message=(f"Param {path!r}: observed {observed:g} vs "
                         f"expected {expected_f:g} ({comparator}, "
                         f"tol pct={tolerance_pct}, abs={tolerance_abs})."),
                metric=path,
                observed=float(observed),
                target=float(expected_f),
                public_hint=(
                    "Re-derive this quantity from first principles and "
                    "set it in the IR."
                ),
            ))
        return ProbeResult(
            probe_id="",
            probe_type=self.type_name,
            passed=passed,
            score=float(max(0.0, min(score, 1.0))),
            metrics=metrics,
            failures=failures,
        )
