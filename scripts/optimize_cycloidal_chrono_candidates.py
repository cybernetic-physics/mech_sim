#!/usr/bin/env python3
"""Run CAD/Chrono-gated cycloidal actuator design baselines.

This is the reproducible hard-result harness for the cycloidal/QDD actuator
track. It separates the cheap CPS actuator reward from the verified reward:
fast-only candidates can score well, but verified reward is zero unless the
candidate survives FreeCAD/OCCT asset generation, trusted DesignIR checks, and
real Chrono SMC contact with ``procedural_cycloidal_fallback=false``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from mech_bench.adapters import _chrono_impl
from mech_bench.geometry.cycloidal_freecad import (
    CycloidalCadExportError,
    CycloidalReducerAssets,
    audit_cycloidal_static_geometry,
    build_chrono_design_ir_from_assets,
    generate_cycloidal_reducer_assets,
)


BASELINE_PARAMS: dict[str, Any] = {
    "pins": 10,
    "line_segment_count": 42,
    "eccentricity": 2.0,
    "clearance": 0.6,
    "driver_hole_diameter": 10.0,
    "driver_circle_diameter": 50.0,
    "driver_pin_collision_shrink_mm": 0.68,
    "roller_diameter": 9.4,
    "roller_circle_diameter": 80.0,
}

FAST_REWARD_DEFAULTS: dict[str, float] = {
    "pins": 10.0,
    "eccentricity": 2.0,
    "clearance": 0.6,
    "driver_hole_diameter": 10.0,
    "driver_circle_diameter": 50.0,
    "driver_pin_collision_shrink_mm": 0.60,
    "line_segment_count": 42.0,
    "roller_diameter": 9.4,
    "roller_circle_diameter": 80.0,
}

METHOD_ORDER = (
    "seed",
    "random",
    "cma_es_fast_only",
    "verifier_gated",
)

TABLE_COLUMNS = (
    "method",
    "best_fast_reward",
    "best_verified_reward",
    "CAD pass rate",
    "Chrono pass rate",
    "lockup rate",
    "mean defect count",
)


@dataclass(frozen=True)
class Candidate:
    id: str
    method: str
    params: dict[str, Any]
    proposer: str


@dataclass(frozen=True)
class VerificationLimits:
    min_output_speed_rad_s: float = 0.5
    max_penetration_mm: float = 1.0
    max_contact_force_rms_N: float = 3000.0
    max_contacts: float = 128.0
    max_ratio_error_pct: float = 25.0
    max_power_balance_error_pct: float = 1.0e12
    max_torque_ripple_pct: float = 1.0e12


@dataclass(frozen=True)
class ChronoTrialConfig:
    input_speed_rad_s: float = 10.0
    output_load_Nm: float = 0.75
    output_load_start_s: float = 0.02
    output_load_ramp_s: float = 0.05
    friction: float = 0.0
    restitution: float = 0.0
    young_modulus: float = 1.0e8
    normal_stiffness: float = 5.0e7
    damping: float = 250.0
    contact_margin: float = 2.0e-5
    contact_envelope: float = 5.0e-5
    solver_iterations: int = 800


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        default="runs/cycloidal_verifier_gated/latest/assets",
        help="Directory for generated CAD assets and per-candidate manifests.",
    )
    parser.add_argument(
        "--results-json",
        default="runs/cycloidal_verifier_gated/latest/results.json",
        help="Path for the experiment summary JSON.",
    )
    parser.add_argument(
        "--results-csv",
        default=None,
        help="Optional path for the compact method table CSV.",
    )
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument(
        "--random-candidates",
        type=int,
        default=4,
        help="Random-search candidates to audit.",
    )
    parser.add_argument(
        "--cma-candidates",
        type=int,
        default=4,
        help="CMA-style fast-only elite candidates to audit.",
    )
    parser.add_argument(
        "--verifier-pool",
        type=int,
        default=10,
        help="Fast proposals generated for verifier-gated search.",
    )
    parser.add_argument(
        "--verifier-audit-k",
        type=int,
        default=4,
        help="Top fast-reward verifier-gated proposals to CAD/Chrono audit.",
    )
    parser.add_argument(
        "--methods",
        default=",".join(METHOD_ORDER),
        help=(
            "Comma-separated methods to run. Defaults to all baseline methods; "
            "use verifier_gated for matched-budget verifier-only comparisons."
        ),
    )
    parser.add_argument(
        "--target-chrono-audits-per-method",
        type=int,
        default=0,
        help=(
            "If positive, continue evaluating each selected method until this "
            "many candidates have real Chrono metric output, or its candidate "
            "plan is exhausted."
        ),
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=41,
        help="Chrono samples per audited candidate.",
    )
    parser.add_argument(
        "--duration-s",
        type=float,
        default=0.15,
        help="Chrono duration per audited candidate.",
    )
    parser.add_argument("--input-speed-rad-s", type=float, default=10.0)
    parser.add_argument("--output-load-Nm", type=float, default=0.75)
    parser.add_argument("--output-load-start-s", type=float, default=0.02)
    parser.add_argument("--output-load-ramp-s", type=float, default=0.05)
    parser.add_argument("--friction", type=float, default=0.0)
    parser.add_argument("--restitution", type=float, default=0.0)
    parser.add_argument("--young-modulus", type=float, default=1.0e8)
    parser.add_argument("--normal-stiffness", type=float, default=5.0e7)
    parser.add_argument("--damping", type=float, default=250.0)
    parser.add_argument("--contact-margin", type=float, default=2.0e-5)
    parser.add_argument("--contact-envelope", type=float, default=5.0e-5)
    parser.add_argument("--solver-iterations", type=int, default=800)
    parser.add_argument("--min-output-speed-rad-s", type=float, default=0.5)
    parser.add_argument("--max-penetration-mm", type=float, default=1.0)
    parser.add_argument("--max-ratio-error-pct", type=float, default=25.0)
    parser.add_argument(
        "--contact-force-limit-N",
        type=float,
        default=3000.0,
        help="Verified gate limit for RMS contact force.",
    )
    parser.add_argument(
        "--max-contacts",
        type=float,
        default=128.0,
        help="Verified gate limit for maximum contact count.",
    )
    parser.add_argument(
        "--power-balance-limit-pct",
        type=float,
        default=1.0e12,
        help="Verified gate limit for Chrono input/output power residual.",
    )
    parser.add_argument(
        "--torque-ripple-limit-pct",
        type=float,
        default=1.0e12,
        help="Verified gate limit for input torque ripple.",
    )
    parser.add_argument(
        "--require-improvement",
        action="store_true",
        help="Exit non-zero unless verifier-gated verified reward beats all baselines.",
    )
    parser.add_argument(
        "--keep-assets",
        action="store_true",
        help="Do not remove the generated CAD asset directory before running.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    if out_dir.exists() and not args.keep_assets:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_json = Path(args.results_json).expanduser().resolve()
    results_csv = (
        Path(args.results_csv).expanduser().resolve()
        if args.results_csv
        else results_json.with_suffix(".csv")
    )
    limits = VerificationLimits(
        min_output_speed_rad_s=max(0.0, float(args.min_output_speed_rad_s)),
        max_penetration_mm=max(0.0, float(args.max_penetration_mm)),
        max_contact_force_rms_N=max(0.0, float(args.contact_force_limit_N)),
        max_contacts=max(1.0, float(args.max_contacts)),
        max_ratio_error_pct=max(0.0, float(args.max_ratio_error_pct)),
        max_power_balance_error_pct=max(
            0.0, float(args.power_balance_limit_pct)),
        max_torque_ripple_pct=max(0.0, float(args.torque_ripple_limit_pct)),
    )
    trial = ChronoTrialConfig(
        input_speed_rad_s=float(args.input_speed_rad_s),
        output_load_Nm=float(args.output_load_Nm),
        output_load_start_s=max(0.0, float(args.output_load_start_s)),
        output_load_ramp_s=max(0.0, float(args.output_load_ramp_s)),
        friction=max(0.0, float(args.friction)),
        restitution=max(0.0, float(args.restitution)),
        young_modulus=max(1.0, float(args.young_modulus)),
        normal_stiffness=max(1.0, float(args.normal_stiffness)),
        damping=max(0.0, float(args.damping)),
        contact_margin=max(0.0, float(args.contact_margin)),
        contact_envelope=max(0.0, float(args.contact_envelope)),
        solver_iterations=max(1, int(args.solver_iterations)),
    )

    selected_methods = _selected_methods(args.methods)
    plans = _experiment_plans(args)
    evaluated: list[dict[str, Any]] = []
    for method in selected_methods:
        method_candidates = plans[method]
        chrono_audits = 0
        for candidate in method_candidates:
            row = _evaluate_candidate_cached(
                candidate,
                out_dir / method / candidate.id,
                samples=max(3, int(args.samples)),
                duration_s=max(1.0e-6, float(args.duration_s)),
                limits=limits,
                trial=trial,
            )
            evaluated.append(row)
            if row.get("metrics"):
                chrono_audits += 1
            print(
                "audited "
                f"method={method} id={candidate.id} "
                f"chrono_audits={chrono_audits} "
                f"verified_reward={row.get('verified_reward', 0.0)}",
                flush=True,
            )
            target = max(0, int(args.target_chrono_audits_per_method))
            if target and chrono_audits >= target:
                break

    method_table = _method_table(evaluated, limits=limits, methods=selected_methods)
    best_by_method = {
        row["method"]: _best_verified_candidate(evaluated, row["method"])
        for row in method_table
    }
    win_condition_met = _win_condition_met(best_by_method)
    summary = {
        "schema": "mech_bench.cycloidal_verifier_gated_experiment.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "claim": (
            "Cycloidal/QDD actuator candidates are scored by a cheap CPS "
            "reward, then audited through FreeCAD/OCCT CAD export, trusted "
            "DesignIR asset checks, and Chrono SMC with procedural fallback "
            "disabled. Verified reward is zero unless all hard gates pass."
        ),
        "out_dir": str(out_dir),
        "audit_config": _chrono_config_summary(args, limits, trial),
        "limits": limits.__dict__,
        "methods": selected_methods,
        "target_chrono_audits_per_method": (
            max(0, int(args.target_chrono_audits_per_method)) or None
        ),
        "method_table": method_table,
        "best_by_method": best_by_method,
        "win_condition_met": win_condition_met,
        "candidate_count": len(evaluated),
        "candidates": evaluated,
    }
    safe_summary = _json_safe(summary)
    results_json.parent.mkdir(parents=True, exist_ok=True)
    results_json.write_text(json.dumps(
        safe_summary, indent=2, sort_keys=True, allow_nan=False))
    _write_table_csv(results_csv, method_table)
    print(_format_table(method_table))
    print(f"\nresults_json={results_json}")
    print(f"results_csv={results_csv}")
    if args.require_improvement and not win_condition_met:
        return 1
    return 0


def _selected_methods(methods: str) -> list[str]:
    selected = [item.strip() for item in str(methods).split(",") if item.strip()]
    if not selected:
        selected = list(METHOD_ORDER)
    unknown = [method for method in selected if method not in METHOD_ORDER]
    if unknown:
        raise SystemExit(f"unknown method(s): {', '.join(unknown)}")
    return selected


def _experiment_plans(args: argparse.Namespace) -> dict[str, list[Candidate]]:
    rng = random.Random(int(args.seed))
    seed = [Candidate(
        id="current_seed",
        method="seed",
        params=dict(BASELINE_PARAMS),
        proposer="hand_seed_current_branch",
    )]
    random_candidates = _random_candidates(
        count=max(0, int(args.random_candidates)),
        rng=rng,
        method="random",
        prefix="random",
        proposer="uniform_parameter_search",
    )
    cma_candidates = _cma_style_fast_only_candidates(
        count=max(0, int(args.cma_candidates)),
        seed=int(args.seed) + 17,
    )
    verifier_pool = _random_candidates(
        count=max(0, int(args.verifier_pool)),
        rng=random.Random(int(args.seed) + 101),
        method="verifier_gated",
        prefix="vg_pool",
        proposer="fast_reward_pool",
    )
    verifier_pool.extend(_verifier_refinement_candidates())
    verifier_candidates = _select_verifier_candidates(
        verifier_pool,
        audit_k=max(0, int(args.verifier_audit_k)),
    )
    return {
        "seed": seed,
        "random": random_candidates,
        "cma_es_fast_only": cma_candidates,
        "verifier_gated": verifier_candidates,
    }


def _random_candidates(
    *,
    count: int,
    rng: random.Random,
    method: str,
    prefix: str,
    proposer: str,
) -> list[Candidate]:
    candidates: list[Candidate] = []
    seen: set[tuple[Any, ...]] = set()
    attempts = 0
    while len(candidates) < count and attempts < max(50, count * 20):
        attempts += 1
        params = _params_from_vector((
            rng.uniform(8.0, 13.5),
            rng.uniform(0.25, 1.15),
            rng.uniform(36.0, 58.0),
            rng.uniform(0.0, 0.82),
            rng.uniform(1.5, 3.0),
        ))
        key = _candidate_key(params)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(Candidate(
            id=f"{prefix}_{len(candidates):03d}",
            method=method,
            params=params,
            proposer=proposer,
        ))
    return candidates


def _cma_style_fast_only_candidates(
    *,
    count: int,
    seed: int,
) -> list[Candidate]:
    if count <= 0:
        return []
    rng = random.Random(seed)
    mean = [10.0, 0.6, 50.0, 0.45, 2.0]
    sigma = [1.25, 0.20, 4.5, 0.22, 0.35]
    bounds = [
        (8.0, 14.0),
        (0.25, 1.15),
        (36.0, 58.0),
        (0.0, 0.82),
        (1.5, 3.0),
    ]
    collected: dict[tuple[Any, ...], Candidate] = {}
    generation = 0
    max_generations = max(12, math.ceil(max(1, count) / 4) + 4)
    while len(collected) < count and generation < max_generations:
        population: list[tuple[float, list[float], dict[str, Any]]] = []
        for _ in range(max(6, count * 2)):
            vector = [
                _bounded(rng.gauss(mu, sd), lo, hi)
                for mu, sd, (lo, hi) in zip(mean, sigma, bounds)
            ]
            params = _params_from_vector(tuple(vector))
            score = fast_cps_actuator_reward(params)["score"]
            population.append((score, vector, params))
        population.sort(key=lambda row: row[0], reverse=True)
        elites = population[:max(2, min(4, len(population)))]
        for score, vector, params in elites:
            key = _candidate_key(params)
            if key in collected:
                continue
            collected[key] = Candidate(
                id=f"cma_fast_{len(collected):03d}",
                method="cma_es_fast_only",
                params=params,
                proposer=(
                    "dependency_free_cma_style_elite_search_fast_reward_only"
                ),
            )
            if len(collected) >= count:
                break
        mean = [
            sum(vector[i] for _, vector, _ in elites) / len(elites)
            for i in range(len(mean))
        ]
        sigma = [max(sd * 0.72, floor) for sd, floor in zip(
            sigma, [0.25, 0.04, 0.65, 0.035, 0.08])]
        generation += 1
    return list(collected.values())[:count]


def _verifier_refinement_candidates() -> list[Candidate]:
    """Deterministic local search around the CAD-valid transmission boundary."""

    boundary_base = {
        "pins": 11,
        "line_segment_count": 44,
        "eccentricity": 1.982,
        "clearance": 0.336,
        "driver_circle_diameter": 50.848,
        "driver_pin_collision_shrink_mm": 0.129,
    }
    boundary_variants = [
        ("driver_circle_0495", {"driver_circle_diameter": 49.500}),
        ("shrink_014", {"driver_pin_collision_shrink_mm": 0.140}),
        ("shrink_016", {"driver_pin_collision_shrink_mm": 0.160}),
        ("shrink_018", {"driver_pin_collision_shrink_mm": 0.180}),
        ("driver_circle_0520", {"driver_circle_diameter": 52.000}),
        ("ecc_190", {"eccentricity": 1.900}),
        ("clearance_034", {"clearance": 0.340}),
    ]
    candidates = [
        Candidate(
            id=f"vg_refine_{suffix}",
            method="verifier_gated",
            params={**boundary_base, **delta},
            proposer="deterministic_boundary_refinement",
        )
        for suffix, delta in boundary_variants
    ]
    strict_base = {
        "pins": 11,
        "line_segment_count": 44,
        "eccentricity": 2.0,
        "clearance": 0.34,
        "driver_circle_diameter": 49.5,
        "driver_pin_collision_shrink_mm": 0.13,
    }
    strict_variants = [
        ("strict_anchor", {}),
        ("strict_ecc_198", {"eccentricity": 1.98}),
        ("strict_ecc_202", {"eccentricity": 2.02}),
        ("strict_clearance_036", {"clearance": 0.36}),
        ("strict_clearance_032", {"clearance": 0.32}),
        ("strict_circle_0500", {"driver_circle_diameter": 50.0}),
        ("strict_shrink_014", {"driver_pin_collision_shrink_mm": 0.14}),
        ("strict_shrink_012", {"driver_pin_collision_shrink_mm": 0.12}),
    ]
    candidates.extend(
        Candidate(
            id=f"vg_refine_{suffix}",
            method="verifier_gated",
            params={**strict_base, **delta},
            proposer="strict_power_ripple_refinement",
        )
        for suffix, delta in strict_variants
    )
    return candidates


def _select_verifier_candidates(
    pool: Iterable[Candidate],
    *,
    audit_k: int,
) -> list[Candidate]:
    """Pick a verifier portfolio instead of blindly auditing fast-score elites."""

    if audit_k <= 0:
        return []
    unique = _unique_candidates(pool)
    strict = [
        c for c in unique if c.proposer == "strict_power_ripple_refinement"
    ]
    boundary = [
        c for c in unique if c.proposer == "deterministic_boundary_refinement"
    ]
    selected: list[Candidate] = []
    selected.extend(strict[:min(len(strict), max(1, audit_k // 2))])
    if boundary:
        selected.extend(boundary[:min(len(boundary), max(1, audit_k // 4))])
    for candidate in sorted(
        unique,
        key=lambda c: fast_cps_actuator_reward(c.params)["score"],
        reverse=True,
    ):
        selected.append(candidate)
        if len(_unique_candidates(selected)) >= audit_k:
            break
    return _unique_candidates(selected)[:audit_k]


def _unique_candidates(candidates: Iterable[Candidate]) -> list[Candidate]:
    seen: set[tuple[Any, ...]] = set()
    unique: list[Candidate] = []
    for candidate in candidates:
        key = _candidate_key(candidate.params)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def _params_from_vector(
    vector: tuple[float, float, float, float, float],
) -> dict[str, Any]:
    pins_raw, clearance, driver_circle, shrink, eccentricity = vector
    pins = max(8, min(14, int(round(pins_raw))))
    line_segments = max(42, pins * 4)
    return {
        "pins": pins,
        "line_segment_count": line_segments,
        "eccentricity": round(float(eccentricity), 3),
        "clearance": round(float(clearance), 3),
        "driver_circle_diameter": round(float(driver_circle), 3),
        "driver_pin_collision_shrink_mm": round(float(shrink), 3),
    }


def _candidate_key(params: dict[str, Any]) -> tuple[Any, ...]:
    p = _scoring_params(params)
    return (
        int(round(p["pins"])),
        round(p["eccentricity"], 3),
        round(p["clearance"], 3),
        round(p["driver_circle_diameter"], 3),
        round(p["driver_pin_collision_shrink_mm"], 3),
    )


def fast_cps_actuator_reward(params: dict[str, Any]) -> dict[str, Any]:
    """Cheap deterministic CPS reward proxy for cycloidal/QDD candidates.

    The reward is intentionally not a verifier. It captures the fast design
    objective a search loop can optimize before CAD and contact validation:
    target ratio, clearance/manufacturability, compactness, and a coarse
    torque-density proxy.
    """

    p = _scoring_params(params)
    pins = int(round(p["pins"]))
    ratio = max(1.0, float(pins - 1))
    ratio_score = _clamped(1.0 - abs(ratio - 9.0) / 4.0)
    eccentricity_score = _triangular_score(
        p["eccentricity"], ideal=2.2, width=1.1)
    clearance_score = _triangular_score(p["clearance"], ideal=0.58, width=0.65)
    shrink_score = _triangular_score(
        p["driver_pin_collision_shrink_mm"], ideal=0.42, width=0.55)
    engagement_score = _clamped(
        1.0 - p["driver_pin_collision_shrink_mm"] / 0.82)
    compact_score = _clamped(
        1.0 - max(0.0, p["driver_circle_diameter"] - 44.0) / 24.0)
    segment_score = _clamped(p["line_segment_count"] / max(42.0, pins * 4.0))
    torque_density_proxy = _clamped(
        (pins / 10.0)
        * (p["driver_circle_diameter"] / 50.0)
        * (p["eccentricity"] / 2.2)
        * _clamped(1.0 - abs(p["clearance"] - 0.55) / 0.9)
    )
    manufacturability = (
        0.50 * clearance_score
        + 0.25 * shrink_score
        + 0.15 * eccentricity_score
        + 0.10 * segment_score
    )
    score = 100.0 * (
        0.24 * ratio_score
        + 0.22 * torque_density_proxy
        + 0.20 * manufacturability
        + 0.12 * compact_score
        + 0.10 * engagement_score
        + 0.10 * _clamped((pins - 7.0) / 5.0)
        + 0.02 * _clamped(1.0 - abs(p["driver_circle_diameter"] - 50.0) / 16.0)
    )
    return {
        "score": round(float(score), 6),
        "components": {
            "target_ratio": round(float(ratio_score), 6),
            "torque_density_proxy": round(float(torque_density_proxy), 6),
            "manufacturability": round(float(manufacturability), 6),
            "compactness": round(float(compact_score), 6),
            "engagement": round(float(engagement_score), 6),
            "eccentricity": round(float(eccentricity_score), 6),
            "segment_resolution": round(float(segment_score), 6),
        },
        "scoring_params": p,
    }


def _evaluate_candidate_cached(
    candidate: Candidate,
    candidate_dir: Path,
    *,
    samples: int,
    duration_s: float,
    limits: VerificationLimits,
    trial: ChronoTrialConfig | None = None,
) -> dict[str, Any]:
    trial = trial or ChronoTrialConfig()
    cache_path = candidate_dir / "evaluation_result.json"
    signature = _evaluation_signature(
        candidate=candidate,
        samples=samples,
        duration_s=duration_s,
        limits=limits,
        trial=trial,
    )
    if cache_path.is_file():
        try:
            cached = json.loads(cache_path.read_text())
            if cached.get("cache_signature") == signature:
                cached["cache_reused"] = True
                return cached
        except Exception:
            pass
    row = _evaluate_candidate(
        candidate,
        candidate_dir,
        samples=samples,
        duration_s=duration_s,
        limits=limits,
        trial=trial,
    )
    row["cache_signature"] = signature
    row["cache_reused"] = False
    candidate_dir.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(_json_safe(row), indent=2, sort_keys=True, allow_nan=False))
    tmp.replace(cache_path)
    return row


def _evaluation_signature(
    *,
    candidate: Candidate,
    samples: int,
    duration_s: float,
    limits: VerificationLimits,
    trial: ChronoTrialConfig,
) -> dict[str, Any]:
    return {
        "candidate": {
            "id": candidate.id,
            "method": candidate.method,
            "params": candidate.params,
            "proposer": candidate.proposer,
        },
        "samples": int(samples),
        "duration_s": float(duration_s),
        "limits": limits.__dict__,
        "trial": trial.__dict__,
        "contact_model": "smc",
        "procedural_cycloidal_fallback": False,
    }


def _evaluate_candidate(
    candidate: Candidate,
    candidate_dir: Path,
    *,
    samples: int,
    duration_s: float,
    limits: VerificationLimits,
    trial: ChronoTrialConfig | None = None,
) -> dict[str, Any]:
    trial = trial or ChronoTrialConfig()
    fast_reward = fast_cps_actuator_reward(candidate.params)
    result: dict[str, Any] = {
        "id": candidate.id,
        "method": candidate.method,
        "proposer": candidate.proposer,
        "params": candidate.params,
        "asset_dir": str(candidate_dir),
        "fast_reward": fast_reward["score"],
        "fast_reward_detail": fast_reward,
        "cad_generated": False,
        "cad_static_ok": False,
        "chrono_real_geometry": False,
        "verified_gate_passed": False,
        "paper_grade_passed": False,
        "verified_reward": 0.0,
        "defects": [],
        "defect_count": 0,
    }
    try:
        assets = generate_cycloidal_reducer_assets(
            candidate_dir,
            candidate.params,
            timeout_s=300.0,
        )
        result["cad_generated"] = True
        result["manifest_path"] = str(assets.manifest_path)
        audit = audit_cycloidal_static_geometry(assets)
        result["cad_static_audit"] = _static_audit_summary(audit)
        result["cad_static_ok"] = _static_audit_ok(audit)
        ir = build_chrono_design_ir_from_assets(
            assets,
            include_secondary_disc=True,
            collision_sweep_radius_m=2.0e-5,
            use_cad_collision_primitives=False,
            use_cad_eccentric_body_frames=True,
            use_cad_outer_sidewall_collision=True,
            cad_outer_sidewall_thickness_mm=0.75,
            cad_outer_sidewall_max_hulls=128,
        )
        out = _chrono_impl.run(
            ir,
            _chrono_config(
                assets,
                samples=samples,
                duration_s=duration_s,
                limits=limits,
                trial=trial,
            ),
        )
        metadata = out.get("metadata", {})
        metrics = out.get("scalar_metrics", {})
        result["chrono_real_geometry"] = _chrono_real_geometry(out)
        result["chrono_preflight_issues"] = metadata.get(
            "preflight_issues", [])
        result["chrono_execution_mode"] = metadata.get("execution_mode")
        result["metrics"] = _metric_summary(metrics)
        result["verified_gate_passed"] = verified_gate_passed(
            cad_generated=result["cad_generated"],
            cad_static_ok=result["cad_static_ok"],
            chrono_real_geometry=result["chrono_real_geometry"],
            metrics=metrics,
            limits=limits,
        )
        result["paper_grade_passed"] = _paper_grade_passed(metrics, limits)
        result["verified_reward"] = verified_reward(
            fast_reward=float(fast_reward["score"]),
            cad_generated=result["cad_generated"],
            cad_static_ok=result["cad_static_ok"],
            chrono_real_geometry=result["chrono_real_geometry"],
            metrics=metrics,
            limits=limits,
        )
    except CycloidalCadExportError as exc:
        result["error_stage"] = exc.stage
        result["error"] = str(exc)
    except Exception as exc:  # noqa: BLE001 - experiment boundary
        result["error_stage"] = "optimizer_audit"
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["defects"] = defect_list(result, limits)
    result["defect_count"] = len(result["defects"])
    return result


def _chrono_config(
    assets: CycloidalReducerAssets,
    *,
    samples: int,
    duration_s: float,
    limits: VerificationLimits,
    trial: ChronoTrialConfig,
) -> dict[str, Any]:
    return {
        "samples": samples,
        "duration_s": duration_s,
        "timestep": 2.5e-5,
        "contact_model": "smc",
        "contact_method": "SMC",
        "procedural_cycloidal_fallback": False,
        "contact_margin": trial.contact_margin,
        "contact_envelope": trial.contact_envelope,
        "friction": trial.friction,
        "restitution": trial.restitution,
        "young_modulus": trial.young_modulus,
        "normal_stiffness": trial.normal_stiffness,
        "damping": trial.damping,
        "solver_iterations": trial.solver_iterations,
        "solver_max_iterations": trial.solver_iterations,
        "collision_filter_named_pairs": True,
        "_mech_bench": {
            "build_root": str(assets.root),
            "task": {
                "id": "cycloidal_lowN_stub_s0001",
                "family": "cycloidal_lowN_stub",
                "difficulty": 3,
                "units": "mm",
            },
            "probe_specs": [
                {
                    "id": "torque",
                    "type": "torque_load_trial",
                    "config": {
                        "input_port": "input_port",
                        "output_port": "output_port",
                        "input_speed_rad_s": trial.input_speed_rad_s,
                        "output_load_Nm": trial.output_load_Nm,
                        "output_load_model": "passive_brake",
                        "output_load_start_s": trial.output_load_start_s,
                        "output_load_ramp_s": trial.output_load_ramp_s,
                        "min_output_speed_rad_s": (
                            limits.min_output_speed_rad_s
                        ),
                        "max_power_error_pct": (
                            limits.max_power_balance_error_pct
                        ),
                        "max_torque_ripple_pct": (
                            limits.max_torque_ripple_pct
                        ),
                    },
                }
            ],
        },
    }


def _chrono_config_summary(
    args: argparse.Namespace,
    limits: VerificationLimits,
    trial: ChronoTrialConfig,
) -> dict[str, Any]:
    return {
        "contact_model": "smc",
        "procedural_cycloidal_fallback": False,
        "output_load_model": "passive_brake",
        "output_load_Nm": trial.output_load_Nm,
        "input_speed_rad_s": trial.input_speed_rad_s,
        "output_load_start_s": trial.output_load_start_s,
        "output_load_ramp_s": trial.output_load_ramp_s,
        "duration_s": float(args.duration_s),
        "samples": int(args.samples),
        "timestep": 2.5e-5,
        "young_modulus": trial.young_modulus,
        "normal_stiffness": trial.normal_stiffness,
        "damping": trial.damping,
        "friction": trial.friction,
        "restitution": trial.restitution,
        "contact_margin": trial.contact_margin,
        "contact_envelope": trial.contact_envelope,
        "solver_iterations": trial.solver_iterations,
        "verified_limits": limits.__dict__,
    }


def _chrono_real_geometry(out: dict[str, Any]) -> bool:
    metadata = out.get("metadata", {})
    return (
        not bool(out.get("__capability_unavailable__"))
        and not bool(out.get("__adapter_error__"))
        and metadata.get("execution_mode") != "procedural_cycloidal_contact_fallback"
        and metadata.get("contact_model") == "smc"
        and not metadata.get("preflight_issues")
    )


def _static_audit_summary(audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "feature_frame_counts": audit.get("feature_frame_counts", {}),
        "ring_pins_to_cycloidalDisk1_distance_mm": audit.get(
            "ring_pins_to_cycloidalDisk1_distance_mm"),
        "ring_pins_to_cycloidalDisk2_distance_mm": audit.get(
            "ring_pins_to_cycloidalDisk2_distance_mm"),
        "driver_pins_to_cycloidalDisk1_min_clearance_mm": (
            audit.get("driver_pins_to_cycloidalDisk1_output_holes", {})
            .get("min_radial_clearance_mm")
        ),
        "driver_pins_to_cycloidalDisk2_min_clearance_mm": (
            audit.get("driver_pins_to_cycloidalDisk2_output_holes", {})
            .get("min_radial_clearance_mm")
        ),
    }


def _static_audit_ok(audit: dict[str, Any]) -> bool:
    for key in (
        "driver_pins_to_cycloidalDisk1_output_holes",
        "driver_pins_to_cycloidalDisk2_output_holes",
    ):
        item = audit.get(key, {})
        try:
            if item.get("status") != "ok":
                return False
            if float(item.get("min_radial_clearance_mm", -math.inf)) < 0.0:
                return False
        except (TypeError, ValueError):
            return False
    for key in (
        "ring_pins_to_cycloidalDisk1_distance_mm",
        "ring_pins_to_cycloidalDisk2_distance_mm",
        "driver_pins_to_cycloidalDisk1_distance_mm",
        "driver_pins_to_cycloidalDisk2_distance_mm",
    ):
        try:
            if float(audit.get(key, math.inf)) < 0.0:
                return False
        except (TypeError, ValueError):
            return False
    return True


def _metric_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "failure_mode",
        "passed",
        "lockup_detected",
        "ratio_observed",
        "ratio_error_pct",
        "in_omega_med",
        "out_omega_med",
        "max_penetration_mm",
        "max_constraint_error_mm",
        "n_contacts_max",
        "contact_force_rms_N",
        "output_torque_Nm_mean",
        "output_torque_Nm_signed_mean",
        "output_power_W_mean",
        "input_power_W_mean",
        "power_balance_error_pct",
        "torque_ripple_pct",
        "top_contact_pairs",
    )
    return {key: metrics.get(key) for key in keys if key in metrics}


def verified_gate_passed(
    *,
    cad_generated: bool,
    cad_static_ok: bool,
    chrono_real_geometry: bool,
    metrics: dict[str, Any],
    limits: VerificationLimits,
) -> bool:
    if not cad_generated or not cad_static_ok or not chrono_real_geometry:
        return False
    return chrono_dynamic_gate_passed(metrics=metrics, limits=limits)


def chrono_dynamic_gate_passed(
    *,
    metrics: dict[str, Any],
    limits: VerificationLimits,
) -> bool:
    ratio = _float_metric(metrics, "ratio_observed", math.inf)
    return (
        _float_metric(metrics, "lockup_detected", 1.0) == 0.0
        and abs(_float_metric(metrics, "out_omega_med", 0.0))
        >= limits.min_output_speed_rad_s
        and math.isfinite(ratio)
        and _float_metric(metrics, "max_penetration_mm", math.inf)
        < limits.max_penetration_mm
        and _float_metric(metrics, "contact_force_rms_N", math.inf)
        <= limits.max_contact_force_rms_N
        and _float_metric(metrics, "n_contacts_max", math.inf)
        <= limits.max_contacts
        and _float_metric(metrics, "ratio_error_pct", math.inf)
        <= limits.max_ratio_error_pct
        and _float_metric(metrics, "power_balance_error_pct", math.inf)
        <= limits.max_power_balance_error_pct
        and _float_metric(metrics, "torque_ripple_pct", math.inf)
        <= limits.max_torque_ripple_pct
    )


def verified_reward(
    *,
    fast_reward: float,
    cad_generated: bool,
    cad_static_ok: bool,
    chrono_real_geometry: bool,
    metrics: dict[str, Any],
    limits: VerificationLimits,
) -> float:
    if not verified_gate_passed(
        cad_generated=cad_generated,
        cad_static_ok=cad_static_ok,
        chrono_real_geometry=chrono_real_geometry,
        metrics=metrics,
        limits=limits,
    ):
        return 0.0
    out_speed = abs(_float_metric(metrics, "out_omega_med", 0.0))
    speed_quality = _clamped(out_speed / max(limits.min_output_speed_rad_s * 2.5, 1e-9))
    penetration_quality = _clamped(
        1.0 - _float_metric(metrics, "max_penetration_mm", math.inf)
        / limits.max_penetration_mm)
    ratio_quality = _clamped(
        1.0 - _float_metric(metrics, "ratio_error_pct", math.inf)
        / limits.max_ratio_error_pct)
    force_quality = _clamped(
        1.0 - _float_metric(metrics, "contact_force_rms_N", math.inf)
        / limits.max_contact_force_rms_N)
    contact_quality = _clamped(
        1.0 - _float_metric(metrics, "n_contacts_max", math.inf)
        / limits.max_contacts)
    quality = (
        0.24 * speed_quality
        + 0.22 * penetration_quality
        + 0.20 * ratio_quality
        + 0.18 * force_quality
        + 0.16 * contact_quality
    )
    return round(float(fast_reward) * (0.55 + 0.45 * quality), 6)


def defect_list(
    result: dict[str, Any],
    limits: VerificationLimits,
) -> list[str]:
    defects: list[str] = []
    if not result.get("cad_generated"):
        defects.append("cad_export_failed")
    if result.get("cad_generated") and not result.get("cad_static_ok"):
        defects.append("trusted_asset_static_clearance_failed")
    if result.get("cad_generated") and not result.get("chrono_real_geometry"):
        defects.append("chrono_real_geometry_failed")
    metrics = result.get("metrics") or {}
    if metrics:
        if _float_metric(metrics, "lockup_detected", 1.0) != 0.0:
            defects.append("lockup")
        if not math.isfinite(_float_metric(metrics, "ratio_observed", math.inf)):
            defects.append("nonfinite_ratio")
        if abs(_float_metric(metrics, "out_omega_med", 0.0)) < limits.min_output_speed_rad_s:
            defects.append("output_speed_below_gate")
        if _float_metric(metrics, "max_penetration_mm", math.inf) >= limits.max_penetration_mm:
            defects.append("penetration_over_gate")
        if _float_metric(metrics, "contact_force_rms_N", math.inf) > limits.max_contact_force_rms_N:
            defects.append("contact_force_over_gate")
        if _float_metric(metrics, "n_contacts_max", math.inf) > limits.max_contacts:
            defects.append("contact_count_over_gate")
        if _float_metric(metrics, "ratio_error_pct", math.inf) > limits.max_ratio_error_pct:
            defects.append("ratio_error_over_gate")
        if (
            _float_metric(metrics, "power_balance_error_pct", math.inf)
            > limits.max_power_balance_error_pct
        ):
            defects.append("power_balance_error_over_gate")
        if (
            _float_metric(metrics, "torque_ripple_pct", math.inf)
            > limits.max_torque_ripple_pct
        ):
            defects.append("torque_ripple_over_gate")
    if result.get("error_stage"):
        defects.append(str(result["error_stage"]).lower().replace(" ", "_"))
    return defects


def _paper_grade_passed(
    metrics: dict[str, Any],
    limits: VerificationLimits,
) -> bool:
    return verified_gate_passed(
        cad_generated=True,
        cad_static_ok=True,
        chrono_real_geometry=True,
        metrics=metrics,
        limits=limits,
    )


def _method_table(
    results: Iterable[dict[str, Any]],
    *,
    limits: VerificationLimits | None = None,
    methods: Iterable[str] = METHOD_ORDER,
) -> list[dict[str, Any]]:
    if limits is None:
        limits = VerificationLimits()
    by_method: dict[str, list[dict[str, Any]]] = {method: [] for method in METHOD_ORDER}
    for row in results:
        by_method.setdefault(str(row.get("method", "")), []).append(row)
    table: list[dict[str, Any]] = []
    for method in methods:
        rows = by_method.get(method, [])
        if not rows:
            table.append(_empty_method_row(method))
            continue
        cad_passes = [bool(r.get("cad_generated")) and bool(r.get("cad_static_ok")) for r in rows]
        chrono_passes = [
            bool(r.get("chrono_real_geometry"))
            and chrono_dynamic_gate_passed(
                metrics=r.get("metrics") or {},
                limits=limits,
            )
            for r in rows
        ]
        lockups = [
            _float_metric(r.get("metrics") or {}, "lockup_detected", 0.0) != 0.0
            for r in rows
            if r.get("metrics")
        ]
        table.append({
            "method": method,
            "best_fast_reward": round(max(float(r.get("fast_reward", 0.0)) for r in rows), 6),
            "best_verified_reward": round(max(
                float(r.get("verified_reward", 0.0)) for r in rows), 6),
            "CAD pass rate": round(sum(cad_passes) / len(rows), 6),
            "Chrono pass rate": round(sum(chrono_passes) / len(rows), 6),
            "lockup rate": round(
                sum(lockups) / len(lockups), 6) if lockups else None,
            "mean defect count": round(
                sum(float(r.get("defect_count", 0.0)) for r in rows) / len(rows),
                6,
            ),
            "candidate_count": len(rows),
        })
    return table


def _empty_method_row(method: str) -> dict[str, Any]:
    return {
        "method": method,
        "best_fast_reward": 0.0,
        "best_verified_reward": 0.0,
        "CAD pass rate": 0.0,
        "Chrono pass rate": 0.0,
        "lockup rate": None,
        "mean defect count": 0.0,
        "candidate_count": 0,
    }


def _best_verified_candidate(
    rows: Iterable[dict[str, Any]],
    method: str,
) -> dict[str, Any] | None:
    method_rows = [r for r in rows if r.get("method") == method]
    if not method_rows:
        return None
    best = max(method_rows, key=lambda r: float(r.get("verified_reward", 0.0)))
    return {
        "id": best.get("id"),
        "method": best.get("method"),
        "params": best.get("params"),
        "fast_reward": best.get("fast_reward"),
        "verified_reward": best.get("verified_reward"),
        "verified_gate_passed": best.get("verified_gate_passed"),
        "defects": best.get("defects", []),
        "metrics": best.get("metrics", {}),
        "asset_dir": best.get("asset_dir"),
    }


def _win_condition_met(best_by_method: dict[str, dict[str, Any] | None]) -> bool:
    vg = best_by_method.get("verifier_gated") or {}
    vg_reward = float(vg.get("verified_reward", 0.0) or 0.0)
    baseline_rewards = []
    for method in ("seed", "random", "cma_es_fast_only"):
        row = best_by_method.get(method) or {}
        baseline_rewards.append(float(row.get("verified_reward", 0.0) or 0.0))
    return vg_reward > max(baseline_rewards or [0.0])


def _write_table_csv(path: Path, table: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(TABLE_COLUMNS))
        writer.writeheader()
        for row in table:
            writer.writerow({key: row.get(key) for key in TABLE_COLUMNS})


def _format_table(table: list[dict[str, Any]]) -> str:
    rows = [
        [str(row.get(col, "")) if row.get(col, "") is not None else ""
         for col in TABLE_COLUMNS]
        for row in table
    ]
    widths = [
        max(len(col), *(len(row[i]) for row in rows))
        for i, col in enumerate(TABLE_COLUMNS)
    ]
    header = " | ".join(col.ljust(widths[i]) for i, col in enumerate(TABLE_COLUMNS))
    sep = "-+-".join("-" * width for width in widths)
    body = "\n".join(
        " | ".join(row[i].ljust(widths[i]) for i in range(len(TABLE_COLUMNS)))
        for row in rows
    )
    return f"{header}\n{sep}\n{body}"


def _scoring_params(params: dict[str, Any]) -> dict[str, float]:
    merged = dict(FAST_REWARD_DEFAULTS)
    merged.update({
        key: float(value)
        for key, value in params.items()
        if key in FAST_REWARD_DEFAULTS
    })
    merged["pins"] = float(max(4, int(round(merged["pins"]))))
    merged["line_segment_count"] = float(max(1, int(round(merged["line_segment_count"]))))
    return merged


def _float_metric(metrics: dict[str, Any], key: str, default: float) -> float:
    try:
        value = float(metrics.get(key, default))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _triangular_score(value: float, *, ideal: float, width: float) -> float:
    return _clamped(1.0 - abs(float(value) - float(ideal)) / float(width))


def _clamped(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, float(value)))


def _bounded(value: float, low: float, high: float) -> float:
    return max(float(low), min(float(high), float(value)))


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


if __name__ == "__main__":
    raise SystemExit(main())
