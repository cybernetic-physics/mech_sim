"""Deterministic fake contact-oracle adapter.

Used by tests and the generated Tier-3 placeholder tasks when the real
`chrono_contact` adapter (which depends on PyChrono) is unavailable.
Same capability surface as the real adapter; outputs are deterministic
synthetic traces derived from the DesignIR plus an optional adapter
config.

Configuration (all optional; defaults are conservative):

* ``contact_pairs``       — list of ``"a:b"`` strings. Force traces are
                            synthesized for each pair.
* ``contact_force_N``     — mean magnitude of the synthesized force.
* ``penetration_mm``      — peak penetration to emit per pair.
* ``ratio_observed``      — observed transmission ratio; falls back to
                            ``ir.params["declared_ratio"]`` when absent.
* ``lockup``              — if True, output speed/motion is zero.
* ``torque_ripple_pct``   — input torque ripple (%) to report.
* ``power_balance_error_pct`` — % power-balance error.
* ``input_speed_rad_s``   — driven input speed.
* ``output_load_Nm``      — output load torque.
* ``samples``             — number of timesteps to emit (default 360).
* ``duration_s``          — total simulated time (default 1.0).
* ``seed``                — RNG seed for the ripple/penetration noise.

The IR is consulted for sensible defaults: ``ir.params["fake_oracle"]``
overrides individual config keys, so a generator's reference solution
can declare what the oracle should report.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from mech_bench.adapters import SimAdapter, register_adapter
from mech_bench.probes import Capability
from mech_bench.schema import DesignIR


_DEFAULT_COST = 1000  # last-resort even when registered.


def _is_test_mode_enabled() -> bool:
    """``True`` when fake_contact_oracle should auto-register.

    Promoted by either:
      * the ``MECH_BENCH_USE_FAKE_ORACLE`` env var (``1``/``true``/``yes``),
      * the ``MECH_BENCH_TEST_MODE`` env var (same values).
    """
    for var in ("MECH_BENCH_USE_FAKE_ORACLE", "MECH_BENCH_TEST_MODE"):
        if os.environ.get(var, "").lower() in ("1", "true", "yes"):
            return True
    return False


if _is_test_mode_enabled():
    _ACTIVE_COST = 50
else:
    _ACTIVE_COST = _DEFAULT_COST


def _normalize_pair(p: str) -> str:
    a, _, b = str(p).partition(":")
    if not b:
        return str(p)
    return ":".join(sorted([a, b]))


def _config_with_overrides(ir: DesignIR, config: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(config)
    ir_overrides = (ir.params or {}).get("fake_oracle") or {}
    if isinstance(ir_overrides, dict):
        for k, v in ir_overrides.items():
            merged.setdefault(k, v)
    return merged


def _resolve_contact_pairs(
    ir: DesignIR, cfg: dict[str, Any],
) -> list[str]:
    pairs = cfg.get("contact_pairs") or []
    if not pairs:
        # Fallback: scan IR for declared contact_pair joints.
        for j in ir.joints:
            if j.type == "contact_pair":
                pairs.append(f"{j.parent}:{j.child}")
    return [_normalize_pair(p) for p in pairs]


def _input_output_ports(ir: DesignIR) -> tuple[str, str]:
    inp = "input_port" if "input_port" in ir.ports else ""
    out = "output_port" if "output_port" in ir.ports else ""
    return inp, out


class FakeContactOracle(SimAdapter):
    """Deterministic fake. Cheap to run, no external dependencies."""

    type_name = "fake_contact_oracle"
    capabilities_provided = frozenset({
        Capability.RIGID_BODY_DYNAMICS,
        Capability.CONTACT_FORCES,
        Capability.JOINT_CONSTRAINTS,
        Capability.MOTOR_DRIVES,
        Capability.LOAD_TORQUES,
        Capability.POSE_TRACES,
        Capability.MESH_OVERLAP,
        Capability.PLANAR_KINEMATICS,
    })
    cost_tier = _ACTIVE_COST

    def run(
        self,
        ir: DesignIR,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        cfg = _config_with_overrides(ir, config)

        n_samples = int(cfg.get("samples", 360))
        duration_s = float(cfg.get("duration_s", 1.0))
        seed = int(cfg.get("seed", 0))
        rng = np.random.default_rng(seed)

        time_s = np.linspace(0.0, duration_s, n_samples, dtype=float)

        in_speed = float(cfg.get("input_speed_rad_s", 1.0))
        out_load = float(cfg.get("output_load_Nm", 0.0))
        lockup = bool(cfg.get("lockup", False))

        # Ratio: prefer explicit override, then declared_ratio, then 1.0.
        ratio = cfg.get("ratio_observed")
        if ratio is None:
            ratio = (ir.params or {}).get("declared_ratio")
        if ratio is None or float(ratio) == 0.0:
            ratio = 1.0
        ratio = float(ratio)

        contact_pairs = _resolve_contact_pairs(ir, cfg)
        contact_force_N = float(cfg.get("contact_force_N", 1.0))
        penetration_mm = float(cfg.get("penetration_mm", 0.001))
        ripple_pct = float(cfg.get("torque_ripple_pct", 5.0))
        power_err_pct = float(cfg.get("power_balance_error_pct", 1.0))
        contact_engagement_fraction = float(
            cfg.get("contact_engagement_fraction", 0.85))

        # ----- joint positions / velocities -----
        # Input joint: constant-speed sweep.
        input_pos = in_speed * time_s
        if lockup:
            output_pos = np.zeros_like(time_s)
            output_vel = np.zeros_like(time_s)
        else:
            out_speed = in_speed / ratio
            output_pos = out_speed * time_s
            output_vel = np.full_like(time_s, out_speed)
        input_vel = np.full_like(time_s, in_speed)

        inp_port, out_port = _input_output_ports(ir)
        joint_positions: dict[str, np.ndarray] = {}
        joint_velocities: dict[str, np.ndarray] = {}
        if inp_port:
            joint_positions[inp_port] = input_pos
            joint_velocities[inp_port] = input_vel
        if out_port:
            joint_positions[out_port] = output_pos
            joint_velocities[out_port] = output_vel
        # Mirror onto the underlying joint ids when ports point to joints.
        for pid, port in ir.ports.items():
            if port.kind != "revolute_joint":
                continue
            if pid == inp_port:
                joint_positions[port.part] = input_pos
                joint_velocities[port.part] = input_vel
            elif pid == out_port:
                joint_positions[port.part] = output_pos
                joint_velocities[port.part] = output_vel

        # ----- contact forces / penetration -----
        contact_forces: dict[str, np.ndarray] = {}
        penetration: dict[str, np.ndarray] = {}
        engagement_mask = (np.linspace(0, 1, n_samples)
                           < contact_engagement_fraction).astype(float)
        # Roll randomly so engaged samples aren't all at the start.
        engagement_mask = np.roll(
            engagement_mask, int(rng.integers(0, n_samples)))
        base_force_trace = contact_force_N * (
            1.0 + 0.05 * np.sin(2.0 * np.pi * time_s / max(duration_s, 1e-9))
        ) * engagement_mask
        for raw in contact_pairs:
            key = _normalize_pair(raw)
            jitter = 1.0 + rng.normal(0.0, 0.01, size=n_samples)
            contact_forces[key] = np.abs(base_force_trace * jitter)
            # Penetration: small ripple around the configured peak.
            pen_trace = penetration_mm * (
                0.5 + 0.5 * np.cos(
                    2.0 * np.pi * time_s / max(duration_s, 1e-9))
            )
            penetration[key] = np.abs(pen_trace)

        # ----- scalar metrics -----
        ratio_error_pct = 0.0
        declared = (ir.params or {}).get("declared_ratio")
        if declared is not None and float(declared) != 0.0:
            ratio_error_pct = float(
                abs(ratio - float(declared)) / abs(float(declared)) * 100.0
            )
        n_contacts_max = float(len(contact_pairs))
        max_pen_mm = float(
            max((np.max(p) for p in penetration.values()), default=0.0))
        scalar_metrics: dict[str, float] = {
            "ratio_observed": float(ratio),
            "ratio_error_pct": float(ratio_error_pct),
            "max_penetration_mm": float(max_pen_mm),
            "max_constraint_error_mm": 0.0,
            "torque_ripple_pct": float(ripple_pct),
            "power_balance_error_pct": float(power_err_pct),
            "lockup_detected": 1.0 if lockup else 0.0,
            "n_contacts_max": n_contacts_max,
            "input_torque_Nm_mean": float(out_load * ratio),
            "output_torque_Nm_mean": float(out_load),
            "input_power_W_mean": float(abs(out_load * ratio * in_speed)),
            "output_power_W_mean": float(
                abs(out_load * (in_speed / ratio)) if not lockup else 0.0
            ),
            "input_torque_ripple_pct": float(ripple_pct),
        }

        # ----- top contact pairs (by max force) -----
        top_pairs: list[dict[str, Any]] = []
        for pair, arr in contact_forces.items():
            top_pairs.append({
                "pair": pair,
                "max_force_N": float(np.max(arr)),
                "rms_force_N": float(np.sqrt(np.mean(arr * arr))),
            })
        top_pairs.sort(key=lambda d: d["max_force_N"], reverse=True)

        metadata: dict[str, Any] = {
            "adapter": self.type_name,
            "simulator": "fake_contact_oracle",
            "trust_level": "synthetic_test_or_demo",
            "is_physical_oracle": False,
            "oracle_is_synthetic": True,
            "solver": "deterministic_synth",
            "contact_method": "synthetic",
            "duration_s": duration_s,
            "dt": float(duration_s / max(n_samples - 1, 1)),
            "seed": seed,
            "preflight_issues": [],
            "build_meta": {
                "n_bodies": len(ir.parts),
                "n_joints": len(ir.joints),
                "n_pairs": len(contact_pairs),
            },
            "top_contact_pairs": top_pairs[:8],
            "lockup": lockup,
        }

        return {
            "time_s": time_s,
            "joint_positions": joint_positions,
            "joint_velocities": joint_velocities,
            "contact_forces": contact_forces,
            "penetration": penetration,
            "body_poses": {},
            "port_traces": {},
            "scalar_metrics": scalar_metrics,
            "metadata": metadata,
            "adapter": self.type_name,
        }


def register_if_test_mode() -> bool:
    """Idempotently register :class:`FakeContactOracle` if test mode is on.

    Tests that want the fake oracle should call this *after* setting
    ``MECH_BENCH_USE_FAKE_ORACLE=1`` so the adapter joins the registry.
    Returns True when the adapter is registered (now or already).
    """
    from mech_bench.adapters import _REGISTRY, register_adapter
    if FakeContactOracle.type_name in _REGISTRY:
        return True
    if not _is_test_mode_enabled():
        return False
    register_adapter(FakeContactOracle)
    return True


def force_register() -> bool:
    """Register the fake oracle regardless of env/test mode.

    Called by the evaluator when ``[adapters.fake_contact_oracle]
    enabled=true`` is set in the eval config or when a mode sets
    ``forced_adapter = "fake_contact_oracle"``. Returns True when the
    adapter ends up registered.
    """
    from mech_bench.adapters import _REGISTRY, register_adapter
    if FakeContactOracle.type_name in _REGISTRY:
        return True
    register_adapter(FakeContactOracle)
    return True


# Best-effort auto-registration at import time. Most test runners set
# the env var before importing mech_bench; CLI runs leave it unset.
if _is_test_mode_enabled():
    from mech_bench.adapters import _REGISTRY, register_adapter as _ra
    if FakeContactOracle.type_name not in _REGISTRY:
        _ra(FakeContactOracle)
