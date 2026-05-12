"""Path-trace probe: Chamfer distance against a target CSV.

Compares the trajectory of a named port frame (typically a coupler
point) against a reference path, after optional normalization (so
the agent does not have to land on the same absolute position).

Symmetric Chamfer distance:

    d_chamfer(A, B) = (1/|A|) Σ_a min_b ‖a - b‖
                    + (1/|B|) Σ_b min_a ‖a - b‖

Requires the simulator adapter to have produced
`sim_outputs["port_traces"][port_id]` as an (N, 2) float array (xy
in the units the task declares).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from mech_bench.feedback import Failure, FailureCode, Severity
from mech_bench.probes import Capability, Probe, register_probe
from mech_bench.schema import DesignIR, ProbeResult


def _load_csv_path(path: Path) -> np.ndarray:
    arr = np.loadtxt(path, delimiter=",", skiprows=1, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 2)
    return arr[:, :2]


def _normalize(p: np.ndarray) -> tuple[np.ndarray, float]:
    """Center at centroid; scale so RMS radius is 1. Returns
    (normalized_points, scale_used).
    """
    c = p.mean(axis=0)
    q = p - c
    rms = float(np.sqrt((q ** 2).sum(axis=1).mean()))
    if rms < 1e-12:
        return q, 1.0
    return q / rms, rms


def _chamfer(a: np.ndarray, b: np.ndarray) -> float:
    """Symmetric Chamfer distance.

    O(|A| · |B|) — fine for paths of a few hundred points each, which
    is all this probe is asked to handle. If we ever need larger,
    switch to scipy.spatial.cKDTree.
    """
    # (Na, Nb) pairwise distances
    d2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=-1)
    return float(np.sqrt(d2.min(axis=1)).mean()
                 + np.sqrt(d2.min(axis=0)).mean()) * 0.5


@register_probe
class PathTraceChamfer(Probe):
    type_name = "path_trace_chamfer"
    capabilities_required = frozenset({Capability.PATH_TRACE})

    def run(
        self,
        ir: DesignIR,
        sim_outputs: dict[str, Any],
        config: dict[str, Any],
    ) -> ProbeResult:
        moving_frame = str(config["moving_frame"])
        target_csv = Path(config["target_csv"])
        normalize = bool(config.get("normalize", True))
        max_chamfer = float(config.get("max_chamfer", 2.0))

        traces = sim_outputs.get("port_traces", {})
        observed = traces.get(moving_frame)
        if observed is None:
            return ProbeResult(
                probe_id="",
                probe_type=self.type_name,
                passed=False,
                score=0.0,
                metrics={},
                failures=[Failure(
                    code=FailureCode.SIMULATOR_DIVERGENCE,
                    severity=Severity.CRITICAL,
                    message=(f"No port trace for moving_frame "
                             f"{moving_frame!r} in simulator output."),
                    public_hint=(
                        "The simulator adapter did not emit a "
                        "trajectory for the named port. Check that "
                        "the port exists in DesignIR and that the "
                        "adapter's capabilities include path_trace."
                    ),
                )],
            )

        target = _load_csv_path(target_csv)
        if normalize:
            observed_n, _ = _normalize(np.asarray(observed, dtype=float))
            target_n, _ = _normalize(target)
        else:
            observed_n = np.asarray(observed, dtype=float)
            target_n = target

        chamfer = _chamfer(observed_n, target_n)
        metrics = {
            "chamfer": chamfer,
            "n_observed": float(len(observed_n)),
            "n_target": float(len(target_n)),
        }
        passed = chamfer <= max_chamfer
        # Score saturates at chamfer=0 → 1.0, decays linearly to
        # 0 at chamfer = max_chamfer, and stays 0 beyond.
        score = max(0.0, 1.0 - chamfer / max_chamfer) if max_chamfer > 0 else 0.0
        failures: list[Failure] = []
        if not passed:
            failures.append(Failure(
                code=FailureCode.PATH_ERROR,
                severity=Severity.MAJOR,
                message=(f"Coupler trace differs from target: "
                         f"Chamfer = {chamfer:.3f} (limit {max_chamfer})."),
                metric="chamfer",
                observed=chamfer,
                target=max_chamfer,
                public_hint=(
                    "Adjust link lengths or coupler-point offset. "
                    "Both observed and target traces are normalized "
                    "to centroid + RMS radius before comparison."
                    if normalize else
                    "Verify absolute position and scale; this probe "
                    "is configured to compare un-normalized paths."
                ),
            ))
        return ProbeResult(
            probe_id="",
            probe_type=self.type_name,
            passed=passed,
            score=score,
            metrics=metrics,
            failures=failures,
        )
