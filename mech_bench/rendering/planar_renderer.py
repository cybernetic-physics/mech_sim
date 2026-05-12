"""Matplotlib-based 2D renderer for planar mechanism traces.

Consumes a ``dashboard_payload`` dict (the same blob the static HTML
dashboard reads) and writes a frame sequence + thumbnail. The
:func:`PlanarRenderer.render` entry point also encodes an mp4 when
ffmpeg is on PATH; if ffmpeg is missing, the frames are retained and a
structured ``PlanarRenderResult`` reports the gap.

The renderer is deliberately mechanism-agnostic — it draws whatever the
payload's ``traces`` block exposes:

* ``coupler_path`` / ``output_path`` — observed traces
* ``target_path`` — task target (dashed)
* ``ground_pivots`` (optional) — small markers
* ``input_angle`` / ``output_angle`` time series — animated dial overlay
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


_LOG = logging.getLogger("mech_bench.rendering.planar_renderer")


try:  # pragma: no cover - exercised by absence of matplotlib
    import matplotlib  # type: ignore[import-not-found]
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore[import-not-found]
    HAS_MATPLOTLIB = True
except ImportError:
    matplotlib = None  # type: ignore[assignment]
    plt = None  # type: ignore[assignment]
    HAS_MATPLOTLIB = False


@dataclass
class PlanarRenderResult:
    ok: bool
    thumbnail_png: Path | None = None
    preview_mp4: Path | None = None
    frames_dir: Path | None = None
    n_frames: int = 0
    reason: str = ""
    warnings: list[str] = field(default_factory=list)


class PlanarRenderer:
    """Render a planar mechanism payload into frames + optional MP4."""

    def __init__(self, *, dpi: int = 100, figsize: tuple[float, float] = (6.0, 6.0)):
        self.dpi = int(dpi)
        self.figsize = figsize

    def available(self) -> bool:
        return HAS_MATPLOTLIB

    def render(
        self,
        payload: dict[str, Any],
        out_dir: Path,
        *,
        fps: int = 30,
        n_frames: int = 60,
        produce_mp4: bool = True,
    ) -> PlanarRenderResult:
        """Render *payload* into *out_dir*.

        Writes:

        * ``frames/frame_%06d.png`` — one PNG per frame
        * ``thumbnail.png`` — single mid-cycle frame
        * ``preview.mp4`` — when ffmpeg is on PATH and ``produce_mp4``
          is True
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if not HAS_MATPLOTLIB:
            return PlanarRenderResult(
                ok=False,
                reason=(
                    "matplotlib is not installed; install "
                    "mech-bench[media] to enable planar rendering."
                ),
            )

        traces = payload.get("traces", {}) or {}
        coupler = _as_xy(traces.get("coupler_path"))
        output_pt = _as_xy(traces.get("output_path"))
        target = _as_xy(traces.get("target_path"))
        # Choose the primary trace for the moving point.
        primary = coupler if coupler is not None else output_pt
        if primary is None or len(primary) < 2:
            return PlanarRenderResult(
                ok=False,
                reason=(
                    "payload has no usable planar traces "
                    "(coupler_path/output_path/target_path)."
                ),
            )

        n_frames = int(min(max(n_frames, 1), len(primary)))
        step = max(1, len(primary) // n_frames)
        frame_indices = list(range(0, len(primary), step))[:n_frames]

        frames_dir = out_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        # Compute axis bounds once so frames are stable.
        all_xy = primary
        if target is not None:
            all_xy = np.vstack([all_xy, target])
        if output_pt is not None and output_pt is not primary:
            all_xy = np.vstack([all_xy, output_pt])
        xmin, ymin = all_xy.min(axis=0)
        xmax, ymax = all_xy.max(axis=0)
        pad = max((xmax - xmin), (ymax - ymin)) * 0.1 or 1.0
        bounds = (xmin - pad, xmax + pad, ymin - pad, ymax + pad)

        # Overlay info.
        run = payload.get("run", {}) or {}
        score = float(
            (payload.get("score") or {}).get("dense", 0.0) or 0.0)
        task_id = str(run.get("task_id", ""))
        family = str(run.get("task_family", ""))
        chamfer = _first_metric(payload,
                                 ("path_trace_chamfer.chamfer",
                                  "coupler_path.chamfer"))
        mobility = _first_metric(payload, ("dof_grubler.observed",
                                            "mobility.observed"))

        # Frames.
        for fi, idx in enumerate(frame_indices):
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
            ax.set_xlim(bounds[0], bounds[1])
            ax.set_ylim(bounds[2], bounds[3])
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, alpha=0.2)

            if target is not None and len(target) > 1:
                ax.plot(target[:, 0], target[:, 1],
                        "--", linewidth=1.2, color="#5b8def",
                        alpha=0.8, label="target")

            # Trace trail up to idx.
            ax.plot(primary[:idx + 1, 0], primary[:idx + 1, 1],
                    "-", linewidth=1.6, color="#d93025",
                    alpha=0.9, label="observed")

            # Current point.
            ax.plot([primary[idx, 0]], [primary[idx, 1]],
                    "o", markersize=8, color="#222",
                    markeredgecolor="white", zorder=5)

            # Ground pivots (if expressed in payload).
            pivots = traces.get("ground_pivots") or []
            for pv in pivots:
                if isinstance(pv, (list, tuple)) and len(pv) >= 2:
                    ax.plot([float(pv[0])], [float(pv[1])],
                            "^", markersize=10, color="#444")

            # HUD.
            hud_lines: list[str] = []
            if task_id:
                hud_lines.append(f"task: {task_id}")
            if family:
                hud_lines.append(f"family: {family}")
            hud_lines.append(f"score: {score:.3f}")
            if chamfer is not None:
                hud_lines.append(f"chamfer: {chamfer:.3f} mm")
            if mobility is not None:
                hud_lines.append(f"mobility: {mobility:g}")
            hud_lines.append(
                f"frame: {fi + 1}/{len(frame_indices)}")
            ax.text(
                0.02, 0.98, "\n".join(hud_lines),
                transform=ax.transAxes, va="top", ha="left",
                fontsize=9, family="monospace",
                bbox=dict(facecolor="white", alpha=0.6,
                          edgecolor="none", pad=4),
            )

            ax.legend(loc="lower right", fontsize=8, framealpha=0.7)
            ax.set_title(f"{family or task_id} preview")
            fig.tight_layout()
            fig.savefig(frames_dir / f"frame_{fi:06d}.png",
                        dpi=self.dpi)
            plt.close(fig)

        # Thumbnail = mid-cycle frame, copied (not re-rendered).
        mid_idx = len(frame_indices) // 2
        thumb_src = frames_dir / f"frame_{mid_idx:06d}.png"
        thumb_path = out_dir / "thumbnail.png"
        if thumb_src.exists():
            thumb_path.write_bytes(thumb_src.read_bytes())

        warnings: list[str] = []
        mp4_path: Path | None = None
        if produce_mp4:
            from mech_bench.rendering.ffmpeg import encode_mp4
            mp4_result = encode_mp4(
                frames_dir, out_dir / "preview.mp4", fps=fps)
            if mp4_result.ok:
                mp4_path = mp4_result.out_path
            else:
                warnings.append(
                    "ffmpeg unavailable or encode failed; frames retained. "
                    f"Reason: {mp4_result.reason}"
                )

        return PlanarRenderResult(
            ok=True,
            thumbnail_png=thumb_path if thumb_path.exists() else None,
            preview_mp4=mp4_path,
            frames_dir=frames_dir,
            n_frames=len(frame_indices),
            warnings=warnings,
        )


def _as_xy(blob: Any) -> np.ndarray | None:
    if blob is None:
        return None
    try:
        arr = np.asarray(blob, dtype=float)
    except (TypeError, ValueError):
        return None
    if arr.ndim != 2 or arr.shape[1] < 2 or arr.shape[0] < 1:
        return None
    return arr[:, :2]


def _first_metric(payload: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    metrics = payload.get("metrics", {}) or {}
    for k in keys:
        v = metrics.get(k)
        if isinstance(v, (int, float)) and math.isfinite(float(v)):
            return float(v)
    return None
