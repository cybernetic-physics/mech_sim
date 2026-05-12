"""MP4 video front-end.

Two backends are wired here:

* ``planar`` — matplotlib-driven 2D renderer for planar mechanism
  payloads (see :mod:`mech_bench.rendering.planar_renderer`). Available
  whenever ``matplotlib`` is installed.
* Placeholders for future Chrono / MuJoCo / Drake / Three.js backends.

The CLI's ``mech-bench video`` flow is:

    1. Load ``dashboard_payload.json`` from ``--report-dir``.
    2. Pick a :class:`RendererBackend` (``--view`` decides; default
       ``planar``).
    3. Have the backend render frames + thumbnail + optional mp4 (if
       ffmpeg is on PATH).

If matplotlib or ffmpeg are missing, the renderer keeps any partial
artifacts (frames) and emits a structured warning rather than crashing
the evaluation.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar


_LOG = logging.getLogger("mech_bench.video")


@dataclass
class RenderResult:
    """Outcome of a video-rendering attempt."""

    ok: bool
    out_path: Path | None = None
    reason: str = ""
    backend: str = ""


class RendererBackend(ABC):
    """Subclasses register themselves via :func:`register_backend`."""

    type_name: ClassVar[str] = ""

    @abstractmethod
    def available(self) -> bool:
        """True if this backend can render right now."""

    @abstractmethod
    def render_frames(
        self,
        payload: dict[str, Any],
        frames_dir: Path,
        *,
        fps: int = 30,
    ) -> int:
        """Render frames to *frames_dir* and return the frame count.

        Frames must be named ``frame_%06d.png`` so ffmpeg can stitch
        them. Implementations should raise on unrecoverable errors;
        the CLI catches and surfaces them as a structured warning.
        """


_BACKENDS: list[RendererBackend] = []


def register_backend(backend: RendererBackend) -> RendererBackend:
    _BACKENDS.append(backend)
    return backend


def available_backends() -> list[RendererBackend]:
    return list(_BACKENDS)


def pick_backend() -> RendererBackend | None:
    for b in _BACKENDS:
        try:
            if b.available():
                return b
        except Exception:  # noqa: BLE001 - defensive
            continue
    return None


# --------------------------------------------------------------------- #
# Frame-sequence renderer                                               #
# --------------------------------------------------------------------- #


class FrameSequenceRenderer:
    """Drive a backend through its frame loop.

    By default uses the planar matplotlib renderer (which is the only
    backend wired today). Future Chrono / MuJoCo backends will register
    via :func:`register_backend` and be picked when their capabilities
    match the payload.
    """

    def __init__(
        self,
        backend: RendererBackend | None = None,
        *,
        view: str = "planar",
    ):
        self.backend = backend
        self.view = view

    def render(
        self,
        payload: dict[str, Any],
        out_mp4: Path,
        *,
        fps: int = 30,
        frames_dir: Path | None = None,
    ) -> RenderResult:
        out_mp4 = Path(out_mp4)
        if self.view in ("planar", "", None):
            from mech_bench.rendering.planar_renderer import (
                HAS_MATPLOTLIB,
                PlanarRenderer,
            )
            if not HAS_MATPLOTLIB:
                return RendererResult_unavailable(
                    "matplotlib is not installed; install "
                    "mech-bench[media] to enable planar rendering."
                )
            renderer = PlanarRenderer()
            out_dir = out_mp4.parent
            out_dir.mkdir(parents=True, exist_ok=True)
            res = renderer.render(payload, out_dir, fps=fps,
                                  produce_mp4=True)
            if not res.ok:
                return RenderResult(
                    ok=False, backend="planar",
                    reason=res.reason or "planar renderer failed",
                )
            if res.preview_mp4 is None:
                return RenderResult(
                    ok=False, backend="planar",
                    reason=("frames + thumbnail written, but ffmpeg "
                            "did not produce an mp4."),
                    out_path=res.frames_dir,
                )
            # Move the renderer's preview.mp4 to the requested path if
            # they differ.
            if res.preview_mp4 != out_mp4:
                out_mp4.write_bytes(res.preview_mp4.read_bytes())
            return RenderResult(
                ok=True, out_path=out_mp4, backend="planar")

        # Other backends: try registered ones.
        backend = self.backend or pick_backend()
        if backend is None:
            return RendererResult_unavailable(
                f"No registered backend for view {self.view!r}.")
        frames_dir = frames_dir or out_mp4.with_suffix("").with_name(
            f"{out_mp4.stem}_frames")
        frames_dir.mkdir(parents=True, exist_ok=True)
        n = backend.render_frames(payload, frames_dir, fps=fps)
        if n <= 0:
            return RenderResult(
                ok=False, backend=backend.type_name,
                reason="backend produced zero frames",
            )
        encoded = encode_mp4_with_ffmpeg(frames_dir, out_mp4, fps=fps)
        if encoded is None:
            return RenderResult(
                ok=False, backend=backend.type_name,
                reason="ffmpeg not available; frames written but not encoded",
            )
        return RenderResult(
            ok=True, out_path=encoded, backend=backend.type_name)


def RendererResult_unavailable(reason: str) -> RenderResult:
    """Helper for the common "no renderer" case."""
    return RenderResult(ok=False, reason=reason, backend="")


# --------------------------------------------------------------------- #
# ffmpeg shim                                                           #
# --------------------------------------------------------------------- #


def encode_mp4_with_ffmpeg(
    frames_dir: Path,
    out_mp4: Path,
    *,
    fps: int = 30,
) -> Path | None:
    """Encode ``frame_%06d.png`` frames into an mp4 with ffmpeg.

    Returns the output path on success, ``None`` if ffmpeg is not on
    PATH or the encode failed. A structured log warning is emitted
    so callers can surface this in the manifest.
    """
    if shutil.which("ffmpeg") is None:
        _LOG.warning(
            "ffmpeg not found on PATH; cannot encode %s. "
            "Install ffmpeg to produce mp4 previews.",
            out_mp4,
        )
        return None
    out_mp4 = Path(out_mp4)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(Path(frames_dir) / "frame_%06d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(out_mp4),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except (subprocess.CalledProcessError, OSError) as e:
        _LOG.warning("ffmpeg encode failed: %s", e)
        return None
    return out_mp4
