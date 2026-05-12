"""MP4 video skeleton.

We don't ship a real renderer today — Chrono / MuJoCo / Drake / Three.js
backends will plug in later. This module provides the interface the
CLI calls and a safe failure mode when no backend is installed.

The flow is:

    1. CLI loads dashboard_payload.json from --report-dir.
    2. CLI asks an available :class:`RendererBackend` for frames.
    3. Frames are encoded into an mp4 with ffmpeg (if installed).

For now every backend reports unavailable, and the CLI emits a
structured warning instead of producing a video. The interfaces are
stable so future backends can register themselves.
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

    Today this is mostly a placeholder so the CLI has a single entry
    point. Real backends will subclass :class:`RendererBackend`.
    """

    def __init__(self, backend: RendererBackend | None = None):
        self.backend = backend or pick_backend()

    def render(
        self,
        payload: dict[str, Any],
        out_mp4: Path,
        *,
        fps: int = 30,
        frames_dir: Path | None = None,
    ) -> RenderResult:
        if self.backend is None:
            return RendererResult_unavailable(
                "No renderer backend is registered. Install a "
                "renderer extra such as `mech-bench[video]` once one "
                "is available."
            )
        if not self.backend.available():
            return RendererResult_unavailable(
                f"Backend {self.backend.type_name!r} is registered "
                f"but not currently runnable."
            )
        frames_dir = frames_dir or out_mp4.with_suffix("").with_name(
            f"{out_mp4.stem}_frames")
        frames_dir.mkdir(parents=True, exist_ok=True)
        n = self.backend.render_frames(payload, frames_dir, fps=fps)
        if n <= 0:
            return RenderResult(
                ok=False,
                backend=self.backend.type_name,
                reason="backend produced zero frames",
            )
        encoded = encode_mp4_with_ffmpeg(frames_dir, out_mp4, fps=fps)
        if encoded is None:
            return RenderResult(
                ok=False,
                backend=self.backend.type_name,
                reason="ffmpeg not available; frames written but not encoded",
            )
        return RenderResult(
            ok=True, out_path=encoded, backend=self.backend.type_name)


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
