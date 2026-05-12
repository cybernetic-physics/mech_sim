"""Thin ffmpeg wrapper for MP4 encoding.

The wrapper exists so callers can probe availability before committing
to a render, and so the failure mode is structured (FFmpegResult) rather
than a bare exception. When ffmpeg is missing, callers typically keep
the frames around and emit a warning in the media manifest.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


_LOG = logging.getLogger("mech_bench.rendering.ffmpeg")


def _probe_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


HAS_FFMPEG = _probe_ffmpeg()


@dataclass
class FFmpegResult:
    ok: bool
    out_path: Path | None = None
    reason: str = ""
    stderr_tail: str = ""


def encode_mp4(
    frames_dir: Path,
    out_mp4: Path,
    *,
    fps: int = 30,
    pattern: str = "frame_%06d.png",
    crf: int = 23,
) -> FFmpegResult:
    """Encode a directory of frames into an mp4.

    Frames must be sequentially numbered (``frame_000000.png``,
    ``frame_000001.png``, …). Returns a structured result so callers
    can keep the frames around even when encoding fails.
    """
    frames_dir = Path(frames_dir)
    out_mp4 = Path(out_mp4)
    if not HAS_FFMPEG:
        _LOG.warning("ffmpeg not found on PATH; skipping encode of %s",
                     out_mp4)
        return FFmpegResult(
            ok=False,
            reason="ffmpeg not found on PATH; frames retained.",
        )
    if not frames_dir.is_dir():
        return FFmpegResult(
            ok=False,
            reason=f"frames_dir does not exist: {frames_dir}",
        )
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(int(fps)),
        "-i", str(frames_dir / pattern),
        "-c:v", "libx264",
        "-crf", str(int(crf)),
        "-pix_fmt", "yuv420p",
        str(out_mp4),
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, check=False)
    except OSError as e:
        return FFmpegResult(
            ok=False,
            reason=f"ffmpeg invocation failed: {e}",
        )
    if proc.returncode != 0:
        return FFmpegResult(
            ok=False,
            reason=f"ffmpeg exit {proc.returncode}",
            stderr_tail=(proc.stderr or "").strip()[-400:],
        )
    return FFmpegResult(ok=True, out_path=out_mp4)
