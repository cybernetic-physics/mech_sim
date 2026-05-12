"""Lightweight rendering backends for run evidence.

The base ``mech_bench`` install has no media dependencies. The
``rendering`` package wraps:

* :mod:`mech_bench.rendering.planar_renderer` — matplotlib-driven 2D
  frame rendering for planar mechanism tasks (four-bar, slider-crank).
* :mod:`mech_bench.rendering.ffmpeg` — thin wrapper around the ffmpeg
  CLI for encoding frame sequences to MP4.

Both modules degrade gracefully when their dependency is missing: the
caller gets a structured result and can decide whether to emit a
warning, keep the frames around, or skip the artifact altogether.
"""

from __future__ import annotations

from mech_bench.rendering.planar_renderer import (
    HAS_MATPLOTLIB,
    PlanarRenderResult,
    PlanarRenderer,
)
from mech_bench.rendering.ffmpeg import (
    FFmpegResult,
    HAS_FFMPEG,
    encode_mp4,
)

__all__ = [
    "HAS_MATPLOTLIB",
    "HAS_FFMPEG",
    "PlanarRenderer",
    "PlanarRenderResult",
    "FFmpegResult",
    "encode_mp4",
]
