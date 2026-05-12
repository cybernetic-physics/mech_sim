"""Negative control: shifts the coupler point so the trace mis-matches.

Same topology and link lengths as the reference, but the coupler
point is at (60, -10) in coupler-local instead of (35, 18). The
resulting trace, after normalization, should differ from the
reference target enough to exceed the Chamfer threshold.
"""

from __future__ import annotations

import sys
from pathlib import Path


def build_design(out_dir: Path) -> dict:
    # Reuse the reference design.py by importing it directly.
    ref_dir = (
        Path(__file__).resolve().parent.parent.parent
        / "reference_solution"
    )
    sys.path.insert(0, str(ref_dir))
    try:
        import design as ref  # noqa: I001
    finally:
        sys.path.pop(0)

    ir = ref.build_design(out_dir)
    ir["ports"]["coupler_point"]["pose_local_mm"] = (60.0, -10.0, 0.0)
    return ir
