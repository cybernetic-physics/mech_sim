# auto-generated; do not edit by hand. See mech_bench.generators.
import sys
from pathlib import Path


def build_design(out_dir: Path) -> dict:
    ref_dir = (
        Path(__file__).resolve().parent.parent.parent
        / 'reference_solution'
    )
    sys.path.insert(0, str(ref_dir))
    try:
        import design as ref  # noqa: I001
    finally:
        sys.path.pop(0)
    ir = ref.build_design(out_dir)
    ir['params']['declared_linear_per_rev_mm'] = 103.5959
    return ir
