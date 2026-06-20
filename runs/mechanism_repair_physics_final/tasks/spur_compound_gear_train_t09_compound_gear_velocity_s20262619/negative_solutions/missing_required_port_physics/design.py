# auto-generated; do not edit by hand. See scripts.prepare_mechanism_repair_physics_benchmark.
import sys
from pathlib import Path


def build_design(out_dir):
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
    ports = ir.get('ports') or {}
    if ports:
        del ports[sorted(ports)[0]]
    return ir
