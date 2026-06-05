"""High-fidelity oracle integration helpers."""

from mech_bench.oracle.reference_cache import (
    REFERENCE_CACHE_VERSION,
    geometry_hash,
    reference_cache_key,
)
from mech_bench.oracle.smoke import run_oracle_smoke

__all__ = [
    "run_oracle_smoke",
    "REFERENCE_CACHE_VERSION",
    "geometry_hash",
    "reference_cache_key",
]
