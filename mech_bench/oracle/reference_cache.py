"""Reference cache for the expensive (Chrono) oracle.

GBA-Eval precomputes Mesen2's per-frame output once per testcase and caches it,
keyed on ``sha256(reference_wasm) ^ sha256(rom) ^ sha256(replay)``, with an
explicit ``CACHE_VERSION`` so any metric change invalidates everything — that is
what makes grading affordable to *iterate* against a slow reference (see
``rl-environment-design-notes.md`` §4).

This module is the mech-bench analogue: a content-addressed key for caching a
real-Chrono ``SimOutput`` per ``(task, design-geometry, oracle-config)``. The
key is deterministic and depends only on the *physically meaningful* geometry —
not on incidental fields (ids, comments, declared answers) — so two designs that
are geometrically identical share a cache entry, and any change to geometry or
the oracle version invalidates it. No silent staleness.

The cache is pure-Python and has no pychrono dependency, so it is testable on
any machine; wiring it into the Chrono adapter is opt-in (``use_reference_cache``
in the adapter config).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

# Bump whenever the oracle's output semantics change so old entries invalidate.
REFERENCE_CACHE_VERSION = 1


def _canonical(obj: Any) -> Any:
    """Round floats and sort keys so logically-equal geometry hashes equally."""
    if isinstance(obj, float):
        # 1e-9 mm resolution: well below any real manufacturing tolerance.
        return round(obj, 9)
    if isinstance(obj, dict):
        return {k: _canonical(obj[k]) for k in sorted(obj, key=str)}
    if isinstance(obj, (list, tuple)):
        return [_canonical(v) for v in obj]
    return obj


def geometry_hash(design_ir: dict[str, Any]) -> str:
    """Hash only the physically-meaningful geometry of a DesignIR.

    Includes parts (mass, com, geometry, params), joints (type, topology,
    axis, anchor), ports, contacts. Excludes provenance/comments and the
    agent's *declared answers* (e.g. ``params['declared_ratio']``) so the cache
    key reflects the mechanism, not the claim about it.
    """
    parts = []
    for p in design_ir.get("parts", []):
        parts.append({
            "id": p.get("id"),
            "role": p.get("role", ""),
            "mass_kg": p.get("mass_kg", 0.0),
            "com_local_mm": p.get("com_local_mm", (0.0, 0.0, 0.0)),
            "fixed": p.get("fixed", False),
            "geometry": p.get("geometry", {}),
            "params": p.get("params", {}),
        })
    joints = []
    for j in design_ir.get("joints", []):
        joints.append({
            "id": j.get("id"),
            "type": str(j.get("type")),
            "parent": j.get("parent"),
            "child": j.get("child"),
            "axis_world": j.get("axis_world"),
            "anchor_world_mm": j.get("anchor_world_mm"),
            "params": j.get("params", {}),
        })
    payload = {
        "parts": parts,
        "joints": joints,
        "ports": design_ir.get("ports", {}),
        "contacts": design_ir.get("contacts", {}),
    }
    blob = json.dumps(_canonical(payload), sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def reference_cache_key(
    task_id: str,
    design_ir: dict[str, Any],
    oracle_config: dict[str, Any] | None = None,
) -> str:
    """Content-addressed cache key for a real-oracle run.

    Combines the cache version, task id, geometry hash, and the oracle config
    (test conditions: input speed, load, samples, solver). Any change to any of
    these invalidates the entry — mirroring GBA-Eval's hash-keyed refcache.
    """
    cfg = _canonical(oracle_config or {})
    parts = [
        f"v{REFERENCE_CACHE_VERSION}",
        str(task_id),
        geometry_hash(design_ir),
        hashlib.sha256(
            json.dumps(cfg, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest(),
    ]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
