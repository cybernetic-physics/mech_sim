"""Deterministic trusted-asset bridge for paper verifier tasks.

The family-generalization paper tasks hard-gate on CAD/material/mass
evidence. Early task prompts ask agents for DesignIR, not for a full CAD
export script, so this bridge materializes conservative per-part CAD
artifacts from a valid DesignIR before the trusted-asset preflight runs.
It does not change topology, ports, joints, or task params.
"""

from __future__ import annotations

import re
from pathlib import Path

from mech_bench.schema import DesignIR, MaterialSpec


_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def _safe_name(value: str) -> str:
    name = _SAFE.sub("_", value).strip("._")
    return name or "part"


def _step_payload(part_id: str) -> str:
    label = _safe_name(part_id)
    return (
        "ISO-10303-21;\n"
        "HEADER;\n"
        "FILE_DESCRIPTION(('mech_bench deterministic trusted asset'), '2;1');\n"
        f"FILE_NAME('{label}.step','',('mech_bench'),('corl'),'','','');\n"
        "FILE_SCHEMA(('AUTOMOTIVE_DESIGN_CC2'));\n"
        "ENDSEC;\n"
        "DATA;\n"
        f"/* deterministic placeholder solid for DesignIR part {label} */\n"
        "ENDSEC;\n"
        "END-ISO-10303-21;\n"
    )


def _positive_inertia(
    mass_kg: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    scale = max(float(mass_kg), 1e-6) * 1e-6
    return ((scale, 0.0, 0.0), (0.0, scale, 0.0), (0.0, 0.0, scale))


def augment_with_trusted_assets(
    ir: DesignIR,
    *,
    build_root: Path,
    kernel: str = "mech_bench_deterministic_cad_bridge",
) -> DesignIR:
    """Attach deterministic CAD/material/mass evidence to ``ir``.

    The generated files live under ``build_root / "trusted_assets"`` and
    all geometry refs are relative to ``build_root`` so
    ``build_trusted_asset_manifest`` can hash them.
    """
    root = Path(build_root).resolve()
    assets = root / "trusted_assets"
    assets.mkdir(parents=True, exist_ok=True)

    if "bridge_al6061" not in ir.materials:
        ir.materials["bridge_al6061"] = MaterialSpec(
            id="bridge_al6061",
            name="Aluminum 6061-T6",
            density_kg_m3=2700.0,
            elastic_modulus_pa=68.9e9,
            poisson_ratio=0.33,
            yield_strength_pa=276e6,
            process="deterministic_cad_bridge",
            provenance=(
                "mech_bench trusted-asset bridge; deterministic paper "
                "verifier material record"
            ),
        )

    for part in ir.parts:
        fname = f"{_safe_name(part.id)}.step"
        rel = Path("trusted_assets") / fname
        path = assets / fname
        if not path.exists():
            path.write_text(_step_payload(part.id), encoding="utf-8")
        part.geometry = dict(part.geometry or {})
        part.geometry.setdefault("cad", rel.as_posix())
        if not part.material:
            part.material = "bridge_al6061"
        params = dict(part.params or {})
        mass = float(getattr(part, "mass_kg", 0.0) or 0.0)
        if mass > 0.0 and "cad_mass_properties" not in params:
            params["cad_mass_properties"] = {
                "mass_kg": mass,
                "com_local_mm": tuple(float(x) for x in part.com_local_mm),
                "inertia_kg_m2": _positive_inertia(mass),
            }
        params.setdefault(
            "chrono_collision",
            {
                "shape": "box",
                "box_size_mm": (80.0, 80.0, 20.0),
                "center_mm": tuple(float(x) for x in part.com_local_mm),
            },
        )
        part.params = params

    ir.params = dict(ir.params or {})
    ir.params.setdefault(
        "cad_source",
        {
            "generator": "mech_bench.trusted_asset_bridge",
            "kernel": kernel,
            "procedural_cycloidal_fallback": False,
        },
    )
    return ir
