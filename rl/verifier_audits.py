from __future__ import annotations

from typing import Any


def _lookup_key_or_path(source: dict[str, Any], key: str) -> Any:
    if key in source:
        return source[key]
    current: Any = source
    for part in key.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _finite_float(raw: Any) -> float | None:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if value != value or value in {float("inf"), float("-inf")}:
        return None
    return value


def adapter_audit_count(score_blob: dict[str, Any], adapter_name: str) -> int:
    """Return 1 iff an evaluator report shows a real adapter attempt.

    ``mech_bench evaluate --full`` exposes adapter execution through
    ``timings.adapter.<name>``. Capability-unavailable probes do not count as
    expensive audits because no simulator run happened.
    """
    timings = score_blob.get("timings") or {}
    if not isinstance(timings, dict):
        return 0
    if _lookup_key_or_path(timings, f"adapter.{adapter_name}") is None:
        return 0

    for item in score_blob.get("feedback") or []:
        if not isinstance(item, dict):
            continue
        if item.get("code") != "capability_unavailable":
            continue
        where = str(item.get("where") or "")
        message = str(item.get("message") or "")
        if adapter_name in where or adapter_name in message:
            return 0
    return 1


def chrono_audit_count(score_blob: dict[str, Any]) -> int:
    return adapter_audit_count(score_blob, "chrono_contact")


def cad_audit_count(score_blob: dict[str, Any]) -> int:
    """Return 1 iff a report contains trusted CAD-kernel evidence.

    This intentionally does not count ordinary ``build_design`` execution.
    The paper contract is about expensive CAD/OCCT validation, so a count is
    emitted only when the full evaluator report includes trusted CAD mass
    evidence from the trusted-asset preflight path.
    """
    metrics = score_blob.get("metrics") or {}
    if not isinstance(metrics, dict):
        return 0
    trusted_mass = _finite_float(
        _lookup_key_or_path(
            metrics,
            "trusted_asset_preflight.trusted_mass_properties_recomputed",
        )
        or _lookup_key_or_path(metrics, "trusted_mass_properties_recomputed")
    )
    trusted_parts = _finite_float(
        _lookup_key_or_path(
            metrics,
            "trusted_asset_preflight.parts_with_trusted_mass_properties",
        )
        or _lookup_key_or_path(metrics, "parts_with_trusted_mass_properties")
    )
    return 1 if (trusted_mass or 0.0) > 0.0 or (trusted_parts or 0.0) > 0.0 else 0
