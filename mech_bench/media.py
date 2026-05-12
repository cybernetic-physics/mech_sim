"""Media manifest for a packaged run.

Today we don't actually render MP4s — but every evaluation still
publishes a manifest that names the slots a future renderer should
populate. That way the dashboard and downstream tooling can link to
``preview.mp4`` even when only the thumbnail is real, and tools can
detect missing media without checking each file.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class MediaManifest:
    """Pointers to media artifacts associated with one run.

    Paths are stored as POSIX strings relative to the manifest file
    where possible, else absolute. ``None`` indicates "this slot is
    planned but has not been rendered yet."
    """

    version: str = "mech_bench.media_manifest.v1"
    run_id: str = ""
    task_id: str = ""
    thumbnail_png: str | None = None
    preview_mp4: str | None = None
    frames_dir: str | None = None
    failure_zoom_mp4: str | None = None
    dashboard_html: str | None = None
    dashboard_payload_json: str | None = None
    trace_h5: str | None = None
    warnings: list[str] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _rel_to(base: Path, target: Path | None) -> str | None:
    if target is None:
        return None
    target = Path(target)
    try:
        return str(target.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(target.resolve())


def write_media_manifest(
    out_dir: Path,
    report: Any,
    *,
    trace_path: Path | None = None,
    dashboard_payload_path: Path | None = None,
    dashboard_html_path: Path | None = None,
    thumbnail_png_path: Path | None = None,
    preview_mp4_path: Path | None = None,
    frames_dir_path: Path | None = None,
    failure_zoom_mp4_path: Path | None = None,
    warnings: list[str] | None = None,
) -> Path:
    """Write ``media_manifest.json`` under *out_dir*.

    ``report`` is an :class:`EvalReport`; only ``run_id`` / ``task_id``
    are read so this module stays decoupled from the full schema.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = MediaManifest(
        run_id=getattr(report, "run_id", "") or "",
        task_id=getattr(report, "task_id", "") or "",
        thumbnail_png=_rel_to(out_dir, thumbnail_png_path),
        preview_mp4=_rel_to(out_dir, preview_mp4_path),
        frames_dir=_rel_to(out_dir, frames_dir_path),
        failure_zoom_mp4=_rel_to(out_dir, failure_zoom_mp4_path),
        dashboard_html=_rel_to(out_dir, dashboard_html_path),
        dashboard_payload_json=_rel_to(out_dir, dashboard_payload_path),
        trace_h5=_rel_to(out_dir, trace_path),
        warnings=list(warnings or []),
    )
    path = out_dir / "media_manifest.json"
    path.write_text(json.dumps(manifest.to_dict(), indent=2))
    return path


# --------------------------------------------------------------------- #
# package-run                                                           #
# --------------------------------------------------------------------- #


# Files we expect to find in a packaged run directory.
PACKAGE_FILES = (
    "scorecard.json",
    "scorecard.public.json",
    "metrics.json",
    "feedback.public.json",
    "dashboard_payload.json",
    "traces.h5",
    "dashboard.html",
    "media_manifest.json",
)


def collect_run_files(report_dir: Path) -> dict[str, Path]:
    """Return the subset of canonical files that actually exist."""
    report_dir = Path(report_dir)
    found: dict[str, Path] = {}
    for name in PACKAGE_FILES:
        p = report_dir / name
        if p.exists():
            found[name] = p
    return found


def package_run(report_dir: Path) -> dict[str, Path]:
    """Ensure a media_manifest.json exists and return the contents
    of the run package.

    If the manifest is missing, write a minimal one referencing
    whichever files are present.
    """
    report_dir = Path(report_dir)
    found = collect_run_files(report_dir)
    if "media_manifest.json" not in found:
        # Reconstruct from whatever artifacts the report-dir contains.
        run_id, task_id = "", ""
        sc = found.get("scorecard.json")
        if sc is not None:
            try:
                blob = json.loads(sc.read_text())
                run_id = str(blob.get("run_id", ""))
                task_id = str(blob.get("task_id", ""))
            except (OSError, json.JSONDecodeError):
                pass

        class _Stub:
            pass
        stub = _Stub()
        stub.run_id = run_id
        stub.task_id = task_id
        write_media_manifest(
            report_dir,
            stub,
            trace_path=found.get("traces.h5"),
            dashboard_payload_path=found.get("dashboard_payload.json"),
            dashboard_html_path=found.get("dashboard.html"),
        )
        found = collect_run_files(report_dir)
    return found
