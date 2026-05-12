"""Trace evidence storage.

Every evaluation can optionally produce a HDF5 ``traces.h5`` bundle
that contains the raw arrays the report numerics were derived from
(joint angles vs time, port traces, contact forces, …). The
benchmark uses these for RLVR debugging, replay, and downstream
video / dashboard rendering — they are NOT part of the scoring
contract and are stripped from any public emission.

The file format is intentionally narrow and stable so that future
adapters (Chrono, MuJoCo, Drake, …) can write the same shape, and
so the reader can be implemented from scratch in any language.

Layout
------

::

    /time_s                          (N,)  float64
    /ports/{port_id}/trace           (N, 2 or 3)
    /ports/{port_id}/velocity        (N, 2 or 3)
    /joints/{joint_id}/position      (N,)
    /joints/{joint_id}/velocity      (N,)
    /bodies/{body_id}/pose           (N, 7)  xyz + quat (w,x,y,z)
    /bodies/{body_id}/twist          (N, 6)
    /contacts/{pair_id}/normal_force (N,)
    /contacts/{pair_id}/penetration  (N,)
    /metrics/{metric_name}           scalar float64
    /                                 attrs: run_id, task_id, adapter, …

If ``h5py`` is not installed, ``write_trace_hdf5`` and the writer
class raise ``TraceUnavailableError`` (a subclass of ImportError) so
callers can detect the missing capability and fall back to writing a
``capability_unavailable.json`` stub.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:  # pragma: no cover - exercised by env, not by tests
    import h5py  # type: ignore[import-not-found]
    HAS_H5PY = True
except ImportError:  # pragma: no cover
    h5py = None  # type: ignore[assignment]
    HAS_H5PY = False


_LOG = logging.getLogger("mech_bench.traces")


class TraceUnavailableError(ImportError):
    """Raised when h5py is required but not installed."""


# --------------------------------------------------------------------- #
# Data container                                                        #
# --------------------------------------------------------------------- #


@dataclass
class TraceData:
    """Replay-grade evidence for one evaluation run.

    All array fields are optional — adapters fill in whatever they
    have. The dashboard / video tooling reads defensively.
    """

    run_id: str = ""
    task_id: str = ""
    adapter: str = ""
    time_s: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    body_poses: dict[str, np.ndarray] = field(default_factory=dict)
    body_twists: dict[str, np.ndarray] = field(default_factory=dict)
    port_traces: dict[str, np.ndarray] = field(default_factory=dict)
    port_velocities: dict[str, np.ndarray] = field(default_factory=dict)
    joint_positions: dict[str, np.ndarray] = field(default_factory=dict)
    joint_velocities: dict[str, np.ndarray] = field(default_factory=dict)
    contact_forces: dict[str, np.ndarray] = field(default_factory=dict)
    penetration: dict[str, np.ndarray] = field(default_factory=dict)
    scalar_metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_sim_output(
        cls,
        sim_output: Any,
        *,
        run_id: str = "",
        task_id: str = "",
        adapter: str = "",
    ) -> "TraceData":
        """Normalize an adapter's output dict (or SimOutput) into a
        TraceData. Unknown keys are ignored. Missing keys are fine.
        """
        if isinstance(sim_output, TraceData):
            td = sim_output
            if run_id:
                td.run_id = run_id
            if task_id:
                td.task_id = task_id
            if adapter:
                td.adapter = adapter
            return td
        if not isinstance(sim_output, dict):
            return cls(run_id=run_id, task_id=task_id, adapter=adapter)

        def _arr_map(key: str) -> dict[str, np.ndarray]:
            raw = sim_output.get(key, {}) or {}
            out: dict[str, np.ndarray] = {}
            for k, v in raw.items():
                if v is None:
                    continue
                try:
                    out[str(k)] = np.asarray(v, dtype=float)
                except (TypeError, ValueError):
                    continue
            return out

        time_s_raw = sim_output.get("time_s")
        if time_s_raw is None:
            time_s = np.zeros(0, dtype=float)
        else:
            time_s = np.asarray(time_s_raw, dtype=float)

        metrics_raw = sim_output.get("scalar_metrics", {}) or {}
        scalar_metrics: dict[str, float] = {}
        for k, v in metrics_raw.items():
            try:
                scalar_metrics[str(k)] = float(v)
            except (TypeError, ValueError):
                continue

        metadata_raw = sim_output.get("metadata", {}) or {}
        metadata: dict[str, Any] = {}
        for k, v in metadata_raw.items():
            if isinstance(v, (str, int, float, bool)):
                metadata[str(k)] = v
            else:
                metadata[str(k)] = json.dumps(v, default=str)

        return cls(
            run_id=run_id or str(sim_output.get("run_id", "")),
            task_id=task_id or str(sim_output.get("task_id", "")),
            adapter=adapter or str(sim_output.get("adapter", "")),
            time_s=time_s,
            body_poses=_arr_map("body_poses"),
            body_twists=_arr_map("body_twists"),
            port_traces=_arr_map("port_traces"),
            port_velocities=_arr_map("port_velocities"),
            joint_positions=_arr_map("joint_positions"),
            joint_velocities=_arr_map("joint_velocities"),
            contact_forces=_arr_map("contact_forces"),
            penetration=_arr_map("penetration"),
            scalar_metrics=scalar_metrics,
            metadata=metadata,
        )

    def is_empty(self) -> bool:
        """True if no array data is present at all."""
        return (
            self.time_s.size == 0
            and not self.body_poses
            and not self.body_twists
            and not self.port_traces
            and not self.port_velocities
            and not self.joint_positions
            and not self.joint_velocities
            and not self.contact_forces
            and not self.penetration
        )


# --------------------------------------------------------------------- #
# Writer / reader                                                       #
# --------------------------------------------------------------------- #


class TraceWriter:
    """Write a :class:`TraceData` to an HDF5 file.

    Use as ``TraceWriter(path).write(trace)`` or via the convenience
    :func:`write_trace_hdf5`. Raises :class:`TraceUnavailableError`
    if h5py is missing.
    """

    def __init__(self, path: Path):
        if not HAS_H5PY:
            raise TraceUnavailableError(
                "h5py is required for trace HDF5 output. Install "
                "with: pip install 'mech-bench[traces]'"
            )
        self.path = Path(path)

    def write(self, trace: TraceData) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(self.path, "w") as f:
            f.attrs["version"] = "mech_bench.trace.v1"
            f.attrs["run_id"] = trace.run_id
            f.attrs["task_id"] = trace.task_id
            f.attrs["adapter"] = trace.adapter
            for k, v in trace.metadata.items():
                f.attrs[f"meta.{k}"] = v

            if trace.time_s.size:
                f.create_dataset("time_s", data=trace.time_s)

            self._write_named(f, "ports", trace.port_traces, "trace")
            self._write_named(f, "ports", trace.port_velocities, "velocity")
            self._write_named(f, "joints", trace.joint_positions, "position")
            self._write_named(f, "joints", trace.joint_velocities, "velocity")
            self._write_named(f, "bodies", trace.body_poses, "pose")
            self._write_named(f, "bodies", trace.body_twists, "twist")
            self._write_named(
                f, "contacts", trace.contact_forces, "normal_force")
            self._write_named(
                f, "contacts", trace.penetration, "penetration")

            if trace.scalar_metrics:
                metrics_grp = f.require_group("metrics")
                for k, v in trace.scalar_metrics.items():
                    metrics_grp.create_dataset(
                        _safe_name(k), data=float(v))
        return self.path

    @staticmethod
    def _write_named(
        root: Any,
        top: str,
        d: dict[str, np.ndarray],
        leaf: str,
    ) -> None:
        if not d:
            return
        grp = root.require_group(top)
        for k, arr in d.items():
            sub = grp.require_group(_safe_name(k))
            if leaf in sub:
                del sub[leaf]
            sub.create_dataset(leaf, data=np.asarray(arr))


class TraceReader:
    """Read a :class:`TraceData` from an HDF5 file."""

    def __init__(self, path: Path):
        if not HAS_H5PY:
            raise TraceUnavailableError(
                "h5py is required to read trace HDF5 files."
            )
        self.path = Path(path)

    def read(self) -> TraceData:
        with h5py.File(self.path, "r") as f:
            run_id = str(f.attrs.get("run_id", ""))
            task_id = str(f.attrs.get("task_id", ""))
            adapter = str(f.attrs.get("adapter", ""))
            metadata: dict[str, Any] = {}
            for k, v in f.attrs.items():
                if not k.startswith("meta."):
                    continue
                metadata[k[len("meta."):]] = _coerce_attr(v)

            time_s = (np.asarray(f["time_s"][...])
                      if "time_s" in f else np.zeros(0, dtype=float))

            port_traces = _read_named(f, "ports", "trace")
            port_velocities = _read_named(f, "ports", "velocity")
            joint_positions = _read_named(f, "joints", "position")
            joint_velocities = _read_named(f, "joints", "velocity")
            body_poses = _read_named(f, "bodies", "pose")
            body_twists = _read_named(f, "bodies", "twist")
            contact_forces = _read_named(f, "contacts", "normal_force")
            penetration = _read_named(f, "contacts", "penetration")

            scalar_metrics: dict[str, float] = {}
            if "metrics" in f:
                for k in f["metrics"]:
                    scalar_metrics[k] = float(f["metrics"][k][()])

        return TraceData(
            run_id=run_id,
            task_id=task_id,
            adapter=adapter,
            time_s=time_s,
            body_poses=body_poses,
            body_twists=body_twists,
            port_traces=port_traces,
            port_velocities=port_velocities,
            joint_positions=joint_positions,
            joint_velocities=joint_velocities,
            contact_forces=contact_forces,
            penetration=penetration,
            scalar_metrics=scalar_metrics,
            metadata=metadata,
        )


def write_trace_hdf5(path: Path, trace: TraceData) -> Path:
    """Write *trace* to *path* in HDF5 format and return *path*."""
    return TraceWriter(path).write(trace)


def read_trace_hdf5(path: Path) -> TraceData:
    """Read a trace file written by :func:`write_trace_hdf5`."""
    return TraceReader(path).read()


def write_capability_unavailable(path: Path, reason: str) -> Path:
    """Emit a stub next to where the trace would have gone.

    Lets downstream tooling (manifest, dashboard) report the gap
    instead of silently producing no artifact.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stub = {
        "version": "mech_bench.trace_stub.v1",
        "status": "capability_unavailable",
        "reason": reason,
    }
    path.write_text(json.dumps(stub, indent=2))
    return path


# --------------------------------------------------------------------- #
# Helpers                                                               #
# --------------------------------------------------------------------- #


def _safe_name(name: str) -> str:
    """HDF5 group/dataset names cannot contain ``/``."""
    return str(name).replace("/", "__")


def _read_named(
    root: Any,
    top: str,
    leaf: str,
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    if top not in root:
        return out
    for k in root[top]:
        sub = root[top][k]
        if leaf in sub:
            out[k] = np.asarray(sub[leaf][...])
    return out


def _coerce_attr(v: Any) -> Any:
    if hasattr(v, "item"):
        try:
            return v.item()
        except (ValueError, TypeError):
            return v
    if isinstance(v, bytes):
        try:
            return v.decode("utf-8")
        except UnicodeDecodeError:
            return v
    return v
