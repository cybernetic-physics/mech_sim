"""Tests for the trace/evidence pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mech_bench.traces import (
    HAS_H5PY,
    TraceData,
    read_trace_hdf5,
    write_trace_hdf5,
)


def _sample_trace() -> TraceData:
    t = np.linspace(0.0, 1.0, 10)
    return TraceData(
        run_id="run_abc",
        task_id="fourbar_path_t001",
        adapter="planar_kinematics",
        time_s=t,
        port_traces={
            "coupler_point": np.column_stack([np.cos(t), np.sin(t)]),
        },
        joint_positions={
            "input_port": t,
            "output_port": 0.5 * t,
        },
        scalar_metrics={"samples": float(len(t))},
        metadata={"topology": "fourbar"},
    )


@pytest.mark.skipif(not HAS_H5PY, reason="h5py not installed")
def test_trace_roundtrip(tmp_path: Path) -> None:
    src = _sample_trace()
    path = write_trace_hdf5(tmp_path / "traces.h5", src)
    assert path.exists()
    back = read_trace_hdf5(path)
    assert back.run_id == src.run_id
    assert back.task_id == src.task_id
    assert back.adapter == src.adapter
    assert np.allclose(back.time_s, src.time_s)
    assert set(back.port_traces) == set(src.port_traces)
    assert np.allclose(
        back.port_traces["coupler_point"],
        src.port_traces["coupler_point"],
    )
    assert np.allclose(
        back.joint_positions["input_port"],
        src.joint_positions["input_port"],
    )
    assert back.scalar_metrics["samples"] == pytest.approx(10.0)
    assert back.metadata.get("topology") == "fourbar"


def test_trace_data_from_dict_normalizes() -> None:
    out = {
        "port_traces": {"coupler_point": [[0.0, 1.0], [1.0, 0.0]]},
        "joint_positions": {"input_port": [0.0, 0.1, 0.2]},
        "time_s": [0.0, 0.5, 1.0],
        "scalar_metrics": {"samples": 3},
        "metadata": {"topology": "fourbar"},
    }
    td = TraceData.from_sim_output(out, run_id="r1", task_id="t1",
                                    adapter="planar_kinematics")
    assert td.run_id == "r1"
    assert td.adapter == "planar_kinematics"
    assert td.time_s.shape == (3,)
    assert "coupler_point" in td.port_traces
    assert td.scalar_metrics["samples"] == 3.0


def test_capability_unavailable_stub(tmp_path: Path) -> None:
    from mech_bench.traces import write_capability_unavailable

    p = write_capability_unavailable(
        tmp_path / "traces.unavailable.json",
        reason="h5py absent",
    )
    assert p.exists()
    blob = json.loads(p.read_text())
    assert blob["status"] == "capability_unavailable"
    assert "h5py" in blob["reason"]
