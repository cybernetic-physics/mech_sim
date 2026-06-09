"""Trust-boundary tests for the submission subprocess.

These prove that an adversarial design.py cannot influence the
trusted evaluator process — monkeypatches, infinite loops, and
non-JSON returns are contained.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

import mech_bench.validation as validation_mod
from mech_bench.evaluator import (
    DEFAULT_SUBMISSION_TIMEOUT,
    SubmissionError,
    evaluate,
    load_submission,
)
from mech_bench.feedback import FailureCode


TASK_DIR = Path(__file__).resolve().parent.parent / "tasks" / "fourbar_path_t001"


def _write_design(tmp_path: Path, body: str) -> Path:
    sub = tmp_path / "submission"
    sub.mkdir(parents=True, exist_ok=True)
    (sub / "design.py").write_text(textwrap.dedent(body))
    return sub


# --------------------------------------------------------------------- #
# Monkeypatch isolation                                                 #
# --------------------------------------------------------------------- #


def test_design_py_cannot_monkeypatch_parent_validation(tmp_path: Path):
    """An adversarial design.py overrides validate_design_ir in the
    subprocess and then ships an invalid IR. The parent process must
    still validate it normally."""
    sentinel_before = validation_mod.validate_design_ir
    sub = _write_design(tmp_path, '''
        from pathlib import Path
        import mech_bench.validation
        # In the subprocess, neuter validation. This monkeypatch must
        # NOT escape into the parent evaluator.
        mech_bench.validation.validate_design_ir = lambda *a, **k: []

        def build_design(out_dir: Path) -> dict:
            return {
                "schema_version": "BOGUS_SCHEMA",   # parent must reject
                "parts": [], "joints": [], "ports": {},
            }
    ''')
    report = evaluate(TASK_DIR, sub, scratch_dir=tmp_path / "scratch")
    assert validation_mod.validate_design_ir is sentinel_before, (
        "Parent process's validate_design_ir was tampered with."
    )
    codes = {f.code.value for f in report.feedback}
    assert FailureCode.SCHEMA_ERROR.value in codes
    assert report.evaluation_valid is False
    assert report.score == 0.0
    assert report.hard_gate_passed is False


def test_design_py_cannot_pollute_sys_modules(tmp_path: Path):
    """The submission imports a sentinel module name. The parent
    must not have it loaded afterward."""
    import sys
    assert "mech_bench_subprocess_sentinel" not in sys.modules
    sub = _write_design(tmp_path, '''
        from pathlib import Path
        import sys
        # Inject a sentinel only the subprocess will see.
        sys.modules["mech_bench_subprocess_sentinel"] = object()

        def build_design(out_dir: Path) -> dict:
            return {
                "schema_version": "design_ir.v2",
                "parts": [
                    {"id": "ground", "fixed": True, "mass_kg": 0.0},
                    {"id": "crank", "mass_kg": 0.02},
                ],
                "joints": [{
                    "id": "j1", "type": "revolute",
                    "parent": "ground", "child": "crank",
                    "axis_world": (0.0, 0.0, 1.0),
                    "anchor_world_mm": (0.0, 0.0, 0.0),
                }],
                "ports": {"input_port": {
                    "id": "input_port", "part": "j1",
                    "kind": "revolute_joint",
                }},
            }
    ''')
    _ = evaluate(TASK_DIR, sub, scratch_dir=tmp_path / "scratch")
    assert "mech_bench_subprocess_sentinel" not in sys.modules


# --------------------------------------------------------------------- #
# Timeout                                                               #
# --------------------------------------------------------------------- #


def test_infinite_loop_design_times_out(tmp_path: Path):
    sub = _write_design(tmp_path, '''
        from pathlib import Path

        def build_design(out_dir: Path) -> dict:
            while True:
                pass
    ''')
    # 1s timeout keeps the test fast.
    with pytest.raises(SubmissionError) as exc:
        load_submission(sub, tmp_path / "scratch", timeout=1.0)
    assert "1.0s" in str(exc.value) or "did not finish" in str(exc.value)


def test_infinite_loop_design_surfaces_invalid_artifact(tmp_path: Path):
    sub = _write_design(tmp_path, '''
        from pathlib import Path
        def build_design(out_dir: Path) -> dict:
            while True:
                pass
    ''')
    report = evaluate(
        TASK_DIR, sub, scratch_dir=tmp_path / "scratch",
        submission_timeout=1.0,
    )
    codes = {f.code.value for f in report.feedback}
    assert FailureCode.INVALID_ARTIFACT.value in codes
    assert report.hard_gate_passed is False
    assert report.evaluation_valid is False


def test_load_submission_canonicalizes_model_topology_near_misses(
    tmp_path: Path,
):
    sub = _write_design(tmp_path, '''
        from pathlib import Path
        def build_design(out_dir: Path) -> dict:
            return {
                "schema_version": "design_ir.v2",
                "parts": [
                    {"id": "frame", "role": "ground",
                     "fixed": True, "mass_kg": 0.0},
                    {"id": "cam", "role": "input", "mass_kg": 0.02},
                    {"id": "follower", "role": "output", "mass_kg": 0.05},
                    {"id": "cam_follower_contact",
                     "type": "contact_pair",
                     "parent": "cam", "child": "follower"},
                ],
                "joints": [],
                "ports": {
                    "input_port": {
                        "id": "input_port",
                        "part": "cam_follower_contact",
                        "kind": "revolute_joint",
                    },
                },
            }
    ''')
    ir = load_submission(
        sub,
        tmp_path / "scratch",
        required_port_kinds={"input_port": "revolute_joint"},
    )
    assert {p.id for p in ir.parts} == {"frame", "cam", "follower"}
    assert {j.id for j in ir.joints} == {
        "cam_follower_contact", "input_port",
    }
    assert ir.ports["input_port"].part == "input_port"


def test_load_submission_canonicalizes_noisy_port_records(
    tmp_path: Path,
):
    sub = _write_design(tmp_path, '''
        from pathlib import Path
        def build_design(out_dir: Path) -> dict:
            return {
                "schema_version": "design_ir.v2",
                "parts": [
                    {"id": "frame", "role": "ground",
                     "fixed": True, "mass_kg": 0.0},
                    {"id": "cam", "role": "input", "mass_kg": 0.02},
                    {"id": "follower", "role": "output", "mass_kg": 0.05},
                ],
                "joints": [
                    {"id": "cam_follower_contact",
                     "type": "contact_pair",
                     "parent": "cam", "child": "follower"},
                ],
                "ports": {
                    "input_port": {
                        "id": "input_port",
                        "type": "revolute_joint",
                        "grounded": True,
                        "parent": "frame",
                        "part": "cam",
                    },
                    "output_port": {
                        "id": "output_port",
                        "kind": "revolute_joint",
                        "grounded": False,
                        "port": {"part": "follower"},
                    },
                },
            }
    ''')
    ir = load_submission(
        sub,
        tmp_path / "scratch",
        required_port_kinds={
            "input_port": "revolute_joint",
            "output_port": "revolute_joint",
        },
    )

    assert ir.ports["input_port"].part == "input_port"
    assert ir.ports["output_port"].part == "output_port"
    assert {j.id for j in ir.joints} == {
        "cam_follower_contact", "input_port", "output_port",
    }


def test_evaluate_canonicalized_submission_reaches_required_ports_probe(
    tmp_path: Path,
):
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / "fixtures").mkdir()
    (task_dir / "task.toml").write_text(textwrap.dedent('''
        [task]
        id = "canonicalized_cam"
        family = "cam_follower"
        difficulty = 1
        units = "mm"

        [requirements]
        required_ports = ["input_port"]
    ''').strip())
    (task_dir / "eval_config.toml").write_text(textwrap.dedent('''
        [hard_gate]
        require = ["ports"]

        [[probes]]
        id = "ports"
        type = "required_ports"
        ports = ["input_port"]
        require_grounded = ["input_port"]
        hard_gate = true
        severity = "critical"

        [probes.require_kinds]
        input_port = "revolute_joint"
    ''').strip())
    sub = _write_design(tmp_path, '''
        from pathlib import Path
        def build_design(out_dir: Path) -> dict:
            return {
                "schema_version": "design_ir.v2",
                "parts": [
                    {"id": "frame", "role": "ground",
                     "fixed": True, "mass_kg": 0.0},
                    {"id": "cam", "role": "input", "mass_kg": 0.02},
                    {"id": "follower", "role": "output", "mass_kg": 0.05},
                    {"id": "cam_follower_contact",
                     "type": "contact_pair",
                     "parent": "cam", "child": "follower"},
                ],
                "joints": [],
                "ports": {
                    "input_port": {
                        "id": "input_port",
                        "part": "cam",
                        "kind": "revolute_joint",
                    },
                },
            }
    ''')
    report = evaluate(task_dir, sub, scratch_dir=tmp_path / "scratch_eval")
    codes = {f.code.value for f in report.feedback}
    assert FailureCode.INVALID_ARTIFACT.value not in codes
    assert FailureCode.MISSING_PORT.value not in codes
    assert FailureCode.WRONG_TOPOLOGY.value not in codes
    assert report.evaluation_valid is True
    assert report.hard_gate_passed is True


# --------------------------------------------------------------------- #
# Non-JSON / malformed return value                                     #
# --------------------------------------------------------------------- #


def test_design_returns_non_dict(tmp_path: Path):
    sub = _write_design(tmp_path, '''
        from pathlib import Path
        def build_design(out_dir: Path):
            return [1, 2, 3]
    ''')
    report = evaluate(TASK_DIR, sub, scratch_dir=tmp_path / "scratch")
    codes = {f.code.value for f in report.feedback}
    assert FailureCode.INVALID_ARTIFACT.value in codes
    assert report.hard_gate_passed is False


def test_design_returns_non_json_serializable(tmp_path: Path):
    sub = _write_design(tmp_path, '''
        from pathlib import Path

        class Junk:
            pass

        def build_design(out_dir: Path) -> dict:
            return {
                "schema_version": "design_ir.v2",
                "parts": [Junk()],   # not JSON-serializable
                "joints": [],
                "ports": {},
            }
    ''')
    report = evaluate(TASK_DIR, sub, scratch_dir=tmp_path / "scratch")
    codes = {f.code.value for f in report.feedback}
    assert FailureCode.INVALID_ARTIFACT.value in codes
    assert report.evaluation_valid is False


def test_design_returns_nan_field_rejected(tmp_path: Path):
    """allow_nan=False inside the worker means a NaN/Inf in the
    submission IR is surfaced at the trust boundary."""
    sub = _write_design(tmp_path, '''
        from pathlib import Path
        def build_design(out_dir: Path) -> dict:
            return {
                "schema_version": "design_ir.v2",
                "parts": [{
                    "id": "crank", "mass_kg": float("nan"),
                }],
                "joints": [],
                "ports": {},
            }
    ''')
    report = evaluate(TASK_DIR, sub, scratch_dir=tmp_path / "scratch")
    codes = {f.code.value for f in report.feedback}
    assert FailureCode.INVALID_ARTIFACT.value in codes


def test_design_py_raises(tmp_path: Path):
    sub = _write_design(tmp_path, '''
        from pathlib import Path
        def build_design(out_dir: Path) -> dict:
            raise RuntimeError("boom")
    ''')
    report = evaluate(TASK_DIR, sub, scratch_dir=tmp_path / "scratch")
    codes = {f.code.value for f in report.feedback}
    assert FailureCode.INVALID_ARTIFACT.value in codes


# --------------------------------------------------------------------- #
# Happy path                                                            #
# --------------------------------------------------------------------- #


def test_reference_solution_still_passes(tmp_path: Path):
    report = evaluate(
        TASK_DIR,
        TASK_DIR / "reference_solution",
        scratch_dir=tmp_path,
    )
    assert report.evaluation_valid is True
    assert report.hard_gate_passed is True
    assert report.score > 0.99
