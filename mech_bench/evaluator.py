"""Generic evaluator.

Pipeline (called once per submission):

  1. Load TaskSpec + EvalConfig from a task directory.
  2. Run the submission's ``design.py`` in an isolated subprocess
     (``mech_bench.submission_worker``) and parse the returned dict
     into a DesignIR. The evaluator process never imports the
     submission's design.py.
  3. Validate the DesignIR. If validation surfaces critical
     structural failures, short-circuit before any probe runs and
     return a zero report whose feedback is the validation failures.
  4. Build an ExecutionPlan: per probe, pick the cheapest adapter
     whose capabilities cover the probe's requirements. Probes that
     need an adapter for which none is registered produce
     CAPABILITY_UNAVAILABLE and mark the whole evaluation invalid.
  5. Run each adapter at most once; pass its outputs to the probes
     that need it. Adapter exceptions also invalidate the evaluation.
  6. Compose the final score (see ``_score`` for the rules) and
     sanitize all numeric values for strict JSON.

The runtime treats the submission as adversarial. Path-policy
enforcement lives in mech_bench.validation; out-of-process execution
lives in mech_bench.submission_worker.
"""

from __future__ import annotations

import json
import logging
import math
import os
import subprocess
import sys
import time
import tomllib
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mech_bench.adapters import (
    SimAdapter,
    all_adapters,
    normalize_sim_output,
)
from mech_bench.feedback import Failure, FailureCode, Severity
from mech_bench.metrics import (
    compute_class_metrics,
    compute_general_metrics,
    compute_tier_metrics,
    derive_tier,
    fill_defaults_for_dashboard,
)
from mech_bench.probes import Capability, Probe, get_probe
from mech_bench.schema import (
    DesignIR,
    EvalConfig,
    EvalReport,
    FeedbackVisibility,
    ProbeResult,
    ProbeSpec,
    TaskSpec,
)
from mech_bench.validation import has_critical_failures, validate_design_ir

_LOG = logging.getLogger("mech_bench.evaluator")

DEFAULT_SUBMISSION_TIMEOUT = 10.0


class SubmissionError(Exception):
    """The submission subprocess failed in a structured way."""


# --------------------------------------------------------------------- #
# Execution planning                                                    #
# --------------------------------------------------------------------- #


@dataclass
class ProbePlan:
    probe_id: str
    probe_type: str
    capabilities: frozenset[Capability]
    adapter_type: str | None = None
    available: bool = True
    reason: str = ""
    probe_known: bool = True


@dataclass
class ExecutionPlan:
    probes: list[ProbePlan] = field(default_factory=list)

    def adapters_to_run(self) -> list[str]:
        seen: list[str] = []
        for p in self.probes:
            if p.adapter_type and p.adapter_type not in seen:
                seen.append(p.adapter_type)
        return seen


def _pick_adapter_for(
    caps: frozenset[Capability],
    *,
    exclude: frozenset[str] = frozenset(),
) -> type[SimAdapter] | None:
    needed = caps - {Capability.NONE}
    if not needed:
        return None
    candidates = [a for a in all_adapters()
                  if a.type_name not in exclude
                  and needed.issubset(a.capabilities_provided)]
    if not candidates:
        return None
    candidates.sort(key=lambda a: a.cost_tier)
    return candidates[0]


def _lookup_adapter(type_name: str) -> type[SimAdapter] | None:
    for a in all_adapters():
        if a.type_name == type_name:
            return a
    return None


def _maybe_register_fake_oracle(
    cfg: EvalConfig, mode: str,
) -> None:
    """Register fake_contact_oracle iff explicitly enabled.

    Triggers:

    * ``[adapters.fake_contact_oracle] enabled = true`` in eval_config.
    * Any mode whose ``forced_adapter = "fake_contact_oracle"``.

    Env-variable triggers (``MECH_BENCH_USE_FAKE_ORACLE``,
    ``MECH_BENCH_TEST_MODE``) are handled at import time by the
    fake-oracle module; this only forces explicit-config opt-in.
    """
    fake_cfg = cfg.adapter_configs.get("fake_contact_oracle", {}) or {}
    enabled = bool(fake_cfg.get("enabled", False))
    forced_in_any_mode = any(
        (m.forced_adapter == "fake_contact_oracle")
        for m in cfg.modes.values()
    )
    # Per-probe overrides also count.
    forced_in_probe = any(
        isinstance(spec.config.get("adapter"), str)
        and spec.config.get("adapter") == "fake_contact_oracle"
        for spec in cfg.probes
    )
    if enabled or forced_in_any_mode or forced_in_probe:
        from mech_bench.adapters import fake_contact_oracle as _fco
        _fco.force_register()


def _resolve_forced_adapter(
    cfg: EvalConfig, mode: str,
) -> str | None:
    if mode and mode in cfg.modes:
        forced = cfg.modes[mode].forced_adapter
        if forced:
            return str(forced)
    return None


def _adapter_runtime_context(
    *,
    task: TaskSpec,
    cfg: EvalConfig,
    plan: ExecutionPlan,
    adapter_name: str,
    build_root: Path,
) -> dict[str, Any]:
    """Serializable task/probe context for one adapter run."""
    by_id = {p.probe_id: p for p in plan.probes}
    probe_specs: list[dict[str, Any]] = []
    for spec in cfg.probes:
        pplan = by_id.get(spec.id)
        if pplan is None or pplan.adapter_type != adapter_name:
            continue
        probe_specs.append({
            "id": spec.id,
            "type": spec.type,
            "config": dict(spec.config),
            "weight": float(spec.weight),
            "severity": spec.severity,
            "hard_gate": bool(spec.hard_gate),
            "tier": spec.tier,
            "class_metric": spec.class_metric,
        })
    return {
        "task": {
            "id": task.id,
            "family": task.family,
            "difficulty": task.difficulty,
            "units": task.units,
        },
        "build_root": str(Path(build_root).resolve()),
        "probe_specs": probe_specs,
    }


def _fake_oracle_is_explicit(
    cfg: EvalConfig, forced_adapter: str | None,
) -> bool:
    """True iff the eval config explicitly opts into fake_contact_oracle.

    Triggers (any one):
      * ``[adapters.fake_contact_oracle] enabled = true``
      * mode-level ``forced_adapter = "fake_contact_oracle"``
      * any probe with ``adapter = "fake_contact_oracle"``
      * env vars ``MECH_BENCH_USE_FAKE_ORACLE`` / ``MECH_BENCH_TEST_MODE``
        — kept for backwards compat with the existing test surface.
    """
    import os as _os
    if forced_adapter == "fake_contact_oracle":
        return True
    fake_cfg = cfg.adapter_configs.get("fake_contact_oracle", {}) or {}
    if bool(fake_cfg.get("enabled", False)):
        return True
    if any(spec.config.get("adapter") == "fake_contact_oracle"
           for spec in cfg.probes):
        return True
    if any(m.forced_adapter == "fake_contact_oracle"
           for m in cfg.modes.values()):
        return True
    for var in ("MECH_BENCH_USE_FAKE_ORACLE", "MECH_BENCH_TEST_MODE"):
        if _os.environ.get(var, "").lower() in ("1", "true", "yes"):
            return True
    return False


def build_execution_plan(
    cfg: EvalConfig,
    *,
    forced_adapter: str | None = None,
) -> ExecutionPlan:
    plan = ExecutionPlan()
    # Fake oracle is opt-in per task. If the current eval config / mode
    # / probe configs don't explicitly request it, exclude it from
    # auto-selection even if it's been registered globally by another
    # task earlier in the process.
    fake_oracle_explicit = _fake_oracle_is_explicit(cfg, forced_adapter)
    exclude: frozenset[str] = frozenset(
        () if fake_oracle_explicit else ("fake_contact_oracle",)
    )
    for spec in cfg.probes:
        try:
            probe = get_probe(spec.type)
        except KeyError as e:
            plan.probes.append(ProbePlan(
                probe_id=spec.id,
                probe_type=spec.type,
                capabilities=frozenset(),
                adapter_type=None,
                available=False,
                reason=str(e),
                probe_known=False,
            ))
            continue
        caps = frozenset(probe.capabilities_required)
        if caps <= {Capability.NONE}:
            plan.probes.append(ProbePlan(
                probe_id=spec.id,
                probe_type=spec.type,
                capabilities=caps,
                adapter_type=None,
                available=True,
            ))
            continue

        # Per-probe adapter override (highest priority).
        probe_override = spec.config.get("adapter")
        adapter: type[SimAdapter] | None = None
        if isinstance(probe_override, str) and probe_override:
            cls = _lookup_adapter(probe_override)
            needed = caps - {Capability.NONE}
            if cls is None:
                plan.probes.append(ProbePlan(
                    probe_id=spec.id,
                    probe_type=spec.type,
                    capabilities=caps,
                    adapter_type=None,
                    available=False,
                    reason=(
                        f"Probe-level adapter {probe_override!r} is "
                        f"not registered."
                    ),
                ))
                continue
            if not needed.issubset(cls.capabilities_provided):
                missing = sorted(
                    c.value for c in needed - cls.capabilities_provided)
                plan.probes.append(ProbePlan(
                    probe_id=spec.id,
                    probe_type=spec.type,
                    capabilities=caps,
                    adapter_type=None,
                    available=False,
                    reason=(
                        f"Probe-level adapter {probe_override!r} does "
                        f"not provide required capabilities: {missing}."
                    ),
                ))
                continue
            adapter = cls
        elif forced_adapter:
            cls = _lookup_adapter(forced_adapter)
            needed = caps - {Capability.NONE}
            if cls is None:
                plan.probes.append(ProbePlan(
                    probe_id=spec.id,
                    probe_type=spec.type,
                    capabilities=caps,
                    adapter_type=None,
                    available=False,
                    reason=(
                        f"Forced adapter {forced_adapter!r} is not "
                        f"registered."
                    ),
                ))
                continue
            if not needed.issubset(cls.capabilities_provided):
                missing = sorted(
                    c.value for c in needed - cls.capabilities_provided)
                plan.probes.append(ProbePlan(
                    probe_id=spec.id,
                    probe_type=spec.type,
                    capabilities=caps,
                    adapter_type=None,
                    available=False,
                    reason=(
                        f"Forced adapter {forced_adapter!r} lacks "
                        f"required capabilities: {missing}."
                    ),
                ))
                continue
            adapter = cls
        else:
            adapter = _pick_adapter_for(caps, exclude=exclude)
        if adapter is None:
            missing = sorted(c.value for c in caps - {Capability.NONE})
            plan.probes.append(ProbePlan(
                probe_id=spec.id,
                probe_type=spec.type,
                capabilities=caps,
                adapter_type=None,
                available=False,
                reason=(f"No registered adapter provides the required "
                        f"capabilities: {missing}."),
            ))
            continue
        plan.probes.append(ProbePlan(
            probe_id=spec.id,
            probe_type=spec.type,
            capabilities=caps,
            adapter_type=adapter.type_name,
            available=True,
        ))
    return plan


# --------------------------------------------------------------------- #
# Tier classification                                                   #
# --------------------------------------------------------------------- #


_TIER_BY_CAP: dict[Capability, str] = {
    Capability.NONE: "topology",
    Capability.PLANAR_KINEMATICS: "kinematics",
    Capability.SPATIAL_KINEMATICS: "kinematics",
    Capability.PATH_TRACE: "kinematics",
    Capability.DOF_DETECTION: "kinematics",
    Capability.MESH_OVERLAP: "geometry",
    Capability.MESH: "geometry",
    Capability.RIGID_BODY_DYNAMICS: "dynamics",
    Capability.CONTACT_FORCES: "dynamics",
    Capability.JOINT_CONSTRAINTS: "dynamics",
    Capability.MOTOR_DRIVES: "dynamics",
    Capability.LOAD_TORQUES: "dynamics",
    Capability.POSE_TRACES: "dynamics",
    Capability.FEA_STATIC: "structural",
    Capability.SAFETY_FACTOR: "structural",
}


def _tier_for(caps: frozenset[Capability]) -> str:
    pruned = caps - {Capability.NONE}
    if not pruned:
        return "topology"
    tiers = {_TIER_BY_CAP.get(c, "other") for c in pruned}
    order = ["topology", "kinematics", "geometry", "dynamics",
             "structural", "other"]
    for t in reversed(order):
        if t in tiers:
            return t
    return "other"


# --------------------------------------------------------------------- #
# Task / submission loading                                             #
# --------------------------------------------------------------------- #


def load_task(task_dir: Path) -> tuple[TaskSpec, EvalConfig]:
    task_dir = Path(task_dir)
    task_toml = task_dir / "task.toml"
    eval_toml = task_dir / "eval_config.toml"
    prompt_md = task_dir / "prompt.md"

    with task_toml.open("rb") as f:
        task_data = tomllib.load(f)
    with eval_toml.open("rb") as f:
        eval_data = tomllib.load(f)

    fixtures_dir = task_dir / "fixtures"
    for entry in eval_data.get("probes", []):
        if "target_csv" in entry:
            entry["target_csv"] = str(
                (fixtures_dir / entry["target_csv"]).resolve()
            )

    prompt = prompt_md.read_text() if prompt_md.exists() else ""
    task = TaskSpec.from_dict(task_data, fixtures_dir=fixtures_dir,
                              prompt=prompt)
    cfg = EvalConfig.from_dict(eval_data)
    return task, cfg


_JOINT_TYPES = {"revolute", "prismatic", "fixed", "contact_pair", "spherical"}
_JOINT_TYPE_ALIASES = {
    "revolute_joint": "revolute",
    "prismatic_joint": "prismatic",
    "fixed_joint": "fixed",
}
_JOINT_FIELDS = {
    "id",
    "type",
    "parent",
    "child",
    "axis_world",
    "anchor_world_mm",
    "limits_rad",
    "params",
}
_PORT_FIELDS = {"id", "part", "kind", "pose_local_mm"}
_PORT_KIND_ALIASES = {
    "frame_port": "frame",
    "revolute": "revolute_joint",
    "prismatic": "prismatic_joint",
}


def _required_port_kinds(cfg: EvalConfig) -> dict[str, str]:
    kinds: dict[str, str] = {}
    for spec in cfg.probes:
        if spec.type != "required_ports":
            continue
        raw = spec.config.get("require_kinds") or {}
        if not isinstance(raw, dict):
            continue
        for pid, kind in raw.items():
            if isinstance(pid, str) and isinstance(kind, str):
                kinds[pid] = kind
    return kinds


def _normalize_joint_type(raw_type: Any) -> str:
    if not isinstance(raw_type, str):
        return ""
    return _JOINT_TYPE_ALIASES.get(raw_type, raw_type)


def _looks_like_joint_record(item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    raw_type = item.get("type")
    joint_type = _normalize_joint_type(raw_type)
    if joint_type in _JOINT_TYPES:
        return True
    return (
        isinstance(item.get("parent"), str)
        and isinstance(item.get("child"), str)
        and isinstance(raw_type, str)
    )


def _joint_record(item: dict[str, Any]) -> dict[str, Any] | None:
    joint_type = _normalize_joint_type(item.get("type"))
    if joint_type not in _JOINT_TYPES:
        return None
    out = {k: v for k, v in item.items() if k in _JOINT_FIELDS}
    out["type"] = joint_type
    if "params" in out and not isinstance(out["params"], dict):
        out.pop("params", None)
    return out


def _normalize_ports(raw_ports: Any) -> dict[str, Any]:
    if isinstance(raw_ports, dict):
        ports = dict(raw_ports)
    elif isinstance(raw_ports, list):
        ports = {}
        for item in raw_ports:
            if not isinstance(item, dict):
                continue
            pid = item.get("id")
            if isinstance(pid, str) and pid:
                ports[pid] = dict(item)
    else:
        return {}
    for pid, port in list(ports.items()):
        if not isinstance(port, dict):
            continue
        port = _canonical_port_record(str(pid), port)
        ports[pid] = port
        port.setdefault("id", str(pid))
        if port.get("id") != str(pid):
            port["id"] = str(pid)
    return ports


def _canonical_port_record(pid: str, port: dict[str, Any]) -> dict[str, Any]:
    out = dict(port)
    out.setdefault("id", pid)
    raw_kind = out.get("kind", out.get("type"))
    if isinstance(raw_kind, str):
        out["kind"] = _PORT_KIND_ALIASES.get(raw_kind, raw_kind)

    if "part" not in out:
        nested = out.get("port")
        if isinstance(nested, dict) and isinstance(nested.get("part"), str):
            out["part"] = nested["part"]
        elif isinstance(nested, str):
            out["part"] = nested
    if "pose_local_mm" not in out and "anchor_world_mm" in out:
        out["pose_local_mm"] = out["anchor_world_mm"]

    return {k: v for k, v in out.items() if k in _PORT_FIELDS}


def _infer_fixed_part(parts: list[dict[str, Any]]) -> str | None:
    for p in parts:
        if isinstance(p, dict) and p.get("fixed") is True:
            pid = p.get("id")
            if isinstance(pid, str) and pid:
                return pid
    for p in parts:
        if not isinstance(p, dict):
            continue
        pid = p.get("id")
        role = str(p.get("role") or "").lower()
        if isinstance(pid, str) and pid and (
            pid.lower() in {"ground", "frame", "base"}
            or role in {"ground", "frame", "base"}
        ):
            return pid
    return None


def _infer_port_child_part(
    *,
    port_id: str,
    port: dict[str, Any],
    parts: list[dict[str, Any]],
) -> str | None:
    part_ids = {
        str(p.get("id"))
        for p in parts
        if isinstance(p, dict) and isinstance(p.get("id"), str)
    }
    raw_target = port.get("part")
    if isinstance(raw_target, str) and raw_target in part_ids:
        return raw_target

    lowered = port_id.lower()
    role_targets: list[str] = []
    if "input" in lowered:
        role_targets = ["input", "drive", "driver"]
    elif "output" in lowered:
        role_targets = ["output", "driven", "follower", "slider"]
    for role in role_targets:
        for p in parts:
            if not isinstance(p, dict):
                continue
            pid = p.get("id")
            prole = str(p.get("role") or "").lower()
            if isinstance(pid, str) and pid and prole == role:
                return pid
    if len(part_ids) == 1:
        return next(iter(part_ids))
    return None


def _canonicalize_submission_raw(
    raw: dict[str, Any],
    *,
    required_port_kinds: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Repair common model-output topology slips before DesignIR parsing.

    This is not a second validator and it does not make arbitrary bad
    artifacts pass. It only fixes deterministic shape mistakes that the
    benchmark prompts make unambiguous: ports-as-list, joint/contact records
    accidentally emitted in ``parts``, aliased joint types, and required
    joint ports that point at a physical body instead of the needed joint id.
    """
    if not isinstance(raw, dict):
        return raw

    out = dict(raw)
    raw_parts = out.get("parts")
    raw_joints = out.get("joints")
    parts_in = list(raw_parts) if isinstance(raw_parts, list) else raw_parts
    joints_in = list(raw_joints) if isinstance(raw_joints, list) else raw_joints
    if not isinstance(parts_in, list) or not isinstance(joints_in, list):
        return out

    parts: list[dict[str, Any]] = []
    joints: list[dict[str, Any]] = []
    for item in parts_in:
        if _looks_like_joint_record(item):
            joint = _joint_record(item)
            if joint is not None:
                joints.append(joint)
            continue
        parts.append(item)
    for item in joints_in:
        if isinstance(item, dict):
            joint = _joint_record(item)
            joints.append(joint if joint is not None else item)
        else:
            joints.append(item)

    deduped_joints: list[Any] = []
    seen_joint_ids: set[str] = set()
    for joint in joints:
        if not isinstance(joint, dict):
            deduped_joints.append(joint)
            continue
        jid = joint.get("id")
        if isinstance(jid, str) and jid:
            if jid in seen_joint_ids:
                continue
            seen_joint_ids.add(jid)
        deduped_joints.append(joint)

    ports = _normalize_ports(out.get("ports"))
    fixed_part = _infer_fixed_part(parts)
    part_ids = {
        str(p.get("id"))
        for p in parts
        if isinstance(p, dict) and isinstance(p.get("id"), str)
    }
    joint_by_id = {
        str(j.get("id")): j
        for j in deduped_joints
        if isinstance(j, dict) and isinstance(j.get("id"), str)
    }
    joint_ids = set(joint_by_id)
    required_port_kinds = required_port_kinds or {}

    for pid, port in list(ports.items()):
        if not isinstance(port, dict):
            continue
        kind = port.get("kind") or required_port_kinds.get(str(pid))
        if kind not in {"revolute_joint", "prismatic_joint"}:
            continue
        target = port.get("part")
        expected_joint_type = (
            "prismatic" if kind == "prismatic_joint" else "revolute"
        )
        target_joint = joint_by_id.get(target) if isinstance(target, str) else None
        if (
            isinstance(target_joint, dict)
            and target_joint.get("type") == expected_joint_type
        ):
            continue
        child = _infer_port_child_part(
            port_id=str(pid),
            port=port,
            parts=parts,
        )
        if fixed_part is None or child is None or child == fixed_part:
            continue
        joint_id = str(pid)
        existing_for_port = joint_by_id.get(joint_id)
        if (
            joint_id in part_ids
            or (
                isinstance(existing_for_port, dict)
                and existing_for_port.get("type") != expected_joint_type
            )
        ):
            joint_id = f"{joint_id}_joint"
        suffix = 2
        while (
            joint_id in joint_by_id
            and joint_by_id[joint_id].get("type") != expected_joint_type
        ):
            joint_id = f"{pid}_joint_{suffix}"
            suffix += 1
        if joint_id not in joint_ids:
            joint = {
                "id": joint_id,
                "type": expected_joint_type,
                "parent": fixed_part,
                "child": child,
                "axis_world": (0.0, 0.0, 1.0),
                "anchor_world_mm": port.get(
                    "pose_local_mm", (0.0, 0.0, 0.0)
                ),
            }
            deduped_joints.append(joint)
            joint_by_id[joint_id] = joint
            joint_ids.add(joint_id)
        port["part"] = joint_id
        port["kind"] = kind

    out["parts"] = parts
    out["joints"] = deduped_joints
    out["ports"] = ports
    return out


def load_submission(
    submission_dir: Path,
    scratch_dir: Path,
    *,
    timeout: float = DEFAULT_SUBMISSION_TIMEOUT,
    required_port_kinds: dict[str, str] | None = None,
) -> DesignIR:
    """Execute the submission's design.py in an isolated subprocess.

    The evaluator process must NOT import the agent's design.py — any
    monkeypatch the agent applies stays inside the subprocess.
    """
    submission_dir = Path(submission_dir).resolve()
    scratch_dir = Path(scratch_dir).resolve()
    scratch_dir.mkdir(parents=True, exist_ok=True)

    design_py = submission_dir / "design.py"
    if not design_py.exists():
        raise SubmissionError(f"Submission missing design.py: {design_py}")

    result_json = scratch_dir / "_design_ir.json"
    if result_json.exists():
        try:
            result_json.unlink()
        except OSError:
            pass

    worker_path = Path(__file__).with_name("submission_worker.py")
    if worker_path.is_file():
        cmd = [
            sys.executable, "-I",
            str(worker_path),
            "--design-py", str(design_py),
            "--out-dir", str(scratch_dir),
            "--result-json", str(result_json),
        ]
    else:
        # Fallback: module-based launch (requires the package to be
        # importable on sys.path already).
        cmd = [
            sys.executable, "-I",
            "-m", "mech_bench.submission_worker",
            "--design-py", str(design_py),
            "--out-dir", str(scratch_dir),
            "--result-json", str(result_json),
        ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        raise SubmissionError(
            f"build_design did not finish within {timeout}s; subprocess "
            f"killed."
        )

    if proc.returncode != 0:
        stderr_tail = (proc.stderr or "")[-1000:].strip()
        raise SubmissionError(
            f"Submission subprocess exited with code "
            f"{proc.returncode}: {stderr_tail or '<no stderr>'}"
        )

    if not result_json.exists():
        raise SubmissionError(
            "Submission subprocess exited 0 but did not write the "
            "result JSON."
        )

    try:
        text = result_json.read_text()
        raw = json.loads(text)
    except (OSError, json.JSONDecodeError) as e:
        raise SubmissionError(f"Could not parse submission JSON: {e}")

    if not isinstance(raw, dict):
        raise SubmissionError(
            f"Submission JSON root must be a dict, got "
            f"{type(raw).__name__}."
        )

    raw = _canonicalize_submission_raw(
        raw,
        required_port_kinds=required_port_kinds,
    )
    try:
        (scratch_dir / "_design_ir_canonicalized.json").write_text(
            _strict_json_dumps(raw, indent=2)
        )
    except OSError:
        pass

    ir, errors = DesignIR.try_from_dict(raw)
    if ir is None:
        raise SubmissionError(
            "Submission JSON does not fit DesignIR schema: "
            + "; ".join(errors)
        )
    return ir


# --------------------------------------------------------------------- #
# Sanitation                                                            #
# --------------------------------------------------------------------- #


def sanitize_metric_value(v: Any) -> float | None:
    """Coerce a metric value for strict JSON.

    Non-finite (NaN / +Inf / -Inf) becomes None. Booleans are
    preserved as 1.0 / 0.0. Strings / dicts / lists pass through
    unchanged — metrics dicts only ever hold floats today, but the
    helper stays liberal so it can be reused on report blobs.
    """
    if v is None:
        return None
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if isinstance(v, (int, float)):
        f = float(v)
        if math.isfinite(f):
            return f
        return None
    return v  # passthrough


def sanitize_metrics_dict(d: dict[str, Any]) -> dict[str, Any]:
    return {k: sanitize_metric_value(v) for k, v in d.items()}


def sanitize_report_for_json(blob: Any) -> Any:
    """Recursively replace NaN/Inf with None so json.dumps with
    ``allow_nan=False`` succeeds.
    """
    if isinstance(blob, dict):
        return {k: sanitize_report_for_json(v) for k, v in blob.items()}
    if isinstance(blob, (list, tuple)):
        return [sanitize_report_for_json(v) for v in blob]
    if isinstance(blob, bool):
        return blob
    if isinstance(blob, (int, float)):
        f = float(blob)
        if math.isfinite(f):
            return blob
        return None
    return blob


def _strict_json_dumps(obj: Any, indent: int = 2) -> str:
    return json.dumps(
        sanitize_report_for_json(obj),
        indent=indent,
        default=str,
        allow_nan=False,
    )


# --------------------------------------------------------------------- #
# Probe execution                                                       #
# --------------------------------------------------------------------- #


def _run_probe(
    spec: ProbeSpec,
    plan: ProbePlan,
    ir: DesignIR,
    sim_outputs_by_adapter: dict[str, dict[str, Any]],
) -> ProbeResult:
    if not plan.probe_known:
        return ProbeResult(
            probe_id=spec.id,
            probe_type=spec.type,
            passed=False,
            score=0.0,
            metrics={},
            failures=[Failure(
                code=FailureCode.CAPABILITY_UNAVAILABLE,
                severity=Severity.CRITICAL,
                message=plan.reason or
                        f"Unknown probe type {spec.type!r}.",
            )],
            skipped_reason=plan.reason,
        )
    if not plan.available:
        return ProbeResult(
            probe_id=spec.id,
            probe_type=spec.type,
            passed=False,
            score=0.0,
            metrics={},
            failures=[Failure(
                code=FailureCode.CAPABILITY_UNAVAILABLE,
                severity=Severity.CRITICAL,
                message=plan.reason or
                        f"No adapter available for {spec.type!r}.",
                public_hint=(
                    "This probe needs a simulator that is not "
                    "registered in this evaluator build."
                ),
            )],
            skipped_reason=plan.reason,
        )
    probe: Probe = get_probe(spec.type)
    sim_outputs: dict[str, Any] = {}
    if plan.adapter_type is not None:
        sim_outputs = sim_outputs_by_adapter.get(plan.adapter_type, {})
    result = probe.run(ir, sim_outputs, spec.config)
    result.probe_id = spec.id
    result.probe_type = spec.type
    return result


def _sanitize_probe_result(
    r: ProbeResult,
) -> tuple[ProbeResult, bool]:
    """Clamp score to [0, 1] and replace NaN/Inf metrics with None.

    Returns the (possibly mutated) result and a flag indicating
    whether anything non-finite was observed, which the caller treats
    as an evaluation_valid invalidation.
    """
    invalid = False
    if not math.isfinite(float(r.score)):
        r.failures.append(Failure(
            code=FailureCode.SIMULATOR_DIVERGENCE,
            severity=Severity.CRITICAL,
            message=(f"Probe {r.probe_id!r} returned a non-finite "
                     f"score ({r.score!r})."),
        ))
        r.score = 0.0
        r.passed = False
        invalid = True
    else:
        r.score = max(0.0, min(1.0, float(r.score)))
    # Metric sanitation: keep dict but replace bad floats with None.
    new_metrics: dict[str, float] = {}
    for k, v in r.metrics.items():
        sv = sanitize_metric_value(v)
        if sv is None and v is not None:
            invalid = True  # metric was non-finite
        new_metrics[k] = sv  # type: ignore[assignment]
    r.metrics = new_metrics
    return r, invalid


def _score(
    probe_results: list[ProbeResult],
    specs_by_id: dict[str, ProbeSpec],
    hard_gate_ids: set[str],
) -> tuple[bool, float]:
    hard_gate_passed = True
    has_gate = False
    for r in probe_results:
        spec = specs_by_id.get(r.probe_id)
        is_gate = r.probe_id in hard_gate_ids or (
            spec is not None and spec.hard_gate)
        if is_gate:
            has_gate = True
            if not r.passed:
                hard_gate_passed = False
                break
    if not hard_gate_passed:
        return False, 0.0

    weighted = 0.0
    total_w = 0.0
    for r in probe_results:
        spec = specs_by_id.get(r.probe_id)
        if spec is None:
            continue
        if spec.hard_gate or spec.id in hard_gate_ids:
            continue
        w = float(spec.weight)
        if w <= 0:
            continue
        weighted += w * float(r.score)
        total_w += w
    if total_w > 0:
        return True, weighted / total_w
    # No non-gate weighted probes. If the hard gate exists and it
    # passed, the task is fully satisfied: score = 1.0.
    if has_gate:
        return True, 1.0
    # No gate, no weighted dense probes — undefined; report 0 instead
    # of pretending there is a verifiable signal.
    return True, 0.0


def _tier_summary(
    probe_results: list[ProbeResult],
    plans: list[ProbePlan],
) -> dict[str, dict]:
    by_id = {p.probe_id: p for p in plans}
    tiers: dict[str, dict] = {}
    for r in probe_results:
        plan = by_id.get(r.probe_id)
        tier = _tier_for(plan.capabilities) if plan else "other"
        bucket = tiers.setdefault(tier, {
            "probe_ids": [],
            "passed": True,
            "score_sum": 0.0,
            "score_count": 0,
        })
        bucket["probe_ids"].append(r.probe_id)
        if not r.passed:
            bucket["passed"] = False
        bucket["score_sum"] += float(r.score)
        bucket["score_count"] += 1
    for b in tiers.values():
        n = b.pop("score_count")
        s = b.pop("score_sum")
        b["score"] = (s / n) if n else None
        b["n"] = n
        b["applicable"] = bool(n)
        if n == 0:
            b["passed"] = None
    return tiers


# --------------------------------------------------------------------- #
# Top-level evaluate()                                                  #
# --------------------------------------------------------------------- #


@dataclass
class RunEvidence:
    """Everything needed to package a run after evaluation.

    The CLI's --report-dir flow turns this into the on-disk
    scorecard / metrics / feedback / dashboard / trace bundle.
    """

    report: EvalReport
    task: TaskSpec
    cfg: EvalConfig
    sim_outputs_by_adapter: dict[str, Any] = field(default_factory=dict)


def evaluate(
    task_dir: Path,
    submission_dir: Path,
    *,
    scratch_dir: Path | None = None,
    run_id: str | None = None,
    submission_timeout: float = DEFAULT_SUBMISSION_TIMEOUT,
    mode: str = "",
) -> EvalReport:
    """Backwards-compatible wrapper: returns only the EvalReport."""
    return evaluate_with_evidence(
        task_dir,
        submission_dir,
        scratch_dir=scratch_dir,
        run_id=run_id,
        submission_timeout=submission_timeout,
        mode=mode,
    ).report


def evaluate_with_evidence(
    task_dir: Path,
    submission_dir: Path,
    *,
    scratch_dir: Path | None = None,
    run_id: str | None = None,
    submission_timeout: float = DEFAULT_SUBMISSION_TIMEOUT,
    mode: str = "",
) -> RunEvidence:
    task_dir = Path(task_dir)
    submission_dir = Path(submission_dir)
    scratch_dir = scratch_dir or (submission_dir / "_scratch")
    rid = run_id or f"run_{uuid.uuid4().hex[:12]}"
    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    task, cfg = load_task(task_dir)
    timings["load_task"] = time.perf_counter() - t0

    if mode:
        from mech_bench.modes import apply_mode
        cfg = apply_mode(cfg, mode)

    # Honor explicit fake-oracle enablement before building the plan.
    _maybe_register_fake_oracle(cfg, mode)
    forced_adapter = _resolve_forced_adapter(cfg, mode)

    def _empty_evidence(
        failures: list[Failure],
        tier_results: dict | None = None,
        *,
        valid: bool = False,
    ) -> RunEvidence:
        return RunEvidence(
            report=EvalReport(
                task_id=task.id,
                task_family=task.family,
                difficulty=task.difficulty,
                run_id=rid,
                score=0.0,
                hard_gate_passed=False,
                probe_results=[],
                metrics={},
                feedback=failures,
                tier_results=tier_results or {},
                timings=dict(timings),
                evaluation_valid=valid,
            ),
            task=task,
            cfg=cfg,
            sim_outputs_by_adapter={},
        )

    # Load submission via isolated subprocess.
    t0 = time.perf_counter()
    try:
        ir = load_submission(
            submission_dir,
            Path(scratch_dir),
            timeout=submission_timeout,
            required_port_kinds=_required_port_kinds(cfg),
        )
    except SubmissionError as e:
        timings["load_submission"] = time.perf_counter() - t0
        return _empty_evidence([Failure(
            code=FailureCode.INVALID_ARTIFACT,
            severity=Severity.CRITICAL,
            message=str(e),
            where=str(submission_dir),
        )])
    timings["load_submission"] = time.perf_counter() - t0

    if os.environ.get("MECH_BENCH_AUTO_TRUSTED_ASSETS"):
        from mech_bench.trusted_asset_bridge import (
            augment_with_trusted_assets,
        )

        t0 = time.perf_counter()
        ir = augment_with_trusted_assets(
            ir,
            build_root=Path(scratch_dir).resolve(),
        )
        timings["trusted_asset_bridge"] = time.perf_counter() - t0

    # Validate DesignIR.
    t0 = time.perf_counter()
    validation_failures = validate_design_ir(
        ir, task=task, build_root=Path(scratch_dir).resolve(),
    )
    timings["validate"] = time.perf_counter() - t0
    if has_critical_failures(validation_failures):
        return _empty_evidence(validation_failures, {
            "validation": {
                "probe_ids": [],
                "passed": False,
                "score": 0.0,
            },
        })

    # Build the per-probe execution plan.
    t0 = time.perf_counter()
    plan = build_execution_plan(cfg, forced_adapter=forced_adapter)
    timings["plan"] = time.perf_counter() - t0

    evaluation_valid = True

    # Run each needed adapter once. Each adapter receives the
    # ``[adapters.<name>]`` table from eval_config.toml (or an empty
    # dict). A registered default of ``samples=360`` is provided for
    # backward compatibility with adapters that expect it.
    sim_outputs_by_adapter: dict[str, dict[str, Any]] = {}
    adapter_failures: list[Failure] = []
    unavailable_adapters: set[str] = set()
    for adapter_name in plan.adapters_to_run():
        adapter_cls = next(
            (a for a in all_adapters() if a.type_name == adapter_name),
            None,
        )
        if adapter_cls is None:
            continue
        adapter = adapter_cls()
        adapter_cfg: dict[str, Any] = {"samples": 360}
        adapter_cfg.update(cfg.adapter_configs.get(adapter_name, {}))
        adapter_cfg["_mech_bench"] = _adapter_runtime_context(
            task=task,
            cfg=cfg,
            plan=plan,
            adapter_name=adapter_name,
            build_root=Path(scratch_dir),
        )
        t0 = time.perf_counter()
        try:
            raw = adapter.run(ir, adapter_cfg)
            normalized = normalize_sim_output(raw)
            sim_outputs_by_adapter[adapter_name] = normalized
            if isinstance(normalized, dict):
                if normalized.get("__capability_unavailable__"):
                    md = normalized.get("metadata") or {}
                    reason = ""
                    issues = md.get("preflight_issues") if isinstance(
                        md, dict) else None
                    if isinstance(issues, list) and issues:
                        reason = str(issues[0])
                    adapter_failures.append(Failure(
                        code=FailureCode.CAPABILITY_UNAVAILABLE,
                        severity=Severity.CRITICAL,
                        message=(
                            f"Adapter {adapter_name!r} reported "
                            f"capability_unavailable"
                            + (f": {reason}" if reason else ".")
                        ),
                        where=f"adapter.{adapter_name}",
                        public_hint=(
                            "This task needs a simulator that is not "
                            "currently available."
                        ),
                    ))
                    unavailable_adapters.add(adapter_name)
                    evaluation_valid = False
                elif "__adapter_error__" in normalized:
                    adapter_failures.append(Failure(
                        code=FailureCode.SIMULATOR_DIVERGENCE,
                        severity=Severity.CRITICAL,
                        message=(
                            f"Adapter {adapter_name!r} reported error: "
                            f"{normalized['__adapter_error__']}"),
                        where=f"adapter.{adapter_name}",
                    ))
                    evaluation_valid = False
        except Exception as e:  # noqa: BLE001 — adapter is internal-ish
            _LOG.warning("adapter %s raised: %s", adapter_name, e)
            sim_outputs_by_adapter[adapter_name] = {
                "__adapter_error__": str(e),
            }
            adapter_failures.append(Failure(
                code=FailureCode.SIMULATOR_DIVERGENCE,
                severity=Severity.CRITICAL,
                message=(f"Adapter {adapter_name!r} raised "
                         f"{type(e).__name__}: {e}"),
                where=f"adapter.{adapter_name}",
            ))
            evaluation_valid = False
        timings[f"adapter.{adapter_name}"] = time.perf_counter() - t0

    # Run probes.
    plan_by_id = {p.probe_id: p for p in plan.probes}
    probe_results: list[ProbeResult] = []
    for spec in cfg.probes:
        pplan = plan_by_id[spec.id]
        t0 = time.perf_counter()
        # If the probe depends on an adapter that reported
        # capability_unavailable, short-circuit so we don't surface
        # spurious physical failures like missing_contact.
        if (pplan.adapter_type
                and pplan.adapter_type in unavailable_adapters):
            r = ProbeResult(
                probe_id=spec.id,
                probe_type=spec.type,
                passed=False,
                score=0.0,
                metrics={},
                failures=[Failure(
                    code=FailureCode.CAPABILITY_UNAVAILABLE,
                    severity=Severity.CRITICAL,
                    message=(
                        f"Adapter {pplan.adapter_type!r} is "
                        f"capability-unavailable in this build; "
                        f"probe {spec.id!r} could not run."
                    ),
                    where=spec.id,
                )],
                skipped_reason=(
                    f"adapter.{pplan.adapter_type} unavailable"
                ),
            )
        else:
            r = _run_probe(spec, pplan, ir, sim_outputs_by_adapter)
        r, bad = _sanitize_probe_result(r)
        if bad:
            evaluation_valid = False
        timings[f"probe.{spec.id}"] = time.perf_counter() - t0
        probe_results.append(r)
        # If any probe surfaced CAPABILITY_UNAVAILABLE, the entire
        # evaluation is structurally invalid: an agent should not be
        # able to earn reward on a partial verifier.
        if any(f.code == FailureCode.CAPABILITY_UNAVAILABLE
               for f in r.failures):
            evaluation_valid = False

    specs_by_id = {s.id: s for s in cfg.probes}
    hard_gate_passed, dense = _score(
        probe_results, specs_by_id, set(cfg.hard_gate_probes)
    )

    if not evaluation_valid:
        # An invalid evaluation can never earn reward and must surface
        # as a CLI failure.
        hard_gate_passed = False
        dense = 0.0

    # Aggregate metrics / feedback.
    agg_metrics: dict[str, float] = {}
    feedback: list[Failure] = list(validation_failures) + list(
        adapter_failures)
    for r in probe_results:
        for k, v in r.metrics.items():
            sv = sanitize_metric_value(v)
            agg_metrics[f"{r.probe_id}.{k}"] = sv  # may be None
        for f in r.failures:
            f.where = f.where or r.probe_id
            feedback.append(f)

    if not math.isfinite(float(dense)):
        feedback.append(Failure(
            code=FailureCode.SIMULATOR_DIVERGENCE,
            severity=Severity.CRITICAL,
            message=f"Aggregate score is non-finite: {dense!r}",
        ))
        dense = 0.0
        evaluation_valid = False
        hard_gate_passed = False

    tier_results = _tier_summary(probe_results, plan.probes)
    if validation_failures:
        tier_results.setdefault("validation", {
            "probe_ids": [],
            "passed": False,
            "score": 0.0,
        })

    # Tier / class / general metric aggregation.
    caps_by_id = {p.probe_id: list(p.capabilities) for p in plan.probes}
    tier_channels = compute_tier_metrics(
        probe_results, specs_by_id, caps_by_id)
    class_channels = compute_class_metrics(probe_results, specs_by_id)
    tier_channels, class_channels = fill_defaults_for_dashboard(
        tier_channels, class_channels)
    # Merge channel scores into tier_results so the dashboard sees both.
    for ch, vals in tier_channels.items():
        merged = tier_results.setdefault(ch, {
            "probe_ids": [],
            "passed": vals.get("passed"),
            "score": vals.get("score"),
        })
        merged.setdefault("probe_ids", merged.get("probe_ids", []))
        merged["score"] = vals.get("score")
        merged["passed"] = vals.get("passed")
        merged["n"] = vals.get("n", 0)
        merged["n_passed"] = vals.get("n_passed", 0)
        if "applicable" in vals:
            merged["applicable"] = vals["applicable"]

    total_runtime = float(sum(timings.values()))
    oracle_is_synthetic = any(
        isinstance(out, dict)
        and bool(out.get("metadata", {}).get("oracle_is_synthetic", False))
        for out in sim_outputs_by_adapter.values()
    )
    general = compute_general_metrics(
        probe_results,
        hard_gate_passed=hard_gate_passed,
        score=float(dense),
        runtime_s=total_runtime,
        oracle_passed=hard_gate_passed if oracle_is_synthetic else None,
    )

    report = EvalReport(
        task_id=task.id,
        task_family=task.family,
        difficulty=task.difficulty,
        run_id=rid,
        score=float(dense),
        hard_gate_passed=hard_gate_passed,
        probe_results=probe_results,
        metrics=agg_metrics,
        feedback=feedback,
        tier_results=tier_results,
        class_metrics=class_channels,
        general_metrics=general,
        timings=timings,
        evaluation_valid=evaluation_valid,
        oracle_is_synthetic=oracle_is_synthetic,
        mode=mode,
    )
    return RunEvidence(
        report=report,
        task=task,
        cfg=cfg,
        sim_outputs_by_adapter=sim_outputs_by_adapter,
    )


# --------------------------------------------------------------------- #
# Report bundle                                                         #
# --------------------------------------------------------------------- #


def _flatten_numeric(d: dict, prefix: str = "") -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, bool):
            out[key] = 1.0 if v else 0.0
        elif v is None:
            out[key] = None
        elif isinstance(v, (int, float)):
            f = float(v)
            out[key] = f if math.isfinite(f) else None
        elif isinstance(v, dict):
            out.update(_flatten_numeric(v, prefix=key + "."))
    return out


def write_report_bundle(
    report: EvalReport,
    out_dir: Path,
    *,
    visibility: FeedbackVisibility,
) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    full = report.to_dict(public=False, visibility=visibility)
    public = report.to_dict(public=True, visibility=visibility)

    paths: dict[str, Path] = {}

    (out_dir / "scorecard.json").write_text(_strict_json_dumps(full))
    paths["scorecard"] = out_dir / "scorecard.json"

    (out_dir / "scorecard.public.json").write_text(_strict_json_dumps(public))
    paths["scorecard_public"] = out_dir / "scorecard.public.json"

    metrics_blob = {
        "score": report.score,
        "hard_gate_passed": report.hard_gate_passed,
        "evaluation_valid": report.evaluation_valid,
        **_flatten_numeric({"metrics": report.metrics}),
        **_flatten_numeric({"timings": report.timings}),
    }
    (out_dir / "metrics.json").write_text(_strict_json_dumps(metrics_blob))
    paths["metrics"] = out_dir / "metrics.json"

    public_failures = [
        (f.public() if hasattr(f, "public") else dict(f))
        for f in report.feedback
    ]
    (out_dir / "feedback.public.json").write_text(
        _strict_json_dumps(public_failures))
    paths["feedback_public"] = out_dir / "feedback.public.json"

    return paths


def write_run_bundle(
    evidence: "RunEvidence",
    out_dir: Path,
    *,
    render_media: bool = False,
    write_dashboard_html: bool = True,
) -> dict[str, Path]:
    """Write the full evidence bundle for one run.

    Always writes: scorecard.json, scorecard.public.json,
    metrics.json, feedback.public.json, dashboard_payload.json,
    media_manifest.json. Optionally writes traces.h5 (when h5py is
    installed) and dashboard.html (when plotly is installed and
    ``write_dashboard_html`` is True). Frames / thumbnail / MP4 are
    only generated when ``render_media=True`` — the CLI flips that
    flag through ``--render-media``.
    Returns a dict of the artifact paths actually written.
    """
    from mech_bench.dashboard_payload import (
        build_dashboard_payload,
        write_dashboard_payload,
    )
    from mech_bench.media import write_media_manifest
    from mech_bench.traces import (
        HAS_H5PY,
        TraceData,
        write_capability_unavailable,
        write_trace_hdf5,
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = write_report_bundle(
        evidence.report, out_dir, visibility=evidence.cfg.visibility)

    # Build per-adapter traces (so multi-adapter tasks preserve both
    # streams) plus a "primary" trace used for the dashboard payload.
    adapter_traces: dict[str, TraceData] = {}
    primary_adapter = ""
    primary_sim: dict | None = None
    for name, sim in evidence.sim_outputs_by_adapter.items():
        if not isinstance(sim, dict) or "__adapter_error__" in sim:
            continue
        if "__capability_unavailable__" in sim:
            # Capability-unavailable adapters contribute no traces.
            continue
        td = TraceData.from_sim_output(
            sim,
            run_id=evidence.report.run_id,
            task_id=evidence.report.task_id,
            adapter=name,
        )
        if td.is_empty():
            continue
        adapter_traces[name] = td
        if primary_sim is None:
            primary_sim = sim
            primary_adapter = name

    trace = (
        adapter_traces.get(primary_adapter)
        or TraceData.from_sim_output(
            primary_sim or {},
            run_id=evidence.report.run_id,
            task_id=evidence.report.task_id,
            adapter=primary_adapter,
        )
    )

    trace_path: Path | None = None
    if adapter_traces:
        if HAS_H5PY:
            trace_path = write_trace_hdf5_multi(
                out_dir / "traces.h5", adapter_traces, trace)
            paths["trace"] = trace_path
        else:
            stub = write_capability_unavailable(
                out_dir / "traces.unavailable.json",
                reason=("h5py is not installed; HDF5 trace not "
                        "written. Install mech-bench[traces] to enable."),
            )
            paths["trace_stub"] = stub

    payload = build_dashboard_payload(
        evidence.report, trace, task=evidence.task)
    if adapter_traces:
        payload["adapter_traces"] = sorted(adapter_traces.keys())
    payload_path = write_dashboard_payload(
        out_dir / "dashboard_payload.json", payload)
    paths["dashboard_payload"] = payload_path

    # Optional planar media rendering — opt-in via render_media.
    thumbnail_path: Path | None = None
    preview_mp4: Path | None = None
    frames_dir: Path | None = None
    if render_media:
        try:
            from mech_bench.rendering.planar_renderer import (
                HAS_MATPLOTLIB,
                PlanarRenderer,
            )
            if HAS_MATPLOTLIB:
                renderer = PlanarRenderer()
                res = renderer.render(payload, out_dir, fps=30,
                                      produce_mp4=True)
                if res.ok:
                    thumbnail_path = res.thumbnail_png
                    preview_mp4 = res.preview_mp4
                    frames_dir = res.frames_dir
        except Exception:  # noqa: BLE001 — media is non-critical
            pass

    dashboard_path: Path | None = None
    if write_dashboard_html:
        try:
            from mech_bench.dashboard import (
                HAS_PLOTLY,
                write_static_dashboard,
            )
            # Refresh payload's media block with whatever was rendered.
            if thumbnail_path or preview_mp4 or frames_dir:
                payload.setdefault("media", {})
                if thumbnail_path:
                    payload["media"]["thumbnail_png"] = str(
                        thumbnail_path.relative_to(out_dir))
                if preview_mp4:
                    payload["media"]["preview_mp4"] = str(
                        preview_mp4.relative_to(out_dir))
                elif frames_dir:
                    payload["media"]["frames_dir"] = str(
                        frames_dir.relative_to(out_dir))
                # Rewrite payload JSON so dashboards see media refs.
                payload_path = write_dashboard_payload(
                    out_dir / "dashboard_payload.json", payload)
            if HAS_PLOTLY:
                dashboard_path = write_static_dashboard(
                    payload, out_dir / "dashboard.html")
                paths["dashboard"] = dashboard_path
        except ImportError:  # pragma: no cover - defensive
            pass

    if thumbnail_path:
        paths["thumbnail_png"] = thumbnail_path
    if preview_mp4:
        paths["preview_mp4"] = preview_mp4
    if frames_dir and not preview_mp4:
        paths["frames_dir"] = frames_dir

    manifest_path = write_media_manifest(
        out_dir,
        evidence.report,
        trace_path=trace_path,
        dashboard_payload_path=payload_path,
        dashboard_html_path=dashboard_path,
        thumbnail_png_path=thumbnail_path,
        preview_mp4_path=preview_mp4,
        frames_dir_path=frames_dir if (preview_mp4 is None
                                        and frames_dir is not None)
                          else None,
    )
    paths["media_manifest"] = manifest_path
    return paths


def write_trace_hdf5_multi(
    path: Path,
    adapter_traces: dict[str, "TraceData"],
    primary: "TraceData",
) -> Path:
    """Write multiple adapter traces under per-adapter groups.

    If there is only one adapter we still nest it under
    ``/adapters/<name>/`` so the file format is uniform, but consumers
    that only know the legacy single-trace shape can still find the
    primary trace at the root attrs.
    """
    from mech_bench.traces import HAS_H5PY  # local to avoid hard dep
    if not HAS_H5PY:
        raise RuntimeError("h5py not installed; cannot write traces")
    import h5py  # type: ignore[import-not-found]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["version"] = "mech_bench.trace.v2"
        f.attrs["run_id"] = primary.run_id
        f.attrs["task_id"] = primary.task_id
        # Keep the legacy root attr for readers that predate the
        # multi-adapter trace layout.
        f.attrs["adapter"] = primary.adapter
        f.attrs["primary_adapter"] = primary.adapter
        adapters_grp = f.create_group("adapters")
        for name, td in adapter_traces.items():
            grp = adapters_grp.create_group(name)
            _h5_write_trace_into_group(grp, td)
        # Also mirror primary at top level for legacy readers.
        _h5_write_trace_into_group(f, primary, include_attrs=False)
    return path


def _h5_write_trace_into_group(
    grp: Any,
    td: "TraceData",
    *,
    include_attrs: bool = True,
) -> None:
    if include_attrs:
        grp.attrs["adapter"] = td.adapter
        for k, v in td.metadata.items():
            grp.attrs[f"meta.{k}"] = v
    if td.time_s.size and "time_s" not in grp:
        grp.create_dataset("time_s", data=td.time_s)
    _h5_named(grp, "ports", td.port_traces, "trace")
    _h5_named(grp, "ports", td.port_velocities, "velocity")
    _h5_named(grp, "joints", td.joint_positions, "position")
    _h5_named(grp, "joints", td.joint_velocities, "velocity")
    _h5_named(grp, "bodies", td.body_poses, "pose")
    _h5_named(grp, "bodies", td.body_twists, "twist")
    _h5_named(grp, "contacts", td.contact_forces, "normal_force")
    _h5_named(grp, "contacts", td.penetration, "penetration")
    if td.scalar_metrics and "metrics" not in grp:
        metrics_grp = grp.create_group("metrics")
        for k, v in td.scalar_metrics.items():
            metrics_grp.create_dataset(
                str(k).replace("/", "__"), data=float(v))


def _h5_named(
    root: Any, top: str, d: dict, leaf: str,
) -> None:
    if not d:
        return
    grp = root.require_group(top)
    for k, arr in d.items():
        safe = str(k).replace("/", "__")
        sub = grp.require_group(safe)
        if leaf in sub:
            continue
        import numpy as np
        sub.create_dataset(leaf, data=np.asarray(arr))
