"""Core schemas.

All five abstractions live here as dataclasses to keep the dependency
surface to numpy + stdlib. JSON / TOML loading is in evaluator.py.

DesignIR shape is ported from phys-sim's design_ir.v1 with two
simplifications:
  * Geometry refs are optional (analytic probes can run on parametric
    descriptions alone).
  * The `params` bag is task-scoped (the task author decides what
    keys mean) rather than schema-enforced.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


# --------------------------------------------------------------------- #
# DesignIR — what the agent submits via build_design(out_dir) -> dict   #
# --------------------------------------------------------------------- #

JointType = Literal["revolute", "prismatic", "fixed", "contact_pair", "spherical"]


@dataclass
class Part:
    id: str
    role: str = ""
    mass_kg: float = 0.0
    com_local_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    inertia_kg_m2: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ] = ((1e-6, 0, 0), (0, 1e-6, 0), (0, 0, 1e-6))
    fixed: bool = False
    geometry: dict[str, str] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class Joint:
    id: str
    type: JointType
    parent: str
    child: str
    axis_world: tuple[float, float, float] | None = None
    anchor_world_mm: tuple[float, float, float] | None = None
    limits_rad: tuple[float, float] | None = None
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class Port:
    id: str
    part: str
    kind: Literal["frame", "revolute_joint", "prismatic_joint"]
    pose_local_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class DesignIR:
    schema_version: str
    parts: list[Part]
    joints: list[Joint]
    ports: dict[str, Port]
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "DesignIR":
        return cls(
            schema_version=d["schema_version"],
            parts=[Part(**p) if not isinstance(p, Part) else p
                   for p in d["parts"]],
            joints=[Joint(**j) if not isinstance(j, Joint) else j
                    for j in d["joints"]],
            ports={k: (Port(**v) if not isinstance(v, Port) else v)
                   for k, v in d["ports"].items()},
            params=d.get("params", {}),
        )

    def part_ids(self) -> set[str]:
        return {p.id for p in self.parts}


# --------------------------------------------------------------------- #
# ProbeSpec — declarative probe config in eval_config.toml              #
# --------------------------------------------------------------------- #


@dataclass
class ProbeSpec:
    """One entry under `[[probes]]` in eval_config.toml.

    `type` is a key in the ProbeRegistry. `config` is the
    probe-type-specific payload. `weight` and `severity` are how
    this probe contributes to the score.
    """

    id: str
    type: str
    config: dict[str, Any] = field(default_factory=dict)
    weight: float = 0.0
    severity: str = "major"
    hard_gate: bool = False


@dataclass
class ProbeResult:
    probe_id: str
    probe_type: str
    passed: bool
    score: float = 0.0
    metrics: dict[str, float] = field(default_factory=dict)
    failures: list = field(default_factory=list)
    artifacts: dict[str, Path] = field(default_factory=dict)
    skipped_reason: str | None = None

    def public_metrics(self, allow: list[str] | None) -> dict[str, float]:
        if allow is None:
            return dict(self.metrics)
        prefix = f"{self.probe_id}."
        out = {}
        for k, v in self.metrics.items():
            key = f"{prefix}{k}"
            if key in allow or k in allow:
                out[k] = v
        return out


# --------------------------------------------------------------------- #
# TaskSpec — stable task definition in task.toml                        #
# --------------------------------------------------------------------- #


@dataclass
class TaskSpec:
    id: str
    family: str
    difficulty: int
    units: str
    prompt: str
    required_ports: list[str] = field(default_factory=list)
    expected_mobility: int | None = None
    envelope_mm: tuple[float, float, float] | None = None
    objective: dict[str, Any] = field(default_factory=dict)
    fixtures_dir: Path | None = None

    @classmethod
    def from_dict(cls, d: dict, *, fixtures_dir: Path | None = None,
                  prompt: str = "") -> "TaskSpec":
        task = d.get("task", d)
        req = d.get("requirements", {})
        obj = d.get("objective", {})
        env = req.get("max_envelope_mm")
        return cls(
            id=task["id"],
            family=task["family"],
            difficulty=task.get("difficulty", 1),
            units=task.get("units", "mm"),
            prompt=prompt,
            required_ports=list(req.get("required_ports", [])),
            expected_mobility=req.get("expected_mobility"),
            envelope_mm=tuple(env) if env else None,
            objective=obj,
            fixtures_dir=fixtures_dir,
        )


# --------------------------------------------------------------------- #
# EvalConfig — probe pipeline, in eval_config.toml                      #
# --------------------------------------------------------------------- #


@dataclass
class FeedbackVisibility:
    public_metrics: list[str] = field(default_factory=list)
    hidden_metrics: list[str] = field(default_factory=list)


@dataclass
class EvalConfig:
    probes: list[ProbeSpec]
    hard_gate_probes: list[str] = field(default_factory=list)
    visibility: FeedbackVisibility = field(default_factory=FeedbackVisibility)

    @classmethod
    def from_dict(cls, d: dict) -> "EvalConfig":
        probes = []
        for entry in d.get("probes", []):
            probes.append(ProbeSpec(
                id=entry["id"],
                type=entry["type"],
                config={k: v for k, v in entry.items()
                        if k not in {"id", "type", "weight",
                                     "severity", "hard_gate"}},
                weight=float(entry.get("weight", 0.0)),
                severity=str(entry.get("severity", "major")),
                hard_gate=bool(entry.get("hard_gate", False)),
            ))
        hard_gate = d.get("hard_gate", {}).get("require", [])
        fb = d.get("feedback", {})
        return cls(
            probes=probes,
            hard_gate_probes=list(hard_gate),
            visibility=FeedbackVisibility(
                public_metrics=list(fb.get("public_metrics", [])),
                hidden_metrics=list(fb.get("hidden_metrics", [])),
            ),
        )


# --------------------------------------------------------------------- #
# EvalReport — the runtime's final output                               #
# --------------------------------------------------------------------- #


@dataclass
class EvalReport:
    task_id: str
    score: float
    hard_gate_passed: bool
    probe_results: list[ProbeResult]
    metrics: dict[str, float] = field(default_factory=dict)
    feedback: list = field(default_factory=list)
    artifacts: dict[str, str] = field(default_factory=dict)
    run_id: str = ""
    task_family: str = ""
    difficulty: int = 0
    tier_results: dict[str, dict] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)
    version: str = "eval_report.v1"

    def to_dict(
        self,
        public: bool = False,
        visibility: "FeedbackVisibility | None" = None,
    ) -> dict:
        """Serialize the report.

        public=False emits everything (private trace pointers and all
        metrics). public=True applies the configured FeedbackVisibility:
        feedback items drop private_trace, and metrics outside the
        allowlist are stripped.
        """
        vis = visibility or FeedbackVisibility()

        feedback_items: list[dict] = []
        for f in self.feedback:
            if public:
                item = f.public() if hasattr(f, "public") else dict(f)
            else:
                item = f.to_dict() if hasattr(f, "to_dict") else dict(f)
            feedback_items.append(item)

        if public:
            metrics_view = _filter_public_metrics(
                self.metrics, vis.public_metrics, vis.hidden_metrics)
        else:
            metrics_view = dict(self.metrics)

        probe_view: list[dict] = []
        for r in self.probe_results:
            entry: dict = {
                "probe_id": r.probe_id,
                "probe_type": r.probe_type,
                "passed": r.passed,
                "score": r.score,
            }
            if public:
                entry["metrics"] = r.public_metrics(
                    vis.public_metrics or None)
            else:
                entry["metrics"] = dict(r.metrics)
                entry["skipped_reason"] = r.skipped_reason
                entry["artifacts"] = {
                    k: str(v) for k, v in r.artifacts.items()
                }
            entry["failures"] = [
                (f.public() if public and hasattr(f, "public")
                 else (f.to_dict() if hasattr(f, "to_dict") else dict(f)))
                for f in r.failures
            ]
            probe_view.append(entry)

        out: dict = {
            "version": self.version,
            "task_id": self.task_id,
            "task_family": self.task_family,
            "difficulty": self.difficulty,
            "run_id": self.run_id,
            "score": self.score,
            "hard_gate_passed": self.hard_gate_passed,
            "metrics": metrics_view,
            "feedback": feedback_items,
            "probe_results": probe_view,
            "tier_results": dict(self.tier_results),
            "timings": dict(self.timings),
        }
        if not public:
            out["artifacts"] = {
                k: str(v) for k, v in self.artifacts.items()
            }
        return out

    def public_dict(
        self,
        vis: "FeedbackVisibility | None" = None,
    ) -> dict:
        return self.to_dict(public=True, visibility=vis)


def _filter_public_metrics(
    metrics: dict[str, float],
    public_allow: list[str],
    hidden: list[str],
) -> dict[str, float]:
    hidden_set = set(hidden or [])
    if not public_allow:
        return {k: v for k, v in metrics.items() if k not in hidden_set}
    out: dict[str, float] = {}
    for k, v in metrics.items():
        if k in hidden_set:
            continue
        if k in public_allow:
            out[k] = v
            continue
        if any(k.startswith(p.rstrip(".") + ".") for p in public_allow):
            out[k] = v
    return out
