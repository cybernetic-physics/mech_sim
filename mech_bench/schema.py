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
    artifacts: dict[str, Path] = field(default_factory=dict)

    def public_dict(self, vis: FeedbackVisibility) -> dict:
        public = []
        for f in self.feedback:
            pf = f.public() if hasattr(f, "public") else dict(f)
            public.append(pf)
        return {
            "task_id": self.task_id,
            "score": self.score,
            "hard_gate_passed": self.hard_gate_passed,
            "metrics": {
                k: v for k, v in self.metrics.items()
                if not vis.public_metrics
                or k in vis.public_metrics
                or any(k.startswith(p.rstrip(".")) for p in vis.public_metrics)
            },
            "feedback": public,
        }
