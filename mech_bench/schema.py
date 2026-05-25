"""Core schemas.

All five abstractions live here as dataclasses to keep the dependency
surface to numpy + stdlib. JSON / TOML loading is in evaluator.py.

DesignIR shape is ported from phys-sim's design_ir.v1 with two
simplifications:
  * Geometry refs are optional (analytic probes can run on parametric
    descriptions alone).
  * The `params` bag is task-scoped (the task author decides what
    keys mean) rather than schema-enforced.

The v2 physical metadata fields are intentionally explicit even when
the current runtime cannot consume all of them yet. They keep
high-fidelity tasks from smuggling units, materials, load cases, and
contact assumptions through opaque params bags.
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
class MaterialSpec:
    id: str
    name: str = ""
    density_kg_m3: float | None = None
    elastic_modulus_pa: float | None = None
    poisson_ratio: float | None = None
    yield_strength_pa: float | None = None
    process: str = ""
    provenance: str = ""
    uncertainty: dict[str, Any] = field(default_factory=dict)
    properties: dict[str, Any] = field(default_factory=dict)


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
    material: str = ""
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
    units: str = "mm"
    frames: dict[str, Any] = field(default_factory=dict)
    materials: dict[str, MaterialSpec] = field(default_factory=dict)
    load_cases: dict[str, Any] = field(default_factory=dict)
    actuators: dict[str, Any] = field(default_factory=dict)
    contacts: dict[str, Any] = field(default_factory=dict)
    tolerances: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "DesignIR":
        materials = {
            str(k): (
                v if isinstance(v, MaterialSpec)
                else MaterialSpec(**({"id": str(k)} | dict(v)))
            )
            for k, v in d.get("materials", {}).items()
        }
        return cls(
            schema_version=d["schema_version"],
            parts=[Part(**p) if not isinstance(p, Part) else p
                   for p in d["parts"]],
            joints=[Joint(**j) if not isinstance(j, Joint) else j
                    for j in d["joints"]],
            ports={k: (Port(**v) if not isinstance(v, Port) else v)
                   for k, v in d["ports"].items()},
            units=d.get("units", "mm"),
            frames=d.get("frames", {}),
            materials=materials,
            load_cases=d.get("load_cases", {}),
            actuators=d.get("actuators", {}),
            contacts=d.get("contacts", {}),
            tolerances=d.get("tolerances", {}),
            provenance=d.get("provenance", {}),
            params=d.get("params", {}),
        )

    @classmethod
    def try_from_dict(
        cls, raw: Any,
    ) -> "tuple[DesignIR | None, list[str]]":
        """Parse *raw* into a DesignIR without ever raising.

        Returns ``(ir, errors)``. When ``errors`` is non-empty, the
        evaluator should surface them as structured failures (typically
        INVALID_ARTIFACT or SCHEMA_ERROR) rather than crashing.
        """
        errors: list[str] = []
        if not isinstance(raw, dict):
            return None, [
                f"DesignIR root must be a dict, got "
                f"{type(raw).__name__}.",
            ]
        sv = raw.get("schema_version")
        if not isinstance(sv, str) or not sv:
            errors.append(
                "Missing or non-string 'schema_version' key.")
        raw_parts = raw.get("parts")
        if not isinstance(raw_parts, list):
            errors.append(
                f"'parts' must be a list, got "
                f"{type(raw_parts).__name__}."
            )
            raw_parts = []
        raw_joints = raw.get("joints")
        if not isinstance(raw_joints, list):
            errors.append(
                f"'joints' must be a list, got "
                f"{type(raw_joints).__name__}."
            )
            raw_joints = []
        raw_ports = raw.get("ports")
        if not isinstance(raw_ports, dict):
            errors.append(
                f"'ports' must be a dict[str, dict], got "
                f"{type(raw_ports).__name__}."
            )
            raw_ports = {}
        raw_params = raw.get("params", {})
        if raw_params is None:
            raw_params = {}
        if not isinstance(raw_params, dict):
            errors.append(
                f"'params' must be a dict if present, got "
                f"{type(raw_params).__name__}."
            )
            raw_params = {}
        raw_materials = raw.get("materials", {})
        if raw_materials is None:
            raw_materials = {}
        if not isinstance(raw_materials, dict):
            errors.append(
                f"'materials' must be a dict if present, got "
                f"{type(raw_materials).__name__}."
            )
            raw_materials = {}
        top_dict_fields: dict[str, dict] = {}
        for field_name in (
            "frames", "load_cases", "actuators", "contacts",
            "tolerances", "provenance",
        ):
            raw_field = raw.get(field_name, {})
            if raw_field is None:
                raw_field = {}
            if not isinstance(raw_field, dict):
                errors.append(
                    f"'{field_name}' must be a dict if present, got "
                    f"{type(raw_field).__name__}."
                )
                raw_field = {}
            top_dict_fields[field_name] = raw_field

        parts: list[Part] = []
        for i, p in enumerate(raw_parts):
            if isinstance(p, Part):
                parts.append(p)
                continue
            if not isinstance(p, dict):
                errors.append(
                    f"parts[{i}] must be a dict, got "
                    f"{type(p).__name__}."
                )
                continue
            try:
                parts.append(Part(**p))
            except (TypeError, ValueError, AttributeError, KeyError) as e:
                errors.append(
                    f"parts[{i}] malformed: "
                    f"{type(e).__name__}: {e}"
                )

        joints: list[Joint] = []
        for i, j in enumerate(raw_joints):
            if isinstance(j, Joint):
                joints.append(j)
                continue
            if not isinstance(j, dict):
                errors.append(
                    f"joints[{i}] must be a dict, got "
                    f"{type(j).__name__}."
                )
                continue
            try:
                joints.append(Joint(**j))
            except (TypeError, ValueError, AttributeError, KeyError) as e:
                errors.append(
                    f"joints[{i}] malformed: "
                    f"{type(e).__name__}: {e}"
                )

        ports: dict[str, Port] = {}
        for k, v in raw_ports.items():
            if isinstance(v, Port):
                ports[str(k)] = v
                continue
            if not isinstance(v, dict):
                errors.append(
                    f"ports[{k!r}] must be a dict, got "
                    f"{type(v).__name__}."
                )
                continue
            try:
                ports[str(k)] = Port(**v)
            except (TypeError, ValueError, AttributeError, KeyError) as e:
                errors.append(
                    f"ports[{k!r}] malformed: "
                    f"{type(e).__name__}: {e}"
                )

        materials: dict[str, MaterialSpec] = {}
        for k, v in raw_materials.items():
            if isinstance(v, MaterialSpec):
                materials[str(k)] = v
                continue
            if not isinstance(v, dict):
                errors.append(
                    f"materials[{k!r}] must be a dict, got "
                    f"{type(v).__name__}."
                )
                continue
            try:
                payload = {"id": str(k)}
                payload.update(v)
                materials[str(k)] = MaterialSpec(**payload)
            except (TypeError, ValueError, AttributeError, KeyError) as e:
                errors.append(
                    f"materials[{k!r}] malformed: "
                    f"{type(e).__name__}: {e}"
                )

        if errors:
            return None, errors
        return cls(
            schema_version=str(sv),
            parts=parts,
            joints=joints,
            ports=ports,
            units=raw.get("units", "mm"),
            materials=materials,
            **top_dict_fields,
            params=raw_params,
        ), []

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

    ``tier`` groups probes into the channel hierarchy
    (artifact, geometry, kinematics, contact, dynamics, structural,
    manufacturability, robustness). ``class_metric`` further routes a
    probe into a task-class channel (``linkage_path_score``,
    ``gearbox_ratio_score``, etc.). Both are optional; when absent,
    the runtime derives the tier from probe capabilities.
    """

    id: str
    type: str
    config: dict[str, Any] = field(default_factory=dict)
    weight: float = 0.0
    severity: str = "major"
    hard_gate: bool = False
    tier: str | None = None
    class_metric: str | None = None


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

    def public_metrics(
        self,
        allow: list[str] | None,
        hidden: list[str] | None = None,
    ) -> dict[str, float]:
        """Filter ``self.metrics`` for public emission.

        ``hidden`` always wins over ``allow``. Both lists may contain
        either the local metric name (``"chamfer"``) or the prefixed
        form (``"coupler_path.chamfer"``); both forms match. Prefix
        keys ending in ``.`` denote "anything under this probe."
        """
        hidden_set = set(hidden or [])
        prefix = f"{self.probe_id}."
        out: dict[str, float] = {}
        for k, v in self.metrics.items():
            full = f"{prefix}{k}"
            if k in hidden_set or full in hidden_set:
                continue
            if any(self._prefix_match(name, full, k)
                   for name in hidden_set):
                continue
            if allow is None:
                out[k] = v
                continue
            if k in allow or full in allow:
                out[k] = v
                continue
            if any(self._prefix_match(name, full, k) for name in allow):
                out[k] = v
        return out

    @staticmethod
    def _prefix_match(pattern: str, full_key: str, local_key: str) -> bool:
        # "probe." or "probe.foo." form: prefix match against full_key.
        if pattern.endswith("."):
            return (full_key.startswith(pattern)
                    or local_key.startswith(pattern))
        return False


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
class ModeConfig:
    """One eval mode (e.g. ``fast``, ``oracle``).

    ``enabled_probe_ids`` is the subset of probes that run in this mode.
    An empty list means "all probes." ``adapter_overrides`` lets a mode
    swap the contact-dynamics adapter (e.g. ``fake_contact_oracle`` in
    ``oracle`` mode for tests).
    """

    enabled_probe_ids: list[str] = field(default_factory=list)
    adapter_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    forced_adapter: str | None = None


@dataclass
class FinalModeConfig:
    """``final`` mode: combines fast + oracle into a single report."""

    require_modes: list[str] = field(default_factory=lambda: ["fast", "oracle"])
    agreement_probes: list[str] = field(default_factory=list)
    ratio_delta_pct_max: float = 5.0
    penetration_delta_mm_max: float = 0.1


@dataclass
class EvalConfig:
    probes: list[ProbeSpec]
    hard_gate_probes: list[str] = field(default_factory=list)
    visibility: FeedbackVisibility = field(default_factory=FeedbackVisibility)
    adapter_configs: dict[str, dict[str, Any]] = field(default_factory=dict)
    modes: dict[str, ModeConfig] = field(default_factory=dict)
    final_mode: FinalModeConfig = field(default_factory=FinalModeConfig)

    @classmethod
    def from_dict(cls, d: dict) -> "EvalConfig":
        probes = []
        for entry in d.get("probes", []):
            probes.append(ProbeSpec(
                id=entry["id"],
                type=entry["type"],
                config={k: v for k, v in entry.items()
                        if k not in {"id", "type", "weight",
                                     "severity", "hard_gate",
                                     "tier", "class_metric"}},
                weight=float(entry.get("weight", 0.0)),
                severity=str(entry.get("severity", "major")),
                hard_gate=bool(entry.get("hard_gate", False)),
                tier=entry.get("tier"),
                class_metric=entry.get("class_metric"),
            ))
        hard_gate = d.get("hard_gate", {}).get("require", [])
        fb = d.get("feedback", {})
        raw_adapters = d.get("adapters", {}) or {}
        adapter_configs: dict[str, dict[str, Any]] = {}
        if isinstance(raw_adapters, dict):
            for k, v in raw_adapters.items():
                if isinstance(v, dict):
                    adapter_configs[str(k)] = dict(v)

        raw_modes = d.get("modes", {}) or {}
        modes: dict[str, ModeConfig] = {}
        final_cfg = FinalModeConfig()
        if isinstance(raw_modes, dict):
            for mname, mraw in raw_modes.items():
                if not isinstance(mraw, dict):
                    continue
                if mname == "final":
                    final_cfg = FinalModeConfig(
                        require_modes=list(
                            mraw.get("require_modes", ["fast", "oracle"])
                        ),
                        agreement_probes=list(
                            mraw.get("agreement_probes", [])
                        ),
                        ratio_delta_pct_max=float(
                            mraw.get("ratio_delta_pct_max", 5.0)
                        ),
                        penetration_delta_mm_max=float(
                            mraw.get("penetration_delta_mm_max", 0.1)
                        ),
                    )
                    continue
                modes[str(mname)] = ModeConfig(
                    enabled_probe_ids=list(
                        mraw.get("enabled_probe_ids", [])
                    ),
                    adapter_overrides={
                        str(k): dict(v) if isinstance(v, dict) else {}
                        for k, v in (mraw.get("adapter_overrides")
                                     or {}).items()
                    },
                    forced_adapter=mraw.get("forced_adapter"),
                )

        return cls(
            probes=probes,
            hard_gate_probes=list(hard_gate),
            visibility=FeedbackVisibility(
                public_metrics=list(fb.get("public_metrics", [])),
                hidden_metrics=list(fb.get("hidden_metrics", [])),
            ),
            adapter_configs=adapter_configs,
            modes=modes,
            final_mode=final_cfg,
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
    class_metrics: dict[str, float] = field(default_factory=dict)
    general_metrics: dict[str, float] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)
    evaluation_valid: bool = True
    oracle_is_synthetic: bool = False
    mode: str = ""
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
                    vis.public_metrics or None,
                    hidden=vis.hidden_metrics,
                )
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
            "evaluation_valid": self.evaluation_valid,
            "oracle_is_synthetic": self.oracle_is_synthetic,
            "mode": self.mode,
            "metrics": metrics_view,
            "feedback": feedback_items,
            "probe_results": probe_view,
            "tier_results": dict(self.tier_results),
            "class_metrics": dict(self.class_metrics),
            "general_metrics": dict(self.general_metrics),
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
    """Apply visibility rules to a flat ``probe.metric`` dict.

    Order of precedence:
      1. If a key (or one of its prefixes) is hidden, drop it.
      2. If a public allowlist is supplied, only emit keys that match
         it (exact or ``"prefix."``).
      3. With no allowlist, emit everything that isn't hidden.
    """
    hidden_set = set(hidden or [])

    def _hidden(key: str) -> bool:
        if key in hidden_set:
            return True
        for h in hidden_set:
            if h.endswith("."):
                if key.startswith(h):
                    return True
            elif "." not in h:
                # Local-name hidden entry like "chamfer" must match
                # any "<probe>.chamfer".
                if key.endswith("." + h):
                    return True
        return False

    def _allowed(key: str) -> bool:
        if not public_allow:
            return True
        if key in public_allow:
            return True
        for p in public_allow:
            if p.endswith("."):
                if key.startswith(p):
                    return True
            elif "." not in p:
                if key.endswith("." + p):
                    return True
        return False

    out: dict[str, float] = {}
    for k, v in metrics.items():
        if _hidden(k):
            continue
        if _allowed(k):
            out[k] = v
    return out
