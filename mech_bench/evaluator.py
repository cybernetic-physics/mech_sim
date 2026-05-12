"""Generic evaluator.

Pipeline (called once per submission):

  1. Load TaskSpec + EvalConfig from a task directory.
  2. Load the submission's `design.py` and call build_design() in a
     sandboxed working directory; parse the returned dict into a
     DesignIR.
  3. Determine the union of capabilities required by all configured
     probes. Pick the cheapest registered adapter whose
     capabilities cover that union. Run the adapter once.
  4. For each probe, instantiate it, run it against the IR +
     adapter outputs. Collect ProbeResult.
  5. Compose the final score:
        hard_gate = AND over hard-gate probes (or probes marked
                    `hard_gate=true`).
        dense     = Σ w_i · s_i across non-gate probes, with weights
                    renormalized to sum to 1.
        final     = 0 if hard_gate fails else dense.
  6. Return EvalReport.

The runtime does **not** trust paths in the agent submission outside
the sandbox `out_dir`. Path-policy enforcement is left as a hook
(implementation deferred to a future iteration).
"""

from __future__ import annotations

import importlib.util
import logging
import sys
import tomllib
from pathlib import Path
from typing import Any

from mech_bench.adapters import SimAdapter, all_adapters
from mech_bench.feedback import Failure, FailureCode, Severity
from mech_bench.probes import Capability, Probe, get_probe
from mech_bench.schema import (
    DesignIR,
    EvalConfig,
    EvalReport,
    ProbeResult,
    ProbeSpec,
    TaskSpec,
)

_LOG = logging.getLogger("mech_bench.evaluator")


def load_task(task_dir: Path) -> tuple[TaskSpec, EvalConfig]:
    task_dir = Path(task_dir)
    task_toml = task_dir / "task.toml"
    eval_toml = task_dir / "eval_config.toml"
    prompt_md = task_dir / "prompt.md"

    with task_toml.open("rb") as f:
        task_data = tomllib.load(f)
    with eval_toml.open("rb") as f:
        eval_data = tomllib.load(f)

    # Resolve fixture paths in eval_config relative to fixtures_dir.
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


def load_submission(submission_dir: Path, scratch_dir: Path) -> DesignIR:
    submission_dir = Path(submission_dir).resolve()
    scratch_dir = Path(scratch_dir).resolve()
    scratch_dir.mkdir(parents=True, exist_ok=True)

    design_py = submission_dir / "design.py"
    if not design_py.exists():
        raise FileNotFoundError(f"Submission missing design.py: {design_py}")

    # Isolated module load — not registered in sys.modules under a
    # well-known name, so a second submission cannot collide.
    spec = importlib.util.spec_from_file_location(
        f"_mech_submission_{abs(hash(submission_dir))}",
        design_py,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {design_py}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.modules.pop(spec.name, None)
    if not hasattr(mod, "build_design"):
        raise AttributeError(
            f"{design_py} does not define build_design(out_dir)."
        )
    raw = mod.build_design(scratch_dir)
    if not isinstance(raw, dict):
        raise TypeError(
            f"build_design must return a dict, got {type(raw).__name__}"
        )
    return DesignIR.from_dict(raw)


def _select_adapter(
    probe_capabilities: set[Capability],
) -> SimAdapter | None:
    """Pick the cheapest adapter whose advertised capabilities cover
    `probe_capabilities`. Capabilities equal to NONE are trivially
    satisfied.
    """
    needed = probe_capabilities - {Capability.NONE}
    if not needed:
        return None
    candidates = [a for a in all_adapters()
                  if needed.issubset(a.capabilities_provided)]
    if not candidates:
        return None
    candidates.sort(key=lambda a: a.cost_tier)
    return candidates[0]()


def _run_probe(
    spec: ProbeSpec,
    ir: DesignIR,
    sim_outputs: dict[str, Any],
) -> ProbeResult:
    try:
        probe: Probe = get_probe(spec.type)
    except KeyError as e:
        return ProbeResult(
            probe_id=spec.id,
            probe_type=spec.type,
            passed=False,
            score=0.0,
            metrics={},
            failures=[Failure(
                code=FailureCode.CAPABILITY_UNAVAILABLE,
                severity=Severity.CRITICAL,
                message=str(e),
            )],
        )
    result = probe.run(ir, sim_outputs, spec.config)
    result.probe_id = spec.id
    result.probe_type = spec.type
    return result


def _score(
    probe_results: list[ProbeResult],
    specs_by_id: dict[str, ProbeSpec],
    hard_gate_ids: set[str],
) -> tuple[bool, float]:
    """Hard gate: every gate probe must pass. Dense: weighted
    average over non-gate probes with weight > 0.
    """
    hard_gate_passed = True
    for r in probe_results:
        if r.probe_id in hard_gate_ids and not r.passed:
            hard_gate_passed = False
            break
        if specs_by_id.get(r.probe_id, ProbeSpec("", "")).hard_gate \
                and not r.passed:
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
    dense = weighted / total_w if total_w > 0 else 0.0
    return True, dense


def evaluate(
    task_dir: Path,
    submission_dir: Path,
    *,
    scratch_dir: Path | None = None,
) -> EvalReport:
    """Run the full evaluation pipeline. Public API.

    `scratch_dir` is where build_design() may write CAD artifacts.
    Defaults to `<submission_dir>/_scratch`.
    """
    task_dir = Path(task_dir)
    submission_dir = Path(submission_dir)
    scratch_dir = scratch_dir or (submission_dir / "_scratch")

    task, cfg = load_task(task_dir)

    # Top-level pipeline failures (before any probe runs) feed into
    # the report's `feedback` and force hard_gate=False.
    pipeline_failures: list[Failure] = []
    try:
        ir = load_submission(submission_dir, scratch_dir)
    except Exception as e:
        pipeline_failures.append(Failure(
            code=FailureCode.INVALID_ARTIFACT,
            severity=Severity.CRITICAL,
            message=f"Failed to load submission: {e}",
            where=str(submission_dir),
        ))
        return EvalReport(
            task_id=task.id,
            score=0.0,
            hard_gate_passed=False,
            probe_results=[],
            metrics={},
            feedback=pipeline_failures,
        )

    # Determine required capabilities.
    required: set[Capability] = set()
    for spec in cfg.probes:
        try:
            probe = get_probe(spec.type)
        except KeyError:
            continue
        required |= set(probe.capabilities_required)

    adapter = _select_adapter(required)
    sim_outputs: dict[str, Any] = {}
    if adapter is not None:
        _LOG.info("dispatching to adapter %s for capabilities %s",
                  adapter.type_name, sorted(c.value for c in required))
        sim_outputs = adapter.run(ir, {"samples": 360})

    # Run probes.
    probe_results: list[ProbeResult] = []
    for spec in cfg.probes:
        r = _run_probe(spec, ir, sim_outputs)
        probe_results.append(r)

    specs_by_id = {s.id: s for s in cfg.probes}
    hard_gate_passed, dense = _score(
        probe_results, specs_by_id, set(cfg.hard_gate_probes)
    )

    # Aggregate metrics / feedback for the report.
    agg_metrics: dict[str, float] = {}
    feedback: list[Failure] = list(pipeline_failures)
    for r in probe_results:
        for k, v in r.metrics.items():
            agg_metrics[f"{r.probe_id}.{k}"] = float(v)
        for f in r.failures:
            f.where = f.where or r.probe_id
            feedback.append(f)

    return EvalReport(
        task_id=task.id,
        score=dense,
        hard_gate_passed=hard_gate_passed,
        probe_results=probe_results,
        metrics=agg_metrics,
        feedback=feedback,
    )
