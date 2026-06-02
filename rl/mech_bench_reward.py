"""Score a model completion via `mech_bench.evaluate`.

Inputs: the raw model completion (text), the task directory.
Outputs: a structured reward record:

    {
        "score": float in [0, 1],
        "verified_score": float in [0, 1],  # 0 when invalid or gate failed
        "hard_gate_passed": bool,
        "evaluation_valid": bool,
        "failure_codes": list[str],
        "submission_path": str,
        "design_py_extracted": bool,
    }

The completion may contain prose; we extract the first triple-
backticked Python block (preferring ```python). If no block is
found, the whole completion is written to design.py as a fallback.
The verifier then runs `python -m mech_bench evaluate --full
--allow-partial` and we parse the score.

This file is the only piece of glue between an RL rollout (model
emits text) and the deterministic reward (mech_bench probes).
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rl.verifier_audits import cad_audit_count, chrono_audit_count


REPO_ROOT = Path(__file__).resolve().parent.parent
PHYSICAL_METRIC_ALIASES = {
    "ratio_observed": (
        "ratio_observed",
        "ratio.observed",
        "chrono_contact.ratio_observed",
        "chrono_contact.ratio.observed",
    ),
    "ratio_error_pct": (
        "ratio_error_pct",
        "ratio.error_pct",
        "chrono_contact.ratio_error_pct",
        "chrono_contact.ratio.error_pct",
    ),
    "out_omega_med": ("out_omega_med", "chrono_contact.out_omega_med"),
    "power_balance_error_pct": (
        "power_balance_error_pct",
        "chrono_contact.power_balance_error_pct",
    ),
    "torque_ripple_pct": ("torque_ripple_pct", "chrono_contact.torque_ripple_pct"),
    "max_penetration_mm": (
        "max_penetration_mm",
        "chrono_contact.max_penetration_mm",
    ),
    "contact_force_rms_N": (
        "contact_force_rms_N",
        "chrono_contact.contact_force_rms_N",
    ),
}
NO_PROCEDURAL_FALLBACK_ALIASES = (
    "procedural_cycloidal_fallback",
    "chrono.procedural_cycloidal_fallback",
    "chrono_contact.procedural_cycloidal_fallback",
)


@dataclass
class RewardResult:
    score: float
    verified_score: float
    hard_gate_passed: bool
    evaluation_valid: bool
    failure_codes: list[str] = field(default_factory=list)
    feedback: list[dict[str, str]] = field(default_factory=list)
    submission_path: str = ""
    design_py_extracted: bool = False
    raw_score_json: str = ""
    cad_audits: int = 0
    chrono_audits: int = 0
    physical_metrics: dict[str, float] = field(default_factory=dict)
    no_procedural_fallback: bool | None = None

    def to_dict(self) -> dict:
        out = {
            "score": self.score,
            "verified_score": self.verified_score,
            "hard_gate_passed": self.hard_gate_passed,
            "evaluation_valid": self.evaluation_valid,
            "failure_codes": list(self.failure_codes),
            "feedback": list(self.feedback),
            "submission_path": self.submission_path,
            "design_py_extracted": self.design_py_extracted,
            "cad_audits": self.cad_audits,
            "chrono_audits": self.chrono_audits,
            "physical_metrics": dict(self.physical_metrics),
        }
        if self.no_procedural_fallback is not None:
            out["no_procedural_fallback"] = self.no_procedural_fallback
        return out


_CODE_FENCE_RE = re.compile(
    r"```(?:python|py)?\s*\n(.*?)```",
    flags=re.DOTALL | re.IGNORECASE,
)
_OPEN_CODE_FENCE_RE = re.compile(
    r"```(?:python|py)?\s*\n(.*)",
    flags=re.DOTALL | re.IGNORECASE,
)


def extract_design_py(completion: str) -> tuple[str, bool]:
    """Pull the intended Python design file out of *completion*.

    Returns (source, extracted). When no fenced block is found, the
    whole completion is returned with extracted=False so the caller
    can still attempt to score (and likely get an invalid_artifact
    failure, which is the desired signal during early training).
    """
    matches = list(_CODE_FENCE_RE.finditer(completion))
    if matches:
        design_matches = [
            m for m in matches
            if "def build_design" in m.group(1)
        ]
        source = (design_matches[-1] if design_matches else matches[0]).group(1)
        extracted = True
    else:
        open_matches = list(_OPEN_CODE_FENCE_RE.finditer(completion))
        design_open_matches = [
            m for m in open_matches
            if "def build_design" in m.group(1)
        ]
        m_open = (
            design_open_matches[-1]
            if design_open_matches else
            (open_matches[-1] if open_matches else None)
        )
        if m_open:
            source = m_open.group(1)
            extracted = True
        else:
            source = completion
            extracted = False
    source = re.sub(r"(?:\s*(?:<\|im_end\|>|<\|endoftext\|>|</s>))*\s*$", "", source, flags=re.IGNORECASE)
    return source.strip() + "\n", extracted


def score_completion(
    completion: str,
    task_dir: Path,
    *,
    scratch_root: Path | None = None,
    timeout_s: float = 60.0,
) -> RewardResult:
    """Score *completion* against *task_dir*.

    The submission directory is materialized inside *scratch_root*
    (a temp dir if not supplied). The mech_bench evaluator is run
    as a subprocess so the agent's design.py never executes in the
    trainer process.
    """
    task_dir = Path(task_dir).resolve()
    source, extracted = extract_design_py(completion)

    cleanup = False
    if scratch_root is None:
        scratch_root = Path(tempfile.mkdtemp(prefix="mech_rl_"))
        cleanup = True
    submission_dir = Path(scratch_root) / "submission"
    submission_dir.mkdir(parents=True, exist_ok=True)
    design_path = submission_dir / "design.py"
    design_path.write_text(source)

    scratch = Path(scratch_root) / "_scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "mech_bench", "evaluate",
        "--task", str(task_dir),
        "--submission", str(submission_dir),
        "--scratch", str(scratch),
        "--full",
        "--allow-partial",
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=timeout_s, check=False,
            cwd=REPO_ROOT,
        )
    except subprocess.TimeoutExpired:
        return RewardResult(
            score=0.0, verified_score=0.0,
            hard_gate_passed=False, evaluation_valid=False,
            failure_codes=["timeout"],
            submission_path=str(submission_dir),
            design_py_extracted=extracted,
        )

    try:
        blob = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return RewardResult(
            score=0.0, verified_score=0.0,
            hard_gate_passed=False, evaluation_valid=False,
            failure_codes=["runner_json_error"],
            submission_path=str(submission_dir),
            design_py_extracted=extracted,
            raw_score_json=proc.stdout[-2000:],
        )

    codes: list[str] = []
    feedback: list[dict[str, str]] = []
    for f in blob.get("feedback") or []:
        c = f.get("code")
        if isinstance(c, str) and c not in codes:
            codes.append(c)
        if isinstance(f, dict):
            item: dict[str, str] = {}
            for key in ("code", "severity", "message", "where"):
                val = f.get(key)
                if val is not None:
                    item[key] = str(val)
            if item and len(feedback) < 8:
                feedback.append(item)

    score = float(blob.get("score") or 0.0)
    valid = bool(blob.get("evaluation_valid"))
    gate = bool(blob.get("hard_gate_passed"))
    verified = score if (valid and gate) else 0.0

    if cleanup:
        # Keep the design.py around for caller inspection; only
        # remove the evaluator scratch (which can be large).
        try:
            for p in scratch.glob("**/*"):
                if p.is_file():
                    p.unlink()
        except OSError:
            pass

    return RewardResult(
        score=score,
        verified_score=verified,
        hard_gate_passed=gate,
        evaluation_valid=valid,
        failure_codes=codes,
        feedback=feedback,
        submission_path=str(submission_dir),
        design_py_extracted=extracted,
        raw_score_json=proc.stdout,
        cad_audits=cad_audit_count(blob),
        chrono_audits=chrono_audit_count(blob),
        physical_metrics=extract_physical_metrics(blob),
        no_procedural_fallback=extract_no_procedural_fallback(blob),
    )


def _finite_float(raw: Any) -> float | None:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if value != value or value in {float("inf"), float("-inf")}:
        return None
    return value


def _lookup_key_or_path(source: dict[str, Any], key: str) -> Any:
    if key in source:
        return source[key]
    current: Any = source
    for part in key.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _lookup_score_value(score_blob: dict[str, Any], key: str) -> Any:
    for container_key in ("metrics", "scalar_metrics", "general_metrics"):
        container = score_blob.get(container_key)
        if isinstance(container, dict):
            value = _lookup_key_or_path(container, key)
            if value is not None:
                return value
    return _lookup_key_or_path(score_blob, key)


def _metric_values_with_suffix(
    score_blob: dict[str, Any],
    *,
    suffix: str,
) -> list[float]:
    values: list[float] = []
    for container_key in ("metrics", "scalar_metrics", "general_metrics"):
        container = score_blob.get(container_key)
        if not isinstance(container, dict):
            continue
        for key, raw in container.items():
            if str(key).endswith(suffix):
                value = _finite_float(raw)
                if value is not None:
                    values.append(value)
    return values


def extract_physical_metrics(score_blob: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for canonical, aliases in PHYSICAL_METRIC_ALIASES.items():
        for alias in aliases:
            value = _finite_float(_lookup_score_value(score_blob, alias))
            if value is not None:
                out[canonical] = value
                break
    if "max_penetration_mm" not in out:
        values = _metric_values_with_suffix(
            score_blob,
            suffix=".max_penetration_mm",
        ) + _metric_values_with_suffix(score_blob, suffix=".max_pen_mm")
        if values:
            out["max_penetration_mm"] = max(values)
    if "contact_force_rms_N" not in out:
        values = _metric_values_with_suffix(score_blob, suffix=".rms_N")
        if values:
            out["contact_force_rms_N"] = max(values)
    if "out_omega_med" not in out:
        output_motion = _finite_float(
            _lookup_score_value(score_blob, "lockup.output_motion_rad")
        )
        if output_motion is not None:
            out["out_omega_med"] = output_motion
    return out


def extract_no_procedural_fallback(score_blob: dict[str, Any]) -> bool | None:
    for alias in NO_PROCEDURAL_FALLBACK_ALIASES:
        raw = _lookup_score_value(score_blob, alias)
        if raw is None:
            continue
        if isinstance(raw, bool):
            return not raw
        value = _finite_float(raw)
        if value is not None:
            return value == 0.0
        text = str(raw).strip().lower()
        if text in {"false", "no", "0"}:
            return True
        if text in {"true", "yes", "1"}:
            return False
    timings = score_blob.get("timings")
    if (
        isinstance(timings, dict)
        and _lookup_key_or_path(timings, "adapter.chrono_contact") is not None
    ):
        return True
    return None


# --------------------------------------------------------------------- #
# CLI for quick smoke testing                                            #
# --------------------------------------------------------------------- #


def _main() -> int:
    import argparse
    p = argparse.ArgumentParser(
        prog="mech_bench_reward",
        description="Score one completion against one task.",
    )
    p.add_argument("--task", required=True, type=Path)
    p.add_argument("--completion-file", type=Path,
                   help="path to a file holding the completion text")
    p.add_argument("--completion", default=None,
                   help="inline completion text (use - for stdin)")
    args = p.parse_args()
    if args.completion == "-":
        text = sys.stdin.read()
    elif args.completion_file:
        text = args.completion_file.read_text()
    else:
        text = args.completion or ""
    r = score_completion(text, args.task)
    print(json.dumps(r.to_dict(), indent=2))
    return 0 if r.verified_score > 0 else 1


if __name__ == "__main__":
    sys.exit(_main())
