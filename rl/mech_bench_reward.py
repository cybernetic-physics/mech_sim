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


REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class RewardResult:
    score: float
    verified_score: float
    hard_gate_passed: bool
    evaluation_valid: bool
    failure_codes: list[str] = field(default_factory=list)
    submission_path: str = ""
    design_py_extracted: bool = False
    raw_score_json: str = ""

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "verified_score": self.verified_score,
            "hard_gate_passed": self.hard_gate_passed,
            "evaluation_valid": self.evaluation_valid,
            "failure_codes": list(self.failure_codes),
            "submission_path": self.submission_path,
            "design_py_extracted": self.design_py_extracted,
        }


_CODE_FENCE_RE = re.compile(
    r"```(?:python|py)?\s*\n(.*?)```",
    flags=re.DOTALL | re.IGNORECASE,
)


def extract_design_py(completion: str) -> tuple[str, bool]:
    """Pull the first Python code block out of *completion*.

    Returns (source, extracted). When no fenced block is found, the
    whole completion is returned with extracted=False so the caller
    can still attempt to score (and likely get an invalid_artifact
    failure, which is the desired signal during early training).
    """
    m = _CODE_FENCE_RE.search(completion)
    if m:
        return m.group(1).strip() + "\n", True
    return completion.strip() + "\n", False


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
    for f in blob.get("feedback") or []:
        c = f.get("code")
        if isinstance(c, str) and c not in codes:
            codes.append(c)

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
        submission_path=str(submission_dir),
        design_py_extracted=extracted,
        raw_score_json=proc.stdout,
    )


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
