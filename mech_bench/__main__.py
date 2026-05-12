"""`mech-bench` CLI.

Subcommands:
  evaluate      Score a submission against a task
  list-probes   Show registered probe types and their capabilities
  list-adapters Show registered adapters and their capabilities
  package-run   Collect/normalize a packaged run directory
  video         Render an mp4 preview for a packaged run (placeholder)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mech_bench.adapters import all_adapters
from mech_bench.evaluator import (
    evaluate_with_evidence,
    load_task,
    sanitize_report_for_json,
    write_run_bundle,
)
from mech_bench.probes import known_probe_types, get_probe


def _cmd_evaluate(args: argparse.Namespace) -> int:
    task_dir = Path(args.task)
    submission_dir = Path(args.submission)
    scratch_dir = Path(args.scratch) if args.scratch else None
    evidence = evaluate_with_evidence(
        task_dir, submission_dir, scratch_dir=scratch_dir)
    report = evidence.report
    cfg = evidence.cfg

    if args.full:
        blob = report.to_dict(public=False, visibility=cfg.visibility)
    else:
        blob = report.to_dict(public=True, visibility=cfg.visibility)
    out = json.dumps(
        sanitize_report_for_json(blob),
        indent=2,
        default=str,
        allow_nan=False,
    )
    print(out)

    if args.out:
        Path(args.out).write_text(out)

    if args.report_dir:
        paths = write_run_bundle(evidence, Path(args.report_dir))
        for k, p in paths.items():
            print(f"# wrote {k}: {p}", file=sys.stderr)

    if args.allow_partial:
        return 0
    ok = (
        report.evaluation_valid
        and report.hard_gate_passed
        and report.score > 0
    )
    return 0 if ok else 1


def _cmd_list_probes(args: argparse.Namespace) -> int:
    for name in known_probe_types():
        probe = get_probe(name)
        caps = sorted(c.value for c in probe.capabilities_required)
        print(f"{name}  requires={caps}")
    return 0


def _cmd_list_adapters(args: argparse.Namespace) -> int:
    for cls in all_adapters():
        caps = sorted(c.value for c in cls.capabilities_provided)
        print(f"{cls.type_name}  cost={cls.cost_tier}  provides={caps}")
    return 0


def _cmd_package_run(args: argparse.Namespace) -> int:
    from mech_bench.media import package_run

    report_dir = Path(args.report_dir)
    if not report_dir.is_dir():
        print(f"error: {report_dir} is not a directory", file=sys.stderr)
        return 2
    files = package_run(report_dir)
    print(json.dumps(
        {"report_dir": str(report_dir),
         "files": {k: str(v) for k, v in files.items()}},
        indent=2,
    ))
    return 0


def _cmd_video(args: argparse.Namespace) -> int:
    from mech_bench.video import FrameSequenceRenderer

    report_dir = Path(args.report_dir)
    payload_path = report_dir / "dashboard_payload.json"
    if not payload_path.exists():
        print(
            f"error: no dashboard_payload.json in {report_dir}. "
            f"Run `mech-bench evaluate --report-dir {report_dir}` first.",
            file=sys.stderr,
        )
        return 2
    try:
        payload = json.loads(payload_path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        print(f"error: cannot read payload: {e}", file=sys.stderr)
        return 2

    renderer = FrameSequenceRenderer()
    out_mp4 = Path(args.out)
    result = renderer.render(payload, out_mp4, fps=args.fps)
    if not result.ok:
        warn = {
            "status": "capability_unavailable",
            "reason": result.reason,
            "backend": result.backend,
            "out_mp4": str(out_mp4),
        }
        print(json.dumps(warn, indent=2), file=sys.stderr)
        return 1
    print(json.dumps(
        {"status": "ok",
         "backend": result.backend,
         "out_mp4": str(result.out_path)},
        indent=2,
    ))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="mech-bench")
    sub = p.add_subparsers(dest="cmd", required=True)

    ev = sub.add_parser("evaluate", help="Score a submission")
    ev.add_argument("--task", required=True, help="task directory")
    ev.add_argument("--submission", required=True,
                    help="submission directory (containing design.py)")
    ev.add_argument("--scratch", default=None,
                    help="scratch dir for build_design output")
    ev.add_argument("--out", default=None,
                    help="write the report JSON to this path too")
    ev.add_argument("--report-dir", default=None,
                    help=("write the full report bundle (scorecard, "
                          "metrics, public feedback, dashboard, trace) "
                          "under this dir"))
    ev.add_argument("--full", action="store_true",
                    help=("print the full internal report instead of "
                          "the public-redacted view"))
    ev.add_argument("--allow-partial", action="store_true",
                    help=("exit 0 even if the hard gate failed or "
                          "score is zero"))
    ev.set_defaults(func=_cmd_evaluate)

    lp = sub.add_parser("list-probes", help="List registered probes")
    lp.set_defaults(func=_cmd_list_probes)

    la = sub.add_parser("list-adapters", help="List registered adapters")
    la.set_defaults(func=_cmd_list_adapters)

    pr = sub.add_parser(
        "package-run",
        help="Normalize a packaged run directory (writes media_manifest)",
    )
    pr.add_argument("--report-dir", required=True,
                    help="directory previously written by `evaluate --report-dir`")
    pr.set_defaults(func=_cmd_package_run)

    vd = sub.add_parser(
        "video",
        help=("Render an mp4 preview from a packaged run "
              "(placeholder; emits a warning if no backend is available)"),
    )
    vd.add_argument("--report-dir", required=True)
    vd.add_argument("--out", required=True, help="output mp4 path")
    vd.add_argument("--fps", type=int, default=30)
    vd.set_defaults(func=_cmd_video)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
