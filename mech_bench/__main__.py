"""`mech-bench` CLI.

Subcommands:
  evaluate      Score a submission against a task
  list-probes   Show registered probe types and their capabilities
  list-adapters Show registered adapters and their capabilities
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mech_bench.adapters import all_adapters
from mech_bench.evaluator import (
    evaluate,
    load_task,
    sanitize_report_for_json,
    write_report_bundle,
)
from mech_bench.probes import known_probe_types, get_probe


def _cmd_evaluate(args: argparse.Namespace) -> int:
    task_dir = Path(args.task)
    submission_dir = Path(args.submission)
    scratch_dir = Path(args.scratch) if args.scratch else None
    report = evaluate(task_dir, submission_dir, scratch_dir=scratch_dir)
    _, cfg = load_task(task_dir)

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
        paths = write_report_bundle(
            report, Path(args.report_dir), visibility=cfg.visibility)
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
                          "metrics, public feedback) under this dir"))
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

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
