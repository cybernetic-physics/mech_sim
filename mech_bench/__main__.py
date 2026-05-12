"""`mech-bench` CLI.

Subcommands:
  evaluate   Score a submission against a task
  list-probes  Show registered probe types and their capabilities
  list-adapters Show registered adapters and their capabilities
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mech_bench.adapters import all_adapters
from mech_bench.evaluator import evaluate, load_task
from mech_bench.probes import known_probe_types, get_probe


def _cmd_evaluate(args: argparse.Namespace) -> int:
    task_dir = Path(args.task)
    submission_dir = Path(args.submission)
    scratch_dir = Path(args.scratch) if args.scratch else None
    report = evaluate(task_dir, submission_dir, scratch_dir=scratch_dir)
    task, cfg = load_task(task_dir)
    public = report.public_dict(cfg.visibility)
    out = json.dumps(public, indent=2, default=str)
    print(out)
    if args.out:
        Path(args.out).write_text(out)
    return 0 if report.hard_gate_passed else 1


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
    ev.set_defaults(func=_cmd_evaluate)

    lp = sub.add_parser("list-probes", help="List registered probes")
    lp.set_defaults(func=_cmd_list_probes)

    la = sub.add_parser("list-adapters", help="List registered adapters")
    la.set_defaults(func=_cmd_list_adapters)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
