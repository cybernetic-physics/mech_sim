"""Procedural task generators.

A generator produces a :class:`GeneratedTask` — a self-contained
on-disk task contract (prompt, task.toml, eval_config.toml, fixtures,
reference solution, negative controls, expected_failures.json,
metadata.json) — for a particular family at a particular difficulty.

See ``benchmark_suite.py`` for the registry used by the
``mech-bench generate-suite`` CLI.
"""

from __future__ import annotations

from mech_bench.generators.base import (
    GeneratedTask,
    TaskGenerator,
    write_task_directory,
)

# Trigger registration of probes that generated tasks may reference.
from mech_bench.probes import analytic_param_check  # noqa: F401,E402

__all__ = ["GeneratedTask", "TaskGenerator", "write_task_directory"]
