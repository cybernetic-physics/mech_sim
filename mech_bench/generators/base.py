"""Base abstractions for procedural task generation.

A :class:`GeneratedTask` is an in-memory description of one task. It
is written to a directory by :func:`write_task_directory`, which
materializes the standard task-contract layout:

::

    <out>/<task_id>/
        prompt.md
        task.toml
        eval_config.toml          # default = public unless hidden-only
        eval_config.public.toml   # always written
        eval_config.hidden.toml   # written when a hidden variant exists
        fixtures/...
        reference_solution/design.py
        negative_solutions/<case>/design.py
        expected_failures.json
        metadata.json

This layout is the contract: the evaluator and the benchmark runner
both rely on file names here.

The TOML emitter is intentionally tiny — generators only feed it
nested dicts of ``str | int | float | bool | list[...]`` values, so a
hand-rolled writer is simpler than pulling in a third-party
dependency for write support (Python ships ``tomllib`` for read
only).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


# --------------------------------------------------------------------- #
# Data model                                                            #
# --------------------------------------------------------------------- #


@dataclass
class GeneratedTask:
    """One generator's output, ready to be written to disk."""

    task_id: str
    family: str
    difficulty: int

    prompt_md: str
    task_toml: dict[str, Any]
    eval_config_toml: dict[str, Any]

    # Files to write under fixtures/. Keys are relative paths; values
    # are either str (text) or bytes.
    fixtures: dict[str, Any] = field(default_factory=dict)

    # Source for reference_solution/design.py.
    reference_solution_py: str = ""

    # name -> design.py source for negative_solutions/<name>/design.py.
    negative_solutions: dict[str, str] = field(default_factory=dict)

    # Expected outcomes for each negative control, plus metadata.
    # The shape is the contents of expected_failures.json.
    expected_failures: dict[str, Any] = field(default_factory=dict)

    # Free-form metadata persisted to metadata.json.
    metadata: dict[str, Any] = field(default_factory=dict)

    # Optional hidden-variant eval config (eval_config.hidden.toml).
    # Public is always written; hidden, if supplied, is written
    # alongside.
    eval_config_hidden_toml: dict[str, Any] | None = None


class TaskGenerator:
    """Subclasses implement :meth:`generate` to emit a GeneratedTask.

    ``family`` is the task family name (e.g. ``"planar_4bar"``);
    ``tier`` is one of ``"artifact_static" | "planar_kinematics" |
    "transmission_analytic" | "contact_dynamics"``.
    """

    family: str = ""
    tier: str = ""

    def generate(self, seed: int, difficulty: int = 1) -> GeneratedTask:
        raise NotImplementedError


# --------------------------------------------------------------------- #
# Directory writer                                                      #
# --------------------------------------------------------------------- #


def write_task_directory(task: GeneratedTask, out_dir: Path) -> Path:
    """Write *task* to ``out_dir / task.task_id`` and return that path.

    The directory is created (parents included) if it does not exist.
    The function overwrites existing files so the same generator can be
    re-run without manual cleanup.
    """
    out_dir = Path(out_dir)
    task_dir = out_dir / task.task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "fixtures").mkdir(exist_ok=True)
    (task_dir / "reference_solution").mkdir(exist_ok=True)
    (task_dir / "negative_solutions").mkdir(exist_ok=True)

    (task_dir / "prompt.md").write_text(task.prompt_md)
    (task_dir / "task.toml").write_text(dumps_toml(task.task_toml))

    # eval_config.toml is the default the evaluator picks up. Generators
    # supply a "public" variant by default; we mirror it as
    # eval_config.public.toml and ALSO leave eval_config.toml so that
    # existing single-config callers keep working.
    public_toml = dumps_toml(task.eval_config_toml)
    (task_dir / "eval_config.toml").write_text(public_toml)
    (task_dir / "eval_config.public.toml").write_text(public_toml)
    if task.eval_config_hidden_toml is not None:
        (task_dir / "eval_config.hidden.toml").write_text(
            dumps_toml(task.eval_config_hidden_toml)
        )

    for rel, payload in task.fixtures.items():
        fpath = task_dir / "fixtures" / rel
        fpath.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(payload, bytes):
            fpath.write_bytes(payload)
        else:
            fpath.write_text(str(payload))

    (task_dir / "reference_solution" / "design.py").write_text(
        task.reference_solution_py
    )

    for name, src in task.negative_solutions.items():
        sub = task_dir / "negative_solutions" / name
        sub.mkdir(parents=True, exist_ok=True)
        (sub / "design.py").write_text(src)

    (task_dir / "expected_failures.json").write_text(
        json.dumps(task.expected_failures, indent=2, default=str)
    )

    meta = dict(task.metadata)
    meta.setdefault("task_id", task.task_id)
    meta.setdefault("family", task.family)
    meta.setdefault("difficulty", task.difficulty)
    (task_dir / "metadata.json").write_text(
        json.dumps(meta, indent=2, default=str)
    )

    return task_dir


# --------------------------------------------------------------------- #
# Minimal TOML emitter                                                  #
# --------------------------------------------------------------------- #
#
# Supports: nested tables ([section] / [section.sub]), array-of-tables
# ([[probes]]), strings, ints, floats, bools, and homogeneous arrays
# of those scalars (including nested arrays). Generators may use ints
# and floats freely; we always preserve numeric type in the output.


def dumps_toml(d: dict[str, Any]) -> str:
    """Render *d* as a TOML document. Only the subset above is emitted."""
    parts: list[str] = []
    _write_section(d, parts, prefix="")
    return "\n".join(parts).rstrip() + "\n"


def _write_section(d: dict[str, Any], parts: list[str], *, prefix: str) -> None:
    # First write scalar entries at this level.
    scalars: list[tuple[str, Any]] = []
    sub_tables: list[tuple[str, dict[str, Any]]] = []
    array_of_tables: list[tuple[str, list[dict[str, Any]]]] = []
    for k, v in d.items():
        if isinstance(v, dict):
            sub_tables.append((k, v))
        elif isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
            array_of_tables.append((k, v))
        else:
            scalars.append((k, v))

    if scalars:
        if prefix:
            parts.append(f"[{prefix}]")
        for k, v in scalars:
            parts.append(f"{_key(k)} = {_emit_value(v)}")
        parts.append("")

    for k, sub in sub_tables:
        new_prefix = f"{prefix}.{_key(k)}" if prefix else _key(k)
        # Empty tables still emit a header so it round-trips.
        if not sub:
            parts.append(f"[{new_prefix}]")
            parts.append("")
            continue
        # Determine if the sub-table has any scalars at top level. If
        # not, we don't need a bare [header]; the nested tables write
        # their own.
        sub_scalars = [(sk, sv) for sk, sv in sub.items()
                       if not isinstance(sv, dict)
                       and not (isinstance(sv, list) and sv
                                and all(isinstance(x, dict) for x in sv))]
        if sub_scalars:
            _write_section(sub, parts, prefix=new_prefix)
        else:
            _write_section(sub, parts, prefix=new_prefix)

    for k, items in array_of_tables:
        new_prefix = f"{prefix}.{_key(k)}" if prefix else _key(k)
        for item in items:
            parts.append(f"[[{new_prefix}]]")
            inner_scalars: list[tuple[str, Any]] = []
            inner_subtables: list[tuple[str, dict[str, Any]]] = []
            inner_aot: list[tuple[str, list[dict[str, Any]]]] = []
            for ik, iv in item.items():
                if isinstance(iv, dict):
                    inner_subtables.append((ik, iv))
                elif isinstance(iv, list) and iv and all(
                        isinstance(x, dict) for x in iv):
                    inner_aot.append((ik, iv))
                else:
                    inner_scalars.append((ik, iv))
            for ik, iv in inner_scalars:
                parts.append(f"{_key(ik)} = {_emit_value(iv)}")
            parts.append("")
            for ik, sub in inner_subtables:
                _write_section(sub, parts, prefix=f"{new_prefix}.{_key(ik)}")
            for ik, sub_items in inner_aot:
                for inner in sub_items:
                    parts.append(f"[[{new_prefix}.{_key(ik)}]]")
                    for kk, vv in inner.items():
                        parts.append(f"{_key(kk)} = {_emit_value(vv)}")
                    parts.append("")


def _key(k: str) -> str:
    # TOML allows bare keys [A-Za-z0-9_-]; we wrap anything else in
    # quotes. Our generators only emit ASCII identifiers so this is
    # mostly a safety net.
    safe = all(c.isalnum() or c in "_-" for c in k)
    if safe and k:
        return k
    return _emit_string(k)


def _emit_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(int(v))
    if isinstance(v, float):
        if v != v:  # NaN
            return "nan"
        if v == float("inf"):
            return "inf"
        if v == float("-inf"):
            return "-inf"
        return repr(float(v))
    if isinstance(v, str):
        return _emit_string(v)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_emit_value(x) for x in v) + "]"
    if v is None:
        return _emit_string("")
    # Fallback — caller probably passed an unsupported type.
    return _emit_string(str(v))


def _emit_string(s: str) -> str:
    # Basic quoting. Escape backslash, double-quote, control chars.
    out = ['"']
    for c in s:
        if c == "\\":
            out.append("\\\\")
        elif c == "\"":
            out.append("\\\"")
        elif c == "\n":
            out.append("\\n")
        elif c == "\r":
            out.append("\\r")
        elif c == "\t":
            out.append("\\t")
        elif ord(c) < 0x20:
            out.append(f"\\u{ord(c):04x}")
        else:
            out.append(c)
    out.append('"')
    return "".join(out)


# --------------------------------------------------------------------- #
# Helpers shared across generators                                      #
# --------------------------------------------------------------------- #


def make_task_id(family: str, seed: int) -> str:
    """Deterministic task id used by all generators."""
    return f"{family}_s{seed:04d}"


def common_metadata(family: str, tier: str, seed: int,
                    difficulty: int, **extras: Any) -> dict[str, Any]:
    out: dict[str, Any] = {
        "family": family,
        "tier": tier,
        "seed": int(seed),
        "difficulty": int(difficulty),
    }
    out.update(extras)
    return out


# Type alias used by benchmark_suite.GENERATORS.
GeneratorFn = Callable[[int, int], GeneratedTask]
