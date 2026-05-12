# mech_bench — Mechanical Design RLVR Benchmark Runtime

A generic runtime that evaluates AI-generated mechanical designs
against task configs, with verifiable rewards. The runtime knows
about probes, adapters, and capabilities — not about specific
mechanism families. Tasks live in `tasks/`; mechanism semantics are
configuration, not code.

See `ARCHITECTURE.md` for the design rationale, and
`mech-sim-state.md` for the prior `phys-sim` distillation that
seeded this design.

## Install

```bash
uv sync                                 # or: pip install -e '.[dev]'
```

## Run the example task

```bash
mech-bench evaluate \
    --task tasks/fourbar_path_t001 \
    --submission tasks/fourbar_path_t001/reference_solution
```

## Layout

```
mech_bench/
    schema.py        DesignIR, TaskSpec, EvalConfig, ProbeSpec, ProbeResult
    feedback.py      FailureCode (closed grammar), Severity, Failure
    probes/          built-in probes; each declares capabilities_required
        dof_grubler.py
        path_trace_chamfer.py
    adapters/        capability-tagged simulator adapters
        planar_kinematics.py
    evaluator.py     hard-gate + dense reward composition
    __main__.py      mech-bench CLI

tasks/
    fourbar_path_t001/
        prompt.md
        task.toml
        eval_config.toml
        reference_solution/design.py
        fixtures/target_path.csv
        expected_failures.json

tests/
    test_schema.py
    test_evaluator.py
```

## Status

- Schemas, feedback grammar, evaluator, probe + adapter bases:
  shipping.
- Probes: `dof_grubler`, `path_trace_chamfer`. More land iteratively.
- Adapters: `planar_kinematics`. `chrono_contact` will port the
  working mesh-contact runner from `phys-sim/mech_harness/simulators/`.
- One end-to-end task (`fourbar_path_t001`) plus a negative-control
  test that proves the probes have teeth.
