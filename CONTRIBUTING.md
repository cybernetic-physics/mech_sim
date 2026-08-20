# Contributing

Thank you for improving `mech-sim`. The project evaluates untrusted,
agent-generated mechanical designs, so changes to trust boundaries and scoring
semantics require particular care.

## Setup

```bash
uv sync --extra dev
scripts/check_core.sh
```

The portable core check runs without a GPU or native physics stack. PyChrono,
CAD-kernel, training, and archived experiment-replay checks have separate
dependencies and should be run when a change touches those surfaces.

## Change guidelines

- Keep synthetic, simulated, and validated evidence clearly distinguished.
- Do not let the synthetic contact adapter satisfy a production physics task
  unless the task explicitly opts in.
- Preserve public/hidden evaluation separation.
- Treat agent submissions, paths, geometry, and declared physical properties as
  untrusted until validated or recomputed.
- Add a negative control for every new hard gate or failure mode.
- Keep frozen result artifacts immutable; generate a new versioned artifact
  when evidence changes.
- Record units, solver versions, settings, and random seeds in new evidence
  paths.

## Validation tiers

| Tier | Use |
|---|---|
| Portable core | Evaluator, schemas, probes, isolation, evidence, and CLI behavior |
| Native solver | PyChrono/CAD imports, kernel operations, contact runs, and diagnostics |
| Training | MLX, PyTorch, PEFT, TRL, and rollout integrations |
| Experiment replay | Frozen manifests, remote artifacts, and historical result audits |

Run `scripts/solver_smoke.sh` for the native solver environment. Install
`training-mlx` or `training-grpo` extras before changing the corresponding
training code.

## Pull-request checklist

- [ ] The change has a focused explanation and test coverage.
- [ ] `scripts/check_core.sh` passes.
- [ ] Relevant optional validation tiers pass or are documented.
- [ ] Documentation reflects the implemented state.
- [ ] New claims link to replayable evidence and state their limitations.
