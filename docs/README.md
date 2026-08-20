# Documentation

This directory separates the project’s current contract from frozen evidence
and historical research notes. Start with the first three documents; use the
others when you need implementation or experiment detail.

## Start here

| Document | Purpose |
|---|---|
| [Project status](project-status.md) | What is implemented, what has been demonstrated, what is incomplete, and what matters next |
| [Autonomous manufacturing vision](autonomous-manufacturing-vision.md) | Why verifier-guided learning for robotics parts is the long-term research direction |
| [Architecture](../ARCHITECTURE.md) | Runtime abstractions, trust boundary, dispatch, scoring, and task layout |

## Implementation references

| Document | Purpose |
|---|---|
| [Chrono backend](chrono-backend.md) | Native rigid-body/contact support, availability checks, evidence policy, and known limits |
| [High-fidelity simulation roadmap](high-fidelity-simulation-roadmap.md) | Ordered work needed to move from solver execution to credible predictive simulation |
| [RL training](../rl/README.md) | Exact GRPO path, legacy trainer, and reward contract |
| [Evaluation snapshots](../evals/README.md) | Frozen agent-evaluation records and their validity caveats |
| [Contributing](../CONTRIBUTING.md) | Setup, validation tiers, and change-review rules |

## Evidence records

Evidence records are dated snapshots. Their task counts and conclusions apply
to the named artifact and commit, not automatically to the current generator
registry.

| Record | What it establishes | Important boundary |
|---|---|---|
| [Benchmark snapshot](paper-results-current.md) | Reference and negative-control behavior on the 51-task May 2026 suite | Eleven reference tasks used the synthetic contact adapter |
| [CAD-to-Chrono proof](cycloidal-real-geometry-proof.md) | Trusted CAD export, mass properties, real collision geometry, and NSC/SMC execution | Solver-path acceptance, not hardware calibration |
| [Later cycloidal acceptance run](cycloidal_real_geometry_chrono_validation.md) | A later real-geometry fixture and refinement packet | Loaded SMC still fails power balance |
| [Equal-budget cycloidal pilot](cycloidal_mechanical_evolve_equal_budget.md) | One matched-audit design-search comparison | Exploratory single-run result with low absolute pass rates |
| [Family-transfer audit](family_transfer_claim_audit.md) | Why an early seed-heldout run did not prove unseen-family transfer | Superseded for Level-1 transfer by the June result bundle |
| [Level-1 family-heldout result](../runs/mechanism_repair_ttrl_final/README.md) | Matched-budget online-learning result across held-out mechanism families | Executable mechanism programs only; no CAD/contact claim |
| [Level-2/3 experiment bundle](../runs/mechanism_repair_physics_final/claim_audit.json) | Benchmark and audit scaffolding for CAD/contact tasks | Not executed: the committed audit records zero result rows |

## Historical notes

The following files are retained for provenance, not as current guidance:

- [`results-and-rl-roadmap.md`](results-and-rl-roadmap.md) — May 2026 agent
  evaluation and first training-cook notes.
- [`comparison-ttt-discover-codex-runtime.md`](comparison-ttt-discover-codex-runtime.md)
  and [`comparison-codex-vs-atropos-stacks.md`](comparison-codex-vs-atropos-stacks.md)
  — point-in-time runtime comparisons.
- [`../mech-sim-state.md`](../mech-sim-state.md) — a distillation of a prior,
  separate simulator repository.
- [`../goals.md`](../goals.md) — the execution contract used to plan the later
  mechanism-repair experiment.

When a historical note disagrees with the project-status page or current code,
the project-status page and code are authoritative.
