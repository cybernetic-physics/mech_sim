# mech-bench — Mechanical-Design RLVR Benchmark Runtime

`mech-bench` is a **generic runtime** that scores AI-generated mechanical
designs against task contracts, with verifiable rewards. The runtime is
mechanism-agnostic — it knows about *probes*, *adapters*, *capabilities*,
and a closed grammar of failure codes. Cycloidal reducers, four-bar
linkages, slider-cranks, and gear trains are all just **task configs**,
not new validator classes.

The deliverables today:

- A generic **DesignIR + TaskSpec + EvalConfig** schema.
- A **capability-tagged probe / adapter dispatcher**.
- A **trusted evaluator** with hard-gate + dense reward separation and
  a closed `FailureCode` grammar.
- A **procedural task generator** spanning four tiers (artifact,
  planar kinematics, transmission analytic, contact dynamics).
- An **HDF5 trace bundle + dashboard payload** for every run.
- **Fast / oracle / final** evaluation modes with cross-mode
  agreement metrics.
- A **compact RLVR reward API** suitable for an agent loop.
- A **deterministic fake contact oracle** for testing the contact /
  dynamics probe pipeline without a real physics engine.
- A **planar-mechanism MP4 renderer** for demos.

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the design rationale and
[`mech-sim-state.md`](mech-sim-state.md) for the distillation of the
prior `phys-sim` harness that seeded this design.

---

## Install

```bash
uv sync                                 # or: pip install -e '.[dev]'
```

Optional extras:

| Extra        | Enables                                                          |
| ------------ | ---------------------------------------------------------------- |
| `traces`     | HDF5 trace evidence (`pip install -e '.[traces]'` adds h5py)     |
| `dashboard`  | Static HTML dashboards with Plotly                               |
| `media`      | Planar MP4 + thumbnail rendering (adds matplotlib)               |
| `dev`        | pytest                                                           |

`ffmpeg` is picked up from `PATH` when present; nothing else is needed
for video encoding.

PyChrono is **not** wired as a real physics oracle in this repo — see
[Chrono status](#chrono-status) below.

---

## Quickstart

### 1. Score one submission

```bash
mech-bench evaluate \
    --task tasks/fourbar_path_t001 \
    --submission tasks/fourbar_path_t001/reference_solution
```

The output is a JSON report. The exit code is 0 only when the
evaluation is valid, the hard gate passes, and the dense score is
positive.

### 2. Score under one mode (`fast`, `oracle`, `final`)

```bash
mech-bench evaluate \
    --task path/to/task --submission path/to/sub \
    --mode fast
```

* `fast` — runs only the probes flagged in `[modes.fast]` of the task's
  `eval_config.toml`. Picks the cheapest adapter the dispatcher finds.
* `oracle` — runs the probes flagged in `[modes.oracle]`. Intended for
  the high-fidelity (contact / dynamics) verifier. Without a real
  Chrono runner, oracle mode either uses the
  [fake contact oracle](#fake-contact-oracle) (when test mode is on)
  or surfaces `capability_unavailable`.
* `final` — runs fast + oracle, computes agreement metrics
  (`ratio_delta_pct`, `penetration_delta_mm`, `lockup_agreement`,
  `contact_presence_agreement`), and gates on **both** passing plus
  agreement.

### 3. Compact reward for an agent loop

```bash
mech-bench rlvr-eval \
    --task path/to/task \
    --submission path/to/sub \
    --mode fast \
    --report-dir runs/attempt_001
```

Emits a small JSON blob with `reward`, `hard_gate_passed`,
`evaluation_valid`, `dense_score`, `public_feedback`, `metrics`,
`scalar_channels`, `retry_suggestions`, and `oracle_is_synthetic`. Or
call it from Python:

```python
from mech_bench.rlvr import evaluate_for_rlvr

result = evaluate_for_rlvr(task_dir, submission_dir, mode="fast")
agent.update(result.reward)
```

`reward` is **forced to 0.0** whenever `evaluation_valid=False` so the
agent can never earn credit on a run the verifier itself couldn't
trust.

### 4. Generate and run a benchmark suite

```bash
mech-bench generate-suite --out suite/ --count-per-family 3 --seed 0
mech-bench run-suite      --tasks suite/ --report-dir runs/ --eval public
mech-bench check-negative-controls --tasks suite/
```

After `run-suite`, a `benchmark_summary.json`, a
`benchmark_dashboard_payload.json`, and (with `plotly` installed) a
`benchmark_dashboard.html` are written next to per-task report bundles.

### 5. Render a preview video for the demo

```bash
mech-bench video \
    --report-dir runs/attempt_001 \
    --out runs/attempt_001/preview.mp4 \
    --view planar
```

With `matplotlib` installed, this writes frames + `thumbnail.png` and
encodes `preview.mp4` (when ffmpeg is on `PATH`). Without
`matplotlib`, you get a structured warning, no crash. The full
evaluation pipeline already calls this renderer automatically when the
optional deps are present.

---

## Architecture in one diagram

```
                         ┌────────────────────────┐
   submission/design.py  │  isolated subprocess   │   trusted side
   ───────────────────▶  │  build_design(out_dir) │   ─────────────────────
                         └────────────┬───────────┘
                                      │ DesignIR (JSON)
                                      ▼
                         ┌────────────────────────┐
                         │  validation.py         │   missing parts,
                         │  (path policy, schema, │   path escapes,
                         │   NaN guards, …)       │   bad mass-properties
                         └────────────┬───────────┘
                                      │
                         ┌────────────▼───────────┐
                         │  Dispatcher            │   for each probe, pick
                         │  (probe ↔ adapter)     │   cheapest adapter whose
                         │                        │   capabilities cover it
                         └────────────┬───────────┘
                                      │
              ┌───────────────────────┼────────────────────────┐
              ▼                       ▼                        ▼
   ┌──────────────────┐   ┌────────────────────┐   ┌────────────────────┐
   │ planar_kinematics│   │ fake_contact_oracle│   │  chrono_contact    │
   │ (analytic, μs)   │   │ (synthetic, test)  │   │  (skeleton; needs  │
   │                  │   │                    │   │   real runner)     │
   └────────┬─────────┘   └─────────┬──────────┘   └─────────┬──────────┘
            │                       │                         │
            └───────────┬───────────┴────────────┬────────────┘
                        │ SimOutput (canonical)  │
                        ▼                        ▼
              ┌─────────────────────┐  ┌────────────────────┐
              │  Probes             │  │  TraceData (HDF5)  │
              │  (verifiable units) │  │  + dashboard JSON  │
              └─────────┬───────────┘  └────────────────────┘
                        │ ProbeResult
                        ▼
              ┌─────────────────────────────────────────────┐
              │  Evaluator: hard-gate + dense score          │
              │  → EvalReport (tier_results, class_metrics,  │
              │    general_metrics, feedback, traces, mp4)   │
              └─────────────────────────────────────────────┘
```

The five core abstractions are documented in `ARCHITECTURE.md`:
`DesignIR`, `TaskSpec` + `EvalConfig`, `Probe`, `SimAdapter`,
`Failure`.

---

## What's shipping

### Probes (`mech_bench/probes/`)

| Probe                  | Capabilities required                       | Purpose                                                  |
| ---------------------- | ------------------------------------------- | -------------------------------------------------------- |
| `dof_grubler`          | none (pure topology)                        | Mobility check via Grübler–Kutzbach.                     |
| `required_ports`       | none                                        | Required-port and grounded-port enforcement.             |
| `path_trace_chamfer`   | `planar_kinematics`                         | Coupler-trace ↔ target chamfer distance.                 |
| `port_velocity_ratio`  | any kinematic adapter                       | Measured ω_out / ω_in vs declared ratio.                 |
| `swept_collision`      | `mesh_overlap`                              | Max penetration over a joint sweep.                      |
| `contact_engagement`   | `contact_forces`                            | Required pair carries ≥ RMS force threshold.             |
| `lockup`               | `planar_kinematics`                         | Output never moves under input drive.                    |
| `torque_load_trial`    | rigid-body dyn + drives + loads             | Load test: motion, power balance, torque ripple.         |
| `printability_dfam`    | `mesh`                                      | Min wall, max overhang per process.                      |
| `safety_factor`        | `safety_factor`                             | FOS check against allowable stress.                      |
| `analytic_param_check` | none                                        | Closed-form parameter relationships (ratios, formulas).  |

Each probe declares `capabilities_required`. The evaluator runs a
probe only when an adapter is registered whose advertised
capabilities cover the requirement.

### Adapters (`mech_bench/adapters/`)

| Adapter                | Capabilities provided                                                    | Cost tier | Notes                                            |
| ---------------------- | ------------------------------------------------------------------------ | --------- | ------------------------------------------------ |
| `planar_kinematics`    | planar_kinematics, path_trace, dof_detection, pose_traces                | 0 (μs–ms) | Always registered.                               |
| `fake_contact_oracle`  | rigid_body_dynamics, contact_forces, joint_constraints, motor_drives, load_torques, pose_traces, mesh_overlap, planar_kinematics | 50 / 1000 | Registers only under `MECH_BENCH_USE_FAKE_ORACLE=1`. Synthetic — never claims to be a physical oracle. |
| `chrono_contact`       | same as above + mesh contact                                             | 100       | Registers only when both `pychrono` and `_chrono_impl` are importable. Skeleton in this repo. |

Adapter outputs share one canonical shape (`SimOutput`): `time_s`,
`joint_positions`, `joint_velocities`, `body_poses`, `contact_forces`,
`penetration`, `scalar_metrics`, `metadata`. Probes never look at the
adapter name — they read the canonical keys.

### Task families (`mech_bench/generators/`)

The procedural generator emits four tiers, ten families:

| Tier                    | Families                                                                  |
| ----------------------- | ------------------------------------------------------------------------- |
| **artifact_static**     | `static_fit_bracket`, `shaft_collar_clearance`, `simple_hinge_fit`        |
| **planar_kinematics**   | `fourbar_path`, `slider_crank_stroke`                                     |
| **transmission_analytic** | `spur_gear_ratio_analytic`, `rack_pinion_conversion`, `belt_pulley_ratio` |
| **contact_dynamics**    | `contact_gear_pair_stub`, `cycloidal_lowN_stub`                           |

Each generated task ships a reference solution that should pass, a
set of negative controls that should fail with specific codes, and an
`expected_failures.json` the runner verifies.

### Failure grammar (`mech_bench/feedback.py`)

A closed enum so an RL agent can pattern-match on `code` without
parsing prose:

```
invalid_artifact           missing_port              schema_error
invalid_mass_properties    wrong_mobility            wrong_topology
wrong_ratio                path_error                collision
insufficient_clearance     missing_contact           lockup
excessive_penetration      excessive_torque_ripple   power_balance_error
insufficient_safety_factor unprintable               simulator_divergence
capability_unavailable
```

Every `Failure` carries `code, severity, message, metric, observed,
target, where, public_hint, private_trace`. Public payloads strip the
private bag.

### SceneGraph IR (`mech_bench/scene_graph.py`)

A simulator-facing IR — `SceneBody`, `SceneJoint`, `SceneMotor`,
`SceneLoad`, `SceneContactPair`, `ScenePort`, `SceneGraph`. Built from
a DesignIR + TaskSpec + EvalConfig via
`build_scene_graph_from_design_ir`, which emits structured preflight
failures for missing bodies referenced by contact pairs, motors
applied to nonexistent joints, etc.

The SceneGraph is what a real Chrono / MuJoCo / Drake adapter would
consume — see [Chrono status](#chrono-status).

### Tier & task-class metrics (`mech_bench/metrics.py`)

Every report now carries three concentric scoring views:

- **General metrics** — `verified_score`, `dense_score`, `pass_at_1`,
  `hard_gate_pass_rate`, `oracle_pass_rate`, `n_probes`, `runtime_s`.
- **Tier channels** — `artifact`, `geometry`, `kinematics`,
  `collision`, `contact`, `dynamics`, `structural`,
  `manufacturability`, `robustness`.
- **Task-class channels** — `linkage_path_score`,
  `gearbox_ratio_score`, `contact_health_score`, `load_trial_score`,
  `printability_score`, `safety_factor_score`.

A probe's tier and class are derived from its `type` by default; an
eval config can override per-probe via `tier = …` and
`class_metric = …` on `[[probes]]`.

### Evidence bundle per run

`mech-bench evaluate … --report-dir out/` writes:

```
out/
  scorecard.json              # full report (trusted side)
  scorecard.public.json       # public-redacted view
  metrics.json                # flat numeric dict
  feedback.public.json        # public failure cards
  dashboard_payload.json      # canonical dashboard input
  dashboard.html              # static HTML (plotly required)
  traces.h5                   # HDF5 evidence (h5py required)
  media_manifest.json         # paths to media artifacts
  thumbnail.png               # mid-cycle frame (matplotlib required)
  preview.mp4                 # encoded video (ffmpeg required)
  frames/                     # PNG frame sequence (fallback when no ffmpeg)
```

Missing optional dependencies degrade gracefully — the manifest
records what was produced and what was skipped with a structured
warning.

---

## Chrono status

PyChrono integration is intentionally **skeleton-only** in this repo.
`mech_bench/adapters/chrono_contact.py` advertises the conceptual
capabilities (rigid-body dynamics, contact forces, joint constraints,
motor drives, load torques, pose traces, mesh overlap) and supports
subprocess execution, but it **only registers** when both `pychrono`
*and* a vendor-out `_chrono_impl` module are importable.

Check the state of the backend:

```python
from mech_bench.adapters.chrono_contact import chrono_diagnostic
chrono_diagnostic()
# {
#   "adapter": "chrono_contact",
#   "status": "unavailable",
#   "pychrono_importable": True,
#   "_chrono_impl_importable": False,
#   "runner_status": "skeleton_only",
#   "reason": "...",
# }
```

When the real runner is unavailable, tasks that require contact
capabilities correctly surface `capability_unavailable`. There is no
silent pass.

The full phys-sim mesh-contact runner (V-HACD, contact-pair
instrumentation, pose sampling, solver / timestepper choices,
worst-violation tracking) is a substantial port and lives on a
separate follow-up: drop a real `_chrono_impl.py` into the adapters
package and the dispatcher picks it up automatically.

### Fake contact oracle

For tests, demos, and procedurally generated Tier-3 tasks, the
deterministic
[`fake_contact_oracle`](mech_bench/adapters/fake_contact_oracle.py)
acts as the test-time stand-in. Enable it explicitly:

```bash
export MECH_BENCH_USE_FAKE_ORACLE=1
mech-bench evaluate --task path/to/contact_task --submission path/to/sub
```

Every report produced with the fake oracle is **clearly labeled**:

```json
{
  "metadata": {
    "simulator": "fake_contact_oracle",
    "is_physical_oracle": false,
    "oracle_is_synthetic": true,
    "trust_level": "synthetic_test_or_demo"
  }
}
```

The `EvalReport.oracle_is_synthetic` flag and the RLVR
`oracle_is_synthetic` field propagate the same signal so downstream
training code can decide whether to use the reward for learning or
only for development.

---

## Layout

```
mech_bench/
  schema.py             DesignIR, TaskSpec, EvalConfig, ProbeSpec, ProbeResult,
                        ModeConfig, FinalModeConfig, EvalReport
  feedback.py           FailureCode (closed grammar), Severity, Failure
  evaluator.py          hard-gate + dense reward composition + report bundle
  validation.py         DesignIR / submission validation (path policy, NaN)
  submission_worker.py  out-of-process build_design() runner
  scene_graph.py        SceneBody/SceneJoint/SceneMotor/SceneContactPair/…
  metrics.py            tier + task-class metric aggregation
  modes.py              fast / oracle / final modes + agreement metrics
  rlvr.py               compact RLVR reward API
  probes/               built-in probes (capability-tagged)
  adapters/             planar_kinematics, fake_contact_oracle, chrono_contact
  generators/           procedural task families (Tiers 1–3)
  rendering/            planar_renderer (matplotlib), ffmpeg shim
  benchmark.py          run-suite, aggregate summaries, negative controls
  dashboard.py          static HTML for one run + suite-level dashboard
  dashboard_payload.py  canonical JSON the dashboard / video consume
  media.py              MediaManifest (thumbnail / preview / frames)
  traces.py             HDF5 trace evidence
  __main__.py           mech-bench CLI

tasks/                  curated example tasks (fourbar_path_t001, …)
tests/                  pytest suite
```

---

## CLI reference

```text
mech-bench evaluate                   Score one submission
                                      Flags: --task, --submission, --scratch,
                                      --out, --report-dir, --full,
                                      --mode {fast|oracle|final}, --allow-partial

mech-bench rlvr-eval                  Compact RLVR-loop result
                                      Flags: --task, --submission, --mode,
                                      --report-dir, --out

mech-bench list-probes                Show registered probe types
mech-bench list-adapters              Show registered adapters and cost tiers

mech-bench generate-suite             Procedurally write a benchmark suite
                                      Flags: --out, --count-per-family, --seed,
                                      --families, --difficulty

mech-bench run-suite                  Score every task in a suite
                                      Flags: --tasks, --submissions, --negative,
                                      --report-dir, --eval {public|hidden|both},
                                      --families

mech-bench check-negative-controls    Verify expected_failures.json
                                      Flags: --tasks, --eval

mech-bench video                      Render an MP4 preview for one run
                                      Flags: --report-dir, --out, --fps, --view

mech-bench package-run                Normalize a packaged run directory
                                      Flags: --report-dir
```

---

## Programmatic usage

Score a submission and inspect the report:

```python
from mech_bench.evaluator import evaluate_with_evidence

evidence = evaluate_with_evidence(
    task_dir="tasks/fourbar_path_t001",
    submission_dir="tasks/fourbar_path_t001/reference_solution",
)
report = evidence.report
print(report.score, report.hard_gate_passed)
for r in report.probe_results:
    print(r.probe_id, r.passed, r.score, r.metrics)
```

Run the agent-loop API:

```python
from mech_bench.rlvr import evaluate_for_rlvr

rlvr = evaluate_for_rlvr(task_dir, sub_dir, mode="fast")
loss = -rlvr.reward
hints = rlvr.retry_suggestions   # short, code-keyed hints for the agent
chans = rlvr.scalar_channels     # per-tier + per-class scalars (for shaping)
```

Build a SceneGraph from a DesignIR for a custom simulator:

```python
from mech_bench.evaluator import load_task, load_submission
from mech_bench.scene_graph import build_scene_graph_from_design_ir

task, cfg = load_task("tasks/fourbar_path_t001")
ir = load_submission("tasks/fourbar_path_t001/reference_solution",
                      scratch_dir="/tmp/scratch")
graph = build_scene_graph_from_design_ir(ir, task, cfg)
assert graph.ok, graph.preflight_failures
```

---

## Tests

```bash
pytest                        # base suite
MECH_BENCH_USE_FAKE_ORACLE=1 pytest tests/test_modes_and_rlvr.py
```

The full suite is 179 tests passing today, with a handful gated on
optional deps (matplotlib / plotly / h5py) that skip cleanly when the
package is missing. No test depends on a real Chrono runtime.

---

## What this is *not*

- **Not** a CAD kernel. Geometry refs are optional; analytic probes
  work on the IR alone.
- **Not** a physics engine. Adapters delegate to real engines (or
  fakes); the runtime stays generic.
- **Not** opinionated about RLVR algorithm. `evaluate_for_rlvr`
  returns scalars; how you use them is up to your trainer.
- **Not** carrying mechanism-specific validators. A check that's only
  meaningful for one mechanism class is implemented as a *probe with
  a config*, not a new code path in the evaluator.
