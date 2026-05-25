# Architecture — Mechanical Design RLVR Benchmark Runtime

> Sister doc: `mech-sim-state.md` (distillation of the prior
> `phys-sim` mechanism-specific harness, with empirical findings).
> This doc proposes the architecture of the **generic runtime** we
> are building here.
>
> Long-horizon note:
> `docs/high-fidelity-simulation-roadmap.md` is the canonical decision
> record for the full high-fidelity simulation stack. This architecture
> document describes the current evaluator/control plane; the roadmap
> defines the CAD, Chrono, FEA, calibration, and V&V work needed to make
> physical simulation claims credible.

## The inversion

The prior harness (`phys-sim/mech_harness`) put mechanism knowledge in
problem plugins. Each new mechanism class required a new
`ProblemPlugin` subclass that knew how to validate that mechanism.
The simulator core had to grow with each new mechanism family.

The runtime in this repo inverts that:

```
phys-sim:
    mechanism plugin   ──▶  validator knows what to check
                            (new mechanism = new plugin)

mech_bench:
    task spec
    + eval config      ──▶  generic evaluator selects probes
                            and simulator adapters from registries
                            (new mechanism = new task config + maybe
                             new probe types, but the runtime stays
                             generic)
```

The runtime knows about a small set of universals — bodies, joints,
ports, drives, contact pairs, swept-volume probes, force probes,
manufacturability constraints, capability tags — and **nothing about
"cycloidal" or "four-bar."** Mechanism knowledge lives in
task config + reference solutions + fixtures, where it can be
authored, versioned, and shared without touching the runtime.

## Five core abstractions

### 1. `DesignIR` — what the agent submits

A JSON-serializable design intermediate. The agent's `design.py`
exposes:

```python
def build_design(out_dir: Path) -> dict:  # DesignIR
    ...
```

The IR carries `parts` (with mass / COM / inertia, optionally
referencing STEP geometry), `joints` (revolute / prismatic / fixed /
contact-pair), `ports` (named frames the task contract references),
and `params` (task-specific data the agent declares). Geometry refs
must resolve under `out_dir`; the runtime never trusts paths the
agent submits outside the sandbox.

### 2. `TaskSpec` + `EvalConfig` — what the runtime evaluates

The task contract is two TOML files:

- `task.toml` — what the task is asking for. Stable; the prompt
  references it. Defines required ports, expected mobility,
  envelope, objectives, and points to fixtures.
- `eval_config.toml` — how scoring happens. Defines which probes
  run, their weights, hard-gate vs dense, public-vs-hidden metric
  visibility. Mostly authored by the benchmark maintainer; agents
  see only the prompt + public probe subset.

Splitting these two lets the same task be re-scored under different
eval configs (cheap vs expensive, training vs eval) without
re-authoring the task.

### 3. `Probe` — the verifiable unit of evaluation

A `Probe` is a function

```python
run(ir: DesignIR, sim_outputs: dict, config: dict) -> ProbeResult
```

that emits a `ProbeResult { passed, metrics, failures, artifacts }`.
Probes declare their `capabilities_required` (planar kinematics,
contact forces, FEA, …). The evaluator only runs a probe when the
selected simulator adapter advertises matching capabilities.

Shipping today:

| Probe | Capabilities | Purpose |
|---|---|---|
| `dof_grubler` | none (pure topology) | Mobility (planar / spatial Grübler-Kutzbach) |
| `required_ports` | none | Required ports exist, kinds match, grounded checks |
| `analytic_param_check` | none | Compare a dotted IR path (`params.declared_ratio`, etc.) to an expected value with `eq` / `ge` / `le` comparators |
| `path_trace_chamfer` | `planar_kinematics` | Compare a moving frame's trace to a target CSV via Chamfer distance |
| `port_velocity_ratio` | any kinematic adapter | Measured ω_out / ω_in vs target |
| `lockup` | `planar_kinematics` | Output never moves while input is driven |
| `swept_collision` | `mesh_overlap` | Maximum penetration over a joint sweep |
| `contact_engagement` | `contact_forces` | Required pair carries ≥ F_min RMS and ≥ engagement fraction |
| `torque_load_trial` | rigid dyn + drives + loads | Motion under prescribed input speed and output load; power balance, torque ripple |
| `printability_dfam` | `mesh` | Min wall, max overhang per process |
| `safety_factor` | `safety_factor` | FOS check against allowable stress |

The bar to add a new probe is "express it as configuration." A probe
should be the same shape no matter which mechanism the task is.

### 4. `SimAdapter` — capability-tagged simulator

Each adapter advertises a `Capability` set. The dispatcher picks the
cheapest adapter whose advertised capabilities cover the union of
the active probes' requirements. The task author does not name an
adapter; the runtime selects one based on capabilities.

Shipping today:

| Adapter | Capabilities | Cost | Registration |
|---|---|---|---|
| `planar_kinematics` | `planar_kinematics`, `path_trace`, `dof_detection`, `pose_traces` | 0 (μs–ms) | always registered |
| `fake_contact_oracle` | `rigid_body_dynamics`, `contact_forces`, `joint_constraints`, `motor_drives`, `load_torques`, `pose_traces`, `mesh_overlap`, `planar_kinematics` | 50 / 1000 | **explicit opt-in only**: `[adapters.fake_contact_oracle] enabled = true`, mode-level `forced_adapter = "fake_contact_oracle"`, probe-level `adapter = "fake_contact_oracle"`, or env var `MECH_BENCH_USE_FAKE_ORACLE=1` / `MECH_BENCH_TEST_MODE=1`. Reports tag `oracle_is_synthetic = true`. |
| `chrono_contact` *(skeleton-only)* | `rigid_body_dynamics`, `contact_forces`, `joint_constraints`, `motor_drives`, `load_torques`, `pose_traces`, `mesh_overlap` | 100 (minutes) | registers only when both `pychrono` and `mech_bench.adapters._chrono_impl` are importable; see `docs/future_chrono_oracle.md`. |

When `chrono_contact` lands, no probe code changes. The dispatcher
picks it automatically for any task whose probes require contact
forces. The fake oracle never satisfies a probe by accident — the
evaluator filters it out unless the active eval config (or env var)
explicitly opts in.

### 5. `Feedback` — structured failures, not strings

Every failure is a `Failure { code, severity, message, metric,
observed, target, where, public_hint, private_trace }`. The codes
come from a closed grammar (the `FailureCode` enum) shared across
all probes, so a generation agent can pattern-match and self-repair
without parsing English. The grammar:

```
invalid_artifact            missing_port
invalid_mass_properties     wrong_mobility
wrong_topology              wrong_ratio
path_error                  collision
insufficient_clearance      missing_contact
lockup                      excessive_penetration
excessive_torque_ripple     power_balance_error
insufficient_safety_factor  unprintable
simulator_divergence        capability_unavailable
```

Public hints are shown to the agent; private traces (HDF5 keys,
hidden-trial pointers) stay on the trusted side.

## Scoring: hard gate + dense reward

```
score = 0                            if hard_gate fails
score = Σ_i w_i · s_i  ∈ [0, 1]      otherwise
```

The hard gate is the set of probes (and pipeline steps) that must
all `passed=True` before any dense reward is computed. Typical
contents: solid validity, schema sanity, no path escape, required
ports, no NaN metrics, no preflight assembly failures.

The dense layer is a weighted convex combination of probe-specific
score functions `s_i ∈ [0, 1]` (each probe maps its metrics into
a normalized score). Weights are declared per task in
`eval_config.toml`.

This separation matters for two reasons:

1. **RLVR signal.** Below the gate, the agent gets dense
   diagnostics with metric deltas (so it can repair) but a reward
   of 0 (so it cannot earn credit for evaluating an invalid design).
2. **Verifier-gaming resistance.** A probe whose proxy metric is
   easy to satisfy in a physically-meaningless way (e.g. "ratio is
   correct, but the contact pair is fake") still costs the agent
   because the contact-engagement probe under the same hard gate
   will fail.

## Procedural task generation

Tasks live in `tasks/<family>_<id>/` as flat files. The directory
layout is the contract:

```
tasks/<id>/
    prompt.md                  # natural-language task statement (agent sees)
    task.toml                  # structured requirements (agent sees abridged)
    eval_config.toml           # default probe pipeline + visibility
    eval_config.public.toml    # public split (looser, agent-visible)
    eval_config.hidden.toml    # hidden split (tighter, for generalization gap)
    fixtures/                  # target paths, envelopes, etc.
    reference_solution/        # known-good design.py (hidden during eval)
    negative_solutions/<case>/ # one design.py per negative control
    expected_failures.json     # negative-control expectations (hidden)
    metadata.json              # family, tier, seed, difficulty
```

A task generator emits these files programmatically. The default
suite registered under `mech_bench/generators/benchmark_suite.py`
ships **four tiers, 50 families** today:

| Tier (`metadata.tier`)   | Adapter dependence                       | Families |
|--------------------------|------------------------------------------|----------|
| `artifact_static`        | none                                     | 13       |
| `planar_kinematics`      | `planar_kinematics`                      | 12       |
| `transmission_analytic`  | none (declared-ratio checks)             | 13       |
| `contact_dynamics`       | `fake_contact_oracle` (opt-in) or real Chrono once ported | 12 |

Tasks under `artifact_static` and `transmission_analytic` evaluate
with no simulator. `planar_kinematics` tasks need only the
always-on `planar_kinematics` adapter. `contact_dynamics` tasks are
test/demo tasks: two surface `capability_unavailable` until a real
Chrono runner ships; the other ten explicitly enable the synthetic
`fake_contact_oracle` and tag reports as synthetic.

The complete suite materializes into `tasks/` (seed 1) and is
exercised end-to-end by `mech-bench check-negative-controls --tasks
tasks`. Add a new family by writing one `TaskGenerator` subclass and
appending it to `SUITE`; nothing else in the runtime changes.

## What is NOT in this runtime

Deliberately out of scope:

- **Mechanism-specific validators.** If a check is only meaningful
  for one mechanism class (e.g. "Hertz pressure on cycloid ring
  pins"), it should be implemented as a *probe with a config*
  (e.g. `hertz_pressure_at_contact` parameterized by which contact
  pair and which material), not as a cycloidal-specific function.
- **Trust model decisions baked into runtime.** Path policy and
  attestation logic are stage-2 work; the runtime exposes hooks for
  them but does not assume a particular sandbox.
- **A new CAD kernel.** We use STEP/build123d when geometry matters;
  for analytic probes, the IR's parametric description suffices.

## How this connects to phys-sim

The phys-sim repo (see `mech-sim-state.md`) has working code we will
port piecewise as probes/adapters mature:

| phys-sim asset | Where it goes here |
|---|---|
| `mech_harness/builder/run_builder.py` | `mech_bench/submission_worker.py` (subprocess-isolated `build_design()` invocation, ported) |
| `mech_harness/validators/assembly.py` (Grübler) | `mech_bench/probes/dof_grubler.py` (ported) |
| `mech_harness/validators/cycloidal.py` (Hertz, FOS) | `mech_bench/probes/safety_factor.py` + parameterized contact probes (in progress) |
| `mech_harness/simulators/_chrono_mesh_runner.py` | `mech_bench/adapters/chrono_contact.py` (skeleton + diagnostic only; vendor `_chrono_impl.py` to enable) |
| `mech_harness/standards/sarif.py` | `mech_bench/feedback.py` (ported in spirit) |
| `mech_harness/standards/hdf5_traces.py` | `mech_bench/traces.py` (ported; per-adapter groups under `/adapters/<name>/`) |

The four-bar task originally shipped in this repo is now one of 50
generated families; the runtime stays generic and the inversion holds
across all of them.
