# Architecture — Mechanical Design RLVR Benchmark Runtime

> Sister doc: `mech-sim-state.md` (distillation of the prior
> `phys-sim` mechanism-specific harness, with empirical findings).
> This doc proposes the architecture of the **generic runtime** we
> are building here.

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

Shipping today (more land iteratively):

| Probe | Capabilities | Purpose |
|---|---|---|
| `dof_grubler` | none (pure topology) | Mobility (planar / spatial Grübler-Kutzbach) |
| `path_trace_chamfer` | `planar_kinematics` | Compare a moving frame's trace to a target CSV via Chamfer distance |
| `port_velocity_ratio` *(planned)* | any kinematic adapter | Measured ω_out / ω_in vs target |
| `swept_collision` *(planned)* | `mesh_overlap` | Maximum penetration over a joint sweep |
| `contact_engagement` *(planned)* | `contact_forces` | Required pair carries ≥ F_min RMS |
| `torque_load_trial` *(planned)* | `rigid_dynamics + drives + loads` | Output motion under prescribed input speed and load torque |
| `printability_dfam` *(planned)* | `mesh` | Min wall, max overhang |

The bar to add a new probe is "express it as configuration." A probe
should be the same shape no matter which mechanism the task is.

### 4. `SimAdapter` — capability-tagged simulator

Each adapter advertises a `Capability` set. The dispatcher picks the
cheapest adapter whose advertised capabilities cover the union of
the active probes' requirements. The task author does not name an
adapter; the runtime selects one based on capabilities.

Shipping today:

| Adapter | Capabilities | Cost |
|---|---|---|
| `planar_kinematics` | `planar_kinematics`, `path_trace` | μs–ms |
| `chrono_contact` *(stub; phys-sim has the real one)* | `rigid_body_dynamics`, `contact_forces`, `joint_constraints`, `motor_drives`, `load_torques`, `pose_traces` | minutes |

When `chrono_contact` lands, no probe code changes. The dispatcher
picks it automatically for any task whose probes require contact
forces.

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
    prompt.md              # natural-language task statement (agent sees)
    task.toml              # structured requirements (agent sees abridged)
    eval_config.toml       # probe weights & visibility (mostly hidden)
    fixtures/              # target paths, envelopes, etc.
    reference_solution/    # known-good design.py (hidden during eval)
    expected_failures.json # negative controls; what should fail (hidden)
```

A task generator emits these files programmatically. Tier-1 (static
fit) tasks can be generated with no simulator dependency. Tier-2
(planar 1-DOF) tasks need only the `planar_kinematics` adapter
shipping here today. Tier-3+ (transmission, contact, integrated)
will need `chrono_contact` ported in from phys-sim.

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
| `mech_harness/builder/run_builder.py` | Sandboxed `build_design()` invocation |
| `mech_harness/validators/assembly.py` (Grübler) | `mech_bench/probes/dof_grubler.py` (already ported) |
| `mech_harness/validators/cycloidal.py` (Hertz, FOS) | Probes parameterized by contact pair |
| `mech_harness/simulators/_chrono_mesh_runner.py` | `mech_bench/adapters/chrono_contact.py` |
| `mech_harness/standards/sarif.py` | Feedback grammar (already ported in spirit) |
| `mech_harness/standards/hdf5_traces.py` | `mech_bench/traces.py` (planned) |

The four-bar task shipping in this PR is the smallest non-trivial
demonstration that the inversion holds.
