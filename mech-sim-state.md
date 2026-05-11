# Mechanical Simulation for RLVR — State of the Build

> **Scope.** This document describes the code state of the **`phys-sim`**
> repository (a worktree of `cybernetic-physics`) at commit
> **`ace9ba925bb081dbb170043c93fe4ab72b449f2a`** — short
> **`ace9ba9`** — `docs(mech-harness): 3-axis design-space sweep
> results` (2026-05-11 05:16:37 -0400). All relative path references
> below are anchored at `/home/freiza/phys-sim/`. The findings here are
> a distilled summary, not the source of truth — if the code moves,
> re-derive against the new HEAD. This file lives outside the worktree
> (in `~/mech_sim/`) intentionally, so it is not version-controlled by
> the same repo it describes.
>
> **Sources read.** Code under `mech_harness/`, `cps/`, `cps_eval/`,
> `cps_newton_actuators/`; design docs `cyber-physical-sim.md`,
> `qdd-rl.md`, `unified-codesign.md`, `evaluator-harness.md`; status
> notes `todo.md`, `MUSTDO.md`, `sum-03.md`, `cps-followup.md`, and
> `simulators/CHRONO_MESH_PLAN.md`.

The repo is reaching for an end-to-end *cyber-physical* RLVR environment:
an agent proposes a 3D-printed quasi-direct-drive (QDD) actuator and the
mechanical assembly that uses it; the harness verifies the design's
physical credibility and returns a verifiable reward. Two complementary
stacks have shipped Phase 1+2 of that vision; the high-fidelity
oracle layers are scaffolded but not yet load-bearing.

---

## 1. Two-stack architecture

The work splits cleanly into two halves that meet at a shared notion of
"trusted design intermediate representation" (DesignIR / genome).

| Stack | Lives in | Purpose | Trust model |
|---|---|---|---|
| **`mech_harness/`** | trusted validation harness | Gate AI-generated *mechanical assemblies* (cycloidal gearboxes, four-bar linkages) before they earn a passing reward. CAD-first, attestation-first. | Pessimistic: assume the agent will lie. Re-derive everything from STEP. |
| **`cps/` + `cps_eval/`** | fast Tier-0 evaluator + Gym env | Score *actuator genomes* (EM + thermal + cycloidal + structural) in <1 ms so an RL/CMA-ES outer loop can co-design morphology and parameters. | Cooperative: the genome is a structured artifact, no adversarial paths yet. |

`cps_newton_actuators/` is a third, smaller staging area: diagnostic
scripts for a planned Newton-on-Warp scene (penetration, contact, ratio
measurement). It is not yet integrated into either stack as a load-bearing
component, but `mech_harness/simulators/kamino_adapter.py` reuses one of
its models (`models.kamino_cycloid_n2`) as the fast simulator.

---

## 2. `mech_harness` — trusted validation pipeline

### 2.1 Execution model

`mech_harness/server.py::validate()` is the synchronous trusted entry
point. A newer standards-first path lives at
`mech_harness/evaluation/pipeline.py::evaluate_package()` (BagIt → USD →
STEP → SARIF), but is not yet the default.

Each call runs an ordered list of **gates**, each producing a
`GateResult { passed, skipped, metrics, failures[] }`. One hard failure
halts and downstream gates are marked skipped. Gate composition by
**tier**:

| Tier | Gates added | Sim used | Wall-time target |
|---|---|---|---|
| `fast` | build → schema → artifacts → geometry_artifacts → assembly → cycloidal → kamino → contact_engagement | Kamino only | 1–2 s |
| `oracle` | + chrono → agreement | Kamino + Chrono | 5–10 s |
| `final` | + gate.final (AND of all upstream) | both | 10–15 s |

### 2.2 What each gate actually checks

- **build** — Loads the agent's `design.py` via `importlib.util` (not via
  `sys.modules`), calls `build_design(out_dir)` where `out_dir` is a
  trusted scratch path. Path-policy verifies every geometry reference in
  the returned DesignIR resolves under `build_root`, blocking escapes.
- **schema** — `schema_version == "design_ir.v1"`, no NaN/Inf, every
  part has positive mass, symmetric 3×3 inertia, principal moments
  positive *and* triangle-inequality (physically realizable).
- **artifacts** — Every STEP/visual mesh/collision mesh/analysis mesh
  referenced by a part actually exists on disk.
- **geometry_artifacts** — (Build123d required.) Imports each STEP,
  asserts volume > 0, recomputes density and matches the agent's
  declared mass ±25% against a material table. Catches "agent declared
  0.4 kg but the STEP is a paper-thin shell."
- **assembly** — Spatial Grübler-Kutzbach DOF must match the task's
  declared mobility; required ports (`housing_frame`, `input_port`,
  `output_port`) must exist and be grounded; required contact pairs
  must be enabled (e.g., for cycloidal both disc↔ring-pins and
  disc↔output-pins).
- **cycloidal** — Analytical oracle on geometry-derived numbers
  (`design_ir.metadata.cycloidal`, populated by the trusted builder):
  - `ring_pin_count = disc_lobe_count + 1` (within 1% tolerance)
  - `eccentricity > 0`
  - **Output-hole orbit clearance**:
    `hole_d ≥ pin_d + 2·e + 2·clearance` — this is the C3 historical
    bug detector.
  - Ring-pin Hertz pressure < 1500 MPa (`p_max = √(F·E*/(π·L·R))`).
  - Output-pin von-Mises FOS ≥ 1.5 (cantilever bending + shear).
  - Input-shaft torsion FOS ≥ 1.5 (`τ = T·c/J`).
- **kamino** — Drives `cps_newton_actuators.models.kamino_cycloid_n2`
  (a Newton+Warp scene). Multi-trial: 4 seeds × varied input speed
  (60–600 rpm), output load (0–5 Nm), friction (0.05–0.09), 0.4 s
  rollout. Emits `ratio_observed`, `torque_ripple_pct`,
  `power_balance_error_pct`, `max_penetration_mm`, per-pair
  `contact_force_rms_N`.
- **contact_engagement** — Each contact pair flagged
  `must_be_real_contact` must carry RMS force ≥ 0.5 N in simulation.
  Skipped if the sim didn't emit per-pair forces (it usually doesn't
  yet — see §5).
- **chrono** — Subprocess shim into a separate conda env (PyChrono is
  conda-only). v0 just enforces `ω_in + (N−1)·ω_out = 0` via
  `ChShaftsPlanetary` and reports it. **This is a tautology** — see §5.
- **agreement** — Kamino vs Chrono: `ratio_delta_pct ≤ 1.0`,
  `torque_ripple_delta_pct ≤ 5.0`, `power_balance_delta_pct ≤ 3.0`,
  `penetration_delta_mm ≤ 0.02`, lockup must agree.
- **final** — Logical AND across all upstream gates.

### 2.3 Trust model

Every emitted report is attested with a digest covering
`design_digest` (sha256 over the agent's `design/` tree), `task_digest`,
`harness_version`, `harness_git_sha`, and `report_digest`. The Stop hook
(`hooks/stop_requires_valid_report.py`) refuses to let the agent
terminate the session until a fresh attested report with `gates.final
== passed` exists and matches the current design digest. The pre-tool
hook (`hooks/pre_tool_guard.py`) blocks agent writes to `reports/`,
`schemas/`, `validators/`, `simulators/`, `.claude/*`, and the harness
itself. The agent's surface area is just `design/` and `scratch/`.

### 2.4 Standards stack

`mech_harness/standards/` formalizes the output contract:

- **`openusd.py`** — OpenUSD I/O, `mech:` namespace metadata,
  units/up-axis/timecode validation, `PartSummary` dataclass.
- **`step_ap242.py`** — STEP → mass / COM / volume / inertia via OCP,
  one `StepProperties` per part with an `is_valid` flag.
- **`tessellate.py`** — STEP → triangle mesh (0.3 mm tol, ~1 k tris per
  cycloid disc) for downstream contact runners.
- **`sarif.py`** — SARIF 2.1.0; public/private result redaction; rule
  registry of ~20 rules (`output_hole_too_small_for_orbit`,
  `invalid_pin_lobe_relation`, `nonpositive_principal_moment`, …).
- **`prov.py`** — Minimal W3C PROV-O JSON-LD lineage.
- **`bagit_pkg.py`** — BagIt validation (manifest checksums + required
  paths `data/scene/root.usda`, `data/cad/*.step`).
- **`hdf5_traces.py`** — *Scaffolded but not wired.* Intended for
  per-trial simulation traces.

A passing run emits, per run dir: `latest_summary.json`,
`findings.sarif` + `findings.public.sarif`, `trusted_metrics.usda`,
`trusted_mass_properties.usda`, `provenance.jsonld`,
`attestation.json`.

### 2.5 Problem plugins

`problems/base.py::ProblemPlugin` defines six lifecycle phases
(validate scene contract → derive problem metrics → create fast-sim
inputs → create oracle-sim inputs → postprocess → UI annotations).
Two problems ship:

- **`cycloidal_gearbox.py`** — emits ratio, hole clearance, Hertz
  pressure, output-pin FOS, torsion FOS, output-torque capacity; enforces
  port topology (`output_port` must point at `output_shaft`, not the
  internal carrier).
- **`four_bar_linkage.py`** — present, less mature. Tests exist in
  `tests/mech_harness/` but the linkage path is not exercised by the
  fixture suite.

### 2.6 Viewers / dashboards

`mech_harness/sim/` ships three actually-runnable tools:

- **`viewer.py`** — MeshCat-backed live WebGL viewer at port 7000;
  tessellates STEPs at 0.2 mm and exposes `set_part_transform()` for
  60 Hz kinematic playback. Used by `cycloidal_player.py`.
- **`snapshot.py`** — PNG renders (iso/top/side) from either analytic
  kinematic poses or recorded Chrono poses. Useful for review packets.
- **`dashboard.py`** — Plotly trial-trace dashboard. `mh dashboard
  --payload PATH --port 7100` or `--out report.html`.

---

## 3. `cps` — fast Tier-0 actuator evaluator + RL env

This is the inner-loop physics for the QDD actuator (Section 5 of
`cyber-physical-sim.md`). `cps.actuator.dynamics.evaluate_actuator(genome)`
returns `ActuatorMetrics` in sub-millisecond, single-threaded CPU.

### 3.1 Electromagnetic — `cps/em/dq_model.py`

Analytic dq-axis model for a coreless axial-flux PMSM with double rotor:

- Surface-charge approximation `B_z ≈ Br · t_m / (t_m + g_eff)` with
  `Br` from N42/N48/N52 lookup.
- Per-pole flux `Φ = B_z · A_pole · pole_arc_factor`.
- Coil linkage fundamentally sinusoidal: `λ_coil = N · Φ · sin(θ_e)`.
- Phase linkage `ψ_f = (2/3) · Σ λ_peak`; `Kt = 1.5 · p · ψ_f`,
  `Kv = 60/(2π Kt)`.
- Resistance from wire geometry with α_Cu = 0.00393.
- Outputs: `Kt`, `Kv`, `R_phase_ohm_20C`, `L_phase_H`, `B_gap_T`,
  pole pairs, coil counts, masses.
- **No saturation, no cogging harmonics, no temperature derating of Br,
  no eddy-current loss, no cross-coupling.** This is the fast lane.

### 3.2 Thermal — `cps/thermal/lptn.py`

Five-node LPTN: copper / potting / heat-spreader / housing / magnet
against ambient. Conductances from geometry × material conductivity
(copper 401, alu 167, NdFeB ~9, plastics 0.20–0.55 W/m·K), capacitances
from `m · c_p`. Convective H = 8 W/m²·K base (still air) with fin-factor
boost. Forward-Euler transient + steady-state solve. The reward uses
`continuous_current_allowed()`, which solves for the I_rms that lands
`T_copper` exactly at limit (typ. 100–110 °C) at steady state — the
**thermally-honest continuous current**.

### 3.3 Gearbox — `cps/gearbox/`

Cycloidal surrogate, not contact-mechanics:

- `ratio = ring_pin_count − 1`.
- Efficiency base 78%, derated linearly by clearance (k=0.20 / mm),
  eccentricity (k=0.04 / mm over 1 mm), and load (k=0.005 / Nm);
  clipped to [20%, 92%].
- No-load friction `τ_f ≈ 0.05 + 0.02 · ecc_mm` (motor side).
- Backlash `θ_b = clearance / r_eff`.
- Output stiffness via crude `K ≈ 200 · disc_t · (1.6 if dual)` Nm/rad.

### 3.4 Structural — `cps/mech/structural_surrogates.py`

Four conservative 1D/2D safety-factor checks, each `SF = σ_allow /
σ_working`:

1. Cycloid web tear-out between output-pin hole and disc OD.
2. Output-pin bending (cantilever) at the carrier root.
3. Magnet pocket wall hoop stress.
4. Bearing seat hoop stress (printed shrink fit).

Printed-material strengths from a `PrintMaterial` table (PETG, PET-CF,
NYLON, NYLON-CF, PC), **all derated by 0.55× globally** to account for
weak-axis (across-layer) loading. No FEA, no fiber-orientation projection
from slicer toolpaths.

### 3.5 Multi-body dynamics — `cps/mbd/`

A custom NumPy planar 4-DOF serial chain (shin/thigh/torso/arm) for
closed-loop tests:

- `forward_dynamics.py` builds M(q) by composite-body recursion,
  G(q) from Jacobian rows on COM positions, C(q,q̇) by finite-difference
  Christoffels; solves `q̈ = M⁻¹(τ − Cq̇ − G)` via `np.linalg.solve`.
- `closed_loop_sim.py` wires PD controller → `ActuatorJointModel`
  (torque demand → thermal-derated delivered torque) → forward dynamics
  → thermal advance.
- `task_suite.py` defines six tasks (hold 60 s, pick 4 kg, lean,
  squat, recover, impact) and scoring (tracking, thermal headroom,
  energy, saturation, success).

The README explicitly chose this over a PyDy wrap "because the
closed-loop sim needs a fast, deterministic FD evaluator more than full
symbolic generality." **No contact, no constraints, no soft bodies.**

### 3.6 Validation orchestrator — `cps/validate/`

Tier folder layout maps to a labelled multi-tier check pipeline:

| Tier | What | State |
|---|---|---|
| `tier_g` | Geometry (z-stack, all-pairs distance, pockets) | working |
| `tier_k` | Kinematics (orbit, output-pin engagement) | working |
| `tier_a` | Assembly / Hertz load | working |
| `tier_s` | Structural (Hertz-driven stress) | partial |
| `tier_t` | Tolerance Monte-Carlo | partial |
| `tier_d` | DfAM (overhang, support volume, min wall) | working |
| `tier_em` | Electromagnetic (quasi-3D, FEMM 2D) | stub |
| `tier_th` | Thermal (Elmer 3D) | stub |
| `tier_kb` | Drake CPU determinism regression | stub |
| `tier_extern` | GetDP / OpenFOAM / preCICE / OrcaSlice | stub |

`orchestrator.py` drives them; `report.py` aggregates `Defect` objects
(INFO / WARNING / MAJOR / CATASTROPHIC); `cache.py` memoizes results by
hash so the RL outer loop doesn't recompute.

`cps_eval/` is a parallel, lighter harness with the same shape:
`ActuatorEvalHarness` with `GeometricTier`, `KinematicSweepTier`,
`AnalyticalHertzTier`, `EMProxyTier`, `ThermalProxyTier`,
`ToleranceMCTier`, `DfAMTier`, `StructuralStubTier`. The fast preset is
G + K + A + EM-lite + Th-lite at <1 ms per candidate.

### 3.7 Reward + RL env — `cps/rl/`

`rewards.py::actuator_reward` is a weighted aggregate of nine scores
(continuous torque, peak torque, speed, efficiency, thermal margin,
structural headroom, gearbox quality, printability, cost) plus hard
constraint penalties.

`envs/actuator_static_env.py::ActuatorStaticEnv` is Gymnasium-shaped:

- 18-dim action ∈ [−1, 1]^18, scaled by `action_scale ·
  Range_width` per variable. Variables span rotor / magnet / airgap /
  coil / heat-spreader / ring-pin / eccentricity / output-pin /
  controller current limits.
- 22-dim observation (electrical, thermal, gearbox, structural,
  printability).
- Episode truncates after `max_design_steps` (default 16), terminates
  on catastrophic constraint violation.

Driven from PureJaxRL, SB3, or `cps cmaes --evals 2000`.

### 3.8 CLI

```
cps eval [genome.yaml]              # Tier-0 metrics + reward breakdown
cps random-search --steps 500       # Phase-3 random baseline
cps cmaes --evals 2000              # CMA-ES baseline (needs cps-rl extra)
cps cad [genome.yaml]               # STEP/STL export (needs cps-cad extra)
cps roll --steps 8                  # one RL rollout
```

---

## 4. Empirical signal — how well does it work?

### 4.1 Fixtures as adversarial tests

`mech_harness/fixtures/` ships six design.py fixtures, each a different
attack/failure mode:

| Fixture | What it does | Which gate catches it |
|---|---|---|
| `golden_cycloidal` | Correct N+1 relation, hole sized for orbit | passes everything |
| `wrong_ratio` | `pins=11, lobes=8` | cycloidal `invalid_pin_lobe_relation` |
| `output_hole_too_small` | Forgets `+2·e` orbit term (the C3 bug) | cycloidal `output_hole_too_small_for_orbit` |
| `missing_contact` | Disables disc↔ring contact | assembly `required_contact_disabled` |
| `nonphysical_inertia` | Negative principal moment | schema `nonpositive_principal_moment` |
| `fake_report_attempt` | Agent writes a fake passing summary | hook + attestation reject |

A test (`test_agent_converges_from_broken_baseline`) deliberately seeds
four simultaneous failures (orbit clearance, FOS, Hertz pressure,
torsion) and shows the agent converges in 2–12 SARIF-guided iterations.

### 4.2 Example design submissions

`cps_eval/examples/` carries three real artifacts:

- `baseline_seed_design.json` — `afcqdd_seed_v1`: OD 116, 58 mm thick,
  10:1 dual-cycloid, 18-coil AFPM, 24 turns, 20 A cont / 35 A peak,
  T_cu 85 °C ceiling, 24 Nm peak. Output-pin clearance **−0.03 mm**
  (intentionally tight; "intended to pass fast gates").
- `v1_bad_design.json` — `af_cqdd_140_v1_bad_negative_control`: OD 140,
  airgap 0.4 mm, zero heat-spreader contact area, structural FOS 1.4,
  thermal resistance 0.20 °C/W. Explicit "negative control."
- `v1_expected_defects.json` — Audit ground-truth: which defect codes
  (A1, B1–3, C2, C3, G1–2, D2–4, EM3, Th1) the validator *must* and
  *should* detect.

### 4.3 Tests

- `tests/mech_harness/` exercises the gate suite end-to-end against the
  fixtures and the attestation/Stop-hook surface. Order ~40 passing.
- `tests/cps/` (≈ 18, < 2 s) covers EM Kt/Kv, LPTN steady/transient,
  joint thermal derating, mass-matrix symmetry/PSD, gravity rollout,
  cycloid orbit geometry, RL env reset/step, CMA-ES baseline.
- Optional-dep gates: `test_v1_pipeline.py` skipped without `pxr`,
  `test_chrono_oracle.py` without PyChrono, `test_cycloid_reference.py`
  without build123d. These are *collection skips, not silent passes.*

### 4.4 Reports

No `latest_summary.json` is checked into the tree — they are minted
per-run. The shape (read from `evaluation/pipeline.py` and `server.py`):

```
reports/
  rpt_<hex>.json          # full report body + attestation
  latest_summary.json     # redacted summary the agent sees
  findings.sarif          # full SARIF (private)
  findings.public.sarif   # redacted SARIF (agent-visible)
  trusted_mass_properties.usda
  trusted_metrics.usda
  provenance.jsonld
  attestation.json
```

The summary is deliberately small so the agent cannot blow its context
on report blobs. Per-gate metrics are exposed; raw failure stack
traces are not.

But the trial-level oracle payloads *are* discoverable in
`/tmp/chrono_diag/` — see §5 below for what they actually contain.

---

## 5. What the artifacts say — empirical state of the oracle

This section was added after reading the actual run outputs and the
recent commit log. **It corrects an outdated claim from earlier
drafts of this doc that the Chrono mesh-contact runner was "deferred
to v1.1."** That claim was based on the now-stale `todo.md:43` note.
Between commits `ba63922` (2026-04-30) and `ace9ba9` (2026-05-11) —
roughly the last two weeks of the repo's history — the mesh-contact
oracle was actually built and is running. The full sequence of fixes
is readable in `git log`:

```
ace9ba9 docs(mech-harness): 3-axis design-space sweep results
baf88c3 feat(cli): mh design-sweep — the RLVR parameter-exploration primitive
b6a7132 docs(mech-harness): N-sweep table — rigid-body cycloidal works for N≤5
a9b0bec docs(mech-harness): record working cycloidal operating point (N=3)
152001c fix(chrono): default to mesh disc contact + tight envelope for cycloidal
909bb16 fix(chrono): disc_collision_mode="mesh" eliminates HACD axial-drift bug
a40de94 fix(chrono): SetPos = world_pos + com_mm — geometry origin lands at authored pos
21477e9 viz(dashboard): add z-drift panel — surfaces axial joint failure visually
0a32f9b feat(dashboard): interactive HTML+Plotly trace UI with embedded MeshCat
6d3f9ae feat(fourbar): second mechanism end-to-end — proves runner is generic
8dc4299 feat(cli): mh trial-summary — single-call agent-facing trial report
1d71236 fix(chrono): cycloidal envelope must exceed lobe clearance
f16f495 feat(chrono): rolling-pin (needle-bearing) modeling option
5c9b491 fix(chrono): expose LCP solver choice; APGD eliminates oscillation
7560d0c fix(chrono): tighten contact envelope/margin to mechanism scale
f72d6ed test(scene-graph): validate runner is mechanism-agnostic + JSON round-trip
8d0e1cf refactor(chrono): generic SceneGraph + cycloidal translator (RLVR-ready)
d61d676 feat(chrono): contact-pair instrumentation + collision-family filter
337d7f7 feat(chrono): generic diagnostics + projected integrator
5595a01 feat(sim): chrono trial pose-replay PNGs
1992125 fix(chrono): bump V-HACD defaults — 5 of 6 disc holes now preserved
```

That is two weeks of small commits that produce a working
mesh-contact oracle. The empirical evidence below confirms it.

### 5.1 Trial payload shape (the real thing)

`/tmp/chrono_diag/dashboard_payload.json` is a 1.7 MB payload from an
actual oracle run on AF-CQDD-140 V1. Its top-level keys are the same
ones the agreement gate consumes:

```
simulator, available, passed, required,
ratio_observed, ratio_error_pct, max_penetration_mm,
max_constraint_error_mm, torque_ripple_pct,
power_balance_error_pct, lockup_detected,
contact_force_rms_N, trials[], failure, failure_detail
```

Per-trial fields (after the recent runner refactor):

```
diverged, seed, wall_s,
in_omega_med, out_omega_med, expected_out_omega_mag,
input_torque_med_Nm, ratio_observed, ripple_pct,
power_err_pct, lockup, n_contacts_max, step_exceptions,
failure, pose_samples[121], preflight_issues[],
worst_violation { norm, step, link },
build_meta { n_bodies, n_joints, n_motors,
             n_excluded_pairs, body_family,
             noncollide_pairs[], contact_method,
             timestepper }
```

`top_contact_pairs` (in the trial-summary view) reports
`{ pair, n_contacts, sum_Fn_N, sum_Ft_N }` per pair, derived from the
simulator — not declared by the agent. This is the contact-engagement
signal that earlier drafts of this doc flagged as "best-effort, usually
skipped."

### 5.2 The canonical 10:1 design honestly fails

Reading `trial_summary.json` for the AF-CQDD-140 V1 default (11 ring
pins, 10 lobes, e=1.4 mm, default clearance) on the mesh-contact oracle:

```
passed:            false
failure:           all_trials_diverged_or_lockup
trial_failure:     lockup_mechanism_jammed
ratio_observed:    Infinity        ← because out_omega_med ≈ 0
target_ratio:      -10.0
in_omega_med:      9.999985 rad/s  ← motor enforces setpoint
out_omega_med:     5.6e-34 rad/s   ← essentially zero
lockup:            true
n_contacts_max:    20
worst_violation:   norm 1.0 mm at step 382,
                   link 'rev:housing↔ring_pin_03'
build_meta:        28 bodies, 26 joints, 1 motor, 71 excluded
                   collision pairs, NSC contact,
                   euler_implicit_projected timestepper
```

Top contact pairs (the highest-loaded interfaces during the trial):

| pair | n_contacts | ΣF_n (N) | ΣF_t (N) |
|---|---:|---:|---:|
| `disc_b ↔ ring_pin_00` | 156 | 36.1 | 4.2 |
| `disc_a ↔ ring_pin_03` | 132 | 51.8 | 9.9 |
| `disc_a ↔ ring_pin_02` | 75 | 0.66 | 0.17 |
| `disc_b ↔ ring_pin_01` | 72 | 22.7 | 2.3 |

Two takeaways:

1. **The harness produces an honest verdict.** The canonical 10:1
   design cannot move in rigid-body NSC simulation, and the oracle
   reports `lockup_mechanism_jammed` rather than confabulating a
   ratio. This is exactly the failure mode the trust model is supposed
   to surface.
2. **Per-pair contact forces are real measurements**, not labels.
   The agent never gets to declare them; they fall out of the
   simulation step.

### 5.3 Why N=10 fails and N=3–5 works (capability ceiling)

`docs/src/mech-harness/working-cycloidal.md` documents this directly.
Rigid-body NSC over-constrains at N ≥ ~6 because every lobe within
the envelope distance of its pin generates an active constraint.
At N=10 that is ~10 simultaneous constraints competing for 1 DOF.
Real cycloidal drives operate at N=20+ but rely on steel elasticity
(distributing load across ~3 loaded lobes at a time) — a regime
**outside** what rigid-contact NSC can model. The doc's honest
conclusion:

> "Real cycloidal drives operate in a regime rigid-body simulation
> cannot reproduce... The fix is to redesign for the simulation
> regime, not to hack the simulator."

A working operating point is recorded explicitly:

```python
CycloidalParams(
    ring_pin_count=4,             # 3:1 reduction
    disc_lobe_count=3,
    disc_lobe_clearance_mm=0.20,  # 200 µm
    disc_profile_samples=720,
)
```

with measured: `ω_in=+10.0`, `ω_disc=−3.0` (90% of theoretical
−3.33), `ω_out=−4.2`, `ratio_observed=5.7:1` vs target 3:1
(consistent ~2× slip from sliding friction at fixed ring pins),
`n_contacts_max=36`, `lockup=False`.

### 5.4 Design sweeps surface real non-monotonic landscapes

The `mh design-sweep` CLI was added in commit `baf88c3`. Three
sweeps in `/tmp/chrono_diag/sweep_{N,e,clr}/sweep.json` cover the
operating point's neighborhood. Per-trial wall time ranges from
~470 s to ~1670 s — these are minutes-to-hours, not seconds.

**`ring_pin_count` sweep** (target ratios 3, 4, 5):

| N | passed | ratio_observed | ω_out | n_contacts | wall (s) |
|---:|---|---:|---:|---:|---:|
| 3 | ✓ | 3.89 | +2.57 | 27 | 1674 |
| 4 | ✓ | 3.21 | −3.11 | 33 | 903 |
| 5 | ✓ | 14.45 | −0.69 | 40 | 786 |

LCP starts struggling at N=5 (ratio observed 14.5 vs target 5), and
fails at N=10 (lockup) — exactly the ceiling described in §5.3.

**`eccentricity_mm` sweep** (at N=4):

| e (mm) | passed | ratio_observed | ω_out | lockup | wall (s) |
|---:|---|---:|---:|---|---:|
| 1.0 | ✗ | 73.9 | +0.13 | **true** | 1288 |
| 1.4 | ✓ | 3.21 | −3.11 | false | 902 |
| 2.0 | ✓ | 10.5 | +0.95 | false | 750 |
| 2.8 | ✓ | 3.10 | +3.23 | false | 470 |

Too-small eccentricity (e=1.0) locks up — the disc lobes can't clear
the ring pins. e=1.4 is the documented optimum.

**`disc_lobe_clearance_mm` sweep** — non-monotonic (this is the
landscape note in the commit message of `ace9ba9`):

| clearance (mm) | passed | ratio_observed | ω_out | contacts | wall (s) |
|---:|---|---:|---:|---:|---:|
| 0.05 | ✓ | 28.8 | +0.35 | 29 | 1408 |
| 0.10 | ✓ | **2.54** | **−3.93** | 29 | 1080 |
| 0.15 | ✓ | 9.43 | −1.06 | 25 | 1244 |
| 0.20 | ✓ | **3.21** | **−3.11** | 33 | 900 |
| 0.30 | ✓ | 6.83 | +1.46 | 31 | 978 |

Two local optima at clearance = 0.10 and 0.20 with a dead zone at
0.15. The commit message frames this as a **deliberate RLVR
observation**: a pure-gradient optimizer would get stuck on the
wrong side of the 0.15 plateau; the sweep-and-select pattern
navigates it fine. This is real mechanical-design landscape
behavior — geometric resonances between profile-sample density and
orbital phasing produce sharp non-smoothness.

### 5.5 Four-bar runs cleanly (proves the runner is mechanism-agnostic)

`/tmp/chrono_diag/payload_fourbar.json` (commit `6d3f9ae`):

```
passed:        true
ratio_observed: 2.91
in_omega_med:  +5.00 rad/s
out_omega_med: −1.72 rad/s    ← sign correct
n_contacts_max: 0             ← purely kinematic
lockup:         false
wall_s:         2.04
build_meta:    4 bodies, 4 joints, 1 motor, 0 excluded pairs
worst_violation: norm 0.37 mm at step 1499, rev:crank↔coupler
```

Same SceneGraph plumbing, no mechanism-specific code in the runner
— the runner is generic over `SceneGraph` (commit `8d0e1cf`).

### 5.6 Codesign integration is now end-to-end

`designs/codesign_run_002/README.md` records a CMA-ES run with
`cps.validate.orchestrator` folded into the reward:

- 22-dim design vector (14 original + 8 per-joint cycloid overrides)
- 312 evals, 17 min wall-clock
- Reward improved `−15.57 → −13.11` (warm-start → winner)
- **Optimizer correctly pushed clearance up and eccentricity down on
  every joint** — the right direction to clear the C3 defect (output-
  pin engagement). Validates that the harness signal is gradient-
  bearing for the optimizer.
- **C2 (cycloid orbit jam) stayed structurally trapped** on all 4
  joints. The README's analysis is precise: the V1 11-pin / 10-lobe /
  e=1.4 mm topology cannot be escaped by scale alone, because the
  e/disc_OD ratio governing C2 is invariant under scale. Need topology
  variables (different `ring_pin_count`, lower eccentricity bound, or
  epicycloid-vs-hypocycloid profile).

### 5.7 Defect ground-truth audit

`designs/af_cqdd_140_v1/v1_defect_audit.md` is a hand-curated list of
20+ defects (A1–A3, B1–B6, C1–C4, D1–D4, E1–E2) that the V1 design
ships with. Each entry has a concrete root cause, e.g. B1:
> "Disc center bore = 28 mm dia accepts a 24 mm bearing OD with 4 mm
> radial slop — way too loose for a press fit. Press fit needs
> interference of −0.02 to −0.05 mm; we have +2 mm clearance."

`cps_eval/examples/v1_expected_defects.json` enumerates which of these
the validator *must* catch (A1, B1–3, C2, C3) vs *should* catch (G1–2,
D2–4, EM3, Th1). This is a useful regression substrate for harness
development — "does the gate suite still find C3 if we move the orbit
math?"

### 5.8 Hard-won runner debugging notes

The `docs/src/mech-harness/index.md` "Known pitfalls" section reads
like a textbook of contact-mechanics simulation errors caught the
hard way. Selected entries:

- **Legacy HACD silently returned every triangle as a "hull."**
  `ChConvexDecompositionHACD` ignored its kwargs through SWIG and
  produced 4080 hulls from a 4080-triangle disc. Diagnosed by hull-vs-
  hole AABB probe in `/tmp/chrono_diag/diag_hull_holes.py`. Switching
  to `ChConvexDecompositionHACDv2` with `mMaxHullCount=64,
  mConcavity=0.001` preserves 5 of 6 disc output-pin holes as gaps
  between adjacent hulls.
- **Polyline sample count gates clearance fidelity.** The lobe profile
  is a Polyline with N samples → effective profile is an N-sided
  polygon. Empirical recipe:
  `disc_profile_samples ≥ 2π·R / (10 × clearance_mm)`. Default
  240 samples gave 0.79 mm chord — too coarse for sub-100 µm
  clearance. Tracked as `disc_profile_samples=720` in the working
  operating point.
- **`body.GetAngVelLocal()` lies if the body tips.** Reading ω in the
  body's local frame produces ω_disc = +9.93 when the world-frame
  value is −2.07, once roll/pitch is non-zero. Use
  `GetAngVelParent().z`. Caught by reading raw signals in the
  dashboard.
- **SMC soft contact beats stiff at this scale.** At 0.5 mm tessellation
  + 0.05 mm clearance, contacts with Young's modulus ~10⁸ Pa (~0.1%
  of steel) give the cleanest kinematic direction. Stiffer contacts
  turn unavoidable mesh penetrations into chaotic restoring impulses.
- **Redundant constraints silently zero the motor.** Pinning a body
  with both `SetFixed(True)` and `ChLinkMateFix` makes the
  BARZILAIBORWEIN solver report the right speed setpoint but zero the
  actual motor angle update. Pick one.
- **NSC has a ~2300-simultaneous-contact fidelity ceiling.** Beyond
  that, the LCP becomes too stiff to converge reliably.

These are not in any TODO file because they have been *solved*. They
are recorded because the next contributor needs them.

---

## 6. Where the system is still rough

Revised list after reading the artifacts. Several earlier items have
moved to §7 (now working).

1. **The harness honestly reports that the canonical 10:1 design
   doesn't move.** This is a feature, not a bug — but it means the
   reward signal for any design at N≥6 is "lockup" rather than a
   gradient. Codesign agents either need to be allowed to mutate
   `ring_pin_count` into the simulator's tractable range, or the
   simulator needs flexible-body modeling (Chrono FEA) to reach N=10+.
2. **Consistent ~2× slip factor in measured ratios** at the working
   N=3–5 operating point. Comes from sliding friction at fixed ring
   pins. `opt.rolling_pins=True` is implemented (commit `f16f495`)
   but currently locks up at all tested clearances. So the simulator
   produces an *honest* but *biased* ratio at this operating regime;
   the reward function would need to correct for that bias or accept
   it.
3. **Per-trial wall time is minutes-to-hours.** Sweep wall times above:
   470 s–1674 s for a single 4-trial mesh-contact run. This is fine for
   audit / batch optimization but **far too slow** for an RL inner
   loop. The CPS Tier-0 reward (<1 ms) and the cps_eval fast harness
   remain the per-step signal; the oracle is for episode-final
   verification at most.
4. **NSC contact has a ~2300-simultaneous-contact fidelity ceiling.**
   Documented in the pitfalls section. Above that the LCP can't
   converge reliably.
5. **Polyline-sample-count is design-coupled.** A particular clearance
   needs at least `2π·R / (10·c)` samples; at 5 µm clearance you need
   ≥3000 samples and even 720 (the default for the working point)
   "should work but doesn't — quadratic effect somewhere." A real
   engineering trade, not a bug, but consumers need to know.
6. **`contact_engagement` gate's strictness still varies by simulator
   adapter.** The Chrono mesh runner emits `top_contact_pairs` with
   `sum_Fn_N` / `sum_Ft_N` — it has the data. But the Kamino path
   (`kamino_adapter.py`) is still the kinematic-constraint variant
   reusing `cps_newton_actuators.models.kamino_cycloid_n2`, and *that*
   does not emit per-pair forces. So the "fast vs oracle agreement"
   gate currently compares Kamino-kinematic-tautology to
   Chrono-mesh-contact — not a fair comparison.
7. **HDF5 trace persistence not wired.** `standards/hdf5_traces.py`
   exists as a contract; current trials persist their pose samples
   and metadata into `dashboard_payload.json` (1.7 MB per trial) but
   not into the standardized HDF5 format. Post-hoc trace querying
   means parsing a per-run JSON.
8. **Standards-first `evaluation/pipeline.py` is not the default
   path.** `server.py` is. Two entry points create a low-grade drift
   risk.
9. **No differentiable mid-fidelity solvers yet.** The `warp.fem` /
   JAX-FEM Phase-3 layer in the design doc is unbuilt; there is no
   coupled magneto-thermal-mechanical kernel. The Tier-2/3 validator
   tiers (`tier_em/quasi3d.py`, `tier_th/elmer_driver.py`,
   `tier_kb/drake_driver.py`, `tier_extern/*`) are stub files.
10. **Anisotropy is a global 0.55× knob.** No mapping from slicer
    toolpath to per-element orthotropic constants.
11. **`cps/mbd/` is planar, NumPy, no contact.** Intentional for fast
    deterministic RL inner-loop dynamics. Does not substitute for a
    full Newton/MuJoCo robot scene; that piece (§3.5 of
    `cyber-physical-sim.md`) hasn't started.
12. **WebGL PyDy 3D viewer is partially broken** (`sum-03.md:148`).
    MeshCat viewer + dashboard (`sim/viewer.py`, `sim/dashboard.py`)
    work fine; PyDy's robot-dynamics viewer does not.
13. **Eureka RL backend "written, not validated"** (`MUSTDO.md`).
14. **C2 (cycloid orbit jam) is a topology-trapped defect.** The V1
    design's e/disc_OD ratio cannot be escaped by scale alone. Codesign
    agents stuck at this defect need exposed topology variables —
    flagged as next iteration in `codesign_run_002/README.md`.

---

## 7. What's solid (updated)

- **The trust model.** Agents cannot fabricate reports (attestation
  + hooks); mass/COM/inertia re-derived from STEP; path policy boxes
  the agent into `design/` and `scratch/`.
- **The analytical oracles** (schema, geometry, density, ratio, orbit
  clearance, Hertz, FOS) — every adversarial fixture caught.
- **The mesh-contact oracle is real.** It runs Bullet/NSC dynamics over
  V-HACDv2 convex decomposition of cycloid discs with `n_clusters=128,
  concavity=0.001`, with named per-joint and per-contact-pair
  instrumentation, projected-Euler timestepper, APGD LCP solver,
  pre-flight assembly checks. It produces a 1.7 MB
  `dashboard_payload.json` per run with 121 pose samples, worst-
  violation tracking, and `top_contact_pairs` force breakdown.
- **The runner is mechanism-agnostic.** A single `SceneGraph`
  dataclass works for cycloidal and four-bar today (commit `f72d6ed`
  is the JSON round-trip test). Adding a new mechanism is the 4-step
  recipe in `docs/src/mech-harness/adding-a-mechanism.md`.
- **The CLI grew real RLVR primitives**: `mh design-sweep` for
  parameter exploration, `mh trial-summary` as the agent-facing
  single-call report, `mh replay` for frame extraction, `mh snapshot`
  for kinematic preview, and an interactive HTML+Plotly dashboard
  with embedded MeshCat (commits `0a32f9b`, `8dc4299`, `baf88c3`).
- **The codesign loop is closed.** `cps codesign-cmaes` runs with
  `cps.validate.orchestrator` in the reward, completes 312 evals in
  17 min, and demonstrably moves clearance/eccentricity in the right
  direction. See `designs/codesign_run_002/README.md`.
- **SARIF as the agent-feedback channel.** Convergence-from-broken-
  baseline is exercised in tests.
- **Tier-0 actuator evaluation in <1 ms.** EM + LPTN + cycloidal +
  structural composed into one reward.
- **Cross-validation against open-source reference.** The cycloid
  profile generator is bit-perfect-anchored
  (`<1e-9` max diff) to the MIT `gittawat/cycloidal_drive_internal_openscad`
  implementation in `tests/mech_harness/test_cycloid_reference.py`.

---

## 8. One-line summary (revised)

The repo has shipped (a) a **trust-first static validation harness**
for AI-generated mechanical assemblies, (b) a **fast actuator
co-design Tier-0 evaluator** plus RL env, and — newer than the
earliest in-tree docs suggest — (c) a **real mesh-contact oracle**
that runs Bullet/NSC dynamics with named per-pair contact
instrumentation, validated mechanism-agnostically across cycloidal
and four-bar. The honest current finding is that **the canonical
10:1 cycloidal design cannot be reproduced in rigid-body NSC**
(LCP over-constrained), and the system reports that fact rather
than confabulating a passing ratio. The next-tier limitations are
*physical-modeling* limitations (flexible bodies for N=20+
cycloidals; differentiable mid-fidelity solvers), not pipeline
gaps.
