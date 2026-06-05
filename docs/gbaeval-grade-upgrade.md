# mech-bench → GBA-Eval grade — upgrade summary

Branch `mech-sim-natalia-gbaeval-grade`. This work turns mech-bench's reward from
a partly self-referential / synthetic signal into an **independent, geometry-
grounded, dense, hard-to-game** one — the property that makes GBA-Eval's signal
good (see `../rl-environment-design-notes.md` and `../mech-sim-rl-improvement-notes.md`).
Driven by an 8-agent inspection workflow whose full plan is in
`gbaeval-grade-implementation-plan.json`.

**Status: core suite green at 331 passed** (was 282; +49 new tests), 12 skipped,
1 pre-existing unrelated failure (`test_resolve_family_tasks_root_materializes_paper_tasks`
— missing `scripts.materialize_paper_family_tasks`, present at baseline).
PyChrono cannot be installed on the macOS-ARM dev box (conda/Docker-only, no
PyPI wheel), so all work is verifiable without it; Chrono is wired as the
high-fidelity tier for capable hardware.

## What landed (per commit)

| # | Fix | What it does | GBA-Eval principle |
|---|-----|--------------|--------------------|
| 1–2 | **Real oracle + dense-scoring helper** | `mech_bench/scoring.py` (quartic sigmoid `1/(1+(d/τ)^4)`, perceptual floor, adaptive τ); `mech_bench/adapters/analytic_mechanics.py` — a **real** closed-form rigid-body oracle computing ratio from teeth, contact engagement from actual inter-body distance vs sum of pitch radii, torque from the lever relation. Gears 100 m apart → zero force (unfakeable). Declines (`capability_unavailable`) on insufficient geometry — no silent pass. | Independent ground-truth oracle; determinism |
| 5 | **Synthetic-oracle reward quarantine** | `rlvr.py` dual-mode `reward_profile`: `eval` reports transparently, `train` forces synthetic-oracle reward to 0 (`dev_reward` keeps the would-be score). `rl/mech_bench_reward.py` default-quarantines on the training surface + scrubs `MECH_BENCH_USE_FAKE_ORACLE`/`TEST_MODE`. | Anti-hack gate (blank-frame→1.0 analogue) |
| 7 | **Undriven-lockup gate** | A static (dead) mechanism no longer scores a degenerate 1.0; emits `DEGENERATE_TEST`. (Geometry-plausibility is already enforced by the real oracle.) | Robustness to optimization pressure |
| 8 | **Chrono provisioning + diagnostics** | Removed stale "vendored out" docstrings (the 119 KB in-repo runner IS present; the gate is pychrono). `chrono_diagnostic()` now returns `remediation` + `reference_cache_version`. `scripts/enable_real_oracle.sh` one-command conda setup. `mech_bench/oracle/reference_cache.py` — content-addressed cache key (excludes declared answers) so the slow oracle is affordable to iterate. | Reference cache; no silent pass |
| 3 | **Dense probe reward** | `analytic_param_check` + `port_velocity_ratio` use the quartic sigmoid (0.5 at tolerance) instead of a linear cliff → a gradient to climb, not a 4-bin plateau. | Dense reward |
| 9 (core) | **De-self-reference probe** | `analytic_derived_check` recomputes the quantity from the agent's declared **primitives** (teeth/pitch/radii) and grades that — never the declared answer. Proven: correct-teeth + lying `declared_ratio=999` → PASS; wrong-teeth → FAIL. | Verifiable oracle, not a self-check |
| 6 | **eval-vs-RL CLI contract** | `mech-bench rlvr-eval --reward-profile {eval,train} [--allow-synthetic-reward]` — the same tasks serve as a transparent eval and a quarantined RL environment. | Both eval AND RL env |

## Remaining mechanical rollout (not yet done)

These are bounded follow-ups; the mechanisms above are in place and tested.

1. **Repoint generators + regenerate task fixtures** (`gear_train.py`,
   `transmission_analytic.py`, `static_analytic.py`, the ~50 `tasks/**` dirs and
   their `expected_failures.json`) from `analytic_param_check` to
   `analytic_derived_check`, and **remove the printed answer from prompts**. The
   probe exists and is tested; this is the per-family wiring + fixture regen the
   inspection plan sequenced last (it touches many fixtures and several negative
   controls). Until done, the live tasks still use the self-referential probe.
2. **Repoint contact tasks to `analytic_mechanics`**: regenerate the Tier-3
   stubs to declare `contact_pair` joints + tooth counts + positions so the real
   oracle grades them (today they carry teeth but no `contact_pair` joint, so the
   conservative adapter declines and they stay `capability_unavailable`).
3. **path_trace_chamfer scale/placement** (Fix F) and **real fast-vs-oracle
   agreement** (Fix L): a wrong-size/displaced coupler curve can still score full
   marks under `normalize=true`; `modes.py` agreement defaults missing terms to
   1.0. Not yet addressed.
4. **Training-side wiring**: have the trainer pass `reward_profile="train"` and,
   per the improvement doc, fix the legacy `EpisodeResult`/`TurnTrace` field
   mismatch and add curriculum / cross-group baseline (these need torch, not
   runnable on this box).
5. **Run the family-held-out paper experiment** from `goals.md` with the real
   oracle in the loop (needs GPU + Chrono hardware).

## Verifying locally

```bash
uv sync --extra dev
uv run python -m pytest -q \
  --ignore=tests/test_chat_rollout.py \
  --ignore=tests/test_sample_and_score_pass_metrics.py \
  --ignore=tests/test_train_sft_peft.py     # 331 passed (1 pre-existing fail)

# New/changed test files:
#   test_scoring.py, test_analytic_mechanics.py, test_analytic_derived_check.py,
#   test_reference_cache.py, test_modes_and_rlvr.py (quarantine),
#   test_probes_library.py (dense + lockup gate)
```
