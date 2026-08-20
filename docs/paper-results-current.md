# Frozen Benchmark Evidence — 2026-05-25

> **Snapshot, not current-suite status.** This record was generated from commit
> `1f18ec5519ea417a56051539728585244d796ae9` on the then-current 51-task
> materialization. The generator registry now contains 58 families. Eleven
> reference tasks in this snapshot used the explicitly labeled synthetic
> contact adapter.

The source artifact is
[`paper-results-current.json`](paper-results-current.json). It records the
commands, repository revision, aggregate metrics, proof output, supported
claims, and unsupported claims.

## Summary

| Check | Result |
|---|---:|
| Reference controls passing | 50 / 51 |
| Reference hard-gate pass rate | 98.0% |
| Mean hidden score | 0.945 |
| Expected negative failures detected | 104 / 104 |
| Synthetic reference tasks | 11 |
| CAD-to-Chrono proof completed | Yes |

The single invalid reference was `contact_gear_pair_stub_s0001`, which reported
`capability_unavailable`. This is an honest missing-capability result, not a
mechanical failure.

## Reference controls by tier

| Tier | Tasks | Hard-gate pass rate | Mean hidden score |
|---|---:|---:|---:|
| Static artifact | 13 | 100.0% | 0.969 |
| Planar kinematics | 12 | 100.0% | 0.982 |
| Analytic transmission | 13 | 100.0% | 1.000 |
| Contact dynamics | 12 | 91.7% | 0.818 |
| Legacy unclassified task | 1 | 100.0% | 1.000 |

Several valid references scored below 0.95. The detailed JSON names them and
records clearance, path, contact, lockup, torque-ripple, and power-balance
feedback. A reference solution passing its hard gate does not imply every
dense objective is ideal.

## Real-geometry fixture

The packet also ran the cycloidal CAD fixture through real Chrono NSC and SMC
paths without procedural geometry fallback.

| Contact model | Maximum penetration | Contact-force RMS | Maximum contacts | Observed ratio | Failure |
|---|---:|---:|---:|---:|---|
| NSC | 3.934 mm | 1.58 MN | 958 | 0.576 | `power_balance_error` |
| SMC | 0.975 mm | 66.4 N | 171 | 14.907 | `power_balance_error` |

These numbers establish runner execution and expose a major formulation/model
difference. They do not establish that either run predicts a physical reducer.

## Supported interpretation

This snapshot supports:

- a working mechanical-design evaluation runtime;
- reference and negative-control execution across four task tiers;
- explicit reporting of unavailable capabilities and synthetic evidence; and
- a real CAD-to-Chrono adapter path with diagnostic output.

It does not support:

- broad high-fidelity or hardware-calibrated simulation;
- a validated cycloidal reducer model; or
- a learning improvement claim.

The last boundary applies to this May snapshot. A later Level-1 learning result
is documented in
[`runs/mechanism_repair_ttrl_final`](../runs/mechanism_repair_ttrl_final/README.md),
but it does not change the physics limits of this packet.
