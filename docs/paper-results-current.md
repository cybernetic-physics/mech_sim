# Current Paper Results
This file summarizes the controlled runs from `docs/paper-results-current.json`. The evidence supports a benchmark/runtime paper, not a hardware-calibrated simulator claim.
## Headline Results
- Reference controls: 50/51 pass rate (0.980); mean hidden score 0.945.
- Negative controls: 104/104 expected failures detected.
- Cycloidal real-geometry proof: ok=True, missing_bridge=None.

## Reference Suite By Tier
| Tier | n | pass rate | mean score |
|---|---:|---:|---:|
| artifact_static | 13 | 1.000 | 0.969 |
| contact_dynamics | 12 | 0.917 | 0.818 |
| planar_kinematics | 12 | 1.000 | 0.982 |
| transmission_analytic | 13 | 1.000 | 1.000 |
| unknown | 1 | 1.000 | 1.000 |

## Known Misses / Caveats
- `contact_gear_pair_stub_s0001`: invalid/hard-gate miss with public=['capability_unavailable', 'capability_unavailable'] hidden=['capability_unavailable', 'capability_unavailable'].
- `brake_caliper_contact_stub_s0001`: valid but low hidden score 0.900; codes=[].
- `cam_follower_contact_stub_s0001`: valid but low hidden score 0.800; codes=[].
- `cycloidal_lowN_stub_s0001`: valid but low hidden score 0.300; codes=['excessive_torque_ripple', 'lockup', 'power_balance_error'].
- `detent_spring_contact_stub_s0001`: valid but low hidden score 0.875; codes=[].
- `fourbar_path_s0001`: valid but low hidden score 0.893; codes=[].
- `fourbar_wiper_arc_s0001`: valid but low hidden score 0.893; codes=[].
- `friction_clutch_torque_stub_s0001`: valid but low hidden score 0.944; codes=[].
- `pulley_bore_alignment_static_s0001`: valid but low hidden score 0.693; codes=['insufficient_clearance'].
- `static_fit_bracket_s0001`: valid but low hidden score 0.909; codes=['insufficient_clearance'].

## Cycloidal NSC vs SMC Real-Geometry Fixture
| Contact model | max penetration mm | contact force RMS N | n_contacts_max | ratio_observed | failure_mode |
|---|---:|---:|---:|---:|---|
| NSC | 3.93418 | 1.58081e+06 | 958 | 0.576348 | power_balance_error |
| SMC | 0.975371 | 66.3579 | 171 | 14.9072 | power_balance_error |

SMC/NSC ratios: force RMS 4.198e-05, penetration 0.248, contact count 0.178.

## Paper Interpretation
Supported: a verifiable mechanical-design benchmark/runtime with reference controls, negative controls, and a real CAD-to-Chrono physics-adapter proof.

Not supported yet: a trustworthy high-fidelity simulator, hardware-calibrated cycloidal reducer, or robot/agent learning result.
