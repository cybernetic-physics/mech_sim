# Cycloidal MechanicalEvolve Equal-Budget Result

All compared methods used equal Chrono audit budget: yes (160 real Chrono audits per method).
All compared methods used identical verifier settings: yes. contact_model=smc, samples=41, duration_s=0.15, limits={'max_contact_force_rms_N': 3000.0, 'max_contacts': 128.0, 'max_penetration_mm': 1.0, 'max_power_balance_error_pct': 90.0, 'max_ratio_error_pct': 25.0, 'max_torque_ripple_pct': 1000.0, 'min_output_speed_rad_s': 0.5}.
All compared methods used procedural_cycloidal_fallback=false: true.
TTRL wins under equal budget: yes.

| method | candidate_count | chrono_audits | best_verified_reward | verified_pass_rate | cad_pass_rate | chrono_real_geometry_rate | lockup_rate | best_id | best_fast_reward | best_out_omega_med | best_ratio_error_pct | best_power_balance_error_pct | best_torque_ripple_pct | best_max_penetration_mm | best_contact_force_rms_N | adapter_updates | trained_tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| verifier_gated | 160 | 160 | 66.8249 | 0.0375 | 1 | 1 | 0.83125 | vg_refine_shrink_018 | 74.6445 | 1.04551 | 4.35272 | 79.9006 | 709.096 | 0.402231 | 462.01 | 0 | 0 |
| llm_evolve_no_update | 374 | 160 | 67.3697 | 0.050802 | 0.427807 | 0.427807 | 0.63125 | llm_evolution_no_update_005 | 75.1365 | 1.04058 | 3.8996 | 66.0908 | 869.535 | 0.404402 | 468.063 | 0 | 0 |
| mechanical_evolve_ttrl | 415 | 160 | 67.9321 | 0.053012 | 0.385542 | 0.385542 | 0.61875 | mechanical_evolve_ttrl_016 | 76.434 | 1.0697 | 6.51565 | 77.4291 | 845.259 | 0.402486 | 531.552 | 5 | 1780 |

## Interpretation

Under equal expensive physics-verification budget, iterative RLVR/TTRL adaptation discovers a stronger verified cycloidal actuator design than the non-updating baselines in this run.
