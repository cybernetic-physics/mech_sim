# Equal-Budget Cycloidal Learning Pilot — 2026-05-26

> **Exploratory single-run evidence.** This pilot matched expensive Chrono
> audits across methods, but absolute verified pass rates were low and the
> winning margin was small. It demonstrates the experiment path; it is not a
> general learning conclusion.

## Controlled setup

- 160 real Chrono audits per method.
- Identical SMC verifier configuration and limits.
- Real geometry required with `procedural_cycloidal_fallback=false`.
- Same maximum penetration, contact-force, contact-count, power-balance,
  ratio-error, torque-ripple, and output-speed thresholds.

## Result

| Method | Candidates | Chrono audits | Best verified reward | Verified pass rate | Lockup rate | Adapter updates |
|---|---:|---:|---:|---:|---:|---:|
| Verifier-gated search | 160 | 160 | 66.825 | 3.75% | 83.1% | 0 |
| LLM evolution, no update | 374 | 160 | 67.370 | 5.08% | 63.1% | 0 |
| Online adaptation | 415 | 160 | 67.932 | 5.30% | 61.9% | 5 |

The online method found the highest-reward candidate under the matched audit
budget. Its advantage over the no-update method was 0.562 reward points and
0.22 percentage points in verified pass rate in this run.

That is a positive pilot signal, not a robust effect estimate. A stronger
conclusion would require repeated seeds, paired uncertainty, and broader
mechanism families. Machine-readable rows remain in:

- [`cycloidal_mechanical_evolve_equal_budget_results.json`](cycloidal_mechanical_evolve_equal_budget_results.json)
- [`cycloidal_mechanical_evolve_equal_budget_results.csv`](cycloidal_mechanical_evolve_equal_budget_results.csv)
