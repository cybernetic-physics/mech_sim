# Family Transfer Claim Audit

Claim status: `does_not_support_family_heldout_transfer`.

Supported claim: seed-heldout multi-family task generalization.
Unsupported claim: RLVR learns reusable mechanical reasoning on unseen mechanism families.

## Split Audit

- Train tasks: 126
- Eval tasks: 42
- Train canonical families: 15
- Eval canonical families: 15
- Train/eval overlapping canonical families: 15
- Eval canonical families unseen in train: 0

The existing paper-grade run is not a family-held-out result: every eval family is also present in training.

Overlapping families:

- `belt`
- `bevel_gear_ratio_analytic`
- `chain`
- `compound_gear_ratio_analytic`
- `fourbar`
- `idler_gear_direction_analytic`
- `lead_screw`
- `planetary`
- `rack_pinion`
- `reciprocating_pump_plunger`
- `rocker_limit_stop_topology`
- `slider_crank`
- `spur_gear_ratio_analytic`
- `toggle_overcenter_margin`
- `worm_gear_ratio_analytic`

## Existing Result

- `baseline_clean_retryfix_sglang`: 20.67/42 mean tasks passed (49.21%), sampler errors=0
- `ref_sft_r8_s250_retryfix_sglang_lora`: 38.67/42 mean tasks passed (92.06%), sampler errors=0
- `rlvr_s8_cap2_retryfix_sglang_lora`: 41.67/42 mean tasks passed (99.21%), sampler errors=0
- RLVR vs prompted baseline: +21.00 tasks, p=3.577867169202165e-18
- RLVR vs SFT: +3.00 tasks, p=0.01171875

## Interpretation

This run is good evidence that RLVR improves verified multi-family mechanical task performance on held-out task instances/seeds. It is not evidence for transfer to unseen mechanism families, because the train and eval split files share all eval canonical families.

The next required experiment is a fresh family-held-out run using the frozen split machinery, with training families disjoint from eval families and matched verifier budget across frozen, SFT, no-update, and RLVR/TTRL methods.
