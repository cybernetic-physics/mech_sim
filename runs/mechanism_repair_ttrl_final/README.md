# MechanismRepair-TTRL final result bundle

This directory contains the lightweight, committed result bundle for the
MechanismRepair-TTRL held-out family experiment described in `goals.md`.

Full raw artifacts remain on MATX:

```text
/matx/u/knatalia/corl_mechanism_repair_ttrl_merged_final/repo/runs/mechanism_repair_ttrl_final
```

The committed bundle keeps the machine-readable summaries and artifact indices:

- benchmark, split, verifier, method, and merge manifests
- benchmark task definitions and split files
- `results.json` and `results.csv`
- `cell_results.jsonl`
- `stats.json`
- `failure_analysis.json`
- `trace_pairs.json`
- `claim_audit.json`
- preflight scratch artifacts used to audit reference and negative solutions
- index files for raw completions, verifier outputs, training logs, and adapter
  checkpoints

The full raw completion files, verifier output files, training logs, and adapter
checkpoints are intentionally not copied into Git. Use the index files and the
MATX path above for those heavy artifacts.

## Result

The final claim audit reports:

```text
claim_status = supports_primary_hypothesis
```

Primary comparison:

- method: `mechanical_evolve_ttrl`
- baseline: `llm_evolve_no_update`
- paired cells: 120
- budget: 32 actual verifier calls per task/seed/method cell
- TTRL success: 85.8%
- no-update verifier-feedback success: 23.3%
- success delta: +62.5 percentage points
- one-sided sign-test p-value: 5.16e-22

Scope: this supports the Level-1 executable mechanism-program repair claim. It
does not claim hardware readiness or Level-3 real contact/dynamics validation.

## Reproduce

The MATX launcher is:

```bash
scripts/submit_mechanism_repair_matx.sh --submit
```

The run can be sharded by overriding `REMOTE_ROOT`, `SPLITS`, and `EVAL_SEEDS`.
The final run used six shard roots:

```text
/matx/u/knatalia/corl_mechanism_repair_ttrl_exact_A_20260607
/matx/u/knatalia/corl_mechanism_repair_ttrl_exact_A_20260608
/matx/u/knatalia/corl_mechanism_repair_ttrl_exact_A_20260609
/matx/u/knatalia/corl_mechanism_repair_ttrl_exact_B_20260607
/matx/u/knatalia/corl_mechanism_repair_ttrl_exact_B_20260608
/matx/u/knatalia/corl_mechanism_repair_ttrl_exact_B_20260609
```

Merge and analysis are handled by:

```bash
python scripts/merge_mechanism_repair_shards.py \
  --out-dir runs/mechanism_repair_ttrl_final \
  --run-analysis \
  --source-dir /path/to/shard_A_20260607 \
  --source-dir /path/to/shard_A_20260608 \
  --source-dir /path/to/shard_A_20260609 \
  --source-dir /path/to/shard_B_20260607 \
  --source-dir /path/to/shard_B_20260608 \
  --source-dir /path/to/shard_B_20260609
```

Verify this bundle with:

```bash
shasum -a 256 -c runs/mechanism_repair_ttrl_final/SHA256SUMS
```
