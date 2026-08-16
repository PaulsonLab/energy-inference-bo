# Task 05A full evidence

This directory contains the compact reviewed evidence from the frozen 20-shard A100
study executed at Git `56e8bc1fd8a4f16a600990211ff86151bc88d224`.

The mechanical result is **FAIL**. No S0/S1/S2 belief met the prespecified credibility
requirement at both low-data sizes, and neither structured model delivered the required
paired sequential-BO improvement. Task 05B is therefore not automatically authorized.

Start with:

- [`IMPORT_AUDIT.json`](IMPORT_AUDIT.json) for the independent integrity and numerical
  audit;
- [`gate_result.json`](gate_result.json) for the frozen mechanical decision;
- [Task 05A summary](../../../tasks/task05a/SUMMARY.md) for the concise scientific
  interpretation;
- [`offline_metrics.csv`](offline_metrics.csv) and
  [`sequential_metrics.csv`](sequential_metrics.csv) for the reviewed aggregate rows;
- [`figures/`](figures/) for the two generated overview plots.

The source package passed all 374 SHA-256 checks, and all 20 shards, 180 offline rows,
60 trajectory summaries, and 1,920 BO fits validated. The complete raw package is kept
under ignored `artifacts/task05a/full/` locally.

One non-gating defect was found during import: historical offline `one_step_regret`
did not retain the incumbent and consequently exceeded one in 118/180 rows. The source
implementation is corrected, the historical CSV is preserved unchanged, and the audit
records the corrected interpretation. Sequential regret and the frozen gate are
unaffected.
