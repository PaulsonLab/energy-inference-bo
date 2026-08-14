# Task 04A evidence

Task 04A's full study is paused. Its implementation, original CPU smoke, and revised
preflight are summarized in the [task summary](../../tasks/task04a/SUMMARY.md); raw
development outputs remain ignored under `artifacts/task04a/`.

A maintainer-requested raw-output diagnostic and the subsequent standardized-child
preflight are summarized there. Raw outputs remain ignored under
`artifacts/task04a/local_diagnostic/` and `artifacts/task04a/preflight_standardized/`.
Standardization repaired the density signal, but the frozen preflight found no
pairwise decision benefit beyond U, so the full A100 run remains paused.

The subsequent [withheld-seed decision diagnostic](decision_diagnostic/README.md)
replicated the density gain but was officially `INVALID`: no I case supplied a single
qualifying natural near-tie pair, and the oracle rarely changed counterfactual EI by
1%. The full study remains disabled.

The reviewed full-result package does not yet exist. After the A100 ZIP is downloaded,
audit it before adding `results/task04a/full/`. Preserve raw cases and resumable output
outside Git history.
