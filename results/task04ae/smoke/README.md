# Task 04A-E reviewed CPU smoke

The oracle and paired decision opportunity passed, but the learned P model did not.
At `r=1`, P improved joint KL over U by only `9.87%` versus the required 20%, its
maximum q=1 EI error was `18.19%`, and its mean r>0 qEI error was `19.92%` versus
`5.89%` for G0. P selected the wrong endpoint for every paired state and doubled
tie-aware regret from `3.41%` to `6.82%`.

The frozen decision is **`LEARNING_NO_GO`**. This is a one-seed mechanism gate, not a
general performance benchmark. P moved upper-tail co-exceedance in the correct
direction, but an unintended standardized-mean shift inflated q=1 EI and overwhelmed
that dependence improvement. The result is specific to this affine, precision-10,
`n=128` correction; it is not a rejection of the oracle or every marginal-preserving
dependence model.

- [Frozen gate](gate_status.json)
- [Oracle preflight](oracle_preflight.json)
- [Reviewed aggregate](smoke_metrics.csv)
- [Task interpretation](../../../tasks/task04ae/SUMMARY.md)
