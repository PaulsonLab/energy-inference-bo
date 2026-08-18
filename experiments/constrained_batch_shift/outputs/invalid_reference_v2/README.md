# Constrained Batch Shift — invalid-reference evidence

## Outcome

The frozen A100 campaign is **INVALID**, not GO or NO-GO. States 3101 and
3102 completed under implementation tag `constrained-batch-shift-gpu-v2`.
State 3103 then failed the prospectively frozen high-budget reference check for
the Gaussian belief after reaching the maximum $2^{18}$ samples per
replicate. The runner stopped before States 3104–3108, as required by the
protocol.

The exact failed convergence components were not serialized before the
exception. The maintainer-provided Colab traceback records
`INVALID_REFERENCE: convergence failed at frozen cap`; the downloaded package
contains State 3103's data and candidate labels but no reference arrays or
summary. This missing failure-detail serialization is an implementation
limitation and prevents post-hoc diagnosis of whether value, gradient cosine,
or gradient-norm convergence was decisive.

## Integrity

- Protocol hash: `4dd20a796d9eb537311ba72eab4f539376b6fcb73a25e0e405500e3c68ff1d03`
- Implementation Git SHA: `0db7ea2b2e45606a488611e286617caec6d93180`
- Runtime: Python 3.12.13, PyTorch 2.13.0+cu130, BoTorch 0.18.1,
  GPyTorch 1.15.2, NVIDIA A100-SXM4-40GB
- Diagnostic archive SHA-256:
  `c15a2d84db2b1e336ee351b961adff04d22177db5a3a7b751afb4f0a5a5dc1ba`
- All 27 archive entries match the extracted files byte-for-byte.
- Both complete states have 56 observations, 448 candidates, 384 practical-QMC
  rows, 96 optimizer rows, finite/bounded arrays, converged diagnostic fits,
  and Gaussian/Student covariance discrepancies below `1.4e-17`.
- State 3103 has 56 finite observations and 448 candidate labels, but correctly
  has no completed summary or raw reference archive.

Raw arrays and the 67.5 MB diagnostic ZIP remain ignored under
`artifacts/constrained_batch_shift/invalid_reference_v2/`. The tracked evidence
here is compact and independently recomputed.

## What the two valid states show

Across the four completed state/belief cells, none of the 180 top-decile
high-acquisition batches had ESS fraction at or below the frozen `0.05`
threshold. Their median ESS fractions were `0.411–0.452`, and their minimum
top-decile ESS fractions were `0.360–0.428`. Low ESS did occur elsewhere, but
only at low acquisition quality: the best low-ESS batch reached at most `7.21%`
of the statewise reference maximum.

At 512 QMC samples:

- median high-set value errors were `0.275–0.487%`;
- ranking disagreement was `3.64–5.35%`;
- the selected panel candidate had zero reference regret in all four cells;
- State 3101 gradient cosines were `0.983` (Gaussian) and `0.978` (Student-t);
- State 3102 gradient cosines were `0.873` and `0.859`, with relative gradient
  errors `0.639` and `0.656`;
- median outer-optimizer regret was `9.49–19.55%`.

Thus State 2 met the frozen *material QMC failure* rule through gradient and
outer-optimizer conditions, but neither state was shift-positive. This suggests
that its optimization difficulty was not explained by the prespecified severe
posterior-to-decision shift. Gaussian and Student-t results were very similar;
there is no visible heavy-tail amplification in these two states.

These observations are descriptive only. Two valid states cannot establish the
eight-state gate, and the State 3 reference failure makes extrapolation
scientifically inappropriate.

## Evidence map

- [`gate_result.json`](gate_result.json): audited invalid classification
- [`IMPORT_AUDIT.json`](IMPORT_AUDIT.json): structural and numerical checks
- [`completed_state_summary.csv`](completed_state_summary.csv): frozen
  state-level conditions
- [`qmc_summary.csv`](qmc_summary.csv): sample-count curves
- [`optimizer_summary.csv`](optimizer_summary.csv): outer-optimizer regret
- [`RAW_SHA256SUMS.txt`](RAW_SHA256SUMS.txt): inventory for ignored raw evidence

## Next decision

No sampler or downstream experiment is authorized. Human review must decide
whether to stop this paper direction or approve a separately frozen numerical-
reference diagnostic. Any revision must preserve this campaign as invalid and
must not reinterpret its two completed states as a gate result.
