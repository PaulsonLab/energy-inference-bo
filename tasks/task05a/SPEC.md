# Task 05A frozen protocol

Protocol version: `task05a-v1`.

## Scientific question

On measured TrpB and CreiLOV pools, is at least one exact-GP belief credibly
calibrated in the high-predicted-utility region, and does biological sequence
structure materially improve BO-relevant decisions?

## Data and preprocessing

The measured `fitness.csv` files come from SGPO commit
`290fa8a4cc99d50980cb8d7cf85ae76744552ead`. TrpB has 111,883 unique length-15
sequences and SHA-256 `f4627857282bef95be5485ff13aef99579c10b5b0e7bf8098ad0cdcbc6a8013a`.
CreiLOV has 167,530 unique length-119 sequences and SHA-256
`e3c09bb02ea6e6da0046ddf3d16b4c370dadeaad79d8e0cb3bcda4e02d5d7f8f`.
Every finite measured fitness is used. SGPO's surrogate split is ignored.

SGPO's processing notebook labels TrpB with the merged dataset's
`norm_fitness_mean`, clips negative values to zero, and retains selected
three-/four-mutation variants. CreiLOV uses experimental `log_mean`; it aligns
the single-mutant experiment to the multi-mutant experiment through their WT
difference, prefers the multi-mutant measurement for duplicates, removes stop
variants, and does not divide by the maximum. Task 05A consumes these published
processed labels unchanged and records that they are not a common raw assay scale.

Sequences are lexicographically canonicalized and integer encoded. Ten
dataset-specific PCG64 permutations, seeds 0–9, define nested train prefixes and
are shared by all models. At every fit, fitness is standardized using only the
currently observed set; predictions are returned to raw units. Observed variance
is used for scoring/calibration and latent variance for LogEI.

## Models and fitting

- **S0:** exact zero-mean GP with isotropic Hamming RBF and learned scale/noise.
- **S1:** exact zero-mean GP with BLOSUM62 positive-spectrum embedding and
  Tanimoto kernel, learned scale/noise.
- **S2:** integer-coded adaptation of LOCK-GP using published BLOSUM50 correlation
  normalization, nonlinear locally weighted product-plus-linear kernel, two
  scales, global and positionwise exponents, and published Gamma/LogNormal priors.

The S2 formulas and priors are audited against the authors' public GPyTorch source at
commit `df384fe24c26ebc3ac4a8aab49809d66104f7e8e`; this compact adaptation is not
represented as an official wrapper. All models use GPyTorch `ExactGP`,
`GaussianLikelihood`, exact marginal likelihood,
BoTorch's SciPy fitting path, double precision, neutral initialization, at most
200 L-BFGS-B iterations, and sequential warm starts. Failed/nonfinite fits remain
failures.

## Frozen full experiment

- Datasets: TrpB and CreiLOV.
- Seeds: 0–9, paired across S0/S1/S2.
- Offline train sizes: 48, 96, 192; test set is the remaining measured pool.
- Sequential BO: 48 initial observations, 32 LogEI selections from the unobserved
  measured pool.
- One resumable shard is one dataset/seed pair. Fits checkpoint after every BO
  iteration. Each shard must stay below 3 A100 hours.
- Primary metrics: high-utility calibration/NLL, one-step regret, selected true
  top-decile rate, normalized-regret AUC, final regret, and top-1%/top-5%
  discovery. Secondary metrics: overall NLL/CRPS/RMSE/Spearman/top-10% recall,
  convergence, wall time, and memory.

The high-utility region is the top 10% of held-out candidates by posterior mean.
Interval error averages absolute 50/80/90/95% coverage errors. A model is credible
on a dataset when at n=48 and 96: median interval error ≤0.10; ≥8/10 splits have
90% coverage error ≤0.20; median high-utility NLL is ≤0.10 nat worse than a
train-only constant Gaussian; and ≥9/10 fits are finite and converged.

Catastrophic calibration means interval error >0.20, NLL gap >0.25 nat, or >2
failed fits. Severe cross-dataset regression means regret AUC worsens by both >20%
relative and >0.05 absolute, or final top-1% discovery falls ≥20 percentage points.

## Prospective gate

- **PASS-A:** credible S1/S2 has ≥20% median paired AUC reduction with ≥7/10
  wins, or ≥15 percentage-point final top-1% discovery gain, without catastrophic
  calibration or severe regression on the other dataset.
- **PASS-B:** S0 is credible on both datasets, neither structured model reaches
  PASS-A, and structured gains stay below 10% AUC and 10 percentage points in
  top-1% discovery.
- **FAIL:** no model supplies a credible belief meeting the control plan.
- **INCONCLUSIVE:** evidence lies between these rules; downstream work remains
  blocked pending human review.

Smoke uses both datasets, seed 0, n=48, two BO steps, 2,048 candidates, and 25
optimizer iterations. It is never gate evidence. There is no automatic fallback.

## Expected full aggregate outputs

`config.yaml`, `metrics.csv`, `offline_metrics.csv`, `sequential_metrics.csv`,
`gate_result.json`, `run_metadata.json`, and reviewed figures under
`results/task05a/<run_id>/`. Downloads and checkpoints remain ignored.
