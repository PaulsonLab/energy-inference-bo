# Task 02A full Colab results

This is the complete three-seed Task 02A result package generated in Google Colab on
2026-08-11. It is deliberately tracked outside the ignored `artifacts/` directory so
the published evidence remains reviewable alongside the code.

The run used the prescribed full configuration: `D=10`, `n=16→40`, seeds `0, 1, 2`,
256 retained SAAS particles, 512 warmup and 512 NUTS samples with thinning 2, 512 test
points, and 2,048 fixed EI candidates. The manifest pins the source to commit
`392937c458f832787f6ffc7b5f53e9d1543c3445`, records Python 3.12.13 and GPU-backed JAX,
and matches this repository's exported-requirements SHA-256.

Start with [the full quantitative summary](TASK_02A_COLAB_SUMMARY.md). Its central
negative result is rapid fixed-support weight collapse: final mean
`-log(ESS/P)=5.268`, final MMD `0.7878`, and q=1 EI decision agreement only `1/3`.
Accordingly, the recorded recommendation is **02B-B**: investigate another particle
transport method before attempting a resample-move SMC implementation.

## Contents

- `colab_manifest.json`: source revision, runtime, JAX device/backend, packages, and
  exact command.
- `task02a_config.json`: per-seed frozen transforms, counters, NUTS times, and aggregate
  metrics.
- `task02a_rounds.csv`: every sequential reweighting/cache update.
- `task02a_checkpoints.csv`: reweighted-versus-fresh posterior, predictive, and EI
  metrics at reference checkpoints.
- `task02a_lengthscales.csv` and `task02a_coordinate_drift.csv`: structural posterior
  summaries and active-versus-irrelevant coordinate drift.
- `task02a_ei.csv`: all candidate-level q=1 EI values for fresh and reweighted mixtures.
- `ess_reuse_horizon.png`, `lengthscale_posteriors.png`, `coordinate_drift.png`,
  `ei_comparison.png`, and `timing_cost.png`: the corresponding figures.

These files are evidence for the Task 02A falsification experiment, not a BO benchmark
and not an implementation of Task 02B.
