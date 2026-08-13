# Task 03A full Colab evidence

This directory contains the compact reviewed evidence from the completed Task 03A
oracle capability study. The run used source commit
[`a671671`](https://github.com/PaulsonLab/energy-inference-bo/commit/a671671ef70112f4aefefc2e1287f9d65cd3ac36),
Python 3.12.13, an NVIDIA A100 runtime, JAX float64, NumPy 2.3.5, SciPy 1.16.3,
PyTorch 2.13.0+cu130, BoTorch 0.18.1, and NumPyro 0.21.0.

Start with the canonical [Task 03A summary](../../../tasks/task03a/SUMMARY.md). The
automatically generated [Colab summary](SUMMARY.md) is retained verbatim.

## Audit and contents

- All expected 240 primary rows, 30 timing rows, 1,200 structural-relevance rows,
  four shrinkage-sensitivity rows, five component-sensitivity rows, and 30 case JSON
  files were present; all recorded numeric values were finite.
- Case keys exactly matched the aggregate metric and timing keys. The final and
  partial metric/timing CSVs were byte-identical, showing a clean completed run.
- [`IMPORT_AUDIT.json`](IMPORT_AUDIT.json) records independently recomputed gate,
  predictive, decision, cost, sensitivity, and integrity diagnostics.
- [`task03a_metrics.csv`](task03a_metrics.csv),
  [`task03a_timing.csv`](task03a_timing.csv),
  [`task03a_structural_relevance.csv`](task03a_structural_relevance.csv), both
  sensitivity tables, the configuration, manifest, device profile, and overview
  figure are the compact numerical record.
- The 30 per-case resume archives and duplicate partial CSVs remain in ignored
  `artifacts/task03a/full/`; their identities are recorded in
  [`RAW_ARTIFACTS_SHA256.txt`](RAW_ARTIFACTS_SHA256.txt).

## Interpretation and caveats

The result is **NO-GO**, not an implementation failure. E0/E1 did not beat the
strongest calibrated baseline under warped truth, and raw Ensemble MAP-SAAS did not
meet the structural-only fallback at warped n=64. The A100 made NUTS unusually fast,
so the absolute and relative wall times are hardware-specific; this does not affect
NLL, calibration, EI, regret, or the predictive-flexibility failure.

Two instrumentation limitations are explicit. The output records only `cuda:0`, not
the GPU model; the A100 identity comes from the maintainer's run record. Also, the
MAP forward-hook counters are all zero despite nonzero optimizer iterations, so those
counters are invalid and wall-time/iteration accounting should be used instead. The
case JSON files do not serialize fold index arrays, so leak-free cross-fitting is
supported by tested construction and runtime assertions rather than independently
reconstructed from the result package.

This remains a bounded q=1 oracle study, not a sequential BO benchmark.
