# Task 02C full Colab evidence

This directory contains the reviewed compact evidence from the completed Task 02C
falsification study. The run used source commit
[`9afa2a1`](https://github.com/PaulsonLab/energy-inference-bo/commit/9afa2a161048ec834f909383eefd27b7c6a2d068),
Python 3.12.13, an NVIDIA GPU, JAX float64, NumPyro 0.21.0, and 256-sample fresh-NUTS
teachers. Later repository changes only simplified the Colab handoff; the scientific
implementation used for this run is unchanged.

Start with the canonical [Task 02C summary](../../../tasks/task02c/SUMMARY.md). The
generated [Colab summary](SUMMARY.md) is retained verbatim.

## Audit and contents

- All expected 222 method rows, 90 fresh-teacher preflight rows, 21,600 trace rows,
  six teacher archives, and 216 final-particle archives were present.
- Every array was loaded with pickle disabled and was finite float64. Positive GP
  scales and unit-box final designs were verified.
- All 108 posterior/decision pairs had identical charged factorization counts,
  structural steps, cache builds, and design attempts.
- [`IMPORT_AUDIT.json`](IMPORT_AUDIT.json) records independently recomputed checks and
  headline metrics.
- [`task02c_methods.csv`](task02c_methods.csv),
  [`task02c_teacher_preflight.csv`](task02c_teacher_preflight.csv), the configuration,
  and manifest are the compact numerical record.
- The three figures show the matched regret/timing result, transport geometry, and
  fresh-teacher tilt ESS.

The 4.4 MB detailed trace, raw teacher states, and final transport particles remain in
ignored `artifacts/task02c/full/`. Their identities, together with the Colab preflight
files, are recorded in [`RAW_ARTIFACTS_SHA256.txt`](RAW_ARTIFACTS_SHA256.txt).

## Interpretation boundary

The exact energy and envelope checks passed and the teacher tilt was not intrinsically
degenerate. Nevertheless, DT-SVGD won only 20/108 matched runs and had worse median
regret and runtime than P-SVGD. Frequent forced initialization tempering, clipping,
and weak kernel repulsion make this a **NO-GO for the tested SVGD configuration**, not
a disproof of the exact decision-energy identities or every possible transport
method. This is not an end-to-end BO benchmark.
