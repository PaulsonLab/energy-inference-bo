# Task 02B full Colab evidence

This directory contains the reviewed, compact evidence from the full three-seed Task
02B study. The run used source commit
[`772f59f`](https://github.com/PaulsonLab/energy-inference-bo/commit/772f59fb029bbebdff3bb22885988c1d37661b89),
Python 3.12.13, GPU-backed JAX 0.9.2, NumPyro 0.21.0, PyTorch 2.13.0, BoTorch 0.18.1,
and GPyTorch 1.15.2. Start with the generated [quantitative summary](SUMMARY.md).

## Configuration and audit

- 3 seeds, D=10, 18 fresh-NUTS checkpoints, 256 retained particles per checkpoint,
  and 2,048 fixed candidates.
- All 18 raw signature archives were read with pickle disabled. Every array was finite
  and float64; lengthscales/outputscales were positive and EI values nonnegative.
- Each archive had lengthscales `[256,10]`, particle EI `[256,2048]`, candidates
  `[2048,10]`, and teacher EI `[2048]`. Recomputed particle-mean versus teacher EI
  differed by at most `5.55e-16`.
- Candidate arrays were identical across checkpoints for each seed. Reconstructing the
  Sobol candidates on the maintainer's ARM Mac differed from the archived x86 Colab
  values by at most `2.98e-08`; all reported metrics use the archived Colab arrays.
- Aggregate tables contain 4,824 spectrum rows and 3,060 coreset rows. Every recorded
  finite-candidate regret-bound check passed.
- Fresh NUTS fits totaled `233.22 s` (`11.19 s` median; `10.20`–`21.76 s`) on the
  recorded Colab GPU runtime.

## Tracked contents

- `colab_manifest.json`: source SHA, environment, backend/devices, and exact commands.
- `IMPORT_AUDIT.json`: machine-readable independent integrity checks and gate result.
- `task02b_config.json`: full configuration and checkpoint-level derived diagnostics.
- `task02b_spectra.csv`, `task02b_coresets.csv`, and
  `task02b_joint_targets.csv`: aggregate evidence used by the summary.
- `acquisition_signature_spectrum.png`, `coreset_quality.png`, and
  `joint_target_validation.png`: reviewed full-run figures.
- `SIGNATURES_SHA256.txt`: hashes and byte sizes of the excluded raw signature files.

The 18 raw `.npz` signature matrices total about 69 MB and are intentionally not
committed. They remain in ignored `artifacts/task02b/full/signatures/` in the importing
workspace. The checksum inventory makes that external/raw evidence identifiable
without bloating Git history. The already-published Task 02A retrospective analysis is
not duplicated here.

## Interpretation boundary

The full evidence passes the prespecified Task 02B gate and justifies designing a
bounded Task 02C transport falsification experiment. It does not demonstrate that a
transport algorithm works, does not establish end-to-end BO gains, and does not itself
implement Task 02C.
