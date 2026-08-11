# Task 02A — fixed-support SAAS reuse diagnostic

**Status:** complete. This falsification experiment tests whether trusted SAAS
hyperparameter particles can be reused for later observations by changing only their
weights. It does not implement particle movement or an end-to-end BO loop.

## Read and run

| Purpose | File |
| --- | --- |
| Bounded implementation contract | [SPEC.md](SPEC.md) |
| Structural-posterior derivation | [MATH.md](MATH.md) |
| Local-smoke conclusions | [SUMMARY.md](SUMMARY.md) |
| Larger-run instructions | [COLAB.md](COLAB.md) |
| Reviewed full evidence | [results/task02a/full](../../results/task02a/full/) |
| Executable notebook | [notebooks/task02a_colab.ipynb](../../notebooks/task02a_colab.ipynb) |

Local CPU smoke command:

```bash
uv run --no-sync python -m energy_bo.experiments.run_task02a --profile smoke
```

This writes only to ignored `artifacts/task02a/smoke/`. The full notebook writes to
`artifacts/task02a/full/` and downloads one ZIP. A reviewed full package is already
tracked under `results/task02a/full/`; treat any rerun as a comparison and do not
overwrite that evidence without deliberately reviewing source SHA, configuration, and
Monte Carlo variation.
