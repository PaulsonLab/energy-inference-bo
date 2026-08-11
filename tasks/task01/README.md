# Task 01 — oracle and q=1 GP validation

**Status:** complete. This task validates scalar oracle shape effects, residual-energy
normalization, expected-utility augmentation, and exact-GP q=1 EI identities. It is not
an end-to-end BO benchmark.

## Read and run

| Purpose | File |
| --- | --- |
| Bounded implementation contract | [SPEC.md](SPEC.md) |
| Quantitative conclusions | [SUMMARY.md](SUMMARY.md) |
| Larger-run instructions | [COLAB.md](COLAB.md) |
| Reviewed three-seed evidence | [results/task01/smoke](../../results/task01/smoke/) |
| Executable notebook | [notebooks/task01_colab.ipynb](../../notebooks/task01_colab.ipynb) |

Task 01 uses the global mathematics in [MATH_AND_SCOPE.md](../../MATH_AND_SCOPE.md), so
it has no duplicate task-local math note.

Local CPU smoke command:

```bash
uv run --no-sync python -m energy_bo.experiments.run_task01
```

This writes only to ignored `artifacts/task01/smoke/`. The notebook expands the same
configuration to 30 seeds on CPU and downloads a ZIP. After review, promote only its
`REPORT.md`, the aggregate metrics/configuration, and a decisive figure into a new
`results/task01/full/` directory; keep redundant plots and raw generated files local.
