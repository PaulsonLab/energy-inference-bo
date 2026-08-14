# Task 04A-E — decision-relevant joint dependence

Task 04A-E is a bounded q=2 oracle mechanism test. It asks whether a small convex
pairwise energy can learn higher-order dependence that is invisible to one-point
marginals and Pearson covariance, then improve batch expected-improvement decisions.

| Document | Purpose |
| --- | --- |
| [SPEC.md](SPEC.md) | Frozen experiment, gates, and scope |
| [MATH.md](MATH.md) | Copula, convexity, normalization, and qEI identities |
| [SUMMARY.md](SUMMARY.md) | Local smoke evidence and current authorization state |
| [COLAB.md](COLAB.md) | Guarded full-study workflow |

Local command:

```bash
uv run --no-sync python -m energy_bo.experiments.run_task04ae \
  --profile smoke --device cpu --output-dir artifacts/task04ae/smoke
```

The historical Task 04A q=1 experiment remains unchanged. Automatic output belongs
under ignored `artifacts/task04ae/`; only reviewed compact evidence belongs under
`results/task04ae/`.
