# Task 03A — fast sparse reference + residual predictive energy

**Status:** complete; the reviewed full study is a **NO-GO for Task 03B**.

| Purpose | File |
| --- | --- |
| Bounded implementation contract | [SPEC.md](SPEC.md) |
| PIT correction and oracle mathematics | [MATH.md](MATH.md) |
| Final eight-question summary | [SUMMARY.md](SUMMARY.md) |
| Full-run procedure and ZIP workflow | [COLAB.md](COLAB.md) |
| Executable notebook | [notebooks/task03a_colab.ipynb](../../notebooks/task03a_colab.ipynb) |
| Reviewed full evidence | [results/task03a/full/](../../results/task03a/full/README.md) |

Local validation is intentionally small:

```bash
uv run --no-sync pytest -q
uv run --no-sync python -m energy_bo.experiments.run_task03a --profile smoke
```

The smoke uses one seed, D=6, n=16/32, both oracle regimes, 128 test points,
256 candidates, and reduced NUTS. It validates wiring and timing, not the scientific
GO/NO-GO gate. The five-seed D=20 Colab run is complete and should not be rerun
routinely; use the reviewed evidence and canonical summary above.
