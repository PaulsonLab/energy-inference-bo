# Task 03A — fast sparse reference + residual predictive energy

**Status:** active implementation; local smoke evidence only until the guarded Colab
study is run and reviewed.

| Purpose | File |
| --- | --- |
| Bounded implementation contract | [SPEC.md](SPEC.md) |
| PIT correction and oracle mathematics | [MATH.md](MATH.md) |
| Provisional eight-question summary | [SUMMARY.md](SUMMARY.md) |
| Full-run procedure and ZIP workflow | [COLAB.md](COLAB.md) |
| Executable notebook | [notebooks/task03a_colab.ipynb](../../notebooks/task03a_colab.ipynb) |

Local validation is intentionally small:

```bash
uv run --no-sync pytest -q
uv run --no-sync python -m energy_bo.experiments.run_task03a --profile smoke
```

The smoke uses one seed, D=6, n=16/32, both oracle regimes, 128 test points,
256 candidates, and reduced NUTS. It validates wiring and timing, not the scientific
GO/NO-GO gate. The full five-seed D=20 run belongs in Colab.
