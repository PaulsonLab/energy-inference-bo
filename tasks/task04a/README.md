# Task 04A — local conditional energy process with oracle geometry

**Status:** local decision diagnostic complete and `INVALID`; full A100 study disabled.
Density learning replicated, but the oracle supplied no valid natural near-tie panel.

| Purpose | File |
| --- | --- |
| Bounded contract and frozen gates | [SPEC.md](SPEC.md) |
| Mathematical derivations | [MATH.md](MATH.md) |
| Current preflight summary | [SUMMARY.md](SUMMARY.md) |
| Completed local decision diagnostic | [DECISION_DIAGNOSTIC.md](DECISION_DIAGNOSTIC.md) |
| Paused Colab status | [COLAB.md](COLAB.md) |
| Disabled full-run driver | [notebooks/task04a_colab.ipynb](../../notebooks/task04a_colab.ipynb) |

Local validation is deliberately CPU-small:

```bash
uv run --no-sync pytest -q
uv run --no-sync python -m energy_bo.experiments.run_task04a --profile smoke
uv run --no-sync python -m energy_bo.experiments.run_task04a --profile preflight
```

The smoke uses one seed, D=6, n=64, all three truth regimes, 128 predictive test
points, and 256 EI candidates. It validates identities, batching, output structure,
and wiring; it cannot evaluate the five-seed GO/NO-GO gates.

The full notebook is retained for provenance but refuses full execution. Do not spend
an A100 session on the current contract or bypass the frozen preflight. A separately
approved decision-relevance diagnostic is required before any larger study.

Automatic outputs belong under ignored `artifacts/task04a/<profile>/`. There is no
reviewed full-result package under the current contract.
