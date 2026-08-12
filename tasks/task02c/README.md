# Task 02C — decision-tilted structural SVGD

**Status:** implementation and local validation complete; the full six-case Colab
falsification study is pending. The current recommendation is **NO-GO pending full
evidence**, not a transport success claim.

| Purpose | File |
| --- | --- |
| Bounded implementation contract | [SPEC.md](SPEC.md) |
| Gibbs, envelope, and annealing mathematics | [MATH.md](MATH.md) |
| Local quantitative status | [SUMMARY.md](SUMMARY.md) |
| Full-run instructions | [COLAB.md](COLAB.md) |
| Local reviewed evidence | [results/task02c](../../results/task02c/) |
| Executable notebook | [notebooks/task02c_colab.ipynb](../../notebooks/task02c_colab.ipynb) |

Run the saved-teacher preflight and the one permitted local wiring smoke:

```bash
uv run --no-sync python -m energy_bo.experiments.run_task02c --profile preflight
uv run --no-sync python -m energy_bo.experiments.run_task02c --profile smoke
```

Both commands require the ignored Task 02B signature archive under
`artifacts/task02b/full/signatures/`. They never fit NUTS. Automatic outputs go to
`artifacts/task02c/`; do not interpret the reduced smoke as performance evidence.

The full study is GPU-preferred and guarded in the notebook. It fits fresh float64
NUTS teachers and writes a resumable artifact package. Do not run it locally.
