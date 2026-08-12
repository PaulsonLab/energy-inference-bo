# Task 02C — decision-tilted structural SVGD

**Status:** complete; **NO-GO for the tested decision-tilted SVGD configuration**.
The exact target and envelope mathematics passed, but decision tilting did not improve
q=1 teacher regret at matched structural compute.

| Purpose | File |
| --- | --- |
| Bounded implementation contract | [SPEC.md](SPEC.md) |
| Gibbs, envelope, and annealing mathematics | [MATH.md](MATH.md) |
| Full quantitative conclusion | [SUMMARY.md](SUMMARY.md) |
| Reproduction-only Colab guide | [COLAB.md](COLAB.md) |
| Reviewed full evidence | [results/task02c/full](../../results/task02c/full/) |
| Executable notebook | [notebooks/task02c_colab.ipynb](../../notebooks/task02c_colab.ipynb) |

The local commands remain available as development checks, but they are not reasons
to rerun the completed scientific study:

```bash
uv run --no-sync python -m energy_bo.experiments.run_task02c --profile preflight
uv run --no-sync python -m energy_bo.experiments.run_task02c --profile smoke
```

Raw full-run teachers, particles, and traces remain under ignored
`artifacts/task02c/full/`; their checksums are tracked with the reviewed evidence.
Do not implement another transport method from this result without a new bounded task
contract.
