# Task 02B — decision-space compression and joint-energy validation

**Status:** complete; **GO for a separately specified Task 02C falsification
experiment**. Full evidence shows decision-space compressibility and passes the exact
M=1/M=2 structural-decision identities. This task does not implement a transport
algorithm or establish that transport will succeed.

## Read and run

| Purpose | File |
| --- | --- |
| Bounded implementation contract | [SPEC.md](SPEC.md) |
| Decision-energy derivation | [MATH.md](MATH.md) |
| Full quantitative conclusions | [SUMMARY.md](SUMMARY.md) |
| Larger-run instructions | [COLAB.md](COLAB.md) |
| Reviewed retrospective, smoke, and full evidence | [results/task02b](../../results/task02b/) |
| Executable notebook | [notebooks/task02b_colab.ipynb](../../notebooks/task02b_colab.ipynb) |

Cheap saved-result analysis:

```bash
uv run --no-sync python -m energy_bo.experiments.run_task02b --profile retrospective
```

Local NUTS smoke (run only when validating that path):

```bash
uv run --no-sync python -m energy_bo.experiments.run_task02b --profile smoke
```

Automatic outputs go to `artifacts/task02b/<profile>/`. The full GPU-preferred run has
now been reviewed and promoted to [`results/task02b/full/`](../../results/task02b/full/).
Its raw per-particle signature matrices remain ignored and are represented by tracked
checksums. Re-run the notebook only for reproduction or a deliberately revised study.
