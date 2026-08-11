# Task 02B — decision-space compression and joint-energy validation

**Status:** active; **NO-GO for Task 02C pending full evidence**. This task tests
whether fresh SAAS posterior uncertainty is compressible in acquisition space and
checks the discrete M=1/M=2 structural-decision identities. It does not implement a
transport algorithm.

## Read and run

| Purpose | File |
| --- | --- |
| Bounded implementation contract | [SPEC.md](SPEC.md) |
| Decision-energy derivation | [MATH.md](MATH.md) |
| Current quantitative conclusions | [SUMMARY.md](SUMMARY.md) |
| Larger-run instructions | [COLAB.md](COLAB.md) |
| Reviewed retrospective/smoke evidence | [results/task02b](../../results/task02b/) |
| Executable notebook | [notebooks/task02b_colab.ipynb](../../notebooks/task02b_colab.ipynb) |

Cheap saved-result analysis:

```bash
uv run --no-sync python -m energy_bo.experiments.run_task02b --profile retrospective
```

Local NUTS smoke (run only when validating that path):

```bash
uv run --no-sync python -m energy_bo.experiments.run_task02b --profile smoke
```

Automatic outputs go to `artifacts/task02b/<profile>/`. The full notebook is
GPU-preferred and extracts fresh signatures at all checkpoints. After review, promote
`SUMMARY.md` and compact aggregate CSV/PNG evidence into `results/task02b/full/`; keep
the compressed per-particle signature matrices in ignored artifacts or external
storage.
