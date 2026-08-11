# Task 02A Colab run

Use the tracked [Task 02A notebook](../../notebooks/task02a_colab.ipynb) or
[open it directly in Colab](https://colab.research.google.com/github/PaulsonLab/energy-inference-bo/blob/main/notebooks/task02a_colab.ipynb).
It runs the full fixed-support SAAS-reuse diagnostic and never authenticates to or
pushes to GitHub.

## What to do

1. **CPU is the reproducible default.** For faster NUTS fits, select a Colab NVIDIA
   GPU before starting and set `ACCELERATOR = "gpu"`. The exact-GP cache remains on
   CPU by design.
2. Replace `REPO_REF = "main"` with a published commit SHA when possible.
3. Run setup, backend validation, and `pytest` cells in order. The GPU branch fails if
   CUDA JAX is not actually visible; it never silently falls back to CPU.
4. Set `RUN_FULL = True`, then run the full experiment cell.
5. Run the manifest/download cells to receive `task02a_full_outputs.zip`.

The full profile uses seeds 0–2, D=10, n=16→40, 256 retained particles, 512 test
points, 2,048 candidates, and 512 warmup/512 NUTS samples with thinning 2. CPU may take
hours; JAX/NumPyro NUTS dominates the runtime.

## After download

A reviewed full package already exists at [`results/task02a/full/`](../../results/task02a/full/).
Keep a rerun under `artifacts/task02a/full/` while comparing its Git SHA, configuration,
summary, and Monte Carlo variation. Do not overwrite the published evidence merely
because a new ZIP exists. If a rerun is intentionally published, retain its
`SUMMARY.md`, compact CSV/PNG files, and `colab_manifest.json`; raw extras remain local.

## Terminal fallback

After installing the project and selecting `JAX_PLATFORMS=cpu` or `cuda`:

```bash
pytest -q
python -m energy_bo.experiments.run_task02a \
  --profile full \
  --seeds 0 1 2 \
  --output-dir artifacts/task02a/full \
  --summary-path artifacts/task02a/full/SUMMARY.md
```

Do not add particle moves, Vecchia, an output EBM, q>1, or a BO loop to Task 02A.
