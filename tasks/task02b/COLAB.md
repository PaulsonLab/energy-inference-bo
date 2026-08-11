# Task 02B Colab run

Use the tracked [Task 02B notebook](../../notebooks/task02b_colab.ipynb) or
[open it directly in Colab](https://colab.research.google.com/github/PaulsonLab/energy-inference-bo/blob/main/notebooks/task02b_colab.ipynb).
The full profile obtains per-particle acquisition signatures from fresh NUTS at all 18
saved Task 02A checkpoints. The retrospective preflight itself requires no new NUTS.

## What to do

1. Select an **NVIDIA GPU** runtime before executing cells; CPU is valid but may take
   hours. Set `ACCELERATOR = "gpu"` only after selecting the GPU runtime.
2. Replace `REPO_REF = "main"` with the published Task 02B commit SHA when possible.
3. Run setup, strict JAX backend validation, `pytest`, and the inexpensive
   retrospective preflight cells.
4. Set `RUN_FULL = True`, then run the full extraction cell.
5. Run the manifest/download cells to receive `task02b_full_outputs.zip`.

The full profile uses D=10, seeds 0–2, 256 retained particles, 2,048 candidates, and
512 warmup/512 NUTS samples with thinning 2 and tree depth 6. Checkpoints are processed
sequentially; JAX/NumPyro NUTS is the dominant cost. A signature matrix is about 4 MB.

## After download

Unzip locally and keep the original `artifacts/task02b/full/` tree for review. If the
full evidence passes its gates, create `results/task02b/full/` and copy only:

- `SUMMARY.md` and `colab_manifest.json`;
- aggregate spectra, coreset, and joint-target CSVs;
- the few plots directly used by the summary.

Do not commit compressed per-particle signature matrices; retain them in ignored
`artifacts/` or external storage. A downloaded ZIP never updates GitHub automatically.

## Terminal fallback

After installing the project and verifying the chosen JAX backend:

```bash
pytest -q
python -m energy_bo.experiments.run_task02b \
  --profile retrospective \
  --task02a-results results/task02a/full \
  --output-dir artifacts/task02b/full/retrospective
python -m energy_bo.experiments.run_task02b \
  --profile full \
  --task02a-results results/task02a/full \
  --output-dir artifacts/task02b/full \
  --retrospective-dir artifacts/task02b/full/retrospective \
  --summary-path artifacts/task02b/full/SUMMARY.md
```

Do not start Task 02C from the Colab runtime.
