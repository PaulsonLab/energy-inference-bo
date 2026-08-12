# Task 03A Colab run

Use the guarded [Task 03A notebook](../../notebooks/task03a_colab.ipynb). Select an
NVIDIA GPU runtime; JAX/NumPyro NUTS and batched prediction benefit most. The notebook
profiles official SciPy MAP fitting on CPU and CUDA rather than assuming CUDA helps.

## Cell order

1. Set `REPO_REF` to the published Task 03A commit SHA and select `ACCELERATOR="gpu"`.
2. Run setup. It checks Python 3.11–3.12, installs the locked revision, enables JAX
   float64, prints CUDA/JAX devices, and runs `pytest -q`.
3. Run the preflight/smoke cell and inspect that it completes without identity or
   provenance failures.
4. Set `RUN_FULL=True` and run the full cell. Completed partial CSVs are written after
   each case; the final cell only packages outputs after the subprocess succeeds.
5. Download `task03a_full_outputs.zip`.

## What to do with the ZIP

Extract it locally as `artifacts/task03a/full/`. Review `TASK_03A_COLAB_SUMMARY.md`,
`SUMMARY.json`, `gate_status.json`, `metrics.csv`, `timing.csv`, `config.json`, and
`colab_manifest.json`. Keep raw/resumable artifacts
ignored. Ask for an import audit; only a reviewed summary, aggregate tables, manifest,
and decisive figures should be copied into `results/task03a/full/` and committed.
The notebook never pushes to GitHub.
