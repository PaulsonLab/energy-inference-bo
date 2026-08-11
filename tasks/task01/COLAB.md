# Task 01 Colab run

Use the tracked [Task 01 notebook](../../notebooks/task01_colab.ipynb) or
[open it directly in Colab](https://colab.research.google.com/github/PaulsonLab/energy-inference-bo/blob/main/notebooks/task01_colab.ipynb).
This is a CPU-only 30-seed reproducibility expansion of the reviewed three-seed smoke
run, not an end-to-end BO benchmark.

## What to do

1. Leave the Colab runtime on **CPU**; expected memory is below 1 GB.
2. In the configuration cell, replace `REPO_REF = "main"` with a published commit SHA
   when possible.
3. Run the setup and test cells in order.
4. Set `RUN_FULL = True` only after the tests pass, then run the clearly marked full
   experiment cell.
5. Run the manifest/download cells to receive `task01_full_outputs.zip`.

The notebook writes `artifacts/task01/full/SUMMARY.md`, configuration JSON, aggregate
CSV, and figures. It does not authenticate to or update GitHub.

## After download

Unzip locally and preserve the `artifacts/task01/full/` directory while reviewing it.
If the run is valid, create `results/task01/full/` in your local clone and copy only:

- `SUMMARY.md`;
- `task01_config.json` and `task01_metrics.csv`;
- the one or two figures needed to support the report;
- `colab_manifest.json`.

Leave redundant generated figures local. Make a normal local commit and push only
after inspection.

## Terminal fallback

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
pytest -q
python -m energy_bo.experiments.run_task01 \
  --seeds $(seq 0 29) \
  --output-dir artifacts/task01/full \
  --report-path artifacts/task01/full/SUMMARY.md
```

Do not add q>1, large-dimensional BO, or Task 02 work to this run.
