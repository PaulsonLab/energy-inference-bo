# Task 04A-E Colab workflow

Use the guarded [Task 04A-E notebook](../../notebooks/task04ae_colab.ipynb) only if
the reviewed smoke gate says `AUTHORIZE_FULL`. The notebook checks that committed
gate itself; editing `RUN_FULL` cannot bypass a negative smoke result.

## Cell order

1. Select an A100 GPU runtime and leave `RUN_FULL=False`.
2. Clone `main` or a full commit SHA and build the locked project `.venv` with `uv`.
   No runtime restart or global scientific-package repair is used.
3. Verify CUDA/A100, use the noninteractive Matplotlib backend, and run `pytest -q`.
4. Inspect the committed smoke summary and gate. Stop unless it authorizes the run.
5. Set `RUN_FULL=True` and execute the full cell. Per-seed/size case JSON makes the
   run resumable when the artifact directory is restored.
6. Download the generated ZIP only after the subprocess completes.

The full command is:

```bash
.venv/bin/python -m energy_bo.experiments.run_task04ae \
  --profile full --device cuda --output-dir artifacts/task04ae/full
```

Extract a completed ZIP locally under `artifacts/task04ae/full/`, request an import
audit, and promote only compact reviewed evidence to `results/task04ae/full/`. The
notebook never authenticates to or pushes to GitHub.
