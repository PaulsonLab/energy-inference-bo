# Task 05A A100 campaign guide

Do not run Task 05B. The preferred driver is
[`notebooks/task05a_colab.ipynb`](../../notebooks/task05a_colab.ipynb). It never
authenticates to or pushes to GitHub. Google Drive is used only to preserve Task 05A
checkpoints across Colab disconnects.

## Before Colab

1. Commit and push the Task 05A implementation and campaign notebook.
2. Open the [Task 05A notebook in
   Colab](https://colab.research.google.com/github/PaulsonLab/energy-inference-bo/blob/main/notebooks/task05a_colab.ipynb).
3. Select **Runtime → Change runtime type → A100 GPU**.
4. Prefer a full 40-character commit SHA for `REPO_REF`. `main` is accepted, but the
   notebook immediately resolves and pins its exact SHA.

The locked `.venv` is created with `uv sync --locked --group dev --no-editable` and
does not require a runtime restart. Setup verifies the A100, runs the unit suite, and
runs the smoke wiring profile. The smoke must remain `INCONCLUSIVE`; it cannot pass
the research gate.

## First session: required runtime profile

1. Set `RUN_PROFILE=True` and leave `RUN_CAMPAIGN=False`.
2. Run all cells.
3. Authorize the Google Drive mount.
4. Let `trpb_seed0` finish. The cell intentionally stops after this one shard.
5. Review `profile_review.json` and the printed 20-row campaign table. Confirm the
   shard is `COMPLETE`, runtime is below 2 h 55 min, and there is no technical error.
   The review also reports finite/converged counts, peak memory, and a naive 20-shard
   runtime projection. If the shard reaches the timeout, stop and return the
   diagnostic package for review.

Seed 0 is a compute and integrity profile, not a preliminary gate. Optimizer
nonconvergence is retained as scientific evidence rather than tuned away. A widespread
exception, nonfinite output, invalid checkpoint, or timeout is a technical reason to
stop; an isolated finite nonconvergence is something the frozen aggregate must count.
Do not infer calibration or structured-model benefit from this one seed.

Persistent evidence lives at:

```text
MyDrive/energy-inference-bo/task05a/<git-sha>/
```

Do not rename or merge anything in this directory.

## Later sessions: automatic resume

1. Set `RUN_PROFILE=False` and `RUN_CAMPAIGN=True`.
2. Run all cells from the top.
3. The notebook validates all twenty dataset/seed locations, skips complete shards,
   resumes partial checkpoints, and starts the next shard in deterministic order.
   `creilov_seed0` runs immediately after the TrpB profile so the longer protein and
   larger pool inform the conservative runtime estimate early; later seeds are paired
   by dataset.
4. The campaign stops cleanly before a new shard is unlikely to fit inside its
   eight-hour soft session budget. Reopen the notebook and repeat these steps.

The status is persisted in `campaign_status.json`, `campaign_status.csv`, and
`CAMPAIGN_STATUS.md`. Each shows `PENDING`, `PARTIAL`, `RUNNING`, `COMPLETE`,
`FAILED`, or `INCOMPATIBLE`, including completed offline fits, total BO steps,
runtime, last update, the next shard, and any error. A runtime disconnect leaves a
stale running shard as `PARTIAL`; the next session resumes it. A real failed or
incompatible shard stops the campaign instead of being silently skipped.

## Automatic aggregation and download

After all twenty shards pass frozen-protocol validation, the same campaign cell:

1. invokes the existing mechanical aggregate evaluator;
2. creates a SHA-256 inventory;
3. writes `task05a_full_results.zip` to Drive; and
4. opens the Colab browser download.

The ZIP contains exactly:

```text
task05a/
├── full_shards/
│   ├── trpb_seed0/ ... trpb_seed9/
│   └── creilov_seed0/ ... creilov_seed9/
├── aggregate/
│   ├── config.yaml
│   ├── metrics.csv
│   ├── gate_result.json
│   ├── run_metadata.json
│   └── figures/
├── campaign_status.json
├── campaign_status.csv
├── CAMPAIGN_STATUS.md
├── profile_review.json
├── colab_manifest.json
└── SHA256SUMS.json
```

Extract the ZIP locally without rearranging it and provide the extracted `task05a/`
folder for repository audit. The next repository call must verify checksums and the
frozen gate before promoting compact evidence or changing the program state. Do not
manually declare PASS or begin Task 05B.

## Failure recovery

On `FAILED` or `INCOMPATIBLE`, the notebook creates
`task05a_diagnostic.zip` in the same Drive directory and prints the precise failing
shard. Return that ZIP for review. Do not delete checkpoints or change the frozen
configuration to make the run continue.

The underlying controller is reproducible outside the notebook:

```bash
.venv/bin/python -m energy_bo.experiments.run_task05a_campaign \
  --mode campaign \
  --campaign-root /content/drive/MyDrive/energy-inference-bo/task05a/<git-sha> \
  --repo-dir /content/energy-inference-bo-task05a \
  --device cuda \
  --session-budget-seconds 28800 \
  --shard-timeout-seconds 10500
```
