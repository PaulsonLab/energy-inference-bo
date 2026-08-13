# Task 03A Colab run

Use the guarded [Task 03A notebook](../../notebooks/task03a_colab.ipynb). Select an
NVIDIA GPU runtime; JAX/NumPyro NUTS and batched prediction benefit most. The notebook
profiles official SciPy MAP fitting on CPU and CUDA rather than assuming CUDA helps.

## Cell order

1. Set `REPO_REF="main"` and select `ACCELERATOR="gpu"`. Setup records the resolved
   immutable SHA in the output manifest. A commit SHA is accepted only if it is already
   reachable from `main`; use `main` rather than asking GitHub to fetch a short SHA.
2. Run setup once. It checks Python 3.11–3.12 and installs the constrained locked
   wheels (including the locked pytest test gate) into `/content/task03a-packages`; every scientific child process runs with
   Colab's global site packages disabled and that directory plus `src/` as its only
   third-party import path. It therefore does **not** use Colab's currently unreliable
   `venv`/`ensurepip` implementation. Setup enables JAX float64, prints CUDA/JAX
   devices, and runs `pytest -q`. **No runtime restart is required or desired.**
   Re-running setup safely rebuilds only the isolated package directory without
   deleting experiment artifacts.
3. Run the preflight/smoke cell and inspect that it completes without identity or
   provenance failures.
4. Set `RUN_FULL=True` and run the full cell. Completed partial CSVs are written after
   each case; the final cell only packages outputs after the subprocess succeeds.
5. Download `task03a_full_outputs.zip`.

The isolated package directory prevents Colab's preinstalled NumPy/SciPy binaries
from mixing with the study dependencies, without relying on `venv`. If setup fails,
use **Runtime → Disconnect and delete runtime**, then begin from the configuration
cell; do not apply ad-hoc global `pip` repairs or restart between setup and preflight.

## What to do with the ZIP

Extract it locally as `artifacts/task03a/full/`. Review `TASK_03A_COLAB_SUMMARY.md`,
`SUMMARY.json`, `gate_status.json`, `metrics.csv`, `timing.csv`, `config.json`, and
`colab_manifest.json`. Keep raw/resumable artifacts
ignored. Ask for an import audit; only a reviewed summary, aggregate tables, manifest,
and decisive figures should be copied into `results/task03a/full/` and committed.
The notebook never pushes to GitHub.
