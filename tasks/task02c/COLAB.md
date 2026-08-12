# Task 02C Colab run

The archival full study is complete and reviewed. Use this notebook only to reproduce
that result; routine reruns are not recommended. See the [full evidence](../../results/task02c/full/README.md)
and [canonical conclusion](SUMMARY.md) first.

Use the tracked [Task 02C notebook](../../notebooks/task02c_colab.ipynb). Select an
**NVIDIA GPU** before starting. JAX/NumPyro NUTS and SVGD use the selected backend;
the notebook fails if GPU was requested but CUDA is unavailable. CPU is valid for
preflight but is not recommended for the full experiment.

## Cell order

1. Set `REPO_REF` to the published Task 02C commit SHA, not a moving branch, and set
   `ACCELERATOR = "gpu"` after selecting a GPU runtime. The notebook explicitly
   fetches that revision before checkout, so a SHA works with Colab's filtered clone.
2. Run setup. The notebook checks Python 3.11–3.12, installs the locked project,
   enables JAX float64 before imports, prints the Git SHA/devices, and runs `pytest`.
3. Set `RUN_PREFLIGHT = True`. This downloads the separately hosted Task 02B raw
   signature archive only if you first make it available in the runtime; the notebook
   otherwise explains the required `artifacts/task02b/full/signatures/` folder. Run
   the six-case saved-teacher preflight and inspect `passed: true`.
4. Only then set `RUN_FULL = True`. The runner fits fresh NUTS teachers and processes
   all cases/K/budgets/repeats sequentially. Existing teacher files are reused after a
   runtime interruption.
5. The same full-run cell writes the manifest, creates the ZIP, and starts the browser
   download of `task02c_full_outputs.zip` after the experiment succeeds. Do not run a
   separate save cell.

The full configuration has six NUTS teachers, 256 retained samples each, K=8/16/32,
32/64 structural steps, and three repeats. Exact GP matrices are at most 32x40x40 for
transport; NUTS and repeated structural gradients dominate runtime. Allow several GPU
hours and keep the browser session active.

## After download

Unzip into `artifacts/task02c/full/` locally. Review `SUMMARY.md`,
`task02c_methods.csv`, `task02c_teacher_preflight.csv`, `task02c_config.json`, and the
manifest. Promote only the reviewed summary, aggregate tables, manifest, and decisive
figures to `results/task02c/full/`. Keep raw NUTS states and traces ignored. A Colab
download never modifies GitHub and the notebook contains no credentials or push step.
