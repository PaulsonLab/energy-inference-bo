# Colab index

Colab never authenticates to or pushes to GitHub. Publish the source revision first,
run the task-specific procedure, download the ZIP through the browser, review it
locally, and commit only deliberately selected compact evidence under `results/`.

| Task | Notebook | Run guide | Expected runtime policy |
| --- | --- | --- | --- |
| Task 01 | [open in Colab](https://colab.research.google.com/github/PaulsonLab/energy-inference-bo/blob/main/notebooks/task01_colab.ipynb) | [instructions](tasks/task01/COLAB.md) | CPU; below 1 GB |
| Task 02A | [open in Colab](https://colab.research.google.com/github/PaulsonLab/energy-inference-bo/blob/main/notebooks/task02a_colab.ipynb) | [instructions](tasks/task02a/COLAB.md) | CPU default; optional NVIDIA GPU for NUTS |
| Task 02B | [open in Colab](https://colab.research.google.com/github/PaulsonLab/energy-inference-bo/blob/main/notebooks/task02b_colab.ipynb) | [instructions](tasks/task02b/COLAB.md) | Full run complete; NVIDIA GPU preferred for reproduction |
| Task 02C | [open in Colab](https://colab.research.google.com/github/PaulsonLab/energy-inference-bo/blob/main/notebooks/task02c_colab.ipynb) | [instructions](tasks/task02c/COLAB.md) | Full run complete; reproduction only, NVIDIA GPU preferred |

Keep regenerated CSV/JSON/PNG files and raw signatures in ignored `artifacts/` until
they have been reviewed. A Colab download is local to the browser and does not change
this Git repository.

Every notebook uses `RUN_FULL = False` by default, checks Python and the selected JAX
backend, installs the chosen Git revision, runs `pytest`, records a manifest, and
downloads one ZIP. After unzipping locally, preserve the task/profile subdirectory;
copy only a reviewed summary and a small number of decisive tables/figures into the
matching `results/<task>/<profile>/` directory, then commit normally.
