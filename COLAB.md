# Colab runs

## Task 01

The checked-in smoke result uses only three seeds locally. Run the following in a new
Google Colab notebook for the planned 30-seed CPU/GPU-optional validation; it is still
only a scalar nine-parameter residual fit, but avoids consuming the local laptop.

```python
!git clone https://github.com/PaulsonLab/energy-inference-bo.git
%cd energy-inference-bo
!pip install -r requirements.txt
!pip install -e .
!pytest -q
!python -m energy_bo.experiments.run_task01 \
    --seeds $(seq 0 29) \
    --output-dir artifacts/task01_colab \
    --report-path reports/TASK_01_COLAB.md
```

The run is CPU-compatible and should remain well below 1 GB RAM. To download outputs:

```python
!zip -r task01_colab_outputs.zip artifacts/task01_colab reports/TASK_01_COLAB.md
from google.colab import files
files.download("task01_colab_outputs.zip")
```

Do not run q>1, large-dimensional BO, particle sweeps beyond the configuration above,
or Task 02 from this notebook.

## Task 02A full SAAS reuse diagnostic

The preferred full-run interface is the tracked
[Task 02A Colab notebook](notebooks/task02a_colab.ipynb) ([open in
Colab](https://colab.research.google.com/github/PaulsonLab/energy-inference-bo/blob/main/notebooks/task02a_colab.ipynb)).
It clones a selected Git
revision, guards the Python/JAX backend, runs the tests before the expensive command,
requires an explicit opt-in for the full study, and downloads one reviewable ZIP. It
never authenticates to or pushes to GitHub. Publish the Task 02A source first: Colab
cannot run local changes that are not present in the selected remote revision.

The notebook defaults to CPU, with an optional NVIDIA-GPU branch that accelerates only
the JAX/NumPyro NUTS references; exact-GP cache calculations remain on CPU by design.
After download, review the ZIP locally, copy `TASK_02A_COLAB_SUMMARY.md` into a tracked
report if desired, optionally choose a figure or compact CSV to track, then make your
normal local Git commit and push. The detailed `artifacts/` directory remains ignored.
The first completed full run is published in
[`results/task02a_full/`](results/task02a_full/); use it as the expected output layout,
not as a replacement for a new reproducible run.

### Terminal fallback

The tracked Task 02A summary is based on intentionally tiny NUTS chains and is not a
scientifically interpretable posterior comparison. Use a separate fresh Colab runtime
for the full configuration: seeds 0–2, D=10, n=16→40, 256 retained particles per fresh
reference, 512 held-out points, and 2,048 fixed EI candidates. The exact GP cache is
about 3.3 MB at n=40 and its largest chunked cross-covariance is about 21 MB in double
precision; JAX compilation and NUTS state dominate total memory and may use a few GB.

CPU is the reproducible default:

```python
!git clone https://github.com/PaulsonLab/energy-inference-bo.git
%cd energy-inference-bo
!pip install -r requirements.txt
!pip install -e .
!pytest -q

%env JAX_PLATFORMS=cpu
!python -c "import jax; print(jax.__version__, jax.default_backend(), jax.devices())"
!python -m energy_bo.experiments.run_task02a \
    --profile full \
    --seeds 0 1 2 \
    --output-dir artifacts/task02a_full \
    --summary-path TASK_02A_COLAB_SUMMARY.md
```

This performs fresh BoTorch SAAS NUTS fits with 512 warmup steps, 512 samples,
thinning 2, and tree depth 6 at n=16, 20, 24, 32, and 40, plus the first otherwise
unscheduled ESS/P < 0.5 crossing. It can take hours on a Colab CPU; do not reduce the
chains and interpret the result as full evidence.

Optional NVIDIA acceleration changes only JAX/NumPyro execution. In a GPU runtime,
install the official CUDA 12 JAX extra after the base requirements, restart the
runtime if requested, verify that a GPU device is listed, and then run the same command:

```python
!pip install --upgrade "jax[cuda12]"
%env JAX_PLATFORMS=cuda
!python -c "import jax; print(jax.__version__, jax.default_backend(), jax.devices())"
```

The run records Python, PyTorch, BoTorch, GPyTorch, JAX, NumPyro, backend, devices,
frozen output-transform parameters, seeds, counters, and runtimes in
`task02a_config.json`. Download all machine-readable metrics, plots, and the generated
summary with:

```python
!zip -r task02a_full_outputs.zip artifacts/task02a_full TASK_02A_COLAB_SUMMARY.md
from google.colab import files
files.download("task02a_full_outputs.zip")
```

The download is local to your browser; it does not change GitHub. Keep detailed files
under ignored `artifacts/` unless you deliberately decide to track a small result.

Do not add particle moves, SMC rejuvenation, selective-coordinate updates, Vecchia,
an output EBM, q>1 acquisition, or a BO loop to this Task 02A run.
