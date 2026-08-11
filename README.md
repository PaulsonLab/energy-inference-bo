# energy-inference-bo

Research prototype for testing energy-based predictive corrections and augmented energy inference in Bayesian optimization.

## Task 01 result

Task 01 has completed its oracle and q=1 GP mathematical validation. Start with the
concise [Task 01 summary](TASK_01_SUMMARY.md), then consult the detailed
[three-seed smoke report](reports/TASK_01_SMOKE.md) and the validated
[oracle EI figure](docs/task01_oracle_ei.png). These are correctness and
learnability checks, not a BO performance benchmark or an implementation of Task 02.

Start with:
1. `AGENTS.md`
2. `MATH_AND_SCOPE.md`
3. `CODEX_TASK_01.md`

The repository is deliberately staged. The first task tests the mathematical hypotheses in oracle and q=1 Gaussian/GP settings before implementing a full energy-based BO method.

## Task 01 quick start

The project uses a locked uv environment with CPython 3.12 for CPU-compatible PyTorch,
BoTorch, and GPyTorch.

```bash
uv sync --python 3.12 --all-groups --no-editable
uv run --no-sync pytest
uv run --no-sync python -m energy_bo.experiments.run_task01
```

The smoke report is written to `reports/TASK_01_SMOKE.md`; regenerated CSV, JSON, and
PNG artifacts are written to the ignored `artifacts/task01/` directory. For a larger
30-seed run, follow [COLAB.md](COLAB.md) rather than running it on the local laptop.

## Task 02A result

Task 02A is a bounded falsification experiment for reusing a trusted SAAS posterior:
particle locations stay fixed while new observations change weights and exact GP
caches. Read the quantitative [Task 02A summary](TASK_02A_SUMMARY.md) and the governing
[structural-energy mathematics](TASK_02_STRUCTURAL_ENERGY_MATH.md). The checked-in
summary is from one deliberately reduced CPU smoke run. The complete three-seed Colab
evidence, including raw CSVs, figures, and reproducibility manifest, is now tracked in
[the Task 02A full-results package](results/task02a_full/README.md). The notebook remains
available as the self-contained [Task 02A Colab notebook](notebooks/task02a_colab.ipynb)
([open in Colab](https://colab.research.google.com/github/PaulsonLab/energy-inference-bo/blob/main/notebooks/task02a_colab.ipynb))
or the terminal fallback in [COLAB.md](COLAB.md).

```bash
uv sync --python 3.12 --all-groups --no-editable
uv run --no-sync pytest -q
uv run --no-sync python -m energy_bo.experiments.run_task02a --profile smoke
```

Generated Task 02A CSV, JSON, and PNG files are written under ignored
`artifacts/task02a_smoke/`. No Task 02B method is implemented or implied.
