# energy-inference-bo

Research prototype for testing whether energy-based predictive corrections and
augmented energy inference provide a real advantage for Bayesian optimization.

## Start here

1. Read [AGENTS.md](AGENTS.md) and [MATH_AND_SCOPE.md](MATH_AND_SCOPE.md).
2. Follow the [active task](tasks/ACTIVE_TASK.md).
3. Use the [task registry](tasks/README.md) for completed-stage specifications,
   mathematics, summaries, and Colab procedures.

| Stage | Status | Evidence |
| --- | --- | --- |
| Task 01 — oracle shape and q=1 GP identities | Complete | [task folder](tasks/task01/README.md) |
| Task 02A — fixed-support SAAS reuse | Complete | [task folder](tasks/task02a/README.md) |
| Task 02B — decision-space compression | Complete; Task 02C gate passed | [task folder](tasks/task02b/README.md) |

The [current roadmap](docs/research/ROADMAP.md) records the completed evidence and the
conditional path to a separately specified Task 02C; [architecture](docs/ARCHITECTURE.md)
maps the source and evidence layout.

## Local quick start

The locked `uv` environment targets CPython 3.12 for CPU-compatible PyTorch, GPyTorch,
and BoTorch.

```bash
uv sync --python 3.12 --all-groups --no-editable
uv run --no-sync pytest -q
```

Task drivers are explicit. These are development checks, not instructions to rerun
every experiment routinely:

```bash
uv run --no-sync python -m energy_bo.experiments.run_task01
uv run --no-sync python -m energy_bo.experiments.run_task02a --profile smoke
uv run --no-sync python -m energy_bo.experiments.run_task02b --profile retrospective
uv run --no-sync python -m energy_bo.experiments.run_task02b --profile smoke
```

## Repository map

| Folder | Contents |
| --- | --- |
| [`tasks/`](tasks/README.md) | One self-contained folder per research task: entry point, contract, math when needed, summary, and run guide |
| [`src/energy_bo/`](src/energy_bo/) | Reusable implementation, organized by mathematical domain rather than task number |
| [`tests/`](tests/) | Unit tests for mathematical identities and reusable code |
| [`notebooks/`](notebooks/README.md) | Thin, guarded Colab drivers—one per runnable task |
| [`results/`](results/README.md) | Reviewed compact evidence committed to Git |
| `artifacts/` | Ignored raw/generated run outputs |

Use [COLAB.md](COLAB.md) for the notebook index and shared download/promotion workflow.
Large runs never push to GitHub: download the ZIP, review it locally, and promote only
selected evidence into `results/`.
