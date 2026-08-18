# Decision-Tilted Acquisition Inference for Bayesian Optimization

Monte Carlo BO samples uncertain worlds from the posterior even when expected utility is controlled by a very different, utility-tilted distribution. This project tests whether adapting the inner integration distribution can reduce acquisition-value and gradient cost by orders of magnitude while keeping the surrogate, acquisition, and gradient-based outer optimizer fixed.

The project studies the **inner Monte Carlo acquisition integral**. It does not replace optimization over the BO decision variable with sampling.

## Repository map

- [`PAPER_PLAN.md`](PAPER_PLAN.md): authoritative scientific contract and gates.
- [`STATUS.md`](STATUS.md): current phase, claim status, and next authorized action.
- [`src/decision_tilt/`](src/decision_tilt/): reusable mathematical and model code.
- [`tests/`](tests/): mathematical identities and implementation tests.
- [`experiments/`](experiments/): prospective paper experiments, organized by result rather than task number.
- [`notebooks/`](notebooks/): readable scientific analyses and future Colab workflows.

## Local setup

The locked environment targets CPython 3.12:

```bash
uv sync --locked --group dev
uv run pytest -q
```

The `rare_mode_mechanism` experiment is complete. The frozen `constrained_batch_shift` diagnostic is implemented and [ready for its external A100 run](experiments/constrained_batch_shift/README.md); no practical-mechanism result exists yet. Read [`STATUS.md`](STATUS.md) before taking the next action.

The previous exploratory project, including Tasks 01–05A and their negative results, is preserved on branch `archive/exploration-v1` and annotated tag `exploration-v1`.
