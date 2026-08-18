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

The `rare_mode_mechanism` experiment is complete. The frozen `constrained_batch_shift` A100 campaign is [invalid because its State 3 high-budget reference failed at the prespecified numerical cap](experiments/constrained_batch_shift/outputs/invalid_reference_v2/README.md); it produced neither GO nor NO-GO. Read [`STATUS.md`](STATUS.md) before taking any further action.

The previous exploratory project, including Tasks 01–05A and their negative results, is preserved on branch `archive/exploration-v1` and annotated tag `exploration-v1`.
