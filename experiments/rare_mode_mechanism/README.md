# Rare-Mode Mechanism

## Question

Can a rare but high-utility predictive mode make ordinary posterior Monte Carlo misrank two BO decisions even when acquisition arithmetic is numerically stable?

## Frozen protocol

The experiment uses an exactly evaluable, non-Gaussian two-component Gaussian mixture on `x in [-2, 2]`. The rare component has fixed posterior weight `0.005`. Candidate A at `x=-0.75` is driven by a common mode with mean `0.02` and standard deviation `0.04`; candidate B at `x=0.85` is driven by a rare mode with mean `7.0` and standard deviation `0.2`. The incumbent is zero. The full prospectively frozen construction, sample counts, seeds, repetitions, and gates are in [`config.json`](config.json).

Exact componentwise truncated-normal first and second moments provide EI, utility second moments, chi-square posterior-to-decision shift, population ESS, and predicted iid Monte Carlo relative variance. A strictly positive softplus utility with temperature `0.01` is evaluated by deterministic one-dimensional quadrature.

## Result

All prospective mechanism gates passed. Candidate B has exact EI `0.0350000`, 25.95% above candidate A's `0.0277885`, yet its decision-tilted distribution puts more than `0.9999999999` mass on a posterior component with probability `0.005`. Its chi-square decision shift is `199.163`, giving a population ESS fraction of `0.004996`.

At 512 posterior samples, iid MC ranks the better candidate correctly in only 56.27% of 10,000 repetitions; scrambled Sobol QMC reaches 59.47% over 1,024 scrambles. The positive-utility control gives 48.99% iid accuracy. Iid accuracy reaches 91.02% only at 8,192 samples. Empirical iid relative variance agrees with `chi-square / N` to a maximum relative discrepancy of 3.16% over the frozen sample counts.

## Reproduce and review

```bash
uv run pytest -q
uv run python experiments/rare_mode_mechanism/run.py
```

- [Human-readable executed notebook](../../notebooks/rare_mode_mechanism.ipynb)
- [Candidate Figure 1 (PNG)](outputs/figure1_rare_mode_mechanism.png), with [PDF](outputs/figure1_rare_mode_mechanism.pdf) and [SVG](outputs/figure1_rare_mode_mechanism.svg)
- [Numerical summary](outputs/summary.json)
- [Prospective gate result](outputs/gate_result.json)

This is a deliberately transparent synthetic mechanism test, not evidence that a decision-adapted sampler beats QMC in realistic BO. QMC becomes reliable much sooner in this construction. No downstream experiment is automatically authorized.

Status: COMPLETE — HUMAN REVIEW REQUIRED
