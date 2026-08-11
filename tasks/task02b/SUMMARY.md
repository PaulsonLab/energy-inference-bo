# Task 02B — Decision-space compression and joint-energy validation

## Scope and evidence

This task tests an oracle/teacher compression hypothesis and the discrete q=1 joint structural-decision identity. It does **not** implement SVGD, MALA, annealed Langevin, SMC rejuvenation, Vecchia, a residual-output EBM, q>1 BO, molecular optimization, or an end-to-end BO loop.

The decision-regret analysis below uses all `18` saved full Task 02A checkpoints without rerunning NUTS. Acquisition-signature results use profile `smoke` with `2` checkpoint(s), D=4, `32` retained particles, and `256` fixed candidates. The signature and coreset findings are smoke diagnostics, not scientific full-run evidence.

## Eight completion questions

1. **How much more robust is decision regret than posterior fidelity?** Across `15` noninitial full Task 02A checkpoints, median/mean/max normalized decision regret was `0` / `0.1028` / `0.9827`; `11` were below 5% and `4` exceeded 10%. Of `10` checkpoints with ESS/P < 0.1, `6` retained <5% decision regret. Exact candidates differed at `7` checkpoints; `0` of those differences cost <1% regret. This supports robustness only at a subset of checkpoints, not uniformly.
2. **Does ESS predict decision regret?** Over the `15` noninitial checkpoints, ESS/P versus normalized regret had Pearson/Spearman correlations `-0.2480` / `-0.1898`; `-log(ESS/P)` had `0.2767` / `0.1898`. These correlations are descriptive because n is small.
3. **What is the acquisition-signature effective rank?** Entropy effective rank ranged `3.383`–`4.395` (mean `3.889`), versus standardized structural-coordinate rank `4.482`–`4.838` (mean `4.660`).
4. **How many particles preserve the decision?** Acquisition-space Frank–Wolfe required K=`4` / `4` / `4` for every analyzed checkpoint to have <1% / <5% / <10% normalized regret.
5. **Does acquisition-space selection outperform the baselines?** Mean normalized regret across tested budgets/checkpoints was `0.002932` for acquisition-space Frank–Wolfe, `0.04349` for 32-repeat random equal thinning, and `0.03076` for posterior-space medoids. This is an oracle compression comparison, not an implementable inference algorithm.
6. **Does M=1 recover full Bayesian EI?** Yes: maximum/L1 normalized-marginal errors were `6.939e-18` / `1.092e-16`, with teacher/M=1 modes `91` / `91`.
7. **Does independent-replica M=2 recover squared EI?** Yes: maximum/L1 errors were `0.000e+00` / `0.000e+00`, with squared-teacher/M=2 modes `91` / `91`. The common-particle negative control differed from independent M=2 by L1 `0.3068`.
8. **Is Task 02C joint-energy transport justified?** **NO-GO pending full evidence.** The acquisition-signature evidence uses only tiny D=4 smoke NUTS chains. No Task 02C method is implemented.

## Reproduction and outputs

- [`results/task02b/retrospective/`](../../results/task02b/retrospective/) contains the saved-result regret table, correlations, and diagnostic plot.
- The profile output contains spectra, coreset metrics, joint-target metrics, plots, environment/configuration JSON, and compressed per-particle signatures.
- [COLAB.md](COLAB.md) gives the exact full extraction command. Full signature matrices remain generated artifacts until deliberately reviewed and imported.
