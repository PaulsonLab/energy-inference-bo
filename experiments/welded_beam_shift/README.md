# Welded Beam q=1 Decision Shift

## Question

Do six ordinary uncertain engineering constraints naturally create large posterior-to-decision shift at the points that constrained EI actually prefers, and does that shift make practical scrambled-Sobol estimation unreliable?

This is a diagnostic experiment. It does not test a decision-adapted sampler, a non-Gaussian belief, or batch acquisition.

## Frozen protocol

The machine-readable contract is [`config.json`](config.json). It is frozen before repeated QMC results are generated.

Use BoTorch 0.18.1's single-objective `WeldedBeamSO` problem with its original four-dimensional bounds and all six black-box constraints. The physical cost is negated for BO maximization. Constraint slack is left unchanged, and slack greater than or equal to zero is feasible.

The standard domain is only about 0.10% jointly feasible under a uniform Sobol design, so an unmodified 64-point design almost surely has no feasible incumbent. Each of the three prospective states therefore contains one fixed, strictly feasible engineering design, `[0.3, 5.0, 8.0, 0.4]`, plus 63 scrambled Sobol points. State seeds are `4101`, `4102`, and `4103`; no state may be replaced after its decision-shift behavior is observed.

Fit seven independent, noiseless-belief Gaussian GPs: one for negated cost and one for each constraint slack. Inputs are normalized to the unit cube. Each output uses train-only population standardization, fixed standardized numerical noise variance `1e-6`, and the pinned BoTorch `SingleTaskGP` default constant-mean ARD-RBF model. Exact marginal likelihood fitting uses SciPy for at most 200 iterations. Acquisition calculations use latent posterior variance.

The fixed candidate set is one shared 16,384-point scrambled Sobol design with seed `5201`. The incumbent is the largest observed negated cost among jointly feasible observations. Candidate quality, top 1%, top 5%, and the exact top 32 are all defined by analytic constrained EI on this finite set.

## Exact mathematics

For objective improvement `I=(Y-best_f)_+`, let `delta=mu-best_f` and `z=delta/sigma`. Then

\[
E[I]=\delta\Phi(z)+\sigma\phi(z),
\]

\[
E[I^2]=(\delta^2+\sigma^2)\Phi(z)+\delta\sigma\phi(z).
\]

With independent objective and constraint beliefs and positive-feasible slacks,

\[
p_F=\prod_{j=1}^6\Phi(\mu_j/\sigma_j),\qquad cEI=E[I]p_F.
\]

For constrained-improvement utility `U=I 1{all constraints feasible}`,

\[
D_2=\log E[I^2]-2\log E[I]-\log p_F,
\quad \chi^2=e^{D_2}-1,
\quad ESS/N=e^{-D_2}.
\]

These quantities are analytic; Monte Carlo is used only to test the practical estimator.

## Practical QMC study

For each state, use 64 independent seven-dimensional scrambled Sobol sequences. Within each scramble, 1,024 standard-normal worlds are shared across all candidates, and prefixes give sample counts `64, 128, 256, 512, 1024`. The sampled utility is exactly

\[
(Y-best_f)_+\prod_j 1\{C_j\ge0\}.
\]

Report exact-best selection, 1%-optimal selection, top-32 pairwise disagreement, Kendall agreement, top-k overlap, and normalized acquisition regret.

## Prospective classification

A state is shift-positive when its exact top-32 median ESS fraction is at most `0.125`. This means 512 posterior worlds provide at most 64 population-effective utility samples across a competitive set.

A state has material QMC failure when, at 512 samples, one-sided 95% bootstrap lower bounds exceed both 1% mean normalized selection regret and 10% mean top-32 pairwise disagreement.

A state is conclusively negative for the required conjunction if either its top-32 median ESS fraction is at least `0.25`, or at both 256 and 512 samples the one-sided 95% upper bounds are below 0.5% mean regret and 5% pairwise disagreement.

- `WELDED_BEAM_SHIFT_POSITIVE_REVIEW_REQUIRED`: the same two or more states are shift-positive and QMC-failure-positive.
- `WELDED_BEAM_SHIFT_NEGATIVE_REVIEW_REQUIRED`: two or more states conclusively fail at least one side of the conjunction.
- `WELDED_BEAM_SHIFT_INCONCLUSIVE_REVIEW_REQUIRED`: neither rule is met, or required numerical/provenance checks fail.

The gap between positive and negative thresholds is deliberate. Exact-best probability is descriptive because harmless near ties can make it pessimistic.

## Reproduction contract

The final CPU notebook will be [`../../notebooks/welded_beam_shift.ipynb`](../../notebooks/welded_beam_shift.ipynb). Durable results will live in `outputs/`. The experiment must not authorize q=4, non-Gaussian beliefs, or a decision-adapted method automatically.

Status: PROTOCOL FROZEN — IMPLEMENTATION AUTHORIZED
