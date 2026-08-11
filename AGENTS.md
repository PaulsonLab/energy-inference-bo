# AGENTS.md

## Purpose

This repository is a research prototype for testing whether energy-based modeling and augmented energy inference provide a real advantage for Bayesian optimization (BO).

Read `MATH_AND_SCOPE.md` before changing mathematical code. Then read
`tasks/ACTIVE_TASK.md` and the canonical task specification it names before
implementing a task. Canonical task contracts, mathematics, run instructions, and
summaries live together under `tasks/<task>/`.

`tasks/ACTIVE_TASK.md` is the navigation source for the current stage. The active
task specification overrides historical roadmaps or stage descriptions when they
conflict; preserve those older documents as research history.

The project is intentionally staged. Do not jump ahead.

## Core research questions

1. **Modeling:** Can a small, strongly regularized residual energy capture decision-relevant predictive structure that remains after a strong Gaussian reference has modeled location and scale?
2. **Decision/inference:** Can augmented probability / energy inference reproduce expected-utility BO decisions and become useful when explicit acquisition construction or optimization is difficult?

These are separate hypotheses. Test them separately before combining them.

## Mathematical invariants

Treat these as non-negotiable unless a task explicitly revises the theory.

### Predictive correction

A corrected scalar predictive density has the form

\[
q(y\mid x,D)
=
\frac{
p_G(y\mid x,D)\exp[-R_\phi(z,c)]
}{
Z_\phi(c)
},
\qquad
z=\frac{y-\mu_G(x)}{\sigma_G(x)}.
\]

- `R_phi = 0` must recover the Gaussian reference exactly.
- The normalizer `Z_phi(c)` must not be dropped when it depends on `x` or context.
- Adding a constant to an energy changes no normalized density. Fix this gauge in any residual parameterization (for example by using reference-centered basis functions or another explicit convention).
- For the first model, keep `R_phi(z)` context-free so the scalar normalizer is independent of `x`.

### Sequential training

For ordered data, use predictive conditionals based only on earlier observations:

\[
q(y_{1:n}\mid x_{1:n})
=
\prod_i q(y_i\mid x_i,D_{i-1}).
\]

Do not compute a “training residual” at a point after conditioning on that same observation.

When `R_phi = 0` and the reference is an exact GP, the product of exact GP predictive conditionals must reduce to the standard GP joint marginal likelihood.

### Expected-utility augmentation

For a normalized predictive model `q`, positive/nonnegative utility `u`, and design prior `p0`,

\[
\pi_1(x,y)
\propto
p_0(x)q(y\mid x,D)u(x,y)
\]

has marginal

\[
\pi_1(x)\propto p_0(x)a(x),
\qquad
a(x)=E_q[u(x,Y)].
\]

With `M` independent replicated latent outcomes,

\[
\pi_M(x,y^{1:M})
\propto
p_0(x)
\prod_{m=1}^M q(y^{(m)}\mid x,D)u(x,y^{(m)})
\]

has marginal

\[
\pi_M(x)\propto p_0(x)a(x)^M.
\]

- With **uniform** `p0`, increasing `M` preserves the acquisition maximizer and sharpens around it.
- With nonuniform `p0`, do not claim that the maximizer is unchanged without checking it.
- Joint MAP over `(x,y)` is not generally equivalent to marginal acquisition maximization over `x`.
- For raw improvement utility, zero utility is mathematically valid but creates infinite log-energy. Use direct weights or a documented small epsilon / smooth positive utility when log-energy calculations require positivity.

### GP q=1 sanity check

For a standard Gaussian GP and `q=1`, the augmented marginal using improvement utility must reproduce EI (or `EI + epsilon` if an epsilon is deliberately added). The selected maximizer must agree with EI / LogEI up to numerical tolerance.

This is a required sanity check before claiming any advantage.

## Scope discipline

Do not claim novelty for:
- rewriting a GP as an energy;
- expected utility as an augmented probability target;
- power/replicated sharpening of an expected-utility target;
- Vecchia GP;
- SAAS priors;
- full Bayesian GP hyperparameter particles;
- GP-referenced function-space EBMs;
- Boltzmann sampling of an acquisition.

Any paper-level claim must come from a demonstrated advantage of the combined method in a regime where strong existing BO methods struggle.

## Initial model hierarchy

Use controlled nested baselines:

1. Gaussian reference.
2. Gaussian location/scale recalibration.
3. Residual energy correction beyond location/scale.

The EBM only earns its complexity if it improves over a fair location/scale Gaussian correction, not merely over an uncalibrated reference.

Start residual energies small:
- scalar standardized residual;
- 7–9 bounded RBF or spline basis functions;
- strong zero-centered shrinkage;
- explicit gauge fixing;
- exact 1-D quadrature where possible.

Do not begin with a large neural EBM.

## Coding rules

- Python 3.11+.
- Use `torch.double` for GP, quadrature, and BO numerical tests unless explicitly justified otherwise.
- Use PyTorch, GPyTorch, and BoTorch when appropriate rather than reimplementing trusted GP/BO primitives.
- Use scipy for independent numerical checks when useful.
- Every experiment must accept and save a random seed.
- Put reusable code in `src/`; notebooks are thin drivers.
- Add tests for mathematical identities before performance experiments.
- Report numerical discrepancies rather than loosening tolerances until tests pass.
- Keep dependencies minimal.
- Never commit API keys, tokens, credentials, or Colab secrets to this public repository.

## Compute policy

The maintainer uses a MacBook Air with 16 GB RAM. Local development should use smoke tests only.

Unless explicitly requested:
- keep local test runs CPU-compatible;
- keep smoke experiments small (roughly a few minutes or less);
- do not launch large seed sweeps, large particle studies, large-dimensional BO runs, or expensive hyperparameter searches locally;
- if a planned experiment is materially more expensive, implement a small smoke version locally and create clear Colab instructions/notebooks for the full run;
- report expected scale (seeds, particles, dimensions, approximate memory-sensitive objects) before recommending the full Colab run.

GPU use must be optional until a later task explicitly requires it.

### Smoke and Colab run policy

- Every task must have a small CPU-compatible local validation path. Local smoke runs
  exercise correctness and wiring; they are not scientific performance evidence.
- Every task with a larger approved study must provide a thin, guarded notebook under
  `notebooks/` and short task-local instructions in `tasks/<task>/COLAB.md`.
- Notebook "Run all" must be safe: expensive cells require an explicit `RUN_FULL =
  True`, record the checked-out Git SHA and environment, run the test gate first, and
  produce one downloadable ZIP. Notebooks never authenticate to or push to GitHub.
- Automatic local and Colab outputs go to ignored `artifacts/<task>/<profile>/`.
  Only compact evidence that has been reviewed and intentionally selected belongs in
  `results/<task>/<profile>/`; raw posterior samples and large signature matrices stay
  ignored.
- A task's `README.md` must state its local command, notebook, expected scale, and the
  exact files worth promoting after review.

## Task completion

Before declaring a coding task complete:

1. Run the requested unit tests.
2. Run only the requested smoke experiments.
3. Summarize files changed.
4. Report quantitative validation results.
5. Report any failed assumption or negative result.
6. State whether the next stage gate is justified; do not automatically implement the next phase.

## Structural-posterior inference rules

For Task 02 and later structural-inference work:

- Distinguish a particle's **location** (GP hyperparameters) from its **weight**. Bayesian updating can change weights without changing hyperparameter locations.
- Do not claim that sequential particles are cheaper than NUTS unless expensive particle moves/rejuvenation are empirically needed less often than a full refit.
- For fixed hyperparameters, prefer exact cached one-step GP updates / rank-one Cholesky augmentation over rebuilding the full factorization.
- When a particle's kernel hyperparameters change, treat its exact GP factorization as invalid and count the resulting full recomputation explicitly.
- Never change output normalization parameters across sequential rounds in an experiment whose math assumes `p_t(theta) ∝ p_{t-1}(theta) p(y_t | D_{t-1}, theta)`.
- Use fresh BoTorch SAAS NUTS as the reference target until the custom structural energy has been independently validated.
- SMC / particle treatment of GP or BO hyperparameters is prior art. Do not claim novelty for particle reuse, data tempering, resampling, or generic rejuvenation.
- The intended research opportunity is information-adaptive, SAAS-specific structural inference: expensive computation should eventually depend on posterior change and unresolved effective dimension.
- Do not implement selective coordinate freezing without a mechanism that can unlock a previously shrunk dimension.
- Task 02A is diagnostic only: no new particle mover, no residual-output EBM, no q>1 acquisition inference.
