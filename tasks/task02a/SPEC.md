# CODEX TASK 02A — Measure the SAAS structural-posterior reuse horizon

Read, in this order:

1. `../../AGENTS.md`
2. `../../MATH_AND_SCOPE.md`
3. `../task01/SUMMARY.md`
4. `MATH.md`
5. this file

Task 02A is a deliberate diagnostic pivot. Where its sequencing differs from the older stage-gate ordering in `MATH_AND_SCOPE.md`, this task-specific plan controls.

## Goal

Test the single assumption required for a computational advantage over repeated SAAS NUTS fitting:

> After one new BO observation, does the SAAS hyperparameter posterior usually remain close enough to the previous posterior that existing hyperparameter particles can be reused by importance reweighting for multiple rounds before expensive rejuvenation is required?

Do **not** implement a new SMC rejuvenation sampler yet.

Use trusted BoTorch SAAS NUTS to initialize particles and to provide reference posteriors at checkpoints.

The output of this task should tell us whether implementing annealed resample-move inference is justified.

---

# Part A — exact sequential weight-update identity

## A1. Fixed-hyperparameter predictive likelihood

For fixed GP hyperparameters \(\theta\), verify numerically that

\[
\log p(D_t\mid\theta)
-
\log p(D_{t-1}\mid\theta)
=
\log p(y_t\mid x_t,D_{t-1},\theta).
\]

Implement two independent calculations:

1. full exact-GP marginal-likelihood difference;
2. one-step predictive log likelihood using the cached \(D_{t-1}\) GP state.

These must agree to strict numerical tolerance in double precision on small test problems.

## A2. Rank-one / block Cholesky append

For unchanged \(\theta\), implement a cached exact-GP state that can append one observation without recomputing the full Cholesky factor.

Unit-test the appended factor against a full fresh factorization.

Also verify that posterior mean/variance at test points agree between:
- cached sequential state;
- fresh full exact GP with the same \(\theta\).

Do not optimize prematurely. Correctness first.

---

# Part B — SAAS NUTS initialization

Use the current BoTorch fully Bayesian SAAS implementation as the trusted source of posterior particles.

Important:
- use fixed known observation noise;
- normalized inputs in `[0,1]^D`;
- do not use round-dependent output standardization;
- standardize once using the initial dataset and freeze that affine transform for all later observations.

Extract operational hyperparameter samples sufficient to evaluate exact GP likelihoods/predictions:
- lengthscales;
- mean;
- outputscale;
- fixed noise is shared.

Task 02A does not need the latent `kernel_tausq` / raw inverse-lengthscale variables because no particle rejuvenation is performed.

Keep enough NUTS samples for a useful particle diagnostic. Do not thin aggressively merely because the standard BoTorch tutorial does so for acquisition-model memory.

---

# Part C — synthetic sparse structural benchmark

## C1. Main benchmark

Use an embedded Branin-style function:

- domain `[0,1]^D`;
- objective depends only on the first two dimensions;
- remaining dimensions are irrelevant;
- maximize the objective (negate Branin if needed);
- deterministic or very low fixed known noise.

Generate one fixed Sobol sequence of inputs and reveal outcomes sequentially.

Suggested full-run setting:
- `D = 10`;
- initial `n0 = 16`;
- final `n = 40`;
- 3 independent seeds initially for Colab validation.

Suggested local smoke:
- `D = 4` or `6`;
- `n0 = 8`;
- add only 3–5 observations;
- one seed;
- reduced NUTS warmup/sample counts explicitly labeled **smoke only**.

Do not interpret low-warmup smoke NUTS scientifically.

## C2. Optional second benchmark

Only if implementation is stable and cheap, add a synthetic function drawn from a known 2-active-dimensional Matérn-5/2 GP.

This is optional; do not delay completion.

---

# Part D — NUTS-seeded sequential replay

At `n0`:

1. fit SAAS with NUTS;
2. obtain `P` hyperparameter particles;
3. set equal weights;
4. construct cached exact-GP state for each particle using `D_n0`.

For each newly revealed observation:

1. compute each particle's one-step predictive log likelihood;
2. update log weights stably;
3. normalize weights;
4. compute
   \[
   ESS = 1/\sum_p w_p^2;
   \]
5. record
   \[
   ESS/P
   \]
   and
   \[
   \widehat D_2 = -\log(ESS/P);
   \]
6. record the variance across particles of the incremental log likelihood / incremental energy;
7. append the new observation to each unchanged cached GP state;
8. do **not** move, resample, or rejuvenate particle locations.

Report the first number of added observations at which ESS/P falls below:
- 0.75;
- 0.50;
- 0.25;
- 0.10.

This is the **reuse horizon**.

---

# Part E — fresh NUTS reference checkpoints

Fit a fresh trusted SAAS NUTS model at selected checkpoints.

For the full Colab experiment, suggested checkpoints are:
- `n = 16` (initialization);
- `n = 20`;
- `n = 24`;
- `n = 32`;
- `n = 40`.

If the reweighted ESS crosses 0.5 substantially earlier, add a reference checkpoint near that crossing.

For the local smoke, use only the smallest feasible number of reference fits.

Compare the reweighted initial-particle posterior to fresh NUTS on:

## E1. Structural marginals
- posterior median lengthscale by dimension;
- posterior interquartile range by dimension;
- rank ordering of inverse lengthscales / relevance;
- top-2 active-dimension recovery;
- weighted probability or fraction of particles in which each dimension is among the top-2 most relevant.

Avoid arbitrary "active if lengthscale < c" thresholds unless clearly labeled.

## E2. Distribution discrepancy
Use at least:
- one-dimensional Wasserstein-1 distance for each log lengthscale;
- a multivariate discrepancy such as MMD on standardized transformed hyperparameters if straightforward.

Do not over-engineer the metric suite.

## E3. Predictive mixture
On a fixed Sobol test set:
- fully Bayesian predictive mean;
- predictive variance;
- predictive log likelihood on held-out noiseless/noisy truth where meaningful.

Compare sequential reweighting versus fresh NUTS.

---

# Part F — BO-decision relevance without running a BO loop

Use one fixed Sobol candidate set, e.g. 2048–4096 points for the full run.

For each hyperparameter particle, compute analytic q=1 EI.

For weighted particles:

\[
EI_{\rm seq}(x)
=
\sum_p
w_p EI(x;\theta_p).
\]

For fresh NUTS:

\[
EI_{\rm NUTS}(x)
=
\frac{1}{P_{\rm ref}}
\sum_r
EI(x;\theta_r^{\rm NUTS}).
\]

At each reference checkpoint report:
- Spearman EI rank correlation;
- top-5% candidate overlap;
- relative error in maximum EI;
- whether the selected best candidate is identical;
- Euclidean distance between selected candidates if they differ.

Use the same candidate set so acquisition optimization is not a confounder.

Do not use q>1.

---

# Part G — computational instrumentation

This part is essential.

Track separately:

### Cheap sequential operations
- one-step predictive likelihood evaluations;
- rank-one Cholesky appends;
- wall time per new observation for all particles.

### Expensive operations
- fresh NUTS wall time at each reference fit;
- full exact-GP factorizations;
- any full recomputations used only for validation.

The code should expose counters where practical for:
- number of full Cholesky factorizations;
- number of rank-one appends;
- number of particle likelihood updates.

The scientific question is not simply "which code path is faster on one laptop." It is whether the sequential method avoids repeated expensive **hyperparameter movement / global refitting** while preserving the posterior well enough.

---

# Part H — SAAS coordinate-stability diagnostic

We want evidence for or against a later selective-rejuvenation idea.

Between fresh NUTS checkpoints, measure for each dimension:
- change in median log inverse lengthscale;
- Wasserstein distance between marginal log-lengthscale posteriors;
- whether the dimension is one of the true active dimensions.

Summarize separately for:
- true active dimensions;
- irrelevant dimensions.

Question:

> Are irrelevant SAAS coordinates substantially more stable across sequential posterior updates than active/uncertain coordinates?

If yes, this supports a later method that rejuvenates only unresolved structural coordinates while retaining a mechanism to unlock previously inactive dimensions.

Do not implement selective coordinate rejuvenation in Task 02A.

---

# Part I — local versus Colab execution

The maintainer uses a 16 GB MacBook Air.

Locally / in the Codex environment:
- run unit tests;
- run only one very small smoke sequence;
- use explicitly reduced NUTS settings;
- do not run the full `D=10, n=40` study.

Create/update this task's `COLAB.md` with exact commands for the full Task 02A experiment.

The full configuration should use NUTS settings consistent with current BoTorch recommendations where feasible (not smoke-test settings), and should clearly report chosen warmup, samples, thinning, device, and runtime.

If CUDA/JAX setup requires a dependency update, make the smallest justified change and document it.

---

# Part J — required tests

Add tests for:

1. full log-marginal-likelihood increment equals one-step predictive log likelihood;
2. rank-one Cholesky append equals full Cholesky;
3. cached prediction equals fresh exact-GP prediction;
4. log-weight normalization is stable;
5. equal weights give `ESS=P`;
6. a deliberately constant incremental likelihood leaves weights unchanged;
7. weighted analytic EI equals an independently computed mixture EI on a small example;
8. fixed preprocessing is reused identically across sequential rounds.

---

# Part K — output files

Create:

- `SUMMARY.md`;
- machine-readable summary CSV/JSON;
- ESS/reuse-horizon plot;
- lengthscale posterior comparison plots;
- active-vs-inactive coordinate drift plot;
- EI comparison plot at reference checkpoints;
- timing/cost summary.

Save experiment configuration and seeds.

---

# Completion questions

Answer these explicitly in `SUMMARY.md`:

1. How many new observations can be added before ESS/P first crosses 0.75, 0.50, 0.25, and 0.10?
2. Does `-log(ESS/P)` track the measured discrepancy between sequential and fresh NUTS posteriors?
3. While ESS is high, how accurately does reweighting reproduce fresh NUTS structural marginals?
4. While ESS is high, how accurately does it reproduce fresh-NUTS predictive mixtures?
5. Does it preserve EI ranking and the selected q=1 decision?
6. How much cheaper is a sequential weight/cache update than a fresh NUTS fit in the measured smoke/full settings?
7. Are irrelevant SAAS dimensions more stable across rounds than active dimensions?
8. Based on the evidence, is Task 02B justified?

Choose one recommendation:

- **02B-A:** implement adaptive annealed resample-move SMC over the full SAAS latent energy;
- **02B-B:** first investigate a different particle transport method because reweighting support collapses too rapidly;
- **02C:** prioritize selective sparse-coordinate rejuvenation because posterior reuse is good and inactive coordinates are clearly stable;
- **STOP/PIVOT:** the sequential structural-posterior reuse hypothesis is not strong enough.

Do not automatically implement the next task.
