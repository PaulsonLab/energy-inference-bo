# CODEX TASK 02B — Decision-relevant structural compression and joint-energy validation

Read the repository state carefully before coding.

Required reading:
1. `../../AGENTS.md`
2. `../../MATH_AND_SCOPE.md`
3. `../task01/SUMMARY.md`
4. `../../results/task02a/full/SUMMARY.md`
5. `../task02a/MATH.md`
6. `MATH.md`
7. this task file

Where older plans conflict with this task, this task is the active specification.

## Main goal

Task 02A showed rapid fixed-support posterior degradation, but BO decision quantities appeared more robust.

Task 02B asks:

> Is the SAAS structural posterior much more compressible in **decision space** than in raw posterior space, and does the exact joint structural-decision energy recover the full Bayesian q=1 acquisition marginal?

This is a bounded bridge task.

Do **not** implement SVGD, MALA, annealed Langevin, a new SMC method, Vecchia, q>1, molecular BO, or a residual-output EBM in Task 02B.

---

# Part 0 — bounded repository cleanup

Before adding research code:

1. inventory current root files, source modules, tests, results, and task documents;
2. do not delete or rewrite Task 01 or Task 02A results;
3. create a concise active-status document such as
   `docs/research/PROJECT_STATUS.md`;
4. create or update an active task pointer, e.g.
   `tasks/ACTIVE_TASK.md`, that points to Task 02B;
5. update `README.md` minimally so a new contributor can tell:
   - Task 01 is complete;
   - Task 02A is complete;
   - Task 02B is active;
   - where the current math and summaries live;
6. add a short `AGENTS.md` rule that the active task file overrides historical stage plans.

Do not perform a broad code refactor merely for aesthetics.

Preserve working Task 01 / 02A modules and tests.

---

# Part A — re-analyze Task 02A using decision regret

Use the existing full Task 02A checkpoint and per-candidate EI outputs. Do not rerun NUTS merely to reproduce already saved numbers if the necessary data are available.

At each fresh-NUTS checkpoint compute:

\[
r_{\rm dec}
=
\frac{
A_{\rm fresh}(x_{\rm fresh}^\star)
-
A_{\rm fresh}(x_{\rm approx}^\star)
}{
\max(|A_{\rm fresh}(x_{\rm fresh}^\star)|,\epsilon)
}.
\]

Also compute absolute decision regret.

Compare decision regret against:
- ESS/P;
- `-log(ESS/P)`;
- posterior MMD if saved;
- mean/max log-lengthscale W1;
- EI Spearman correlation;
- top-5% overlap.

Report correlation plots, but be cautious about the small sample count.

The goal is to verify quantitatively whether posterior degradation is a poor proxy for actual BO decision degradation.

Also report how often:
- exact best candidate differs but decision regret is <1%;
- decision regret is <5%;
- decision regret is >10%.

---

# Part B — acquisition-signature matrix

At each fresh-NUTS checkpoint, obtain or reconstruct per-particle analytic EI values on the exact same fixed candidate set:

\[
V_{pj}=EI_{\theta_p}(x_j).
\]

If the current result files do not contain per-particle EI signatures, add a small utility that loads the saved/reference NUTS particle state where available or reruns only the minimum fresh-NUTS checkpoints needed for this analysis.

Do not silently approximate a fresh-NUTS signature matrix from the fixed-support Task 02A particles.

For each checkpoint:

1. center/scale the acquisition-signature matrix in a documented way;
2. compute singular values;
3. report cumulative explained squared singular value / effective rank;
4. visualize the first few singular values.

Also compare with analogous dimensionality in standardized transformed hyperparameter space if straightforward.

Question:

> Is the decision representation substantially lower-dimensional than the structural-posterior representation?

Do not overclaim from one benchmark.

---

# Part C — oracle decision-space particle compression

Treat fresh NUTS as the teacher.

For coreset sizes

\[
K\in\{4,8,16,32\}
\]

(and 64 if useful), approximate the teacher acquisition vector using only K weighted structural particles.

Implement at least:

### C1. Random baseline
Randomly choose K NUTS particles; use equal weights or optimized nonnegative weights, clearly distinguish the variants.

### C2. Posterior-space baseline
A simple clustering/medoid selection in standardized transformed hyperparameter space if straightforward.

### C3. Acquisition-space coreset
Select particles to minimize error in the acquisition vector over the candidate set.

A simple greedy Frank-Wolfe / conditional-gradient procedure is preferred because the full teacher acquisition is a convex combination of particle signatures.

Requirements:
- nonnegative weights;
- weights sum to one;
- no exotic optimizer;
- deterministic given seed.

For each K report:
- max absolute acquisition error;
- normalized max error;
- RMSE;
- Spearman rank correlation;
- top-5% overlap;
- selected best candidate;
- absolute and normalized decision regret.

Evaluate over all available seeds/checkpoints.

This is an oracle diagnostic only. Do not present it as an implementable inference method yet.

---

# Part D — empirical decision-regret bound check

For every approximate coreset compute

\[
\delta_{A,C}
=
\max_{x\in C}
|A_{\rm teacher}(x)-A_{\rm core}(x)|.
\]

Verify numerically that

\[
A_{\rm teacher}(x_{\rm teacher}^\star)
-
A_{\rm teacher}(x_{\rm core}^\star)
\le
2\delta_{A,C}
\]

up to floating-point tolerance.

Add a unit test for the finite-candidate version of this bound using synthetic acquisition vectors.

---

# Part E — exact joint decision-energy identity with NUTS as teacher

Do not yet implement the unnormalized SAAS log joint.

Use fresh NUTS particles as a trusted quadrature representation of \(p(\theta\mid D)\).

On one representative checkpoint and candidate set:

## E1. M = 1

Construct the discrete teacher joint target

\[
\pi(x_j,\theta_p)
\propto
w_p EI_{\theta_p}(x_j)
\]

under a uniform candidate prior.

Marginalize particles and verify numerically that

\[
\pi(x_j)
\propto
A_{\rm FB}(x_j).
\]

## E2. M = 2

Using independent structural-particle replicas, verify that the x marginal is proportional to

\[
A_{\rm FB}(x_j)^2.
\]

Be careful: using the same structural particle in both factors computes a different target involving \(E[EI_\theta^2]\).

## E3. Sampling smoke

Optionally draw discrete samples from the teacher joint distribution and confirm empirically that the x histogram converges to the known marginal.

This is a correctness/intuition check, not a proposed algorithm.

---

# Part F — recommendation gate for an actual EBM-native sampler

Task 02B should end with a recommendation for or against implementing direct joint energy inference.

Recommend **Task 02C: joint-energy transport** only if most of the following are true:

1. Task 02A decision regret is materially more robust than posterior diagnostics;
2. acquisition-signature effective rank is modest;
3. K <= 16 or 32 acquisition-space coresets preserve decisions well across a meaningful fraction of checkpoints;
4. the joint-energy identities pass exactly;
5. results suggest that focusing particles on decision-relevant structural variation could reduce required particle count substantially.

If these are not supported, recommend another pivot rather than implementing an advanced EBM sampler merely because it is interesting.

---

# Part G — if Task 02C is recommended, specify but do not implement it

Add a short section to `SUMMARY.md` describing the proposed next algorithm.

The default candidate should be a GPU-parallel particle transport method such as SVGD operating on the unnormalized joint structural-decision energy

\[
E(x,\theta)
=
-\log p_0(x)
-\log p(\theta)
-\log p(D\mid\theta)
-\log EI_\theta(x).
\]

But do not hard-code SVGD as the only future option.

Compare conceptually:
- SVGD;
- annealed MALA/Langevin;
- resample-move SMC.

Discuss:
- gradient availability;
- parallelization;
- particle diversity;
- need for full GP factorizations when theta moves;
- how a decision-space kernel or preconditioner might exploit the acquisition-signature result.

Do not implement any of these in Task 02B.

---

# Part H — compute policy

Use the saved Task 02A artifacts wherever possible.

Local/Codex:
- tests;
- retrospective metrics;
- SVD/compression on saved results;
- small teacher-joint identity checks.

If per-particle fresh-NUTS acquisition signatures require rerunning substantial NUTS:
- implement and smoke-test locally;
- provide/update a Colab command for the full extraction;
- do not perform a large full rerun locally.

GPU Colab is acceptable for the full analysis if needed.

---

# Part I — required outputs

Create:

- `SUMMARY.md`;
- `results/task02b/...` machine-readable metrics;
- decision-regret-vs-ESS/posterior-discrepancy plots;
- acquisition-signature spectrum plot;
- coreset quality vs K plots;
- joint-target marginal validation plot/table;
- updated `docs/research/PROJECT_STATUS.md`;
- minimal README / active-task updates.

The summary must explicitly answer:

1. How much more robust is BO decision regret than posterior fidelity in Task 02A?
2. Does ESS meaningfully predict decision regret?
3. What is the effective rank of the fresh-NUTS acquisition-signature matrix?
4. How many structural particles are needed to preserve the fully Bayesian EI decision to <1%, <5%, and <10% normalized regret?
5. Does acquisition-space selection materially outperform random thinning and posterior-space selection?
6. Does the M=1 joint structural-decision target exactly recover full Bayesian EI?
7. Does the independent-replica M=2 target exactly recover squared full Bayesian EI?
8. Is a direct joint-energy sampler justified as Task 02C?

Stop after Task 02B.
