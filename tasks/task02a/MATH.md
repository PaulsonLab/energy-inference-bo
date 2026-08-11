# Task 02 Mathematical Note: Sequential Structural-Energy Reuse

## 1. Why Task 02 changes focus

Task 01 established two useful facts:

1. higher-order predictive shape can change an EI decision even when a Gaussian model has the correct first two moments;
2. the augmented expected-utility construction exactly reproduces q=1 EI / LogEI decisions.

Task 01 also showed that naive replicated importance sampling degenerates quickly.

For Task 02, do **not** yet add a more complicated output residual EBM or try to replace qLogEI. Instead, investigate a computational bottleneck that is already important in high-dimensional BO: fully Bayesian inference over GP structure, especially SAAS lengthscales.

The central Task 02 hypothesis is:

> Because BO adds only one observation at a time, the structural posterior may often change only modestly from one round to the next. A particle representation can then update by reweighting existing hyperparameter states, paying for expensive parameter movement only when the posterior has changed enough to require it.

This is an **information-adaptive computation** hypothesis.

It is not a novelty claim by itself. Sequential Monte Carlo for GP hyperparameters and BO hyperparameters already exists. The purpose of Task 02A is to determine whether the SAAS setting gives enough posterior continuity and sparse structure to justify developing a specialized method.

---

# 2. What a structural particle is

Let

\[
\theta
\]

denote GP structural / nuisance hyperparameters. For the current BoTorch SAAS model this includes, after postprocessing, quantities such as:

- ARD lengthscales \(\ell_{1:D}\);
- constant mean;
- outputscale;
- noise if it is inferred.

In Task 02A, fix the observation noise to a known small value, so the operational particle state can be

\[
\theta^{(p)}
=
(\ell_{1:D}^{(p)}, m^{(p)}, s_f^{(p)}).
\]

A particle is a **fixed hyperparameter hypothesis plus a weight**.

The approximate posterior is

\[
p_t(\theta)
\approx
\sum_{p=1}^P
w_t^{(p)}
\delta_{\theta^{(p)}}.
\]

Crucially, the posterior can change even when particle locations do not:

\[
\theta_t^{(p)}=\theta_{t-1}^{(p)}
\]

but

\[
w_t^{(p)}\ne w_{t-1}^{(p)}.
\]

Changing the weights is already Bayesian updating.

---

# 3. Exact sequential posterior update

Let

\[
D_t=D_{t-1}\cup\{(x_t,y_t)\}.
\]

Bayes' rule gives

\[
p_t(\theta)
=
p(\theta\mid D_t)
\propto
p(\theta\mid D_{t-1})
p(y_t\mid x_t,D_{t-1},\theta).
\]

Define the one-step predictive likelihood

\[
L_t(\theta)
=
p(y_t\mid x_t,D_{t-1},\theta).
\]

Then particle weights update exactly as

\[
\tilde w_t^{(p)}
=
w_{t-1}^{(p)}
L_t(\theta^{(p)}),
\]

followed by normalization.

In energy notation, if

\[
E_{t-1}(\theta)
=
-\log p(\theta\mid D_{t-1})+C,
\]

then

\[
E_t(\theta)
=
E_{t-1}(\theta)
+
\Delta E_t(\theta)
+
C_t,
\]

where

\[
\Delta E_t(\theta)
=
-\log L_t(\theta).
\]

One new BO observation therefore adds **one new energy factor**.

---

# 4. Why unchanged particles can be computationally useful

For fixed \(\theta^{(p)}\), the existing GP covariance factorization does not become invalid merely because a new observation arrives.

Suppose

\[
K_{t-1}^{(p)}
=
L_{t-1}^{(p)}
(L_{t-1}^{(p)})^\top
\]

has already been factored.

The one-step predictive mean and variance at \(x_t\) can be obtained using the cached factorization. After \(y_t\) is observed, the Cholesky factor can be augmented by a block / rank-one update rather than recomputed from scratch.

For each unchanged particle:

- incremental predictive likelihood: roughly \(O(t^2)\) linear algebra after kernel-vector construction;
- Cholesky append: roughly \(O(t^2)\);
- no new hyperparameter search is required.

By contrast, if a rejuvenation move proposes a new \(\theta'\), the kernel matrix changes globally and an exact GP factorization generally must be recomputed:

\[
O(t^3)
\]

for that proposal.

Therefore distinguish carefully between:

### Cheap structural update
same particle locations, new weights, rank-one GP state update.

### Expensive structural move
change \(\theta\), requiring a new kernel/factorization.

The intended computational advantage exists only if expensive structural moves are **not required on every BO round**.

---

# 5. ESS has a clean information-theoretic interpretation

Assume for analysis that particles are iid from the exact previous posterior

\[
q(\theta)=p_{t-1}(\theta).
\]

The unnormalized importance ratio for the new posterior is

\[
r(\theta)
=
\frac{p_t(\theta)}{p_{t-1}(\theta)}
\propto
L_t(\theta).
\]

For large particle count, the normalized effective-sample-size fraction satisfies

\[
\frac{\mathrm{ESS}}{P}
\longrightarrow
\frac{
(\mathbb E_q[L_t])^2
}{
\mathbb E_q[L_t^2]
}.
\]

The order-2 Rényi divergence between successive posteriors is

\[
D_2(p_t\Vert p_{t-1})
=
\log
\mathbb E_{p_{t-1}}
\left[
\left(
\frac{p_t}{p_{t-1}}
\right)^2
\right].
\]

Therefore

\[
\boxed{
\frac{\mathrm{ESS}}{P}
\approx
\exp\left[
-D_2(p_t\Vert p_{t-1})
\right].
}
\]

This is central to Task 02.

The ability to reuse particles is controlled by the **information distance between consecutive structural posteriors**.

For a mild posterior update,

\[
D_2\ll1
\quad\Rightarrow\quad
\mathrm{ESS}/P\approx1.
\]

For an informative observation that strongly changes the structural posterior,

\[
D_2\gg1
\quad\Rightarrow\quad
\mathrm{ESS}/P\ll1.
\]

Equivalently, since

\[
L_t=e^{-\Delta E_t},
\]

the ESS depends on the variability of the incremental energy over the previous posterior.

For modest variation, a cumulant expansion gives the leading approximation

\[
D_2
\approx
\operatorname{Var}_{p_{t-1}}
[
\log L_t(\theta)
]
=
\operatorname{Var}_{p_{t-1}}
[
\Delta E_t(\theta)
],
\]

up to higher-order cumulants.

So the variance of the new energy factor is itself a useful diagnostic for whether structural rejuvenation will be needed.

---

# 6. The strongest possible computational hypothesis

A standard NUTS refit spends substantial computation exploring the full hyperparameter posterior even if the posterior has changed very little since the previous BO round.

An information-adaptive particle method could instead use:

1. **reweight** existing particles by the new likelihood factor;
2. **do nothing else** if ESS remains high;
3. if ESS falls, bridge gradually with
   \[
   p_{t,\beta}(\theta)
   \propto
   p_{t-1}(\theta)
   L_t(\theta)^\beta,
   \qquad
   0\le\beta\le1;
   \]
4. resample / rejuvenate only when needed;
5. later, exploit SAAS sparsity so that strongly shrunk coordinates can be moved less often than uncertain/active coordinates.

This gives an eventual target architecture in which compute is adaptive to:

- how much the posterior changes between rounds;
- how many dimensions remain structurally uncertain;
- how many particles actually require rejuvenation.

The ambitious long-term goal is therefore not merely "SMC instead of NUTS." It is:

> **information-adaptive sparse structural inference whose expensive computation scales with posterior change and unresolved effective dimension rather than blindly with every BO round and every ambient dimension.**

Task 02A tests only the first necessary condition: posterior continuity.

---

# 7. SAAS connection

Current BoTorch SAAS uses a hierarchical shrinkage prior. In the current implementation:

\[
\tau_{\rm sq}\sim \mathrm{HalfCauchy}(0.1),
\]

\[
r_j\sim \mathrm{HalfCauchy}(1),
\]

\[
\rho_j^2
=
\tau_{\rm sq}r_j,
\qquad
\ell_j
=
1/\sqrt{\rho_j^2}.
\]

The prior strongly shrinks inverse lengthscales toward zero while retaining heavy tails that permit important dimensions to escape shrinkage.

This is an energy-based structural-complexity mechanism already:

\[
E_{\rm struct}
=
-\log p(\tau_{\rm sq})
-\sum_j\log p(r_j)
-\log p(\text{mean})
-\log p(\text{outputscale})
-\log p(D\mid\theta).
\]

Task 02A does **not** need to reimplement this prior. Trusted NUTS samples from BoTorch will initialize the particle approximation.

Only if sequential reuse is promising should Task 02B implement the full latent SAAS structural energy for resample-move / annealed inference.

---

# 8. Important preprocessing invariant

The exact sequential identity

\[
p_t(\theta)
\propto
p_{t-1}(\theta)L_t(\theta)
\]

assumes the probabilistic model and parameterization remain fixed as data arrive.

Therefore:

- inputs may use fixed known normalization to \([0,1]^D\);
- outputs must **not** be re-standardized with a new empirical mean/std at every round.

For synthetic experiments:

1. generate all raw outcomes;
2. define a fixed output affine transform using only the initial dataset \(D_{n_0}\);
3. apply that same transform to every later outcome;
4. keep the transformed noise variance fixed consistently.

Do not use a round-dependent `Standardize` outcome transform in the sequential-reweighting experiment.

Otherwise the target changes for reasons beyond the newly added likelihood factor and the weight-update identity no longer applies directly.

---

# 9. Task 02A: NUTS-seeded sequential replay

Task 02A deliberately avoids implementing a new sampler.

At an initial size \(n_0\):

1. fit trusted BoTorch SAAS NUTS;
2. retain \(P\) posterior hyperparameter samples;
3. initialize equal particle weights;
4. cache the exact-GP state for each particle.

Then add observations one at a time.

At each step:

1. compute
   \[
   \log L_t^{(p)}
   =
   \log p(y_t\mid x_t,D_{t-1},\theta^{(p)});
   \]
2. update log weights;
3. compute ESS;
4. estimate
   \[
   \widehat D_2
   =
   -\log(\mathrm{ESS}/P)
   \]
   as the particle approximation diagnostic;
5. update each unchanged GP cache by rank-one Cholesky augmentation;
6. compare weighted predictions and EI to fresh NUTS reference fits at selected checkpoints.

No particle locations move in Task 02A.

The result measures the **reuse horizon**: how many new observations can be assimilated before the initial structural particle support becomes inadequate.

---

# 10. Decision-relevant comparison

For each fixed hyperparameter particle,

\[
Y(x)\mid D,\theta^{(p)}
\]

is Gaussian, so q=1 EI has the usual analytic form.

The fully Bayesian EI under a weighted structural posterior is simply

\[
\boxed{
EI_{\rm FB}(x)
=
\sum_{p=1}^P
w^{(p)}
EI(x;\theta^{(p)}).
}
\]

Therefore Task 02A can compare the sequential particle approximation to NUTS without introducing any acquisition-function novelty.

Use the same fixed candidate set.

Compare:

- EI values;
- Spearman EI ranking;
- top-k / top-5% overlap;
- selected next candidate.

This directly answers whether structural-posterior reuse is accurate enough for BO decisions.

---

# 11. What would make Task 02A encouraging

The strongest evidence would be:

1. ESS remains reasonably high for multiple new observations after an NUTS initialization;
2. when ESS is high, weighted structural marginals agree with fresh NUTS;
3. predictive mixtures and EI rankings remain close to fresh NUTS;
4. inactive dimensions remain structurally stable longer than active/uncertain dimensions;
5. rank-one sequential updates are far cheaper than a fresh NUTS refit.

Do not predeclare an arbitrary paper-success threshold, but report at least:

- number of new observations until ESS/P first falls below 0.75, 0.5, 0.25;
- posterior accuracy at those points;
- measured update time versus fresh NUTS time.

If ESS collapses after essentially every new observation, the proposed information-adaptive computational advantage is weak.

---

# 12. What comes after Task 02A

## If posterior reuse is promising: Task 02B

Implement the **full latent SAAS energy** in unconstrained coordinates and add adaptive tempered resample-move inference.

Important requirements for Task 02B:

- exact transformed-prior Jacobians;
- adaptive \(\beta\)-schedule using conditional ESS;
- a correct rejuvenation kernel;
- cached states retained for particles that do not move;
- instrument number of full GP factorizations.

Do not assume HMC is the best rejuvenation kernel. Compare at least a simple gradient-based parallel kernel such as MALA before considering more elaborate transport.

## If inactive coordinates appear especially stable: Task 02C

Test **selective structural rejuvenation**:

- freeze coordinates whose posterior remains strongly shrunk/stable;
- rejuvenate active or uncertain coordinates;
- include birth/unlocking moves so frozen dimensions can re-enter when evidence changes.

This is a more distinctive potential contribution because the cost could depend on unresolved effective dimension rather than ambient dimension.

## If exact GP factorizations become dominant: later scalable reference

Replace the exact GP data energy with a local/factorized approximation such as Vecchia.

This addresses scaling in \(n\) separately from structural scaling in \(D\).

---

# 13. Prior-art boundary

Do not claim novelty for SMC treatment of BO hyperparameters.

Relevant precedents include:

- Gramacy & Polson, "Particle learning of Gaussian process models for sequential design and optimization" (2011): online GP updating with particle learning for sequential design.
- Svensson, Dahlin & Schön, "Marginalizing Gaussian Process Hyperparameters using Sequential Monte Carlo" (2015): SMC marginalization of GP hyperparameters, emphasizing online problems and multimodal posteriors.
- Kim, Park & Kim, "A Sequential Monte Carlo Treatment of Bayesian Optimization Hyperparameters" (2020 preprint): explicitly uses SMC and a data-tempering view to reuse BO hyperparameter samples across iterations.
- Eriksson & Jankowiak, "High-Dimensional Bayesian Optimization with Sparse Axis-Aligned Subspaces" (2021): SAASBO and HMC/NUTS structural inference.
- Jimenez & Katzfuss, "Scalable Bayesian Optimization Using Vecchia Approximations of Gaussian Processes" (2023): scalable local GP likelihood/prediction and an explicit suggestion that Vecchia could reduce SAASBO cost.

Task 02A is therefore a **research diagnostic**, not the final contribution.

The possible later novelty would need to come from SAAS-specific, information-adaptive sparse structural transport, selective unlocking/rejuvenation, scalable local energy evaluation, and/or integration with the broader compositional-energy framework.
