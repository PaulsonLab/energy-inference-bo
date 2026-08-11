# Mathematical Core and Initial Scope

## 1. Goal

Test whether an energy-based formulation provides a **real** advantage for Bayesian optimization (BO), rather than only a change of notation.

There are two separate hypotheses:

1. **Modeling hypothesis:** a strongly regularized residual energy can correct predictive structure that a Gaussian reference cannot represent, while collapsing back to that reference when the data do not support extra complexity.
2. **Decision/inference hypothesis:** expected-utility acquisition can be represented as a marginal of an augmented energy model, which may allow particle/annealed inference to replace the usual nested procedure of repeatedly estimating and optimizing an acquisition in difficult settings.

The first task tests these hypotheses in idealized settings with known ground truth.

---

# 2. Reference predictive model and residual energy

Let

\[
p_{G,\theta}(y\mid x,D)
\]

be a Gaussian reference predictive distribution with hyperparameters \(\theta\).

For scalar output define

\[
z=\frac{y-\mu_\theta(x;D)}
{\sigma_\theta(x;D)}.
\]

A corrected predictive density is

\[
q_{\theta,\phi}(y\mid x,D)
=
\frac{
p_{G,\theta}(y\mid x,D)\,
\exp[-R_\phi(z,c)]
}{
Z_\phi(c)
},
\]

with

\[
Z_\phi(c)
=
\mathbb E_{Z\sim N(0,1)}
[
\exp(-R_\phi(Z,c))
].
\]

Important:

- \(R_\phi=0\) recovers the Gaussian reference exactly.
- A scalar \(Z_\phi\) can be computed accurately by one-dimensional quadrature.
- \(Z_\phi(c)\) cannot be omitted if it varies with \(x\) or context.
- The residual energy is only defined up to an additive constant. Fix this gauge explicitly.

For the first experiment use a context-free \(R_\phi(z)\). Then \(Z_\phi\) is independent of \(x\).

---

# 3. What the residual can and cannot fix

A non-Gaussian residual does not automatically fix a bad stationary kernel.

Separate three types of predictive error:

1. location / mean error;
2. scale / uncertainty error;
3. higher-order shape error.

Use nested baselines.

## G0 — Gaussian reference

\[
Y\mid x,D\sim N(\mu,\sigma^2).
\]

## G1 — Gaussian location/scale correction

For small corrections \(a_\psi(c)\) and \(b_\psi(c)\),

\[
\tilde\mu=\mu+\sigma a_\psi(c),
\qquad
\tilde\sigma=\sigma\exp[b_\psi(c)].
\]

Then

\[
Y\mid x,D\sim N(\tilde\mu,\tilde\sigma^2).
\]

## E — higher-order residual energy

Define

\[
r=\frac{y-\tilde\mu}{\tilde\sigma}
\]

and

\[
q(y\mid x,D)
=
\frac{1}{\tilde\sigma}
\frac{
\varphi(r)e^{-R_\phi(r,c)}
}{
Z_\phi(c)
}.
\]

The EBM is scientifically interesting only if it improves beyond a fair Gaussian location/scale correction.

For Task 01, isolate shape with a ground-truth residual whose mean is 0 and variance is 1.

---

# 4. Proper sequential training when a GP is introduced

Do not train the residual using posterior residuals at points after conditioning on those same observations.

For ordered data,

\[
D_i=\{(x_1,y_1),\ldots,(x_i,y_i)\},
\]

use the one-step predictive conditional

\[
q_i(y_i)
=
q_{\theta,\phi}(y_i\mid x_i,D_{i-1}).
\]

Then

\[
q_{\theta,\phi}(y_{1:n}\mid x_{1:n})
=
\prod_{i=1}^n
q_i(y_i).
\]

Its negative log likelihood is

\[
\mathcal L(\theta,\phi)
=
-\sum_i
\log p_{G,\theta}(y_i\mid x_i,D_{i-1})
+
\sum_i
[
R_\phi(z_i,c_i)
+
\log Z_\phi(c_i)
]
+
\Omega_\theta+\Omega_\phi.
\]

When \(R_\phi=0\) and the reference is the same exact GP throughout, the product of exact Gaussian predictive conditionals equals the GP joint marginal likelihood.

Recommended low-data strategy later:

A. fit \(\theta\) with ordinary GP marginal likelihood;

B. freeze \(\theta\), fit a strongly regularized \(\phi\) initialized at zero;

C. only if stable, optionally perform a short joint fine-tuning of \((\theta,\phi)\).

This staged fit should be compared with direct joint training because kernel parameters and residual-energy parameters can otherwise compete to explain the same errors.

---

# 5. Initial residual parameterization and gauge

Start with

\[
R_\phi(z)
=
\sum_{k=1}^{K}
\phi_k\tilde b_k(z),
\]

where \(K=7\) or \(9\) and \(b_k\) are bounded RBF or spline functions.

Use reference-centered basis functions

\[
\tilde b_k(z)
=
b_k(z)
-
\mathbb E_{N(0,1)}[b_k(Z)]
\]

as a simple gauge convention. The centering does not change the expressive normalized family but removes an unnecessary constant-energy direction from the parameterization.

Use:
- centers spanning roughly \([-3,3]\);
- a fixed, documented bandwidth;
- strong L2 / Gaussian shrinkage toward \(\phi=0\);
- 32–64 point Gauss-Hermite quadrature.

Because the Gaussian reference dominates the tails and the basis functions are bounded, the scalar partition function remains finite.

---

# 6. Modeling metrics

RMSE alone is insufficient for the intended mechanism.

On problems with known predictive truth, measure:

1. held-out predictive log score / NLL;
2. CRPS where straightforward;
3. PIT / quantile calibration;
4. error in probability of improvement;
5. error in expected improvement;
6. rank correlation of acquisition across candidate points;
7. recall/overlap of the true top-acquisition region;
8. eventual BO simple regret.

The key question is:

> Does higher-order predictive shape improve decision-relevant tails after location and scale are already correct or well calibrated?

---

# 7. Oracle shape experiment

Let the Gaussian residual reference be

\[
Z_G\sim N(0,1).
\]

Use the ground-truth mixture

\[
Z_\star
\sim
0.8N(-0.3,0.8^2)
+
0.2N(1.2,0.8^2).
\]

Check analytically/numerically:

\[
E[Z_\star]=0,
\qquad
\operatorname{Var}(Z_\star)=1.
\]

Define

\[
Y_\star(x)=\mu(x)+\sigma(x)Z_\star,
\]

and the moment-matched Gaussian

\[
Y_G(x)=\mu(x)+\sigma(x)Z_G.
\]

Thus both predictive models have exactly the same mean and variance at every \(x\), but different higher-order shape.

Compare:
- oracle true expected improvement;
- Gaussian expected improvement;
- expected improvement from a residual energy fitted to samples from \(Z_\star\).

Use several predetermined \(\mu(x),\sigma(x),f_{\rm best}\) scenarios rather than unconstrained search for a favorable example. It is acceptable to include one clearly labeled illustrative scenario in which the acquisition ranking changes, but also report scenarios where it does not.

Train the residual on sample sizes such as 20, 50, 100, 200, 500.

Fair baselines:
1. fixed \(N(0,1)\);
2. Gaussian location/scale MLE fitted to the same residual samples;
3. optionally Student-t location/scale;
4. residual-energy model.

Also report the fitted EBM mean and variance so any finite-sample gain due merely to location/scale adjustment is visible.

---

# 8. Expected utility as augmented inference

Let \(q(y\mid x,D)\) be a normalized predictive distribution and \(u(x,y)\ge0\) a utility.

Define

\[
a(x)=
\mathbb E_q[u(x,Y)].
\]

With design prior \(p_0(x)\),

\[
\pi_1(x,y)
\propto
p_0(x)
q(y\mid x,D)
u(x,y)
\]

has marginal

\[
\pi_1(x)
\propto
p_0(x)a(x).
\]

For a uniform design prior, the marginal mode is the usual expected-utility acquisition maximizer.

This identity is an existing augmented-probability idea; it is a mathematical tool, not a novelty claim.

---

# 9. Replicated sharpening

Introduce \(M\) conditionally independent latent outcomes:

\[
y^{(1)},\ldots,y^{(M)}
\sim q(\cdot\mid x,D).
\]

Define

\[
\pi_M(x,y^{(1:M)})
\propto
p_0(x)
\prod_{m=1}^M
q(y^{(m)}\mid x,D)
u(x,y^{(m)}).
\]

Then

\[
\pi_M(x)
\propto
p_0(x)a(x)^M.
\]

If \(p_0(x)\) is uniform, increasing \(M\) sharpens the marginal around the acquisition maximizers without changing them.

If \(p_0(x)\) is nonuniform, the mode of \(p_0(x)a(x)^M\) need not equal the acquisition-only maximizer.

---

# 10. Exact Gaussian / GP q=1 sanity check

This is required.

For a Gaussian predictive distribution

\[
Y(x)\mid D
\sim
N(\mu(x),\sigma^2(x))
\]

and improvement utility

\[
u(x,y)=(y-f_{\rm best})_+,
\]

the expected utility is ordinary EI.

Therefore

\[
\pi_1(x)\propto p_0(x)EI(x).
\]

With uniform \(p_0\),

\[
\arg\max_x \pi_1(x)
=
\arg\max_x EI(x)
=
\arg\max_x \log EI(x)
\]

where EI is positive.

A small exact-GP smoke experiment must numerically verify that:
- analytic EI agrees with direct quadrature of expected improvement;
- the augmented \(x\)-marginal agrees with normalized EI;
- the selected maximizer agrees with standard EI / LogEI.

If a small \(\epsilon>0\) is added to make the utility strictly positive for log-energy computations, the marginal becomes proportional to \(EI(x)+\epsilon\). The argmax is unchanged for uniform \(\epsilon\), but the normalized density is not identical to normalized EI.

---

# 11. Batch extension is not automatic

For a batch

\[
X=(x_1,\ldots,x_q),
\]

a joint utility requires a coherent joint predictive distribution

\[
q(Y_X\mid X,D).
\]

A scalar marginal residual correction at each \(x\) does not by itself define the joint dependence required for qEI.

Therefore Task 01 and the first real GP model remain \(q=1\).

Possible later routes include:
- retaining a Gaussian copula/reference dependence while transforming marginal residuals;
- a sequential/factorized joint correction;
- a genuinely multivariate residual energy with valid normalization/sampling.

Do not claim a batch advantage until a coherent joint predictive construction has been implemented and tested.

For any valid joint \(q(Y_X\mid X,D)\),

\[
a_q(X)
=
E_q[u_q(X,Y_X)]
\]

can be embedded in the same augmented construction, but the augmented state and inference cost grow with both \(q\) and the number of replicas \(M\).

---

# 12. Why acquisition-side inference might help

For standard Gaussian \(q=1\) BO, analytic EI / LogEI is already strong. Do not expect the augmented sampler to beat it.

The acquisition-side hypothesis is more plausible when:
- the predictive model is non-Gaussian or implicit;
- expected utility is expensive to evaluate repeatedly;
- the design space is discrete or structured;
- a valid large-batch joint predictive model is available;
- particle/annealed inference can reuse and parallelize latent calculations.

The augmented formulation turns a nested

\[
\text{estimate expected utility at proposed }x
\rightarrow
\text{optimize it}
\]

problem into one joint inference problem. Whether this is computationally better is an empirical question.

---

# 13. Stage gates

## Gate 1 — oracle shape value

Does non-Gaussian shape materially alter correct BO decisions when mean and variance are fixed?

If no, stop/rethink the modeling direction.

## Gate 2 — low-data learnability

Can a tiny strongly regularized residual energy recover useful shape better than fair Gaussian location/scale baselines?

If no, stop or restrict the method to higher-data regimes.

## Gate 3 — q=1 acquisition identity

Does augmented inference reproduce ordinary GP EI/LogEI decisions exactly/numerically?

This must pass; failure indicates an implementation or mathematical error.

## Gate 4 — acquisition inference practicality

Does particle inference approximate the target without catastrophic weight degeneracy?

If no, either develop annealed SMC or keep the EBM only as a surrogate idea.

## Gate 5 — exact GP integration

Only after Gates 1–4, introduce corrected prequential GP training.

## Gate 6 — scalable reference

Only after the exact-GP version works, introduce Vecchia/local factorization.

## Gate 7 — batch and structured domains

Only after the single-point formulation is stable, define a coherent joint predictive correction and test batch / molecular optimization.
