# Task 02B Mathematical Pivot: Decision-Relevant Structural Inference

## 1. Why Task 02A changes the target

Task 02A showed that fixed-support reuse of an old SAAS posterior is **not** a reliable approximation of the full structural posterior for very long:

- ESS can collapse after only a few new observations;
- structural posterior discrepancies grow substantially;
- raw inactive-coordinate log-lengthscale marginals can drift strongly.

However, the same experiments showed that **BO decision quantities can remain much more stable than the full posterior**. In several checkpoints, EI ranking remained very high even after ESS became extremely small.

Therefore the next question should not be:

> Can we cheaply approximate the entire fresh SAAS posterior?

It should be:

> Can we cheaply approximate the parts of structural uncertainty that matter for the next BO decision?

This is a different approximation target.

Task 02B is a bounded diagnostic and mathematical bridge. It does **not** yet implement a new EBM sampler.

---

# 2. Full Bayesian acquisition as a functional of the structural posterior

Let

\[
p(\theta\mid D)
\]

be the fully Bayesian posterior over GP structural hyperparameters.

For fixed \(\theta\), let

\[
a_\theta(x)
\]

be the q=1 acquisition, initially analytic EI.

Then the fully Bayesian acquisition is

\[
\boxed{
A_p(x)
=
\mathbb E_{\theta\sim p(\theta\mid D)}
[a_\theta(x)].
}
\]

For a weighted approximation

\[
q(\theta)
=
\sum_{k=1}^{K} w_k\delta_{\theta_k},
\]

the approximate acquisition is

\[
A_q(x)
=
\sum_{k=1}^{K}w_k a_{\theta_k}(x).
\]

For BO, the relevant object is often \(A_p\), not the full posterior \(p\).

---

# 3. A decision-relevant discrepancy and regret bound

Define the acquisition-function discrepancy

\[
\delta_A(p,q)
=
\sup_{x\in\mathcal X}
|A_p(x)-A_q(x)|.
\]

Let

\[
x_p\in\arg\max_x A_p(x),
\qquad
x_q\in\arg\max_x A_q(x).
\]

Then

\[
\boxed{
A_p(x_p)-A_p(x_q)
\le
2\delta_A(p,q).
}
\]

Proof:

\[
A_p(x_p)
\le
A_q(x_p)+\delta_A
\le
A_q(x_q)+\delta_A
\le
A_p(x_q)+2\delta_A.
\]

Therefore a posterior approximation can be poor in Wasserstein, MMD, or ESS while still producing a good BO decision if it preserves the acquisition functional.

This is the mathematical motivation for Task 02B.

For finite candidate sets, use

\[
\delta_{A,C}
=
\max_{x\in C}
|A_p(x)-A_q(x)|.
\]

Also report normalized decision regret

\[
r_{\rm dec}
=
\frac{
A_p(x_p)-A_p(x_q)
}{
\max(|A_p(x_p)|,\epsilon)
}.
\]

Exact argmax agreement alone is too brittle when the acquisition surface is flat or multimodal.

---

# 4. Acquisition signatures of structural particles

For a fixed candidate set

\[
C=\{x_1,\ldots,x_J\},
\]

associate each structural particle \(\theta_p\) with an **acquisition signature**

\[
v_p
=
[
a_{\theta_p}(x_1),
\ldots,
a_{\theta_p}(x_J)
]
\in\mathbb R^J.
\]

The fresh-NUTS fully Bayesian acquisition vector is

\[
\bar v
=
\sum_p w_p v_p.
\]

Two structural particles that are far apart in hyperparameter space can be nearly equivalent for the pending BO decision if their acquisition signatures are similar.

Conversely, particles that are close in raw lengthscale space can matter differently if they imply different acquisition behavior near promising candidates.

Task 02B will measure whether the acquisition-signature matrix has low effective rank and whether a small weighted structural coreset can reproduce \(\bar v\).

If yes, this is evidence that **decision-relevant structural complexity is substantially smaller than posterior complexity**.

There is also a useful geometric consequence. If all acquisition signatures lie in an affine subspace of dimension \(r\), then the full-Bayesian acquisition vector \(\bar v\), being a convex combination of those signatures, can be represented exactly by a convex combination of at most

\[
r+1
\]

particle signatures (Carathéodory's theorem). Thus an exactly low-dimensional decision representation implies an exactly small structural coreset. In the approximate low-rank case this motivates testing approximate coresets and measuring their decision regret.

This is not a claim that the SAAS acquisition signatures will actually be low rank; Task 02B tests that assumption.

---

# 5. Structural-posterior compression as an oracle diagnostic

Given fresh NUTS particles \(\theta_1,\ldots,\theta_P\), define matrix

\[
V\in\mathbb R^{P\times J},
\qquad
V_{pj}=a_{\theta_p}(x_j).
\]

The full Bayesian acquisition is the weighted row average.

Task 02B should test whether

\[
\bar v
\approx
\sum_{k=1}^{K}\alpha_k v_{i_k},
\qquad
\alpha_k\ge0,\quad\sum_k\alpha_k=1
\]

with

\[
K\ll P.
\]

This is only a **teacher/oracle compression test**. It does not yet provide a way to discover the right particles without NUTS.

Compare:
- random thinning;
- posterior-space clustering / medoids if straightforward;
- acquisition-space weighted coreset selection.

A simple greedy conditional-gradient / Frank-Wolfe style selection or nonnegative least-squares refinement is sufficient. Avoid excessive optimization engineering.

Key metrics:
- max acquisition-vector error;
- RMSE;
- Spearman ranking;
- top-5% overlap;
- normalized decision regret;
- number of retained particles.

---

# 6. Exact joint decision-energy formulation

The broader EBM direction becomes more compelling if we can avoid first approximating a globally accurate \(p(\theta\mid D)\).

For fixed structural hyperparameters \(\theta\), q=1 Gaussian EI is

\[
a_\theta(x)=EI_\theta(x).
\]

Define the unnormalized joint target

\[
\boxed{
\tilde\pi(x,\theta\mid D)
\propto
p_0(x)\,
p(\theta)\,
p(D\mid\theta)\,
a_\theta(x).
}
\]

Its energy is

\[
E(x,\theta)
=
-\log p_0(x)
-\log p(\theta)
-\log p(D\mid\theta)
-\log a_\theta(x)
+C.
\]

Marginalizing \(\theta\),

\[
\begin{aligned}
\pi(x\mid D)
&\propto
p_0(x)
\int
p(\theta)p(D\mid\theta)a_\theta(x)
\,d\theta \\
&\propto
p_0(x)
\mathbb E_{p(\theta\mid D)}
[a_\theta(x)].
\end{aligned}
\]

Thus, for uniform \(p_0\),

\[
\boxed{
\pi(x\mid D)
\propto
A_p(x),
}
\]

the full Bayesian acquisition.

Important:

- This does **not** require the normalized posterior \(p(\theta\mid D)\).
- The likelihood normalizing evidence \(p(D)\) cancels because it is independent of \(x\).
- This does **not** mean joint MAP in \((x,\theta)\) equals full Bayesian acquisition maximization. The correct object is the marginal over \(x\).
- This is an ephemeral decision target; it does not permanently retrain the structural belief toward the acquisition.

This is the core mathematical bridge toward a genuinely EBM-native BO method.

---

# 7. Latent-utility version: no closed-form acquisition required

The analytic-EI version above is useful for validation, but the larger EBM opportunity is that a closed-form acquisition is not necessary.

For fixed \(\theta\), write a predictive sample as

\[
y=g_\theta(x,z),
\qquad
z\sim p_0(z)
\]

(e.g. \(z\sim N(0,1)\) for a scalar Gaussian predictive model).

For positive/nonnegative utility \(u(x,y)\), define

\[
\tilde\pi(x,\theta,z\mid D)
\propto
p_0(x)
p(\theta)
p(D\mid\theta)
p_0(z)
u(x,g_\theta(x,z)).
\]

Then

\[
\pi(x\mid D)
\propto
p_0(x)
\mathbb E_{\theta\mid D}
\mathbb E_z[
u(x,g_\theta(x,z))
].
\]

That marginal is the full Bayesian expected-utility acquisition.

This formulation could eventually combine:
- SAAS structural energy;
- learned residual predictive energy;
- constraints;
- localization/design priors;
- non-Gaussian or implicit predictive models.

That is where the EBM perspective becomes more than notation.

Task 02B only verifies the analytic-EI structural version.

---

# 8. Replicated sharpening under structural uncertainty

If a sharper marginal over \(x\) is desired, introduce independent replicas

\[
\theta_1,\ldots,\theta_M
\]

tied to the same design \(x\):

\[
\tilde\pi_M(x,\theta_{1:M})
\propto
p_0(x)
\prod_{m=1}^{M}
[
p(\theta_m)p(D\mid\theta_m)a_{\theta_m}(x)
].
\]

Marginalizing all structural replicas gives

\[
\pi_M(x)
\propto
p_0(x)
[p(D)A_p(x)]^M.
\]

With uniform \(p_0\),

\[
\boxed{
\pi_M(x)
\propto
A_p(x)^M.
}
\]

The same identity holds for replicated \((\theta_m,z_m)\) utility latents.

Do not accidentally reuse one common \(\theta\) across replicas if the goal is \(A_p(x)^M\). A common \(\theta\) would yield a different quantity involving \(\mathbb E[a_\theta(x)^M]\).

---

# 9. Why this could be a real computational advantage

The conventional fully Bayesian BO pipeline is conceptually:

1. approximate \(p(\theta\mid D)\) globally;
2. integrate acquisition over that approximation;
3. optimize the integrated acquisition.

The joint-energy alternative would be:

\[
\boxed{
\text{perform inference directly on }
E_{\rm structural}
+
E_{\rm decision}
}
\]

so the \(x\)-marginal already represents the full Bayesian decision.

If successful, this can avoid spending large computation approximating parts of the hyperparameter posterior that have negligible effect on the next decision.

The potential gain is therefore **decision-targeted inference without permanently decision-targeted model training**.

This is related to, but distinct from, approximation-aware / loss-calibrated inference:
- those methods alter the surrogate approximation objective using downstream utility;
- the proposed joint target leaves the persistent Bayesian model conceptually unchanged and performs an ephemeral decision-conditioned inference step.

This distinction must be maintained carefully.

---

# 10. Why Task 02B is diagnostic rather than the final method

A modern EBM/particle algorithm such as SVGD, annealed Langevin, or MALA could operate using gradients of this unnormalized joint target.

However, before implementing that machinery, Task 02B should establish:

1. decision error is substantially less sensitive than global posterior error in the existing Task 02A results;
2. fresh SAAS acquisition signatures are compressible with \(K\ll P\);
3. the joint-energy marginal identity is numerically correct using NUTS particles as a teacher.

If these fail, a sophisticated joint-energy sampler is unlikely to be worthwhile.

If they succeed, Task 02C can implement a GPU-parallel particle transport algorithm directly on the joint structural-decision energy.

---

# 11. Important prior-art boundaries

Do not claim novelty for:
- utility-aware approximate Bayesian inference;
- loss-calibrated inference;
- generic SVGD;
- generic SMC;
- particle/Stein batch BO;
- acquisition sampling;
- SAAS priors.

The potential research contribution would need to be the **specific decision-conditioned structural-energy formulation and inference architecture**, together with evidence that it avoids unnecessary full-posterior computation while retaining full-Bayesian BO decisions and later composes naturally with richer energy terms.

Task 02B is meant to determine whether that direction is empirically plausible.
