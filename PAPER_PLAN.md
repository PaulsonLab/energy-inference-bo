# Decision-Tilted Acquisition Inference — Paper Plan

## 0. Purpose

This repository is being reset around one narrow paper hypothesis.

We are **not** trying to build a general EBM framework for all of Bayesian optimization. We are testing one predicted failure mode of Monte Carlo acquisition computation and one possible energy-based remedy.

### Working thesis

> Monte Carlo BO estimates expected-utility acquisitions using samples from the posterior, even though the acquisition is controlled by a utility-tilted distribution over uncertain worlds. The relative variance of the standard Monte Carlo estimator is exactly determined by the divergence between these two distributions. When this posterior-to-decision shift is large, ordinary MC acquisition values and gradients can require very large sample counts. Decision-adapted energy inference should reduce this integration cost while preserving the same surrogate, acquisition, and gradient-based outer optimizer.

Short version:

> **LogEI fixes the arithmetic; decision tilting may fix the samples.**

The paper lives or dies on whether this is an **orders-of-magnitude problem in realistic BO regimes**, not on whether a new estimator saves a modest constant factor.

---

# 1. What the paper is and is not

## We keep fixed

- the surrogate belief;
- the mathematical acquisition function;
- gradient-based outer acquisition optimization such as multi-start L-BFGS / Adam;
- the underlying BO objective.

## We change only

The distribution used to perform the **inner expectation over uncertainty**.

## We are not proposing

- MCMC or diffusion sampling over the decision variable \(X\);
- a replacement for qLogEI's numerical stabilization;
- a new surrogate model;
- a new asymptotic BO policy;
- a claim that energy notation is intrinsically more expressive than probability notation.

The potential gain is computational/statistical:

> obtain accurate acquisition values and gradients with much less integration effort when decision-relevant posterior worlds are rare or highly concentrated.

If brute-force Sobol/QMC posterior sampling is as fast and reliable at matched wall-clock, the project should stop.

---

# 2. General mathematical setup

Let \(Z\) denote the uncertain world relevant to a BO decision. It can be arbitrarily non-Gaussian and may include:

- a latent function;
- objective and constraint functions;
- model parameters;
- predictive noise;
- latent variables of a diffusion/EBM surrogate;
- joint outcomes for a batch.

Let \(P\) be the current posterior probability measure over \(Z\).

For a candidate decision \(X\), let

\[
U_X(Z)\ge 0
\]

be a measurable utility and define the acquisition

\[
\boxed{
\alpha(X)=\mathbb E_P[U_X(Z)].
}
\]

Assume \(0<\alpha(X)<\infty\). For variance statements also assume \(\mathbb E_P[U_X^2]<\infty\).

Examples include EI, smoothed qEI/qLogEI utilities, constrained improvement, noisy improvement, and other expected-utility acquisitions.

---

# 3. Decision-tilted posterior

Define

\[
\boxed{
\frac{d\Pi_X}{dP}(Z)
=
\frac{U_X(Z)}{\alpha(X)}.
}
\]

Equivalently,

\[
\Pi_X(dz)
=
\frac{U_X(z)P(dz)}{\alpha(X)}.
\]

Interpretation:

> \(\Pi_X\) is the distribution of posterior worlds reweighted by how much they contribute to the decision at \(X\).

No Gaussian assumption is required.

The central diagnostic quantity is the divergence between ordinary posterior worlds and decision-relevant worlds.

---

# 4. Core theorem: standard Monte Carlo difficulty

For iid posterior samples

\[
Z_i\sim P,
\]

the standard estimator is

\[
\widehat\alpha_N(X)
=
\frac1N\sum_{i=1}^N U_X(Z_i).
\]

Then

\[
\boxed{
\frac{
\operatorname{Var}[\widehat\alpha_N(X)]
}{
\alpha(X)^2
}
=
\frac1N
\chi^2(\Pi_X\Vert P).
}
\]

Since

\[
\frac{d\Pi_X}{dP}
=
\frac{U_X}{\alpha},
\]

we also have

\[
\boxed{
\chi^2(\Pi_X\Vert P)
=
\frac{\mathbb E_P[U_X^2]}
{\mathbb E_P[U_X]^2}
-1
=
CV_P^2(U_X).
}
\]

Define the order-2 Rényi decision shift

\[
\boxed{
D_2(X)
=
\log\left(1+\chi^2(\Pi_X\Vert P)\right).
}
\]

Then

\[
\boxed{
\operatorname{RelVar}(\widehat\alpha_N)
=
\frac{e^{D_2(X)}-1}{N}.
}
\]

This is completely distribution-free.

---

# 5. Effective sample size interpretation

For posterior utility weights \(w_i=U_X(Z_i)\),

\[
ESS_N(X)
=
\frac{
\left(\sum_i w_i\right)^2
}{
\sum_i w_i^2
}.
\]

In the population limit,

\[
\boxed{
\frac{ESS}{N}
\rightarrow
\frac{1}
{1+\chi^2(\Pi_X\Vert P)}
=
e^{-D_2(X)}.
}
\]

Thus the same decision shift controls:

- relative MC variance;
- effective sample fraction;
- how many ordinary posterior samples materially contribute to the acquisition.

This gives us a practical diagnostic before we build a new sampler.

---

# 6. Rare-event special case

For unsmoothed improvement, suppose

\[
p_{\rm imp}(X)
=
P\{U_X>0\}.
\]

Then

\[
\boxed{
\chi^2(\Pi_X\Vert P)
\ge
\frac1{p_{\rm imp}(X)}-1,
}
\]

so

\[
\boxed{
\operatorname{RelVar}(\widehat\alpha_N)
\ge
\frac1N
\left[
\frac1{p_{\rm imp}(X)}-1
\right].
}
\]

Rare improvement is one mechanism producing large shift.

For constrained improvement, the relevant event may be

\[
\{\text{improvement}\}
\cap
\{\text{feasibility}\},
\]

which can be substantially rarer.

The broader quantity remains \(\chi^2(\Pi_X\Vert P)\); literal zero-utility events are not required.

---

# 7. General decision-adapted proposal theorem

Let \(Q_X\) be any proposal satisfying the necessary support condition \(\Pi_X\ll Q_X\).

For

\[
Z_i\sim Q_X,
\]

use the importance-corrected estimator

\[
\widehat\alpha_{Q,N}(X)
=
\frac1N
\sum_{i=1}^N
U_X(Z_i)
\frac{dP}{dQ_X}(Z_i).
\]

Then

\[
\boxed{
\frac{
\operatorname{Var}[\widehat\alpha_{Q,N}(X)]
}{
\alpha(X)^2
}
=
\frac1N
\chi^2(\Pi_X\Vert Q_X).
}
\]

Therefore the ideal proposal is

\[
\boxed{
Q_X^\star=\Pi_X,
}
\]

which gives zero-variance acquisition estimation.

The algorithmic problem is therefore precise:

> construct a cheap \(Q_X\) that is substantially closer to \(\Pi_X\) than the original posterior \(P\).

---

# 8. Energy / free-energy form

Suppose the posterior is represented with respect to some base measure as

\[
P(dz)
=
\frac{
e^{-E_B(z)}
}{
Z_B
}\,dz.
\]

Then the decision tilt has energy

\[
\boxed{
E_{\Pi_X}(z)
=
E_B(z)-\log U_X(z)
}
\]

up to an additive constant.

The acquisition is the ratio

\[
\boxed{
\alpha(X)
=
\frac{Z_{B+U_X}}{Z_B},
}
\]

and therefore

\[
\boxed{
\log\alpha(X)
=
F_B-F_{B+U_X}.
}
\]

This is where explicit energy access may matter computationally: \(\Pi_X\) can be targeted without knowing its normalizing constant \(\alpha(X)\).

---

# 9. Annealed path / thermodynamic integration

For strictly positive utility, including a smoothed LogEI-style utility, define

\[
\pi_{\lambda,X}(dz)
\propto
P(dz)U_X(z)^\lambda,
\qquad
0\le\lambda\le1.
\]

Then

\[
\pi_{0,X}=P,
\qquad
\pi_{1,X}=\Pi_X.
\]

The log acquisition satisfies

\[
\boxed{
\log\alpha(X)
=
\int_0^1
\mathbb E_{\pi_{\lambda,X}}
[
\log U_X(Z)
]
\,d\lambda.
}
\]

This suggests AIS / SMC / tempering / transport as possible integration tools.

Important:

- \(\lambda\) is an **algorithmic annealing coordinate**, not a tunable model weight;
- the target at \(\lambda=1\) remains the original acquisition.

No annealed sampler is authorized until the mechanism experiments show that the posterior-to-decision shift is genuinely problematic.

---

# 10. Variational characterization

For \(U_X>0\) and appropriate integrability,

\[
\boxed{
\log\alpha(X)
=
\sup_Q
\left\{
\mathbb E_Q[\log U_X(Z)]
-
KL(Q\Vert P)
\right\}.
}
\]

The optimizer is exactly

\[
\boxed{
Q^\star=\Pi_X.
}
\]

Moreover,

\[
\boxed{
\log\alpha(X)
-
\left[
\mathbb E_Q\log U_X
-
KL(Q\Vert P)
\right]
=
KL(Q\Vert\Pi_X).
}
\]

This relationship is useful for constructing proposals, but the variational identity itself is not claimed as novel.

---

# 11. Acquisition-gradient identity

When \(P\) is a distribution over worlds independent of \(X\) and differentiation can pass through the expectation,

\[
\boxed{
\nabla_X\log\alpha(X)
=
\mathbb E_{\Pi_X}
[
\nabla_X\log U_X(Z)
].
}
\]

The cleanest interpretation is function-space or fixed-base-randomness:

\[
Z\sim P
\]

is held fixed while \(X\) changes how that world is interrogated.

For predictive distributions written directly as \(P_X\), extra dependence on \(X\) must be handled correctly; do not use this identity blindly.

The intended outer loop remains standard gradient-based optimization.

---

# 12. Novelty boundaries that must remain explicit

The paper must be distinguished from:

## qLogEI

qLogEI stabilizes acquisition arithmetic and gradients. It does not adapt the posterior sampling distribution toward decision-relevant worlds.

Our proposed failure mode is **statistical integration mismatch**, not floating-point underflow.

## SBBO / B3O / GenBO

These methods sample or generate the **decision variable \(X\)** from acquisition-related targets.

We keep L-BFGS / Adam over \(X\).

We adapt samples of the **uncertain world \(Z\)** inside acquisition evaluation.

## Approximation-Aware BO / utility-calibrated VI

Those methods adapt an approximate surrogate/inference model toward decision utility.

Our intended method keeps the surrogate belief \(P\) fixed and uses a temporary integration proposal \(Q_X\), with correction back to the same acquisition whenever possible.

## Generic importance sampling / rare-event methods

Importance sampling is not novel.

The paper requires a BO-specific contribution:

1. the exact posterior-to-decision divergence characterization;
2. evidence that the divergence is large where good BO decisions actually live;
3. an acquisition estimator/gradient procedure that exploits the tilted energy;
4. a regime where this makes an otherwise impractical modern MC acquisition practical.

If those do not materialize, stop.

---

# 13. Evidence sequence

The first experiments are designed to answer questions in the order required by the paper.

We do **not** build a sophisticated energy sampler first.

---

# 14. Experiment: `rare_mode_mechanism`

## Purpose

Produce a final-paper-quality mechanism figure that:

- verifies the variance/divergence identities;
- is explicitly non-Gaussian;
- shows acquisition misranking from posterior MC;
- visualizes how the utility tilt changes the relevant posterior worlds;
- establishes that numerical LogEI stability does not fix sample scarcity.

This should run on CPU.

## Model

Use a one-dimensional decision domain \(x\in[-2,2]\).

The predictive belief is a two-component mixture:

\[
P_x
=
(1-\varepsilon)
\mathcal N(\mu_0(x),\sigma_0^2)
+
\varepsilon
\mathcal N(\mu_1(x),\sigma_1^2).
\]

Choose fixed smooth component means:

- a broad common mode with a modest improvement peak near \(x_A\);
- a rare mode with a much larger performance peak near \(x_B\).

Use a fixed incumbent \(f^\star\).

A reasonable starting construction is:

\[
\varepsilon=0.005,
\]

with the rare component sufficiently high that

\[
EI(x_B)>EI(x_A)
\]

despite its low posterior probability.

The exact numeric parameters must be frozen in the experiment specification before generating final results.

## Why this is useful

The posterior is non-Gaussian, but EI is analytically available because each Gaussian component has analytic first and second truncated moments.

For \(Y\sim\mathcal N(\mu,\sigma^2)\), \(a=(\mu-f^\star)/\sigma\),

\[
EI
=
(\mu-f^\star)\Phi(a)
+
\sigma\phi(a),
\]

and

\[
\mathbb E[(Y-f^\star)_+^2]
=
\left[(\mu-f^\star)^2+\sigma^2\right]\Phi(a)
+
(\mu-f^\star)\sigma\phi(a).
\]

Mixture moments are the weighted component moments.

Therefore we can compute exactly:

- \(\alpha(x)\);
- \(\chi^2(\Pi_x\Vert P_x)\);
- asymptotic ESS fraction;
- predicted MC relative variance.

The utility-tilted mixture weights are also explicit:

\[
\widetilde w_k(x)
\propto
w_k EI_k(x).
\]

This gives a direct visualization of a rare posterior component becoming dominant under the decision tilt.

## Required outputs

### Figure 1A — posterior versus decision worlds

At \(x_A\) and \(x_B\), show:

- predictive mixture;
- original mixture weights;
- utility-tilted mixture weights.

### Figure 1B — exact acquisition landscape

Plot:

- exact log EI;
- low-sample MC LogEI estimates for several frozen seeds;
- high-budget MC reference.

Show whether finite posterior samples create a false acquisition mode.

### Figure 1C — theorem verification

Across sample counts \(N\), compare:

- empirical relative variance;
- theoretical \(\chi^2/N\).

### Figure 1D — decision reliability

Across repeated MC estimates, show:

\[
P\{
\widehat\alpha(x_B)
>
\widehat\alpha(x_A)
\}
\]

as a function of \(N\).

Also include a positive smoothed utility version to verify that the effect is not an artifact of exact zero utility.

## Mechanism expectations

This experiment is considered successful only if:

1. mathematical identities agree with simulation to numerical tolerance;
2. \(x_B\) is truly better by a nontrivial margin;
3. low/moderate posterior MC frequently misranks \(x_A\) and \(x_B\);
4. the failure probability follows the predicted decision-shift/sample-size relation;
5. the phenomenon survives a smooth positive utility.

This experiment alone does **not** authorize a paper.

---

# 15. Experiment: `constrained_batch_shift`

## Purpose

Answer the most important pre-algorithm question:

> Do large posterior-to-decision shifts naturally occur near high-acquisition decisions in a recognizable batch/constrained BO model?

If not, stop before implementing a new energy sampler.

## Model system

Use the constrained Hartmann6 setting as the base optimization problem because it is a standard, recognizable batch BO model system.

Use:

- \(d=6\);
- batch \(q=4\) initially;
- the standard Hartmann objective;
- at least one constraint;
- a late-stage BO data state generated by a fully frozen protocol.

## Non-Gaussian surrogate from the beginning

Do **not** use a plain Gaussian posterior as the main model.

Use a Student-t-process-style posterior generated by integrating out an uncertain GP scale parameter under a conjugate inverse-gamma prior.

This gives a multivariate Student-t predictive distribution that is:

- genuinely non-Gaussian;
- heavy-tailed;
- density/energy evaluable;
- exactly sampleable by a Gaussian-scale-mixture representation;
- differentiable with respect to candidate locations;
- simple enough to obtain a very high-budget reference.

The purpose is not to claim that Student-t processes are the final target application.

The purpose is to isolate the acquisition-integration problem in a realistic non-Gaussian Bayesian surrogate without confounding it with an approximate posterior implementation.

## Data state

Generate one or more frozen late-stage states using a fixed procedure, e.g.:

1. fixed Sobol initialization;
2. a small number of ordinary BO steps;
3. freeze all observations and incumbent values;
4. fit the Student-t-process objective/constraint beliefs.

Do not choose states after inspecting decision-shift results.

## High-budget reference

On A100, use a very large Sobol/QMC sample count to approximate:

- reference qLogEI / constrained qLogEI values;
- reference acquisition gradients;
- \(\chi^2(\Pi_X\Vert P)\) from utility moments;
- ESS fraction.

The high-budget reference is diagnostic and can be expensive relative to the eventual method because it is run once.

## Candidate batch set

Evaluate:

- batches near the high-budget acquisition optimizer;
- batches from multiple optimizer restarts;
- Sobol/random batches spanning low and high acquisition values;
- local perturbations around top batches.

The critical analysis is the relationship between

\[
\alpha(X)
\]

and

\[
D_2(X)
=
\log(1+\chi^2(\Pi_X\Vert P)).
\]

## Sample-count study

For

\[
N\in\{64,128,256,512,1024,2048\},
\]

measure against the high-budget reference:

- relative acquisition-value error;
- acquisition ranking error;
- gradient relative error;
- gradient cosine similarity;
- optimizer outcome under matched multi-start L-BFGS;
- wall-clock.

Use scrambled Sobol/QMC and common random numbers wherever appropriate. Do not compare only against iid random MC.

## Primary GO condition

There must be a meaningful high-acquisition region where ordinary posterior sampling has low effective sample fraction.

A provisional gate is:

- at least 20% of candidate batches in the top decile of reference acquisition have population ESS fraction \(\le 0.05\) (\(\chi^2\ge19\)); and
- at 256–512 standard QMC samples, this causes a visible acquisition-ranking, gradient, or optimizer-decision error.

The numerical thresholds may be refined **before** the full experiment, but never after seeing final results.

## Strong NO-GO

Stop the paper direction if:

- high-acquisition batches generally have ESS fraction \(>0.2\);
- 512-sample qLogEI/qLogCEI is already highly reliable;
- large decision shift appears only in irrelevant low-acquisition regions;
- errors vanish under ordinary scrambled Sobol/QMC at negligible cost.

This is the most important early falsification.

---

# 16. CPU versus A100 strategy

The project should not manufacture a compute advantage by making the baseline inefficient.

## Local MacBook

Use CPU for:

- all unit tests;
- analytic mixture mechanism;
- small Student-t posterior checks;
- notebook development;
- reduced candidate sets;
- low-budget reference runs.

## A100

Use GPU only when large vectorized posterior sampling is useful:

- high-budget q/constrained acquisition reference;
- many candidate batches;
- gradient reliability study;
- later sampler comparisons.

The fact that brute-force sampling is cheap for the initial model system is acceptable.

At this stage we are testing the **statistical phenomenon**, not claiming wall-clock superiority.

A later paper claim must move to a posterior model where each world/sample is sufficiently costly that reduction in integration effort matters in practice.

---

# 17. Only if both mechanism experiments pass: `tilted_inference`

Do not implement this yet.

The next algorithmic question would be:

> Can we approximate \(\Pi_X\) cheaply enough that reduced integration variance exceeds proposal-adaptation overhead?

Candidate approaches, in preferred order:

1. a lightweight variational proposal in a fixed latent representation, with exact importance correction;
2. adaptive importance sampling;
3. annealed SMC / AIS along
   \[
   \pi_{\lambda,X}\propto P U_X^\lambda;
   \]
4. only then consider more advanced learned energy transport.

The first method that works is preferable. Algorithmic novelty is not obtained by choosing the fanciest EBM sampler.

Outer acquisition optimization remains L-BFGS / Adam.

---

# 18. Later flagship requirement

Even if the Student-t model system passes, the paper eventually needs one setting where posterior samples are materially more expensive than Gaussian GP draws.

Candidate later surrogates include:

- a nonconjugately conditioned GP / FlowGP-style belief;
- an energy-based function model;
- a Bayesian neural network or hierarchical Bayesian surrogate with expensive posterior sampling;
- another modern implicit/non-Gaussian surrogate with density or energy access.

This is not selected yet.

The first two experiments must establish the phenomenon before this engineering investment is made.

---

# 19. Paper claims tracker

## C1 — General MC variance identity

**Status:** mathematical claim to verify formally and with unit tests.

\[
RelVar(\widehat\alpha_N)
=
\chi^2(\Pi_X\Vert P)/N.
\]

## C2 — Proposal variance identity

**Status:** mathematical claim to verify formally and with unit tests.

\[
RelVar(\widehat\alpha_{Q,N})
=
\chi^2(\Pi_X\Vert Q_X)/N.
\]

## C3 — Free-energy / variational equivalence

**Status:** known mathematical machinery; supporting mechanism, not standalone novelty.

## C4 — High-value BO decisions can exhibit large posterior-to-decision shift

**Status:** untested. This is the key empirical mechanism claim.

Target evidence: `rare_mode_mechanism` + `constrained_batch_shift`.

## C5 — Decision-adapted integration materially reduces acquisition computation

**Status:** unauthorized until C4 passes.

## C6 — Reduced integration error changes real sequential BO performance for a complex non-Gaussian belief

**Status:** future flagship; unauthorized.

---

# 20. Intended paper structure if the idea survives

## 1. Introduction

Problem:

Monte Carlo acquisitions use posterior samples even when the expected utility is concentrated in a very different subset of posterior worlds.

Contrast:

- LogEI stabilizes arithmetic;
- decision tilting addresses sampling mismatch.

## 2. Posterior-to-decision shift

Define \(\Pi_X\).

Derive:

- variance identity;
- ESS identity;
- rare-event lower bound;
- proposal identity.

## 3. Acquisition as energy / free-energy inference

Derive:

- utility-tilted energy;
- partition-function ratio;
- annealed path;
- variational characterization;
- gradient identity.

## 4. Decision-adapted acquisition inference

Introduce the simplest method that survives empirical testing.

Do not introduce a sophisticated sampler unless necessary.

## 5. Experiments

1. exact non-Gaussian mechanism figure;
2. constrained batch non-Gaussian model system;
3. complex expensive-posterior flagship;
4. runtime/sample-complexity and failure-regime analysis.

## 6. Limitations

Explicitly state:

- same asymptotic acquisition/policy as standard MC;
- no expected gain when decision shift is small;
- proposal adaptation has overhead;
- explicit density/energy access may limit which posterior models can use some estimators;
- QMC can be extremely strong and must be the baseline.

---

# 21. Project stop conditions

Stop this paper direction if any of the following is established:

1. The core variance/divergence identities are wrong or irrelevant to the estimator actually used by qLogEI variants.
2. High posterior-to-decision shift does not occur near high-acquisition decisions in the constrained batch model system.
3. Scrambled Sobol/QMC with a few hundred samples is already reliable in the predicted hard regime.
4. Decision-adapted proposal construction costs more than brute-force posterior sampling.
5. Existing prior work already contains essentially the same BO-specific diagnosis and corrected estimator.
6. Gains remain a modest constant-factor sample reduction rather than a regime change.

Negative results are a valid outcome. Do not rescue the project by repeatedly redesigning synthetic cases.

---

# 22. Repository organization

Keep the repository paper-centric and small.

```text
README.md
AGENTS.md
PAPER_PLAN.md
STATUS.md

src/
    decision_tilt/

tests/

experiments/
    README.md
    rare_mode_mechanism/
    constrained_batch_shift/

notebooks/
    README.md
```

Do not recreate numbered task folders.

Every experiment folder should correspond to a prospective paper figure/result, not an exploratory branch.

---

# 23. Notebook standard

Notebooks are intended for human review, not merely as thin launchers.

Every substantive notebook should contain:

1. **Question**
2. **Why it matters to the paper**
3. **Mathematical object being tested**
4. **Frozen protocol**
5. **GO / NO-GO expectation**
6. **Readable code cells**
7. **Figures produced**
8. **Numerical result summary**
9. **Interpretation**
10. **Next authorized action**

Long reusable implementation still belongs in `src/`.

A notebook should be understandable without reading the entire codebase.

---

# 24. Immediate next action after repository reset

The reset itself should not implement experiments.

After the reset is reviewed, the next Codex run should implement only:

```text
rare_mode_mechanism
```

That run should be designed to produce a candidate Figure 1 and mathematical unit tests.

Only after human review of Figure 1 should `constrained_batch_shift` be implemented.
