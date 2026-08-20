# Related Work and Novelty Boundaries

> **Role of this file:** compact novelty map for paper positioning and reviewer checks. This is not a literature review. Keep only work that changes how we should state the contribution, design a baseline, or defend novelty.
>
> **Last checked:** 20 August 2026.

## Safe central distinction

The paper should **not** claim that decision-aware inference, factor subsampling, non-Gaussian GP conditioning, utility-aware approximation, or sequential Monte Carlo is new.

The intended contribution is narrower:

> **Select original structured-conditioning factors according to their possible effect on the next BO action, perform non-Gaussian inference only on the active conditioned target, and stop when an optimistic global-challenger bound certifies small acquisition regret relative to the fully conditioned model.**

The strongest novelty comes from the combination of:

1. **where approximation occurs:** before/during conditioning by omitting original factors;
2. **what is preserved:** the next BO action rather than the full posterior;
3. **how sufficiency is assessed:** a bound on full-target acquisition gaps against the worst unresolved challenger;
4. **what is demonstrated theoretically:** a regime where decision-relevant conditioning can stay small while the total factor set grows.

Do not claim novelty for any one of these ingredients in isolation without qualification.

---

# Tier 1 — papers that directly constrain the framing

## Approximation-Aware Bayesian Optimization (AABO)

**Reference:** Natalie Maus, Kyurae Kim, Geoff Pleiss, David Eriksson, John P. Cunningham, Jacob R. Gardner. *Approximation-Aware Bayesian Optimization*. NeurIPS 2024.

**What it establishes:** GP approximation in BO should be aligned with the downstream acquisition/decision rather than global posterior fidelity. AABO uses utility-calibrated variational inference to jointly improve a sparse GP approximation and BO data acquisition under limited model capacity.

**Why it is close:** it already makes the important conceptual move from posterior fidelity to BO decision quality. We cannot sell “BO only needs decision-relevant posterior accuracy” as the novelty.

**Boundary:** AABO changes the objective used to construct an approximate GP representation (e.g. an SVGP). The present project instead starts from a richly conditioned, generally non-Gaussian target and chooses which **original conditioning factors** must be instantiated to certify the next acquisition decision relative to the full conditioned model. The optimistic-challenger certificate and factor-activation problem are different objects.

**How it should appear in the paper:** early in the introduction. Use it to motivate a broader trend toward decision-aligned BO computation, then state that rich conditioning creates a distinct approximation axis: how much side information must be incorporated at a given BO step?

**Baseline implication:** include AABO only when an experiment also has an approximate GP representation for which AABO is technically meaningful. Do not force it into a factor-selection benchmark where the underlying GP is exact.

**Link:** https://proceedings.neurips.cc/paper_files/paper/2024/hash/257b9a6a0e3856735d0e624e38fb6803-Abstract-Conference.html

---

## Decision-Theoretic Sparsification for Gaussian Process Preference Learning

**Reference:** M. Ehsan Abbasnejad, Edwin V. Bonilla, Scott Sanner. *Decision-theoretic sparsification for Gaussian process preference learning*. ECML PKDD 2013. DOI: 10.1007/978-3-642-40991-2_33.

**What it establishes:** GP sparsification can be driven by the downstream loss rather than an information-only criterion. The Valuable Vector Machine selects useful users/items/inducing structure for preference-learning decisions and can recover several decision/information criteria through different losses.

**Why it is the closest historical conceptual threat:** it explicitly combines GP sparsification with downstream decision theory and even discusses UCB-style criteria.

**Boundary:** the project should not claim “decision-theoretic GP sparsification” as new. The intended method sparsifies the **original non-Gaussian conditioning factors before active-target inference**, repeatedly updates the factor set as the BO decision changes, and stops using a bound on acquisition regret under the **fully conditioned target**. The VVM is an approximation/sparsification method for GP preference learning; it does not provide the same full-target BO challenger certificate or the proposed decision-versus-conditioning scaling statement.

**How it should appear:** name explicitly in related work and likely in the preference-BO experiment discussion. Hiding it would create an avoidable reviewer problem.

**Baseline implication:** the planned preference experiment should include the closest practical VVM/decision-sparsification analogue or explain precisely why an exact implementation is not comparable.

**Link:** https://doi.org/10.1007/978-3-642-40991-2_33

---

## FlowGP — Conditioning Gaussian Processes on Almost Anything

**Reference:** Henry Moss, Lachlan Astfalck, Thomas Cowperthwaite, Colin Doumont, Sam Willis, Philipp Hennig, Christopher Nemeth, Andrew Zammit-Mangion. *Conditioning Gaussian Processes on Almost Anything*. arXiv:2605.21041, 2026.

**What it establishes:** GP predictive sampling can be recast through a diffusion/ODE representation with likelihood-dependent guidance, enabling conditioning on nonlinear physics and other nonconjugate information without deriving a bespoke inference scheme for every likelihood.

**Why it matters to this paper:** it makes the motivating regime more plausible and timely. If rich GP conditioning is becoming much more general, BO needs a principled way to decide how much of that information is worth resolving at each sequential decision.

**Boundary:** FlowGP is an inference method for the conditioned GP target. The present project is about deciding **which conditioning information must be included before the BO action is determined**. FlowGP is therefore a possible inference backend for the active target, not the competing algorithmic idea.

**Claim to avoid:** do not claim general GP conditioning, diffusion guidance, whitening, or likelihood-based guidance as contributions.

**Link:** https://arxiv.org/abs/2605.21041

---

## LatentFlow — A General Framework for Conditioning Stochastic Processes

**Reference:** Louis Sharrock, Lachlan Astfalck, Henry Moss. *LatentFlow: A General Framework for Conditioning Stochastic Processes*. arXiv:2607.12922, 2026.

**What it establishes:** stochastic-process conditioning can be reduced to latent-space inference through a guided probability flow, covering nonlinear/non-Gaussian/global constraints across a broad class of process models.

**Why it matters:** it broadens the enabling technology beyond GPs and reinforces the distinction between **how to sample a richly conditioned target** and **how much conditioning a BO decision needs**.

**Boundary:** LatentFlow targets the desired conditioned law; this project changes the target itself during the BO decision by activating only enough original factors to certify the action.

**Link:** https://arxiv.org/abs/2607.12922

---

# Tier 2 — conceptual ancestors and comparison classes

## Loss-calibrated approximate inference

**Reference:** Simon Lacoste-Julien, Ferenc Huszár, Zoubin Ghahramani. *Approximate inference for the loss-calibrated Bayesian*. AISTATS 2011.

**What it establishes:** approximate inference should account for the downstream decision loss rather than approximate the posterior independently of the task.

**Boundary:** this is a general decision-aware inference principle. The project must not claim the principle itself. Its narrower object is adaptive omission of structured conditioning factors plus a full-target BO action certificate.

**Link:** https://proceedings.mlr.press/v15/lacoste_julien11a.html

### Loss-calibrated BNN work

Later work such as Cobb, Roberts, and Gal (2018), *Loss-Calibrated Approximate Inference in Bayesian Neural Networks*, further develops task-specific variational objectives. It reinforces the same boundary: utility-aware posterior approximation is prior art.

**Link:** https://arxiv.org/abs/1805.03901

---

## Target-Aware Bayesian Inference (TABI)

**Reference:** Tom Rainforth, Adam Goliński, Frank Wood, Sheheryar Zaidi. *Target-Aware Bayesian Inference: How to Beat Optimal Conventional Estimators*. JMLR 21(88), 2020.

**What it establishes:** when a target expectation is known in advance, inference can directly target that expectation rather than first approximate the entire posterior; target-aware estimators can outperform conventional posterior-sampling pipelines.

**Boundary:** TABI targets efficient estimation of a specified posterior expectation. The current project targets a changing **BO argmax decision** and asks which original conditioning factors need to be included before any challenger can materially change that decision.

**Important distinction:** do not claim that directly targeting acquisition expectations rather than posterior fidelity is novel.

**Link:** https://www.jmlr.org/papers/v21/19-102.html

### Generalized thermodynamic integration for target-aware inference

F. Llorente, L. Martino, D. Delgado (2025), *Target-aware Bayesian inference via generalized thermodynamic integration*, develops another target-aware path-based estimator. This is relevant because our proof also uses an interpolation path, but the purpose is different: their path supports expectation estimation; ours is used to bound how omitted conditioning perturbs a BO acquisition gap.

**Link:** https://arxiv.org/abs/2502.02206

---

## Bayesian coresets

Representative references:

- Trevor Campbell, Tamara Broderick. *Bayesian Coreset Construction via Greedy Iterative Geodesic Ascent*. ICML 2018.
- Trevor Campbell, Tamara Broderick. *Automated Scalable Bayesian Inference via Hilbert Coresets*. JMLR 2019.
- Jacky Y. Zhang, Rajiv Khanna, Anastasios Kyrillidis, Oluwasanmi Koyejo. *Bayesian Coresets: Revisiting the Nonconvex Optimization Perspective*. AISTATS 2021.

**What they establish:** a small weighted subset of likelihood/data terms can approximate a posterior, with algorithms and guarantees tied to log-likelihood geometry or posterior approximation quality.

**Boundary:** coreset selection is ordinarily posterior-oriented. Our factor subset is allowed to produce a poor posterior away from the current BO decision as long as the omitted-factor influence is bounded tightly enough to certify the full-target action.

**Baseline implication:** include a posterior-oriented coreset/subsampling baseline in at least one factor experiment where the mapping is technically clean. Compare action quality at matched active factor count, not only posterior metrics.

**Links:**
- https://proceedings.mlr.press/v80/campbell18a.html
- https://www.jmlr.org/papers/v20/17-613.html
- https://arxiv.org/abs/2007.00715

---

## Firefly Monte Carlo and tall-data likelihood subsampling

**Reference:** Dougal Maclaurin, Ryan P. Adams. *Firefly Monte Carlo: Exact MCMC with Subsets of Data*. 2014.

**What it establishes:** auxiliary variables and per-factor lower bounds can allow MCMC updates that inspect only a subset of likelihood terms while retaining the full posterior as the exact invariant target.

**Boundary:** Firefly reduces per-iteration factor evaluation while still targeting the complete posterior. The present project intentionally uses a reduced conditioned target until a decision-level bound says the omitted factors cannot change the BO action materially.

**Claim to avoid:** do not imply that evaluating only some factors inside Bayesian inference is itself new.

**Link:** https://arxiv.org/abs/1403.5693

---

## Simulation Based Bayesian Optimization (SBBO)

**Reference:** Roi Naveiro, Becky Tang. *Simulation based Bayesian Optimization*. Statistics and Computing 35, 176 (2025); earlier arXiv:2401.10811.

**What it establishes:** BO acquisition optimization can work with sampling-based access to posterior predictive distributions and nonstandard surrogate models, including MCMC-based models in combinatorial spaces.

**Boundary:** SBBO changes how the acquisition is optimized/evaluated when only simulation access to the posterior is available. It does not determine which structured conditioning factors can be omitted while guaranteeing the full-conditioned BO action.

**Link:** https://arxiv.org/abs/2401.10811

---

# Tier 3 — enabling modeling and inference literature

## Constrained and virtual-point Gaussian processes

Shape-, monotonicity-, sign-, or inequality-constrained GP methods often introduce non-Gaussian virtual observations or derivative constraints and sometimes seek sparse virtual-point representations.

A canonical example is Riihimäki and Vehtari (AISTATS 2010), *Gaussian processes with monotonicity information*.

**Relevance:** these methods provide realistic sources of large factorized conditioning sets.

**Boundary:** their objective is to represent or enforce the constrained GP model. The present paper asks which of the available conditioning statements are needed for one BO decision and certifies against the full conditioned acquisition.

**Paper-writing implication:** discuss this work as a problem source, not primarily as a competing BO algorithm.

---

## Physics-informed GPs / probabilistic PDE conditioning

Representative reference: Marvin Pförtner, Ingo Steinwart, Philipp Hennig, Jonathan Wenger. *Physics-Informed Gaussian Process Regression Generalizes Linear PDE Solvers*. 2022/2023.

**Relevance:** physics residuals, operator observations, and discretized mechanistic information motivate large structured factor collections and the PDE experiments.

**Boundary:** the project is not a probabilistic PDE solver and should not claim improved physical inference. The PDE system is a model problem for BO with expensive rich conditioning.

**Link:** https://arxiv.org/abs/2212.12474

---

## Standard non-Gaussian inference machinery

### Elliptical slice sampling

Murray, Adams, and MacKay, *Elliptical slice sampling*, AISTATS 2010. Natural for latent-Gaussian models with non-Gaussian likelihood factors.

**Link:** https://proceedings.mlr.press/v9/murray10a.html

### HMC / NUTS

Standard gradient-based MCMC options for smooth active targets. Their invariant-target correctness does not automatically give a useful finite-run BO acquisition-gap confidence bound.

### Sequential Monte Carlo / Feynman--Kac

Del Moral, Doucet, and Jasra, *Sequential Monte Carlo Samplers*, JRSS B 2006, provides the standard machinery for moving through a sequence of targets. Adding decision-selected factors through reweighting/resampling/rejuvenation is therefore a natural implementation but not, by itself, sampler novelty.

**Link:** https://www.stats.ox.ac.uk/~doucet/delmoral_doucet_jasra_sequentialmontecarlosamplersJRSSB.pdf

### AIS and thermodynamic integration

Useful for bridging targets, normalizing constants, and diagnostics. The acquisition-gap interpolation identity should not be sold as a new annealing/thermodynamic-integration idea.

### FlowGP / LatentFlow

Potentially valuable active-target backends when standard importance sampling or MCMC struggles. Their numerical guidance/discretization errors should be treated empirically unless a rigorous bound is separately constructed.

---

# Reviewer-facing novelty table

| Prior-art family | Their approximation/target | Downstream objective | What remains different here |
|---|---|---|---|
| AABO / loss-calibrated VI | posterior representation / variational objective | decision utility / BO acquisition | select original conditioning factors and certify action against full conditioned target |
| TABI | estimator for a known expectation | target expectation | changing BO argmax; adaptive factor activation; global challenger certificate |
| VVM decision sparsification | sparse GP preference representation | preference decision loss | omit original non-Gaussian factors before inference; sequential BO; full-target action certificate |
| Bayesian coresets | weighted subset approximating full posterior/log likelihood | posterior fidelity | factor subset chosen for action sufficiency, not posterior fidelity |
| Firefly / tall-data MCMC | exact full posterior with subset factor evaluations per iteration | posterior sampling | deliberately reduced active target until decision is certified |
| FlowGP / LatentFlow | inference for rich conditioned process | conditioned-law sampling | decide how much conditioning is needed before the BO action is determined |
| SBBO | simulation-based acquisition optimization | BO action | does not screen structured conditioning factors |
| constrained/PIGP methods | constrained/physics-informed process model | inference / prediction | use them as sources of structured BO conditioning; certify decision with subset |

---

# Claims we can safely make if the theory/experiments pass

- Richly conditioned BO creates a distinct computational problem: repeated non-Gaussian inference may involve many structured factors at each sequential decision.
- Existing decision-aware inference work motivates prioritizing downstream utility, but does not by itself provide an adaptive original-factor certificate for the full conditioned BO action.
- For screenable factorized latent-Gaussian models, omitted-factor influence on acquisition gaps can be bounded without full-target inference.
- The resulting worst-challenger procedure can certify action regret under the fully conditioned acquisition when structural, inference, and optimization bounds are valid.
- Under explicit localized-influence assumptions, decision-relevant factor count can remain independent/sublinear in the total factor count; the experiments test whether this regime occurs in representative BO problems.

# Claims to avoid

- “First decision-aware Bayesian inference method.”
- “First task-aware/posterior approximation for BO.”
- “First method to subsample likelihood/factor terms.”
- “First way to condition GPs on non-Gaussian physics/preferences.”
- “New SMC/importance sampling/diffusion method” unless something genuinely new is later proved.
- “Posterior approximation is unnecessary” without the qualifier that the result concerns the next BO action under the stated screenability assumptions.
- “Arbitrary conditioning factors can be screened cheaply.”

# Literature work still worth doing later

Before final submission, run one focused update search for papers published after this file's date and citation chains around:

1. AABO and utility-aware BO inference;
2. decision-theoretic GP sparsification / preference learning;
3. Bayesian coresets with decision-aware objectives;
4. factor-subsampling MCMC with decision-specific stopping;
5. 2026 follow-ups to FlowGP and LatentFlow.

Do not continuously reopen the literature search during normal proof/experiment execution unless a new paper materially threatens the central distinction above.
