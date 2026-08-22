# Project Handoff

> **Read this first in a new ChatGPT Project chat.** This is the concise replacement for the earlier `PRO_HANDOFF_DEC_BO.md`. It records the current paper direction after the Pro audit without reproducing the full exploratory history.
>
> If files conflict, use this precedence:
> 1. `PAPER_STORY.md` for narrative and claims;
> 2. `PROBLEM_FORMULATION.tex` for notation/problem definition;
> 3. `THEORY.tex` for theorem/proof status;
> 4. `EXPERIMENTS.md` for empirical status;
> 5. `RELATED_WORK.md` for novelty boundaries.

## Canonical project files

| File | Role | Status |
|---|---|---|
| `PAPER_STORY.md` | Paper narrative and claims | LOCKED |
| `PROBLEM_FORMULATION.tex` | Mathematical formulation | ACTIVE |
| `THEORY.tex` | Theorems and proof status | ACTIVE |
| `EXPERIMENTS.md` | Experimental plan/results | ACTIVE |
| `RELATED_WORK.md` | Novelty boundaries | STABLE |

## Project objective

Develop one strong Bayesian optimization paper about **decision-relevant use of rich structured conditioning**.

**Working title:** *Decision-Relevant Conditioning for Bayesian Optimization*.

The project has moved beyond open-ended ideation. Do not generate unrelated directions unless a blocker shows that the present thesis is untenable.

## BO-first problem

At BO iteration \(t\), let the ordinary data-conditioned GP/reference law be

\[
P_{0,t}(df).
\]

Additional structured information is represented by a large factorized energy

\[
E_C(f)=\sum_{j=1}^N e_j(f),
\]

which produces the generally non-Gaussian fully conditioned law

\[
\pi_C(df)\propto e^{-E_C(f)}P_{0,t}(df).
\]

The factors can encode symmetry, monotonicity/shape, preferences, PDE or physical residuals, conservation laws, or related structured knowledge.

For BO utility \(u_x(f)\), define

\[
\alpha_C(x)=\mathbb E_{\pi_C}[u_x(f)],
\qquad
x_C^\star\in\arg\max_x\alpha_C(x).
\]

The computational problem is that evaluating and optimizing \(\alpha_C\) may require expensive non-Gaussian inference involving all \(N\) factors at every BO step.

### Central question

> **How much of the available structured conditioning must be incorporated before the next BO action is already determined, to tolerance \(\epsilon\), relative to the fully conditioned model?**

The desired output is an action \(\widehat x\) satisfying

\[
\alpha_C(x_C^\star)-\alpha_C(\widehat x)\le\epsilon
\]

without performing full conditioned inference when only a smaller, decision-specific subset of factors matters.

### Locked reference-model scope

Each decision-certification problem treats the supplied GP/reference
hyperparameters as fixed. Their upstream calibration is outside scope, although
structured factors may inform that calibration in a fully joint model. This is
not a remaining paper blocker or a new workstream.

## Current thesis

> **In Bayesian optimization with richly structured conditioning, the next query can often be determined using only a decision-specific subset of the available information; for screenable factorized latent-Gaussian models, we can adaptively identify enough of that information to certify the action relative to the fully conditioned acquisition.**

The paper should be framed as a BO capability/problem, not as generic posterior sparsification or a new inference algorithm.

## Core mechanism

Maintain an active factor set \(S\),

\[
\pi_S(df)\propto e^{-\sum_{j\in S}e_j(f)}P_{0,t}(df).
\]

For current action \(\widehat x\) and challenger \(x\), define the acquisition-gap observable

\[
F_{x,\widehat x}(f)=u_x(f)-u_{\widehat x}(f).
\]

Along the interpolation from \(\pi_S\) to \(\pi_C\), the exact gap perturbation is

\[
G_C(x,\widehat x)-G_S(x,\widehat x)
=
-\sum_{j\notin S}\int_0^1
\operatorname{Cov}_{\pi_{S,s}}(F_{x,\widehat x},e_j)\,ds.
\]

For a screenable factor class with covariance/influence bound

\[
|\operatorname{Cov}(g,h)|\le L(g)^\top C_S L(h),
\]

the omitted-factor effect is bounded by

\[
B_{\rm struct}(x,\widehat x)
=
L(F_{x,\widehat x})^\top C_S
\sum_{j\notin S}L(e_j).
\]

Inference supplies a modular action-gap error bound

\[
|\widehat G_S-G_S|\le B_{\rm infer},
\]

and the optimistic challenger is

\[
\Psi_S(x;\widehat x)
=
\widehat G_S(x,\widehat x)
+B_{\rm infer}(x,\widehat x)
+B_{\rm struct}(x,\widehat x).
\]

If a certified challenger optimizer gives

\[
\sup_x\Psi_S(x;\widehat x)
\le
\Psi_S(x_c;\widehat x)+\eta_{\rm opt}
\le\epsilon,
\]

then the returned action has full-conditioned acquisition regret at most \(\epsilon\) on the stated inference-confidence event.

If the test fails, activate factors contributing most to the structural bound for the worst challenger. If inference or optimization error dominates, refine that component instead of blindly adding factors.

## Frozen contribution structure

Use 2--3 introduction contributions, not a bullet for every theorem or sampler.

1. **Decision-relevant conditioning for BO:** formulate rich structured information as something that can be activated adaptively according to its effect on the next experiment rather than incorporated uniformly for posterior fidelity.
2. **Certified adaptive BO procedure:** combine an exact acquisition-gap perturbation identity, a structural influence bound, modular inference error, and one global optimistic challenger to certify the action under the fully conditioned acquisition.
3. **Decision/conditioning complexity separation:** under localized decisions and decaying influence, the number of factors needed for the next BO action can grow much more slowly than the total conditioning set; experiments test the scaling and its inference consequences.

Sampling is a modular component, not a contribution unless later evidence changes this conclusion.

## Theory status

### T1 — exact acquisition-gap transport

**Status: PROVED.**

Standard exponential-tilting calculus specialized to BO acquisition gaps. Applies to EI without requiring classical smoothness.

### T2 — screenable omitted-factor bound

**Abstract status: PROVED.**

If a valid covariance/influence operator \(C_S\) exists uniformly along the active-to-full path, the structural bound follows directly from T1.

**Reflection-symmetry analytic status: PROVED.**

For the block-local `logcosh` reflection factors, Menz's Theorem 2.3 gives one
uniform conservative operator \(C_S\equiv A^{-1}\), with
\(A_{ii}=\lambda_{\min}(Q_{ii})\) and
\(A_{ij}=-\lVert Q_{ij}\rVert_{\mathrm{op}}\). The archived OU specialization
has an analytic positive row-dominance margin
\((1-e^{-\Delta x/\ell})/(1+e^{-\Delta x/\ell})\). See
`T2B_SYMMETRY_AUDIT.md` and `THEORY.tex`.

**Reflection-symmetry EI numerical status: PASSED (narrow validation).**

The prospective EI screen stopped with 15 of 40 factors active and left 62.5%
unevaluated before held-out validation. The structural envelope was
\(0.005084<0.01\), and eight fresh empirical full-target replicates had maximum
observed EI regret \(3.78\times10^{-4}\). This establishes useful non-vacuity,
not a rigorous end-to-end inference or continuous-action certificate.

**Nonlinear-PDE analytic status: PROVED.**

For the archived sine-residual factor family, the exact two-term factor Hessian
has worst-case negative curvature \(\gamma\eta/\tau\). Complete overlap
accounting gives a uniformly positive comparison matrix for every active set and
interpolation parameter. The rigorous matrix equals the archived notebook
matrix to floating-point roundoff, so its structural term remains
\(0.0387440330\) and the rigorous correction factor is one. See
`T2B_NONLINEAR_PDE_AUDIT.md` and `THEORY.tex`.

> **The structural-influence blocker is closed for the main factor families.**

### T3 — full-target action certificate

**Status: PROVED.**

The full regret is bounded by the optimized challenger envelope when the structural, inference, and acquisition-optimization bounds are valid.

### T4 — decision/conditioning separation

**Abstract status: PROVED under explicit assumptions.**

For a localized decision region, bounded factor density, polynomial graph growth, and exponentially decaying influence, activating a radius-\(r\) neighborhood gives exponentially small omitted influence; hence the required factor count can depend polylogarithmically on \(1/\epsilon\) and not on total domain size. A separate product construction shows that posterior-faithful original-factor subsets can require \(\Omega(N)\) factors.

**Nonlinear-PDE concrete status: PROVED FOR THIS FAMILY.**

The expanding-domain comparison matrices are uniformly SPD, fixed-range sparse,
and uniformly conditioned. Standard sparse-matrix inverse decay therefore
supplies the domain-independent exponential influence required by T4A. This is
a family-level mapping; it does not prove that the adaptive experiment must
choose exactly \(M=40\).

### Inference certification

**Status: PROVED; PROSPECTIVE FINITE-GRID PILOT PASSED; the end-to-end blocker
is closed for this exact reflection-symmetry instantiation.**

The canonical flagship instantiation is now fixed: exact Gaussian-reference
rejection sampling for the convex active symmetry target, exact i.i.d. accepted
samples, a Rao--Blackwellized EI-gap sample mean, and the whitened
strong-log-concavity/log-Sobolev radius with a two-sided union bound across the
finite action grid and adaptive rounds.  Every certification batch is drawn
only after its active set and leader are fixed and is expended after use.  See
`THEORY.tex` and `INFERENCE_CERTIFICATION_IMPLEMENTATION_HANDOFF.md`.

The locked pilot was run once from preregistration commit
`5da6fec6c0a645ba56f555062a3adb4139a1782d` and satisfied every predeclared
condition.  It stopped with 15 of 40 factors active,
\(B_{\rm infer}=0.0038485111\), \(B_{\rm struct}=0.0057717865\), and
\(U_{\rm cert}=0.0093103112\le0.01\).  ESS or ordinary Monte Carlo error bars
remain diagnostics, not substitutes for this certificate.  The guarantee is
limited to the exhaustive 401-action reflection-symmetry grid with exact
rejection samples and does not automatically transfer to HMC, SMC, FlowGP,
importance sampling, or another inference backend.

## Canonical algorithm

At each refinement round:

1. infer under the active target;
2. optimize the active acquisition to obtain a leader;
3. aggregate omitted structural influence;
4. solve one optimistic global challenger problem;
5. stop if the total upper bound is \(\le\epsilon\);
6. otherwise determine whether structural, inference, or optimization error dominates;
7. if structural error dominates, activate the factors contributing most to the worst challenger and repeat.

The algorithm does **not** require all pairwise action comparisons. The leader itself need not have a certified global optimization error; an inadequate leader simply cannot pass the global challenger test.

## Existing evidence

### Reflection symmetry

- \(N=40\);
- structural loop certified with \(M=12\);
- approximately \(M=15\) with existing conservative Monte Carlo allowance for \(\epsilon=0.03\);
- held-out full conditioning selected the same action;
- earlier \(N=20,40,80,160\) scaling used approximately \(M=10\) factors at the same structural threshold.

Notebook: `DEC_Symmetry_Continuous_BO_Demo.ipynb`.

### Linear PDE

- \(N=576\), \(M=50\);
- active GP-reference IS ESS \(85.0\%\), full-target ESS \(12.0\%\);
- expanding-domain \(N=324\to1600\): \(M\approx50\to60\), active ESS \(\sim81\!\text{--}\!86\%\), full ESS \(34.1\%\to0.78\%\).

Notebook: `DEC_PDE_Certified_BO_Demo.ipynb`.

### Nonlinear PDE

- \(N=576\), \(M=40\);
- active GP-reference IS ESS \(84.3\%\), full-target ESS \(6.2\%\);
- same action as held-out full conditioning;
- expanding-domain \(N=324,576,900,1296,1600\): \(M=40\) throughout;
- active ESS stayed approximately \(84\!\text{--}\!86\%\); full ESS fell from \(30.0\%\) to \(1.8\%\).

Notebook: `DEC_Nonlinear_PDE_BO_Demo.ipynb`.

These are mechanism results, not yet a finished NeurIPS experimental package.

## Frozen experimental direction

### Main essential

1. **Symmetry mechanism/certificate:** polish existing evidence with repeated coverage and random/local/static baselines.
2. **Preference-conditioned sequential BO:** add one end-to-end non-PDE example in which the relevant factor set changes with the BO state.
3. **Nonlinear PDE scaling:** strengthen existing expanding-domain evidence across source fields/BO states and compare against fixed local/random/static/posterior-oriented factor selection.

### Supplement

- linear PDE control;
- influence-bound derivations and tightness;
- sampler comparisons;
- factor-selection ablations;
- continuous-action certification details;
- extra dimensions/couplings/seeds;
- continuous-factor/quadrature extension only if used.

### Do not add automatically

Generic Hartmann/Ackley/Rosenbrock benchmark suites, batch/multiobjective variants, molecule/image case studies, or a new SMC method are not required for the central claim.

## Novelty boundaries

The project must explicitly distinguish itself from:

- **Approximation-Aware Bayesian Optimization:** utility-aware approximation is prior art; our approximation axis is the original rich-conditioning factor set plus a full-target action certificate.
- **Decision-theoretic GP sparsification / VVM:** downstream-loss-aware GP sparsification is prior art; our method omits original non-Gaussian conditioning factors before inference and adaptively certifies a sequential BO decision.
- **Target-aware/loss-calibrated inference:** directly targeting an expectation or decision is prior art.
- **Bayesian coresets / Firefly:** factor/data subsampling for posterior approximation or exact full-posterior MCMC is prior art.
- **FlowGP / LatentFlow:** general rich conditioning is complementary inference technology, not our novelty.
- **SBBO:** simulation-based acquisition evaluation/optimization is distinct from conditioning-factor selection.
- **PIGP/constrained GP:** sources of structured conditioning; this project is not a new PDE or constraint-inference method.
- **AIS/SMC/HMC/ESS/diffusion guidance:** standard inference machinery; keep modular.

See `RELATED_WORK.md` for the concise paper-by-paper map.

## Main-paper philosophy

The main manuscript should lead with the BO problem and capability:

1. rich structured information is increasingly usable in GP/latent-Gaussian BO;
2. full conditioning can be expensive inside every sequential decision;
3. BO only needs enough information to settle the next experiment;
4. acquisition-gap influence gives a principled way to determine sufficiency;
5. certification and scaling theory support this idea;
6. experiments show the mechanism, sequential adaptation, and scaling consequence.

Do not let the manuscript become a paper about covariance inequalities with BO as an application. Long influence-operator proofs, inference confidence constructions, and continuous-factor extensions belong primarily in the supplement.

## Immediate execution gates

### CLOSED — Gp2 rejected as E3

The preregistered target-blind Gp2 proxy structural preflight at commit
`1e81cd1c0f2ffe3c8d5347fdd1dd0007c2a5ff96` terminated with
`PREPROCESSING_INVALID`: the connected 279-action library supplied only 47
candidate-local proxy factors (16.85% coverage), below the frozen minimums of
75 factors and 40% coverage.  No theory matrices, structural-sparsity
distribution, inference, or BO were run.  Gp2 is abandoned as E3; its earlier
pairwise-preference P1 negative result remains unchanged.

### CLOSED — structural influence validity

Reflection symmetry and the nonlinear-PDE family now have proved concrete
comparison constructions. The nonlinear-PDE structural replay is unchanged at
\(0.03874403301354687\).

### CLOSED — rigorous finite-grid end-to-end certificate

The committed prospective reflection-symmetry finite-grid pilot passed without
post-hoc changes.  The finite-sample end-to-end empirical blocker is closed for
that exact instantiation; the next work is the broader frozen empirical program.

**Freeze the technical idea and execute the experimental program.** Further
mathematical generalization is optional unless required by a reviewer-facing
gap.

## Compute constraints

Development:

- MacBook Air;
- 16 GB RAM.

Cloud:

- Google Colab Pro;
- single A100 available manually.

Use cheap/local falsification first. GPU experiments should be manually runnable, print progress, and save structured results for GitHub/Codex workflows. Avoid plans requiring multi-GPU training.

## Instructions for future ChatGPT/Pro chats

- Do not brainstorm unrelated research directions unless a blocker invalidates the core thesis.
- Keep framing Bayesian-optimization-first.
- Treat `PAPER_STORY.md` as the narrative source of truth.
- Distinguish **established**, **proved under assumptions**, **empirically observed**, and **proposed** claims.
- When working on theory, modify only unresolved proof tasks rather than redesigning proved results for elegance.
- When proposing an experiment, state which paper claim it tests and the failure condition.
- When interpreting new results, update `EXPERIMENTS.md` before changing the paper story.
- Do not claim sampler novelty for standard reweighting, tempering, rejuvenation, MCMC, or diffusion guidance.
