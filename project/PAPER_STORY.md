# Paper Story

> **Role of this file:** source of truth for the paper narrative. Keep this BO-first and short. Mathematical details belong in `PROBLEM_FORMULATION.tex` and `THEORY.tex`; experimental bookkeeping belongs in `EXPERIMENTS.md`.
>
> **Status:** story LOCKED. The structural-influence blocker is closed; finite-sample end-to-end certification is the sole remaining technical blocker. Do not reopen the research direction unless that blocker invalidates the central claim.

## The BO problem

Many expensive scientific and engineering experiments come with a large bank of structured information about the unknown objective: local physics or conservation residuals, symmetry and shape relations, simulator-derived statements, or auxiliary expert comparisons. Conditioning a Gaussian-process or latent-Gaussian model on this information can materially change the BO acquisition, but nonlinear or non-Gaussian factors generally produce a non-Gaussian conditioned model whose acquisition must be reevaluated as new observations arrive.

At each BO round, both the data-conditioned reference model and the set of competitive experiments change. Incorporating every conditioning factor at every round can therefore make rich conditioning costly inside the sequential loop, while a subset fixed in advance may cease to contain the information relevant to the current decision.

**The paper therefore asks whether all of this information must be incorporated at every BO decision, or only the information capable of changing the next experiment.**

## Core question

> **How much of the available structured information must BO incorporate before its next action is already determined, to a prescribed tolerance, relative to the fully conditioned model?**

This is a decision problem rather than a posterior-reconstruction problem. BO ultimately needs to choose the next query, not reproduce every aspect of the fully conditioned posterior.

## Key insight

Suppose a subset of conditioning factors has been incorporated and produces a current BO leader \(\widehat x\). For an omitted factor to matter for the next action, it must be capable of changing the acquisition comparison between \(\widehat x\) and some challenger \(x\).

The exact perturbation of this pairwise acquisition gap can be written as a covariance along a path from the active conditioned model to the fully conditioned model. For a structured class of factors, this covariance can be bounded from inexpensive sensitivity and influence information without performing inference under every omitted factor.

This gives the operational principle:

> **Condition only until no unresolved factor can make another experiment materially better than the current choice.**

The fully conditioned acquisition remains the fixed reference; the active model is only an internal computational surrogate for selecting the current query. Approximation occurs in which **original conditioning factors** are instantiated, not by redefining the BO target or requiring a globally faithful sparse posterior.

## Proposed solution

Maintain an active subset of conditioning factors. At each BO iteration:

1. perform inference under the active conditioned model;
2. optimize the acquisition to obtain a candidate next experiment;
3. solve one optimistic global-challenger problem that combines the active acquisition gap with bounds for omitted-factor influence and inference/optimization error;
4. if no challenger can beat the candidate by more than \(\epsilon\), query it, observe the outcome, and update the BO model;
5. otherwise activate the factors that contribute most to the unresolved challenger and repeat the refinement.

At the next BO round, factor relevance is reassessed for the new acquisition landscape. Any inference method that supplies the active acquisition and the required error bound may be used; the inference backend is not a contribution of the paper.

## One-sentence thesis

> **In Bayesian optimization with richly structured conditioning, the next query can often be determined using only a decision-specific subset of the available information; for screenable factorized latent-Gaussian models, we can adaptively identify enough of that information to certify the action relative to the fully conditioned acquisition.**

## Intended contributions

1. **A BO formulation for decision-relevant structured conditioning.** We formulate the goal as selecting a query with at most \(\epsilon\) acquisition regret under the fully conditioned model while adaptively instantiating only the original conditioning factors needed for that decision.

2. **A certified adaptive BO procedure.** An exact acquisition-gap perturbation identity and structural influence bounds yield an optimistic worst-challenger test that combines omitted-factor, inference, and acquisition-optimization error in a single full-target action certificate.

3. **A decision-versus-conditioning complexity separation.** Under localized decision dependence and decaying structural influence, the number of factors required to determine the BO action can grow much more slowly than the total conditioning set; experiments test both this scaling and the resulting change in non-Gaussian inference difficulty.

These are paper-level contributions. Individual covariance inequalities, samplers, tempering schemes, or PDE constructions are supporting tools rather than separate contributions.

## Why the BO community should care

- **Make rich conditioning usable inside sequential BO.** The fully conditioned acquisition remains the reference, but each BO round resolves only the structured information needed to determine the current experiment.
- **Exploit changing relevance.** As observations and acquisition competitors change, a factor can matter at one BO round and not another; a subset fixed for the trajectory cannot capture this decision dependence.
- **Separate action difficulty from conditioning difficulty.** A fully conditioned posterior may be difficult to construct even when the next BO action is easy to certify, so screening can change the inference regime rather than merely avoid factor evaluations.

## Canonical motivating settings

The paper should use examples that make the BO problem tangible rather than presenting factorized energies as an abstract construction.

### Physics- or mechanism-informed experimentation

A scientific objective may be constrained by many local residual statements from a differential equation, conservation law, simulator, or mechanistic model. The full set can be large even though the next experiment is concentrated in one region of the design space.

### Preference- or expert-informed BO

Offline pairwise preferences or expert comparisons can create many non-Gaussian relational factors. Which preferences matter should change with the current competitive set of experiments. This is a particularly useful non-PDE example and connects directly to historical decision-theoretic GP sparsification.

### Symmetry, monotonicity, and shape information

Large collections of relational or derivative constraints can encode scientific structure. The symmetry example is especially useful because relevant factors can couple inputs that are far apart in Euclidean space, demonstrating that the method is not merely spatial truncation.

## What the paper is not about

- a generic method for approximating arbitrary non-Gaussian posteriors;
- a new PDE solver;
- a new SMC, HMC, importance-sampling, or diffusion-guidance algorithm;
- posterior coresets or factor subsampling whose objective is posterior fidelity;
- a generic relevance test for arbitrary black-box conditioning factors with no structural side information;
- proving that the full posterior is unnecessary for every Bayesian decision problem.

## Main-paper narrative

The main paper should be organized around the BO problem, not the theorem hierarchy.

1. **Motivation:** BO can increasingly exploit rich structured knowledge, but repeated full conditioning is expensive.
2. **Question:** how much of that knowledge is needed for the next experiment?
3. **Mechanism:** omitted information matters only through its possible effect on acquisition comparisons.
4. **Method:** active conditioning + one optimistic challenger + refinement until certification.
5. **Theory:** enough mathematics to justify the perturbation bound, certificate, and scaling implication; longer comparison-inequality and model-specific arguments go to the supplement.
6. **Evidence:** a nonlocal symmetry example that exposes the mechanism and certificate; a sequential preference-BO example showing that structured conditioning materially improves on data-only BO, adaptive conditioning retains that benefit, and the relevant factor set changes across rounds; and a nonlinear-PDE scaling example showing that the required factor count can remain small while full-target inference becomes harder.

Do **not** structure the manuscript as “Theorem 1, Theorem 2, Theorem 3, experiments.”

## Current empirical evidence

The following results already exist and should be treated as evidence to refine and validate, not rediscovered:

- **Reflection symmetry:** \(N=40\) factors; the structural loop certified with \(M=12\), or approximately \(M=15\) after a conservative Monte Carlo allowance for an \(\epsilon=0.03\) target; the held-out fully conditioned acquisition chose the same action.
- **Symmetry scaling:** in an earlier test with \(N=20,40,80,160\), approximately \(M=10\) factors were sufficient for the same structural threshold.
- **Linear PDE:** \(N=576\), \(M=50\); active-target GP-reference importance-sampling ESS was about \(85\%\), versus \(12\%\) for the full target; expanding-domain tests kept \(M\) roughly \(50\!\to\!60\) as \(N\) grew from 324 to 1600.
- **Nonlinear PDE:** \(N=576\), \(M=40\), with the same action as held-out full conditioning; across \(N=324,576,900,1296,1600\), \(M=40\) throughout, while full-target GP-reference IS ESS fell from about \(30\%\) to \(1.8\%\) and active-target ESS stayed near \(84\!\text{--}\!86\%\).

These are promising mechanism results, not yet the final experimental package.

## What belongs in the main theory

The main text likely needs only three ideas visibly:

1. the exact acquisition-gap perturbation identity;
2. a screenable omitted-factor bound leading to the full-action certificate;
3. a clean statement explaining when required conditioning can remain local/sublinear while the available factor set grows.

The block Poincaré/Brascamp--Lieb/Dobrushin machinery, inference-confidence construction, continuous-factor extension, and most model-specific proofs should live in the supplement unless one becomes essential to the central insight.

## Current blockers

The structural-influence blocker is closed for the reflection-symmetry and
nonlinear-PDE factor families. The sole remaining technical blocker is:

1. **End-to-end certification:** implement at least one finite-sample inference
   plus global-challenger certificate on a finite action set/grid, with errors
   bounded rigorously enough that the reported full-target BO certificate is
   genuinely valid rather than only an empirical diagnostic.

Everything else is a proof or execution task, not a reason to reopen the research direction.

## Framing discipline

- Keep **Bayesian optimization** in the foreground.
- Introduce “decision complexity” only after the BO problem is clear; do not lead with it as an abstract inference concept.
- Avoid treating computational savings as the sole value proposition. The stronger message is that **richer conditioning becomes practical inside a sequential BO loop because only decision-relevant information needs to be resolved at each step**.
- Do not optimize the title until the end-to-end experiment and scaling theorem are fixed.

**Working title: Decision-Relevant Conditioning for Bayesian Optimization.**
