# Project Handoff

> **Read this first in a new Codex/ChatGPT session.** This file states the
> locked thesis, current theory and evidence, closed failures, and next task.
> It intentionally does not repeat derivations from `THEORY.tex` or run logs
> from `EXPERIMENTS.md` and committed output directories.

## Canonical sources and reading order

| Order | File | Role | Status |
|---:|---|---|---|
| 1 | [`PROJECT_HANDOFF.md`](PROJECT_HANDOFF.md) | Session entry point | CURRENT |
| 2 | [`PAPER_STORY.md`](PAPER_STORY.md) | Narrative and claims | LOCKED |
| 3 | [`PROBLEM_FORMULATION.tex`](PROBLEM_FORMULATION.tex) | Notation and problem definition | ACTIVE |
| 4 | [`THEORY.tex`](THEORY.tex) | Theorems, proofs, and proof status | ACTIVE |
| 5 | [`EXPERIMENTS.md`](EXPERIMENTS.md) | Experiment registry and execution plan | ACTIVE |
| 6 | [`RELATED_WORK.md`](RELATED_WORK.md) | Novelty boundaries | STABLE |

Completed audits live in [`reference/`](reference/); closed implementation and
gate handoffs live in [`archive/`](archive/). They are linked where relevant
but are not default required reading.

## Locked thesis and scope

**Working title:** *Decision-Relevant Conditioning for Bayesian Optimization*.

> In Bayesian optimization with richly structured conditioning, the next query
> can often be determined using only a decision-specific subset of the
> available information; for screenable factorized latent-Gaussian models, we
> can adaptively identify enough of that information to certify the action
> relative to the fully conditioned acquisition.

The paper is about a BO capability, not generic posterior sparsification or a
new sampler. At BO iteration \(t\), the fixed-hyperparameter reference law is
\(P_{0,t}\), the full conditioned law is

\[
\pi_C(df) \propto \exp\{-\textstyle\sum_{j=1}^N e_j(f)\}P_{0,t}(df),
\]

and the target is an action whose acquisition regret relative to the fully
conditioned action is at most \(\epsilon\), without full conditioned inference
when fewer factors suffice. Upstream reference-model calibration is outside
the decision-certification scope. Sampling remains a modular implementation
component; standard IS, SMC, HMC, ESS, and diffusion methods are not claimed as
novel contributions.

The contribution structure remains frozen:

1. decision-relevant conditioning for BO;
2. a certified adaptive action-selection procedure;
3. separation between decision complexity and total conditioning complexity.

## Theory status

| Result | Status | Current scope |
|---|---|---|
| T1, exact acquisition-gap transport | `PROVED` | Standard exponential-tilting identity specialized to BO gaps; includes EI. |
| T2, screenable omitted-factor bound | `PROVED` abstractly | Valid when a uniform covariance/influence operator exists along the active-to-full path. |
| T2, reflection symmetry | `PROVED` | The block-local `logcosh` family has the conservative Menz comparison operator \(C_S=A^{-1}\); the OU specialization has a positive analytic row-dominance margin. |
| T2, nonlinear PDE | `PROVED` | Complete overlap accounting gives a uniformly positive comparison matrix; the clean matrix matches the archived construction to roundoff. |
| T3, full-target action certificate | `PROVED` | Combines structural, inference, and acquisition-optimization bounds; the finite-grid pilot below instantiates it with zero grid optimization remainder. |
| T4, decision/conditioning separation | `PROVED UNDER EXPLICIT ASSUMPTIONS` | Localized decisions plus bounded factor density, graph growth, and inverse-decay assumptions yield sublinear/polylogarithmic active-factor scaling. |
| T4, nonlinear-PDE family mapping | `PROVED FOR THIS FAMILY` | Uniform SPD, sparsity, and conditioning supply the required domain-independent inverse decay; this does not force the adaptive experiment to select exactly \(M=40\). |

Technical records: [reflection-symmetry audit](reference/T2B_SYMMETRY_AUDIT.md),
[nonlinear-PDE audit](reference/T2B_NONLINEAR_PDE_AUDIT.md), and the completed
[finite-grid implementation contract](archive/INFERENCE_CERTIFICATION_IMPLEMENTATION_HANDOFF.md).

The structural-influence blocker is closed for the main factor families. The
finite-sample end-to-end blocker is closed only for the exact finite-grid
reflection-symmetry instantiation described below.

## Current empirical evidence

### E1 — reflection symmetry

- The narrow EI structural validation passed with 15/40 factors active,
  structural envelope \(0.00508439<0.01\), and maximum observed held-out EI
  regret \(3.78\times10^{-4}\) across eight fresh replicates.
- The prospective finite-sample pilot was run once from preregistration commit
  `5da6fec6c0a645ba56f555062a3adb4139a1782d` and passed every frozen condition.
  It certified at 15/40 active factors with
  \(B_{\rm infer}=0.003848511075377576\),
  \(B_{\rm struct}=0.005771786534479693\), and
  \(U_{\rm cert}=0.00931031122157721\le0.01\). The exact-rejection acceptance
  rate was `0.3342708888888889`, using 5,375,000 Gaussian proposals.
- The guarantee is limited to the exhaustive 401-action grid and exact
  rejection samples. It is not a continuous-action certificate and does not
  transfer automatically to another inference backend.

Details: [`experiments/symmetry/`](../experiments/symmetry/) and its immutable
[`RESULTS.md`](../experiments/symmetry/outputs/inference_certification_pilot/RESULTS.md).

### E2 — nonlinear PDE

- The clean T2-B structural validation passed at \(M/N=40/576\). The
  theorem-backed structural value is `0.03874403301354687`; the clean and
  archived comparison matrices differ by at most \(1.11\times10^{-16}\).
- The associated inference allowance `0.0235285003` and total envelope
  `0.0495224797` are empirical prototype diagnostics, not a rigorous
  finite-sample certificate.
- Historical expanding-domain evidence reports \(M=40\) for
  \(N=324,576,900,1296,1600\), active GP-reference IS ESS near 84–86%, and
  full-target ESS declining from about 30.0% to 1.8%. This remains prototype
  evidence pending clean repeated reproduction and baselines.

Details: [`experiments/nonlinear_pde/`](../experiments/nonlinear_pde/) and its
immutable [`RESULTS.md`](../experiments/nonlinear_pde/outputs/t2b_structural_validation/RESULTS.md).

### E3 — Sun oxide source gate and current-NLR benchmark

- Source recovery closed with terminal verdict `JOIN_AMBIGUOUS`; this is a
  historical data-provenance result, not evidence for the paper method, and it
  remains unchanged.
- Current authoritative queries reproduce 2,142 unique PBE compositions and
  194 unique GW compositions. The separately specified
  `CURRENT_NLR_PBE_GW_V1` benchmark uses authoritative `wave`-parent energy and
  family provenance rather than claiming an exact historical polymorph match.
- The frozen benchmark contains 2,142 deterministic PBE/FERE legacy rows and
  191 strict GW actions. The initial GW split is 166 single-row and 28
  multi-polymorph compositions; all 28 multi-polymorph selections resolve with
  zero exact energy ties. Strict mapping excludes exactly CdO, Ga2O3, and
  Sb2O3, and every retained action maps to one legacy composition key.
- Terminal benchmark verdict: `PASS_CURRENT_NLR_BENCHMARK`. GW magnitudes are
  confined to the isolated two-column oracle and were not used for selection
  or mapping. No descriptors, graph, factors, theory calculation, inference,
  or BO were produced.

Details: [`experiments/sun_oxide/`](../experiments/sun_oxide/) and its immutable
[`SOURCE_AUDIT.md`](../experiments/sun_oxide/outputs/source_recovery/SOURCE_AUDIT.md),
plus the current-NLR
[`benchmark_manifest.json`](../experiments/sun_oxide/benchmark/benchmark_manifest.json).

### E4 — linear PDE supplementary control

Historical prototype evidence reports \(N=576\), \(M=50\), active/full
GP-reference IS ESS of 85.0%/12.0%, and expanding-domain active counts near
50–60 as \(N\) grows from 324 to 1600. It remains a supplementary control, not
a main-paper result.

The full current registry, prospective criteria, and execution plan are in
[`EXPERIMENTS.md`](EXPERIMENTS.md); the directory-level index is
[`experiments/README.md`](../experiments/README.md).

## Closed negative cases

The scientific record is preserved; none of these runs should be tuned or
quietly reinterpreted.

- **Minimal synthetic preference pilot — `FAIL-P2`.** P1 and P2 performance
  passed, but median factor use was `0.875`, above the preregistered `0.65`
  sparsity threshold. Starting commit:
  `5f7f7cfa66f4d6ec33ebf5a5feb3d8f77ba2110a`.
- **Redundant-bank synthetic preference pilot — `FAIL-P2`.** P1 and P2
  performance passed, but median factor use was `0.8888888889`, above its
  independently frozen `0.80` threshold. Starting commit:
  `b68525952f58693461ac32d4658a8d611a594706`.
- **Gp2 P1 gate — `PREPROCESSING_INVALID`.** Only 169 candidates survived,
  below the frozen minimum of 250; no P1 scientific comparison ran.
- **Gp2 target-blind proxy structural preflight —
  `PREPROCESSING_INVALID / GP2_ABANDONED`.** From preregistration commit
  `1e81cd1c0f2ffe3c8d5347fdd1dd0007c2a5ff96`, only 47 of 279 actions supplied
  candidate-local proxy factors (`0.16845878136200718` coverage), below both
  frozen minimums of 75 factors and 40% coverage. No theory matrices,
  structural-sparsity distribution, inference, or BO were run.

Gp2 is recorded only as a closed negative case and is abandoned as E3. All
code, configs, handoffs, failed/intermediate attempts, and committed outputs
remain in place; see [`experiments/README.md`](../experiments/README.md) and
[`archive/e3/`](archive/e3/).

## Next work

Immediate scientific task after this hygiene commit: upgrade E1 into the
reproducible repeated mechanism/coverage experiment with its already specified
random, Euclidean-local, and static baselines. Do not substitute a broad generic
benchmark suite.

The exact historical Sun et al. reproduction remains closed as
`JOIN_AMBIGUOUS`. The distinct `CURRENT_NLR_PBE_GW_V1` benchmark is frozen, but
no downstream descriptor, graph, factor, inference, or BO gate has begun; any
such gate requires a separate prospective decision. E2 robustness and the
smallest necessary baselines remain later work; broad supplementary
sampler/ablation work comes last.

## Operating rules

- Keep the framing Bayesian-optimization-first and do not reopen unrelated
  research directions without a blocker.
- Distinguish proved theory, empirical evidence, diagnostics, failures, and
  planned work.
- Map every new experiment to a claim and prospective failure criterion in
  `EXPERIMENTS.md` before running it.
- Preserve prototype notebooks and committed experiment outputs as immutable
  provenance.
- Use CPU smoke tests locally; reserve the single Colab A100 for larger runs.
- Do not silently change behavior that conflicts with `PROBLEM_FORMULATION.tex`
  or `THEORY.tex`.
