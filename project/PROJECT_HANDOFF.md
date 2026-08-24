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
- Replacement stress-test preregistration
  `0dcb76f5f2d053e098b472ac9984182b837295b5` remains unexecuted; no prospective
  E2 source seed has been accessed. A development-only FULL-shadow diagnostic
  using seed `2026082401` found that n=24 passes the unchanged shadow rule after
  the frozen 16,384 escalation, but all three n=40 checkpoints fail its ESS-
  fraction requirement at both 8,192 and 16,384. Increasing to 32,768 raises
  absolute ESS but does not repair relative proposal/target overlap. One early
  16,384 pair also has material cross-batch acquisition regret `0.024442`.
  Recommendation: improve and development-validate the FULL-reference backend
  before prospective execution; do not merely increase samples or replace the
  reliability rule from these development cases. Exact record:
  [`RESULTS.md`](../experiments/nonlinear_pde/outputs/full_shadow_reliability_diagnostic/RESULTS.md).

Details: [`experiments/nonlinear_pde/`](../experiments/nonlinear_pde/) and its
immutable [`RESULTS.md`](../experiments/nonlinear_pde/outputs/t2b_structural_validation/RESULTS.md).

### E3 — Sun oxide source, benchmark, reference graph, PBE model, and BO value

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
  or mapping.
- The separately preregistered Magpie/reference-graph gate passed as
  `PASS_DESCRIPTOR_GRAPH`: the finite raw descriptor matrix is `(2142, 132)`;
  the deterministic 14,072-edge 10-NN-plus-MST graph is connected; sparse `Q0`
  passed its sign, symmetry, eigenvalue, positive-definiteness, and three-solve
  checks; and all 191 actions map uniquely. No GW value was read and no
  benchmark PBE value entered descriptor or graph construction. No preference
  factors, theory calculation, inference, or BO were produced.
- The frozen `ADJACENT_STRICT_PBE_ORDER_V1` bank passed the model-specific
  existing-Menz-theory gate as `PASS_PBE_FACTOR_THEORY`: 1,681 strict adjacent
  factors remain after 460 exact-tie omissions; the path-subgraph adjacency has
  maximum degree two and norm `1.988275914308721`; and
  `A0=Q0-0.25R` has analytic eigenvalue floor `0.5` and numerical minimum
  `0.5896249844278044`. Sparse solves and all 18,145 target-blind action-pair
  diagnostics completed without a dense inverse. No GW value, inference, or
  BO entered this gate, and `THEORY.tex` was unchanged. This remains the valid
  sparse-compression baseline.
- The target-blind normalized replacement model passed as
  `PASS_NORMALIZED_PBE_MODEL`. `PBE_SUPPORT_500_V1` contains every one of the
  191 actions plus 309 deterministic descriptor-space farthest-point nodes;
  `NORMALIZED_ALL_PAIRS_PBE_500_V1` contains 124,718 strict PBE pairs at weight
  `1/499`, omitting 32 exact-tie pairs. Its weighted adjacency has maximum row
  sum `0.9999999999999997` and norm `0.9997442203602465`; the existing Menz
  matrix has analytic eigenvalue floor `0.75` and numerical minimum
  `0.9003424446737381`. PBE-only MAP and 256-pair influence diagnostics
  completed without reading GW values, inference, BO, a dense inverse, or a
  theory-ledger edit. This is the proposed E3 full-conditioning model.
- The preregistered 12-seed GW BO value pilot passed as `PASS_PBE_VALUE` from
  run SHA `44f58f100f41247afe0937e42eebe58055104225` with frozen config SHA-256
  `6cc47d41dfbdbf88187d535d405ca6afd971e4b07f91932d55dbbbf5c101ef0f`.
  Median 12-query AURC was 19.4445 eV for `NO_PBE` and 1.0170 eV for
  `FULL_PBE`; median final regret was 0.8310 and 0.0000 eV. `FULL_PBE` won
  strictly on 10/12 paired seeds, tied the two seeds whose shared initial set
  already contained the optimum, and never lost; it found the optimum in
  10/12 seeds versus 4/12. All three frozen Laplace-proposal SNIS checks
  passed with ESS fractions about 0.905--0.907, and an independent audit found
  no oracle leakage. This establishes the narrow full-PBE value claim on the
  frozen E3 benchmark, not adaptive speedup or cross-dataset generalization.
- The original adaptive engineering smoke ran locally on already-consumed
  seeds 0--2 from implementation SHA
  `7fbfb202268dd0fd92d35defbea2cc4990f089e2`; it is not scientific evidence.
  Its cumulative cross-iteration factor lifecycle made the first fallback
  permanent. Commit `70a9686b143c09f9f970306cc4489a2ce2b6e173`
  preregistered a fresh validation, but it was never scientifically executed
  and is now superseded. No seed 12--31 oracle value was accessed.
- A lifecycle-only follow-up starts each BO decision from an empty factor mask,
  retains cumulative activation within that decision, and warm-starts from the
  preceding adaptive MAP. Its seeds 0--2 smoke remained mechanically correct:
  optimized FULL reproduced 36/36 prior FULL actions, every adaptive decision
  terminated by explicit full fallback, and shadow FULL agreed 36/36 with zero
  EI regret. All 36 decisions exhausted eight stages and ended with all 124,718
  factors. Decisions 2--12 had a median paired conditioning-time ratio of about
  4.00, about 3.64 times FULL energy-gradient work, and about 8.40 times FULL
  Hessian work. This is `ADAPTIVE_ENGINEERING_PATHOLOGICAL`; the fresh adaptive
  efficiency validation is blocked and the fresh seeds remain unspent.
- A development-only full-bank scaling probe then compared normalized all-pairs
  supports of 500, 1,000, and all 2,142 legacy materials at three already-used
  seed-0 FULL states. The full archive stayed SPD (`lambda_min(A0)=0.7520475515`)
  and retained strong PBE rank signal, but median pre-fallback active fraction
  increased from `0.931662` to `0.949890` to `0.972771`; all nine states still
  exhausted eight stages and full-fallbacked. Median active counts scaled from
  `116195` to `2230019` while total factors scaled from `124718` to `2292440`.
  This is the development classification `FULL_ARCHIVE_NOT_HELPFUL`, not new
  scientific evidence or a reinterpretation of either immutable smoke.
- The final development-only decision-sparsity diagnostic evaluated the frozen
  500-support model on the same three already-consumed seed-0 FULL states. Its
  primary reranked influence path first stabilized on the FULL action at active
  fractions `0.70`, `0.20`, and `0.10`, whereas the theorem certificate required
  fraction `1.00` in every state. The prospective terminal classification is
  `MIXED_DECISION_SPARSITY`: certificate conservatism is clear, especially after
  observations, but empirical exact-action sparsity is not uniformly strong
  enough to justify a major new theory effort. No fresh seed was accessed and
  no scientific preregistration was created.

Details: [`experiments/sun_oxide/`](../experiments/sun_oxide/) and its immutable
[`SOURCE_AUDIT.md`](../experiments/sun_oxide/outputs/source_recovery/SOURCE_AUDIT.md),
plus the current-NLR
[`benchmark_manifest.json`](../experiments/sun_oxide/benchmark/benchmark_manifest.json)
and independently checked descriptor/graph
[`VERIFICATION.md`](../experiments/sun_oxide/outputs/descriptor_graph/VERIFICATION.md),
plus the frozen factor/theory
[`RESULTS.md`](../experiments/sun_oxide/outputs/pbe_factor_theory/RESULTS.md),
and the normalized-model
[`RESULTS.md`](../experiments/sun_oxide/outputs/normalized_pbe_model/RESULTS.md),
plus the BO-value
[`VERIFICATION.md`](../experiments/sun_oxide/outputs/bo_value_pilot/VERIFICATION.md).

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

Do not execute prospective E2 source seeds from replacement preregistration
`0dcb76f5f2d053e098b472ac9984182b837295b5`. The immediate E2 planning task is
development-only improvement of FULL proposal/target overlap, followed by an
independent reliability diagnostic. The current development evidence does not
justify changing the scientific methods, thresholds, factor selection, or
FULL-reference criterion yet.

Do not run `experiments/sun_oxide/colab_adaptive_e3_validation.ipynb` or access
seeds 12--31. The decision-reset smoke and full-archive scaling probe remain
engineering blockers, and the final frozen 500-support diagnostic classified
empirical decision sparsity as `MIXED_DECISION_SPARSITY`, not the prospectively
defined strong-bound-conservatism case. There is no active follow-on E3
adaptive-preference experiment plan or fresh validation authorization. The
superseded preregistration remains unrun.

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
