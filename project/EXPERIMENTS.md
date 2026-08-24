# Experiments

> **Role:** current empirical registry and execution plan. Detailed run history
> belongs in immutable experiment output directories and closed handoffs, not
> here. Every new experiment must test a paper claim and have a prospective
> failure criterion.

## Operating principles

1. Experiments are theory-implied, not a generic BO benchmark suite.
2. Compare against the fully conditioned BO decision whenever feasible.
3. Report structural savings, inference quality/cost, and action regret
   separately.
4. A claimed certificate requires coverage checks against stronger held-out
   full-target calculations.
5. Use the same inference backend across factor-selection baselines when
   possible.
6. If a prospective failure occurs, preserve it and weaken the corresponding
   claim; do not tune the gate after the fact.
7. ESS, split-half checks, and ordinary Monte Carlo error bars are diagnostics,
   not rigorous certificates.

Experiments that isolate decision-relevant conditioning use the same fixed
hyperparameters, or the same prospectively specified calibration protocol,
across compared methods. Record calibration cost separately from decision-time
conditioning cost when nontrivial fitting is used.

## Registry

| ID | Experiment | Paper role | Current status |
|---|---|---|---|
| E1 | Nonlocal reflection symmetry | Main mechanism + certificate | `ACTIVE`; T2-B validation and finite-grid pilot passed; repeats/baselines remain |
| E2 | Nonlinear PDE expanding-domain scaling | Main scaling/inference consequence | `ACTIVE`; T2-B and family T4 passed; robustness/baselines remain |
| E3 | Realistic non-PDE sequential BO | Main end-to-end sequential test | `PASS_PBE_VALUE`; adaptive path ends `MIXED_DECISION_SPARSITY` after pathological smoke and unhelpful full-archive probe; fresh seeds remain unspent |
| E4 | Linear PDE factor graph | Supplementary control | `EXISTING EVIDENCE / SUPPLEMENT` |
| A1 | Factor-selection ablations | Supplement | `PLANNED` after primary pipelines |
| A2 | Inference-backend comparison | Supplement/modularity | `PLANNED` after primary pipelines |
| A3 | Continuous-factor/quadrature example | Optional supplement | `NOT REQUIRED` |

Directory-level status and immutable output links are indexed in
[`../experiments/README.md`](../experiments/README.md).

## E1 — Nonlocal reflection symmetry

**Status:** `EXISTING EVIDENCE / PASSED NARROW GATES`; the polished repeated
mechanism/coverage experiment remains active.

**Claim tested:** C1/C2 and T1–T3: relational, non-Euclidean conditioning can
be screened according to the next BO decision, with a valid full-target action
certificate using materially fewer than all factors.

### Established evidence

- Historical prototype: \(N=40\); structural loop stopped at \(M=12\); an
  earlier conservative Monte Carlo allowance required about \(M=15\) for
  \(\epsilon=0.03\); held-out full conditioning selected the same action.
  Historical scaling for \(N=20,40,80,160\) used about \(M=10\) at the same
  structural threshold. Source:
  [`DEC_Symmetry_Continuous_BO_Demo.ipynb`](../notebooks/prototypes/DEC_Symmetry_Continuous_BO_Demo.ipynb).
- **T2-B EI validation — `PASS`.** The screen stopped with 15/40 factors,
  omitted 62.5% before validation, and had sparse gap `-0.00025484`, structural
  bound `0.00533922`, and optimistic envelope `0.00508439 < 0.01`. Across
  eight fresh 20,000-sample full-target replicates, maximum observed EI regret
  was `0.00037839759`. These inference/grid checks are diagnostics, not a
  rigorous finite-sample or continuous-action certificate. See the immutable
  [`RESULTS.md`](../experiments/symmetry/outputs/t2b_ei_validation/RESULTS.md)
  and [technical audit](reference/T2B_SYMMETRY_AUDIT.md).
- **Prospective finite-grid inference-certification pilot — `PASS`.** It was
  run once from preregistration commit
  `5da6fec6c0a645ba56f555062a3adb4139a1782d` with frozen configuration SHA-256
  `c006a02581a6e586c793ce116476ccdd311509c31ece85790deedf7fdb33b639`.
  It certified on round 6 with 15/40 active factors, leader `-0.2082`, worst
  challenger `-0.2173`, estimated gap `-0.0003099863882800591`, inference bound
  `0.003848511075377576`, structural bound `0.005771786534479693`, and total
  certificate `0.00931031122157721 <= 0.01`. Final exact-rejection acceptance
  was `0.3342708888888889`; total Gaussian proposals were 5,375,000. Every
  frozen mechanical condition and the preregistered test suite passed. See the
  immutable [`RESULTS.md`](../experiments/symmetry/outputs/inference_certification_pilot/RESULTS.md),
  [full output directory](../experiments/symmetry/outputs/inference_certification_pilot/),
  and [archived implementation contract](archive/INFERENCE_CERTIFICATION_IMPLEMENTATION_HANDOFF.md).

The finite-sample guarantee is only for the exhaustive 401-action grid and
exact rejection backend. It does not imply a continuous-action certificate or
transfer automatically to HMC, SMC, FlowGP, importance sampling, or another
backend.

### Next E1 experiment

Convert the mechanism into a seeded repeated experiment that saves per-round
leader/challenger, active set, bound decomposition, held-out full-target
validation, and baseline results in tidy machine-readable tables.

Required comparisons: full conditioning; random activation at matched \(M\);
Euclidean-nearest factors; a static graph/symmetry neighborhood; and static
top-influence selection without challenger reoptimization.

Primary metrics: \(M\), \(M/N\), certified bound, held-out full acquisition
regret, empirical coverage at nominal \(1-\delta\), factor evaluations,
wall-clock cost, and full-target action agreement.

**Prospective success criterion:** across repeated trials, coverage is
consistent with the nominal level, materially fewer than all factors are used,
and adaptive selection improves factor count or bound tightness over at least
random and Euclidean-local selection at matched action quality.

**Prospective failure criterion:** if this fails, remove the claim that the
method identifies genuinely decision-relevant nonlocal conditioning rather
than ordinary locality or a favorable single instance.

## E2 — Nonlinear PDE expanding-domain scaling

**Status:** `EXISTING EVIDENCE / FULL-SHADOW BACKEND DEVELOPMENT BLOCKER`;
the T2-B construction passed and T4 is proved for this family, while the
replacement paper-level stress preregistration remains unexecuted.

**Claim tested:** C3/T4, with T2/T3 validation: the factor count needed for a
localized BO decision grows much more slowly than total conditioning size, and
active-target inference degrades more slowly than full-target inference.

### Established evidence

- **Clean structural validation — `PASS`.** The implementation-base commit was
  `fe8d994119a47b0709651f26a418c946032e90f5`. For the frozen \(24\times24\)
  model, \(M/N=40/576\), sparse gap is `-0.0127500536`, theorem-backed
  structural bound is `0.03874403301354687`, and the clean comparison matrix
  matches the literal notebook construction to maximum error
  `1.11e-16`. The rigorous correction factor is one. See the immutable
  [`RESULTS.md`](../experiments/nonlinear_pde/outputs/t2b_structural_validation/RESULTS.md)
  and [technical audit](reference/T2B_NONLINEAR_PDE_AUDIT.md).
- The associated archived inference allowance `0.0235285003` and empirical
  total envelope `0.0495224797 < 0.06` are diagnostics, not a rigorous
  finite-sample certificate. The replay reports active GP-reference IS ESS
  `84.3855%`, full GP-reference IS ESS about `6.2%`, and full Laplace-IS ESS
  about `58.9%`; active/full actions are both `(14, 12)` on the grid.
- The earlier 24×24 notebook record is retained as historical evidence: it
  reported structural/inference/total values `0.03626`/`0.02347`/`0.04758`,
  active/full GP-reference IS ESS `84.3%`/`6.2%`, full Laplace-IS ESS `58.9%`,
  and the same held-out full action. The clean replay does not silently
  overwrite those legacy values.
- Historical expanding-domain runs at
  \(N=324,576,900,1296,1600\) selected \(M=40\) throughout; active
  GP-reference IS ESS stayed about 84–86%, while full-target ESS fell from
  about 30.0% to 1.8%. Source:
  [`DEC_Nonlinear_PDE_BO_Demo.ipynb`](../notebooks/prototypes/DEC_Nonlinear_PDE_BO_Demo.ipynb).
- **Development-only FULL-shadow reliability diagnostic.** No prospective E2
  seed was accessed; replacement preregistration
  `0dcb76f5f2d053e098b472ac9984182b837295b5` remains unexecuted. Using only
  development seed `2026082401`, independent FULL Laplace-SNIS pairs at n=24
  had ESS fractions `0.510`--`0.559`: 8,192-sample action changes were
  decision-negligible near ties, and all early/middle/late states passed the
  unchanged rule after the frozen 16,384 escalation. At n=40, all three states
  failed the frozen ESS-fraction rule at both 8,192 and 16,384; mean ESS
  fractions over all batches through 32,768 were about `0.199`, `0.202`, and
  `0.179` for early/middle/late, even as absolute ESS grew into the thousands.
  Every Laplace mode converged in three steps with gradient infinity norm below
  `5e-14`. Most action changes had reciprocal regret below `0.006`, but one
  early 16,384 pair had material cross-regret `0.024442`. At early 32,768 the
  actions and ESS passed and top-five-union difference was `0.008649`, while
  the uniform vector difference `0.011589` failed, confirming one secondary
  global-vector-only rejection. The primary recommendation is to improve and
  development-validate FULL proposal/target overlap before prospective
  execution; neither more samples alone nor a post hoc decision-aligned rule
  is justified as the sole correction. The optional 65,536 level was not used
  because the cause was unambiguous. See the development-only
  [`RESULTS.md`](../experiments/nonlinear_pde/outputs/full_shadow_reliability_diagnostic/RESULTS.md).

### Next E2 experiment

After the E1 upgrade and the realistic-E3 gate, reproduce the scaling study
over multiple source fields/BO states per \(N\). Add fixed geometric-local and
random baselines first, then only the smallest additional static/adaptive
ablations needed to test challenger reoptimization. Save one summary table
with \(N,M\), regret/certificate, ESS, wall time, and selected-factor geometry.

**Prospective success criterion:** active factor count remains strongly
sublinear over the tested expanding-domain regime, action-quality/certificate
behavior is stable, active-target inference degrades materially more slowly
than full-target inference, and adaptation adds value beyond fixed locality in
at least some instances.

**Prospective failure criterion:** if this fails, remove the strong empirical
claim that decision-relevant conditioning stays small as the full problem
grows or changes the inference regime beyond simply dropping remote PDE
factors.

## E3 — Realistic non-PDE BO candidate

**Status:** `PASS_PBE_VALUE / ADAPTIVE_ENGINEERING_PATHOLOGICAL / FULL_ARCHIVE_NOT_HELPFUL / MIXED_DECISION_SPARSITY / FRESH ADAPTIVE VALIDATION NOT PREREGISTERED / NORMALIZED PBE MODEL PASSED / ADJACENT BASELINE VALID / DESCRIPTOR-REFERENCE GRAPH PASSED / CURRENT NLR BENCHMARK FROZEN / HISTORICAL REPRODUCTION JOIN_AMBIGUOUS`.

The exact historical **Sun et al. PBE→GW oxide legacy-data problem** failed its
first source-recovery gate. The current authoritative NREL/NLR queries reproduce
5,604 finite PBE rows / 2,142 unique compositions and 244 finite GW rows / 194
unique compositions, but only 193 GW compositions overlap the finite-PBE set.
The 28 duplicate GW composition groups all span multiple stable material
families, and the live raw rows differ from the author's surviving PrefInt
notebook evidence. The exact historical formula-only de-duplication therefore
cannot be reconstructed without unresolved polymorph choices.

Terminal verdict: `JOIN_AMBIGUOUS`. Implementation commit
`46a08e09bce33d2189f3f3666035a72e08e608cb`; immutable result:
[`experiments/sun_oxide/outputs/source_recovery/SOURCE_AUDIT.md`](../experiments/sun_oxide/outputs/source_recovery/SOURCE_AUDIT.md).
No normalized benchmark or descriptor matrix was emitted, and no target
ranking, descriptor regeneration, factor construction, inference, BO design,
or performance baseline was run. That historical reconstruction remains closed
unless the exact author CSV/matrix is recovered.

A separate prospective current-data benchmark is now frozen as
`CURRENT_NLR_PBE_GW_V1`: “A current, reproducible NLR PBE→GW oxide benchmark
inspired by the legacy-data setting of Sun et al. (2020), not an exact
reproduction of their historical polymorph selection.” Starting from 194
current GW compositions, its target-blind rule uses the sole authoritative
`wave` parent's finite GGA(+U) total energy per atom and stable MatDB ID only as
an exact-tie breaker. The 166 single-row and 28 multi-polymorph composition
counts reproduce; all 28 multi-polymorph cases resolve with zero exact energy
ties. Strict composition/family-provenance mapping excludes CdO, Ga2O3, and
Sb2O3 and leaves 191 one-to-one GW actions over 2,142 canonical PBE/FERE legacy
composition keys. Canonical PBE rows use lowest available source total energy
per atom with stable MatDB ID as the exact-tie breaker. The prospective freeze
criterion was exact agreement with all of these counts, exclusions, mappings,
deterministic reruns, provenance hashes, and GW-target isolation; it passed with
terminal verdict `PASS_CURRENT_NLR_BENCHMARK`.

Committed benchmark record:
[`experiments/sun_oxide/benchmark/benchmark_manifest.json`](../experiments/sun_oxide/benchmark/benchmark_manifest.json).
GW magnitudes live only in the isolated two-column `gw_oracle.csv` and did not
enter selection or mapping.

The separately preregistered descriptor/reference-graph compatibility gate ran
from implementation SHA `843b2173454b70cf12a6199b4d1a32740e60315e` in a
fully pinned Python 3.12.13 environment and passed as
`PASS_DESCRIPTOR_GRAPH`. It produced a finite `(2142, 132)` raw Magpie matrix
with 15 zero-variance features, a connected deterministic 10-NN-plus-MST graph
with 14,072 unique edges, and a sparse positive-definite `Q0` whose independent
extremal eigenvalue checks are approximately `1.0000000000000024` and
`2.424183846275552`. All three sparse-solve residuals are below `1e-10`, and
all 191 actions map uniquely. No GW value was read and no benchmark PBE value
entered descriptor or graph construction. This is a representation/reference
compatibility result, not theory, inference, or BO evidence. Committed result:
[`outputs/descriptor_graph/VERIFICATION.md`](../experiments/sun_oxide/outputs/descriptor_graph/VERIFICATION.md).

The prospectively frozen `ADJACENT_STRICT_PBE_ORDER_V1` gate passed as
`PASS_PBE_FACTOR_THEORY`. It produced 1,681 strict consecutive factors after
omitting 460 exact-tie adjacencies in 369 tie groups; the factor graph has
maximum degree two and spectral norm `1.988275914308721`. The existing scalar-
block Menz construction `A0=Q0-0.25R` has analytic eigenvalue floor `0.5` and
numerical smallest eigenvalue `0.5896249844278044`; sparse action-row solves,
later-observation diagonal-monotonicity checks, and all 18,145 target-blind
action-pair influence diagnostics completed without a dense inverse. The
90/95/99%-influence fractions remain diagnostics, not sparsity gates. No GW
value was read, and no posterior inference, BO, or covariance-theory extension
was performed. Committed result:
[`outputs/pbe_factor_theory/RESULTS.md`](../experiments/sun_oxide/outputs/pbe_factor_theory/RESULTS.md).

The prospectively frozen replacement-model gate passed as
`PASS_NORMALIZED_PBE_MODEL`. `PBE_SUPPORT_500_V1` contains all 191 actions plus
309 PBE-only nodes from deterministic farthest-point sampling in the committed
standardized Magpie space. `NORMALIZED_ALL_PAIRS_PBE_500_V1` retains 124,718
strict PBE pairs with global weight `1/499` and omits 32 exact-tie pairs in 23
tie groups. The maximum weighted row sum is `0.9999999999999997`, the numerical
weighted-adjacency norm is `0.9997442203602465`, and
`A0=Q0-0.25W_pbe` has analytic eigenvalue floor `0.75` and numerical minimum
`0.9003424446737381`. Sparse solves, factor calculus, and later-observation
M-matrix/resolvent checks passed without a dense inverse. The PBE-only MAP
shows a nontrivial global ordinal signal relative to the adjacent-chain sparse
baseline; the 256-pair 50/75/90/95%-influence summaries remain diagnostics
without a signal or sparsity threshold. No GW value, inference, BO, or new
covariance theorem entered this gate. The adjacent chain remains a valid sparse
baseline; the normalized 500-support model is the proposed E3 full-conditioning
model. Committed result:
[`outputs/normalized_pbe_model/RESULTS.md`](../experiments/sun_oxide/outputs/normalized_pbe_model/RESULTS.md).

The preregistered first GW-target value pilot passed as `PASS_PBE_VALUE` from
run SHA `44f58f100f41247afe0937e42eebe58055104225`; frozen config SHA-256
`6cc47d41dfbdbf88187d535d405ca6afd971e4b07f91932d55dbbbf5c101ef0f`.
Across 12 paired seeds and 12 sequential queries, median AURC was 19.4445 eV
for `NO_PBE` versus 1.0170 eV for `FULL_PBE`, and median final regret was
0.8310 versus 0.0000 eV. The full method won 10/12 paired AURCs and tied the
other two because their shared initial observations already contained the
optimum; it never lost. Global-optimum discovery was 4/12 versus 10/12.

All three frozen 4,096-sample Laplace-proposal SNIS validations passed. ESS
fractions were 0.905102 initially, 0.905340 after six queries, and 0.906766
after twelve; the corresponding IS-estimated regrets of the Laplace actions
were 0.00289358, 0.00003532, and 0.00008743, all below the frozen 0.02-dominated
threshold. Independent replay reproduced the selected actions and numeric
diagnostics, and the oracle-access audit found no unobserved target value in
either acquisition. Median routine decision time was 0.240 seconds for
`NO_PBE` and 1.630 seconds for `FULL_PBE`; these are diagnostics, not an
adaptive-speedup result. Immutable record:
[`outputs/bo_value_pilot/VERIFICATION.md`](../experiments/sun_oxide/outputs/bo_value_pilot/VERIFICATION.md).

The outcome makes scientific sense: the allowed post-run PBE-vs-GW Spearman
diagnostic is 0.8333 over the 191 actions, and the GW optimum is also the
highest-PBE action. This explains why the frozen ranking bank is highly useful
without changing the fact that the model and protocol were fixed before GW
evaluation. The supported claim is limited to full-conditioning value on this
benchmark; it does not establish adaptive savings, a rigorous Laplace
certificate, or cross-dataset generalization.

### E3 adaptive engineering status

The engineering implementation is frozen from SHA
`7fbfb202268dd0fd92d35defbea2cc4990f089e2`. Its local seeds 0--2 smoke passed
mechanically: optimized `FULL_PBE_OPT` reproduced all 36 committed `FULL_PBE`
decisions; every adaptive decision certified or explicitly fell back; and
shadow FULL agreed on all 36 adaptive states. This is not scientific evidence.
Each seed's first adaptive decision fell back to the full bank after eight
stages, making the active fraction 1.0 thereafter. Median ADAPTIVE/FULL
conditioning time was about 0.999, so the conjunctive engineering-pathology
stop did not trigger because the timing ratio was not above 1.25. Immutable
engineering record:
[`outputs/adaptive_e3_smoke/RESULTS.md`](../experiments/sun_oxide/outputs/adaptive_e3_smoke/RESULTS.md).

Commit `70a9686b143c09f9f970306cc4489a2ce2b6e173` preregistered a fresh
seeds 12--31 validation, but that preregistration was never scientifically
executed and is superseded. No fresh-seed oracle access occurred.

The lifecycle-only follow-up implementation starts every BO decision with an
empty factor mask and re-screens the complete bank, while keeping factor
activation cumulative across stages within the decision and warm-starting the
first active-target fit from the preceding adaptive MAP. The seeds 0--2 smoke
remained mechanically correct: optimized FULL reproduced 36/36 prior FULL
decisions, every adaptive decision explicitly full-fallbacked, and shadow FULL
agreed 36/36 with zero EI regret. It is not scientific evidence.

The efficiency outcome is a blocker. All 36 decisions exhausted eight stages
and ended with all 124,718 factors. For decisions 2--12 alone, 33/33 decisions
full-fallbacked; median final active fraction was 1.0; median paired
ADAPTIVE/FULL conditioning-time ratio was about 4.00; median factor
energy-gradient work was about 3.64 times FULL; and median Hessian work was
about 8.40 times FULL. The frozen algorithm therefore supplies no credible
factor-work or wall-clock reduction after observations arrive. Immutable
engineering record:
[`outputs/adaptive_e3_decision_reset_smoke/RESULTS.md`](../experiments/sun_oxide/outputs/adaptive_e3_decision_reset_smoke/RESULTS.md).

There is no active fresh adaptive validation preregistration and no authorized
seed 12--31 run. The existing config/notebook are retained as engineering
artifacts only.

**Development-only full-bank scaling probe — `FULL_ARCHIVE_NOT_HELPFUL`.** The
500-, 1,000-, and 2,142-support probe completed locally at the three
already-consumed seed-0 FULL states with `0.425639936` GB peak RSS. The full
archive remained mathematically healthy and target-blind PBE rank signal stayed
strong, but median pre-fallback active counts were `116195`, `474338`, and
`2230019`, corresponding to fractions `0.931662`, `0.949890`, and `0.972771`.
All nine primary states exhausted eight stages and full-fallbacked. At 2,142
support the active action first matched FULL only after the first bound-selected
batch had already activated `0.964427`--`0.969912` of factors. Median FULL
conditioning times were `0.2250`, `0.8838`, and `5.5806` seconds. Thus enlarging
the bank did not reveal decision/conditioning separation under the current
certificate. This used no fresh seed, changed no theorem or paper claim, and
created no preregistration. Immutable record:
[`outputs/full_bank_scaling_probe/RESULTS.md`](../experiments/sun_oxide/outputs/full_bank_scaling_probe/RESULTS.md).

**Final development-only decision-sparsity diagnostic —
`MIXED_DECISION_SPARSITY`.** From starting main SHA
`5f49f140acc3532b0231b1d1c446d22cd0e168d8`, implementation SHA
`2ac5dd3576c548f0c9999cbea3bdf7d6f626656d`, and config SHA-256
`281e9b173a234029563f8ed876b5d252befb6706849bf100cd61f281e65662e6`,
the frozen 500-support diagnostic used only the three already-consumed seed-0
FULL states. The primary reranked path had first stable FULL-action agreement
fractions `0.70`, `0.20`, and `0.10`; all three theorem certificate fractions
were `1.00`. STATIC stabilized at `0.10`, `0.20`, and `0.40`, so reranking was
materially state-dependent. At the middle state, reranking recovered FULL from
fraction `0.20` onward while none of 20 matched random subsets agreed at
fractions `0.10`, `0.20`, or `0.40`; after 12 queries, both ranked and random
subsets agreed at all three matched fractions. The theorem envelope is clearly
conservative, but the initial reranked path's `0.70` requirement prevents the
prospectively defined strong classification. Peak RSS was `0.222232576` GB.
No fresh seed was accessed, no preregistration was created, and no additional
E3 adaptive-preference experiment is planned. Immutable record:
[`outputs/decision_sparsity_diagnostic/RESULTS.md`](../experiments/sun_oxide/outputs/decision_sparsity_diagnostic/RESULTS.md).

### Closed E3 evidence

| Case | Frozen provenance | Exact verdict and interpretation | Immutable record |
|---|---|---|---|
| Minimal synthetic preference bank | Start `5f7f7cfa66f4d6ec33ebf5a5feb3d8f77ba2110a`; config SHA-256 `4eafe727b67864632d25d30b526cbdb5c0a83989bf31e7b0cf41f7129c8a89da` | `FAIL-P2`: P1 passed (median \(T_{0.10}\) full 3 vs standard 7) and P2 performance passed (adaptive 3 vs full 3), but median factor ratio `0.875` failed the `0.65` sparsity gate. Evidence points to conservative structural bounds, not IS failure; maximum held-out acquisition regret was `9.2476e-4`. | [RESULTS](../experiments/preference_bo/outputs/minimal_pilot/RESULTS.md), [handoff](archive/e3/E3_PREFERENCE_BO_PILOT_HANDOFF.md) |
| Redundant 24-edge synthetic bank | Start `b68525952f58693461ac32d4658a8d611a594706`; config SHA-256 `5c53c8de7144d1d0bc3a66b1ca4233b2c17811f726e977e1a5c90f103140b260` | `FAIL-P2`: P1 passed (full 2.5 vs standard 7) and P2 performance passed (adaptive 2.5 vs full 2.5), but median factor ratio `0.8888888889` failed the independently frozen `0.80` gate. Structural terms again dominated empirical inference allowances; maximum held-out acquisition regret was `0.00261658`. | [RESULTS](../experiments/preference_bo/outputs/redundant_bank_pilot/RESULTS.md), [handoff](archive/e3/E3_PREFERENCE_BO_REDUNDANT_BANK_PILOT_HANDOFF.md) |
| Gp2 pairwise-preference P1 gate | Run provenance SHA `5e9ed896956a937c0ed24e58b17499699e7734ad`; config SHA-256 `56cf2ab3e9d6a97b39d75b55161054924eab2e5b0c153880c87c5fbe96e93dc8` | `PREPROCESSING_INVALID`: 169 candidates survived, below the frozen minimum of 250. No smoke or scientific P1 comparison ran. | [verdict](../experiments/gp2_preference_bo/outputs/preflight/PREPROCESSING_INVALID.json), [handoff](archive/e3/E3_GP2_P1_GATE_HANDOFF.md) |
| Gp2 target-blind proxy structural preflight | Preregistration `1e81cd1c0f2ffe3c8d5347fdd1dd0007c2a5ff96`; config SHA-256 `332107f9a6c98af7135d627d39ea6883e3c2098d87feb5cdb0446a23151f78cc` | `PREPROCESSING_INVALID / GP2_ABANDONED`: 47/279 actions supplied proxy factors (`0.16845878136200718` coverage), below the frozen 75-factor and 40% minimums. No theory matrices, structural distribution, inference, or BO ran. | [RESULTS](../experiments/gp2_proxy_bo/outputs/structural_preflight/RESULTS.md), [handoff](archive/e3/E3_GP2_STRUCTURAL_PREFLIGHT_HANDOFF.md) |

All failed attempts, schema/provenance-incomplete attempts, raw trajectories,
configs, and final outputs remain committed under their original experiment
paths. They are provenance, not alternative successful results. Gp2 is closed
and abandoned as E3.

## E4 — Linear PDE control

**Status:** `EXISTING EVIDENCE / SUPPLEMENT`.

Historical 24×24 evidence reports \(N=576\), \(M=50\), total empirical EI
envelope `0.0563`, structural component `0.0398`, inference component `0.0246`,
active/full GP-reference IS ESS `85.0%`/`12.0%`, full Laplace-IS ESS `69.4%`,
and the same active/full BO action. Expanding-domain runs report \(M\) near
50–60 as \(N\) grows 324→1600, while full-target GP-reference IS ESS falls
34.1%→0.78%. Source:
[`DEC_PDE_Certified_BO_Demo.ipynb`](../notebooks/prototypes/DEC_PDE_Certified_BO_Demo.ipynb).

Retain E4 as a supplement/control and proof-aligned debugging environment. It
does not receive a main-paper figure unless the nonlinear case becomes
unreliable.

## Cross-cutting work

- **A1 factor selection:** random, Euclidean/geometric local, graph local,
  static top influence, adaptive influence without challenger reoptimization,
  full adaptive, and cost-aware variants where relevant. Run only after the
  primary pipelines stabilize.
- **A2 inference backends:** compare standard backends only to establish
  modularity and the inference-regime consequence. Report action error and wall
  time as well as ESS/mixing; claim no sampler novelty.
- **A3 continuous factors/quadrature:** optional, only if the continuous-factor
  formulation becomes important to the paper narrative.

## Certification and result contract

For an experiment marketed as certified:

1. freeze total failure probability \(\delta\) and its allocation across
   candidates/adaptive rounds;
2. record structural, inference, and optimization components separately;
3. validate the returned action with a materially stronger held-out full-target
   computation;
4. report repeated empirical coverage of the advertised bound; and
5. state the precise action-domain and inference-backend scope.

Every run directory must contain machine-readable results plus a concise
`RESULTS.md` recording run ID/date, Git SHA, frozen config and hash, seeds,
primary metric, coverage status, \(M/N\), active/full inference metrics, wall
time, exact verdict, one-sentence interpretation, and next action. Do not rely
on screenshots or notebook output alone.

## Execution order

1. **Completed:** reflection-symmetry T2-B validation and rigorous finite-grid
   end-to-end pilot.
2. **Completed:** current-NLR oxide benchmark freeze and descriptor/reference-
   graph compatibility gate.
3. **Completed:** sparse PBE-order legacy factor bank and existing-theory
   compatibility gate.
4. **Completed:** minimal sequential GW BO value pilot comparing no legacy
   factors with full normalized PBE-order conditioning (`PASS_PBE_VALUE`).
5. **Blocked:** decision-reset adaptive PBE conditioning preserved FULL actions
   but required full fallback at all 36 smoke decisions and increased factor
   work/time. Enlarging normalized all-pairs support to all 2,142 materials did
   not help: median pre-fallback factor fraction increased to `0.972771` and all
   nine development states still fell back. Fresh seeds 12--31 remain unspent.
6. Run the E1 repeated mechanism/coverage experiment and frozen baselines.
7. Expand E2 across source fields/BO states with the smallest necessary
   baseline set.
8. Run broad supplementary sampler/factor-selection ablations only afterward.
