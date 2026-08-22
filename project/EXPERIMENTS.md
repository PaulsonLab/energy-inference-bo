# Experiments

> **Role of this file:** living registry of empirical claims, code, results, pass/fail criteria, and next actions. Every experiment must test a paper claim. Do not grow a generic BO benchmark suite.
>
> **Status labels:** `EXISTING EVIDENCE`, `RUNNING`, `PLANNED`, `PASSED`, `FAILED`, `SUPPLEMENT`.

## Experimental principles

1. **Theory-implied, not benchmark-driven.** Each experiment must answer a specific question raised by the paper story or theory.
2. **Compare against the fully conditioned BO decision whenever feasible.** The target is not agreement with the active posterior; it is action quality relative to the full conditioned acquisition.
3. **Separate structural savings from inference savings.** Report active factor count, factor-evaluation cost, inference quality/cost, and action regret separately.
4. **A certificate must be checked for coverage.** For experiments that claim certification, compare the stated upper bound with held-out full-target regret over repeated trials.
5. **Use the same inference backend across factor-selection baselines when possible.** Otherwise improvements can be attributed to sampler differences rather than decision-directed conditioning.
6. **Every experiment has a failure statement.** If the stated failure occurs, weaken or remove the corresponding claim rather than adding ad hoc tests.

## Experiment registry

| ID | Experiment | Paper role | Status | Compute |
|---|---|---|---|---|
| E1 | Nonlocal reflection symmetry | Mechanism + certificate figure | EXISTING EVIDENCE; T2-B PASS; prospective finite-sample pilot PASS; repeats/baselines remain | MacBook Air |
| E2 | Nonlinear PDE expanding-domain scaling | Main scaling / inference consequence | EXISTING EVIDENCE; T2-B and family T4 PASS; needs robustness + baselines | Colab A100 |
| E3 | Preference-conditioned sequential BO | Main end-to-end non-PDE BO test | TWO PROSPECTIVE SYNTHETIC PILOTS FAILED-P2 (P1 and performance passed; sparsity failed; final suite deferred) | MacBook Air / A100 sweeps |
| E4 | Linear PDE factor graph | Supplementary control | EXISTING EVIDENCE | MacBook Air / A100 sweeps |
| A1 | Factor-selection ablations | Supplement / supports E1--E3 | PLANNED | Mixed |
| A2 | Inference/sampler comparison | Supplement / modularity | PLANNED | A100 |
| A3 | Continuous-factor/quadrature example | Optional | NOT REQUIRED | MacBook Air |

---

## E1 — Nonlocal reflection symmetry

**Status:** `EXISTING EVIDENCE / PASSED` — the reflection-symmetry T2-B
construction, narrow EI non-vacuity check, and locked prospective finite-sample
pilot pass.  The rigorous end-to-end result is scoped to the committed
finite-grid instantiation; repeated trials and baselines remain later E1 work.

**Paper claim tested:** C1/C2; Theory T1--T3.

### Question

Can decision-relevant factor selection identify relational conditioning that is nonlocal in Euclidean input space and certify the next BO action without instantiating the full conditioned target?

### Construction

Reflection factors have the form

\[
e_j(f)
=
\gamma\log\cosh\!\left(\frac{f(r_j)-f(-r_j)}{\tau}\right).
\]

Each factor couples potentially distant inputs, so the example tests influence through a symmetry/function graph rather than a naive spatial radius around the current leader.

### Existing result

- total factors: \(N=40\);
- structural-only adaptive loop certified after \(M=12\) active factors;
- sparse leader: approximately \(\widehat x=-0.2082\);
- final worst challenger: approximately \(-0.359\);
- structural log-acquisition certificate: approximately \(0.0261\);
- held-out full conditioning selected the same action;
- adding the existing conservative Monte Carlo gap allowance required approximately \(M=15\) factors for an \(\epsilon=0.03\) target;
- earlier scaling test with \(N=20,40,80,160\) required approximately \(M=10\) factors at the same structural threshold.

**Existing notebook:** `DEC_Symmetry_Continuous_BO_Demo.ipynb`

### T2-B EI validation — 2026-08-20

**Status:** `PASSED` for the narrow structural non-vacuity question only.

The prospective failure criterion was inability to stop at EI tolerance
\(0.01\) while omitting at least 20% of the 40 factors, or an empirical
held-out full-target EI regret above that fixed tolerance. The screen stopped
with 15 active factors and 25 (62.5%) unevaluated before validation. Its final
sparse gap was \(-0.00025484\), structural term \(0.00533922\), and optimistic
envelope \(0.00508439\). The screened action was \(-0.2082\).

Across eight fresh 20,000-sample full-target replicates, the maximum empirical
EI regret at that action was \(3.78\times10^{-4}\); all replicates were below
the fixed tolerance. Four of eight selected the exact same coarse-grid action,
and the maximum action displacement was \(0.0065\). A 401-to-1601 action-grid
refinement moved the leader by \(0.00065\). These inference and grid checks are
empirical diagnostics, not a rigorous finite-sample or continuous-action
certificate. See
[`../experiments/symmetry/outputs/t2b_ei_validation/`](../experiments/symmetry/outputs/t2b_ei_validation/)
and [`T2B_SYMMETRY_AUDIT.md`](T2B_SYMMETRY_AUDIT.md).

### Prospective finite-sample inference-certification pilot — passed

**Status:** `PASSED` — the end-to-end blocker is closed for the committed
reflection-symmetry finite-grid instantiation and exact-rejection backend.

**Claim tested:** In the committed clean reflection-symmetry EI problem, the
complete finite-grid T3 certificate can remain below \(\epsilon=0.01\) while a
materially sparse subset of the 40 conditioning factors is active, after the
empirical Monte Carlo allowance is replaced by an adaptivity-safe rigorous
finite-sample bound from exact active-target samples.

The machine-readable source is
[`../experiments/symmetry/configs/inference_certification_pilot.json`](../experiments/symmetry/configs/inference_certification_pilot.json),
and the exact implementation/preregistration contract is
[`INFERENCE_CERTIFICATION_IMPLEMENTATION_HANDOFF.md`](INFERENCE_CERTIFICATION_IMPLEMENTATION_HANDOFF.md).
The locked configuration is:

- 40 symmetry factors and the committed clean-EI OU/model parameters;
- 401 actions on \([-0.58,-0.06]\), incumbent \(0.50\);
- \(\epsilon=0.01\), \(\delta=0.05\), at most 15 certification rounds;
- three factors per failed refinement;
- one reusable 80,000-sample working batch with seed 123;
- independent certification children from
  `SeedSequence(314159265).spawn(15)`;
- exact rejection sampling in chunks of 25,000 proposals;
- 100,000 accepted samples for \(M<15\), otherwise 1,500,000.

The pilot is **PASS if and only if** a reached round has
\(U_{\rm cert}\le0.01\), certification occurs with \(M\le18\), the inference
radius at the maximizing optimistic challenger is at most \(0.0045\), total
generated Gaussian proposals are at most 20,000,000, the final-round acceptance
rate is at least 0.20, and every required mathematical, regression,
seed-separation, no-reuse, synthetic end-to-end, and output-integrity test
passes.  Failure of any condition is a prospective **FAIL**; no threshold,
seed, sample schedule, grid, or confidence allocation may be changed afterward
in this task.

Session-4 planning calculations predict stopping near \(M=15\), final
acceptance roughly 0.27--0.34, \(B_{\rm infer}\) roughly 0.0038--0.0040, and
\(U_{\rm cert}\) roughly 0.0089--0.0091.  These are planning estimates, not
results.

**Prospective failure statement:** if the pilot fails, the finite-sample
end-to-end empirical blocker remains open and the failed predeclared conditions
will be reported without redesigning the experiment in this task.

**Prospective result (2026-08-21): PASS.**  The pilot was run once from the
pre-result preregistration commit
`5da6fec6c0a645ba56f555062a3adb4139a1782d`, using the frozen configuration
SHA-256
`c006a02581a6e586c793ce116476ccdd311509c31ece85790deedf7fdb33b639`.
It reached six rounds and certified at \(M=15\), omitting 25 of 40 factors
(62.5%).  The leader was \(-0.2082\), the worst optimistic challenger was
\(-0.2173\), and the final decomposition was
\[
  \widehat G_S=-0.0003099863882800591,
  \qquad B_{\rm infer}=0.003848511075377576,
  \qquad B_{\rm struct}=0.005771786534479693,
\]
so \(U_{\rm cert}=0.00931031122157721\le0.01\).  The final exact-rejection
acceptance rate was \(0.3342708888888889\); 5,375,000 Gaussian proposals were
generated in total.  All six mechanical conditions passed, as did the 43-test
pre-pilot suite and the saved-output integrity checks.  Machine-readable
manifests, every 401-row challenger table, diagnostics, summary, and figure are
in
[`../experiments/symmetry/outputs/inference_certification_pilot/`](../experiments/symmetry/outputs/inference_certification_pilot/).

This is not a continuous-action certificate and does not confer the same
finite-sample guarantee on HMC, SMC, FlowGP, importance sampling, or another
inference backend.

### Final hypothesis

The adaptive influence/challenger procedure reaches a valid full-target action certificate using substantially fewer than \(N\) factors, including factors that would not be selected by Euclidean locality alone.

### Required baselines

1. full conditioning;
2. random factor activation at matched \(M\);
3. factors nearest to the leader in Euclidean input space;
4. static graph/symmetry-neighborhood selection;
5. static top-influence selection without re-solving the worst challenger.

### Primary figure

A mechanism figure with aligned panels:

1. BO domain, current leader/challenger, and symmetry pairs;
2. factors activated over refinement rounds;
3. active acquisition and held-out full acquisition, with optimistic challenger envelope;
4. decomposition of the certificate into active gap, structural error, inference error, and optimization error versus \(M\).

### Metrics

- active factors \(M\) and \(M/N\);
- certified bound versus observed held-out full acquisition regret;
- empirical certificate coverage at nominal confidence \(1-\delta\);
- factor-evaluation and wall-clock cost;
- action agreement with the full target.

### Pass criterion

Across repeated trials, the certificate has coverage consistent with its nominal level, the adaptive method uses materially fewer factors than the full target, and it outperforms at least random and Euclidean-local selection in factor count or tightness at matched action quality.

### Failure statement

> **If this fails, we can no longer claim that the method identifies genuinely decision-relevant nonlocal conditioning rather than simply exploiting ordinary locality or a favorable single example.**

### Next E1 task

Convert the existing notebook into a reproducible seeded experiment that saves
per-round leader/challenger, active set, bound decomposition, full-target
validation, and baseline results in a tidy result table.

---

## E2 — Nonlinear PDE expanding-domain scaling

**Status:** `EXISTING EVIDENCE` — the nonlinear-PDE T2-B construction and
family-level T4 mapping pass; the scaling evidence still needs robustness and
selection baselines.

**Paper claim tested:** C3; Theory T4, with T2/T3 validation.

### Construction

The nonlinear residual is based on

\[
f-\kappa\Delta f+\lambda\sin f=s,
\]

with discrete residual

\[
r_{ij}(f)
=
f_{ij}
-c\sum_{(k,\ell)\in\mathcal N(i,j)}f_{k\ell}
+\eta\sin f_{ij}
-b_{ij},
\]

and conditioning factor

\[
e_{ij}(f)=\gamma\log\cosh(r_{ij}(f)/\tau).
\]

### Legacy reported result: 24 x 24 case

These values predate the locked-environment provenance replay and are retained
as historical claims rather than silently overwritten:

- total factors: \(N=576\);
- active factors: \(M=40\);
- reported total empirical EI certificate: \(0.04758\);
- structural component: \(0.03626\);
- inference component: \(0.02347\);
- sparse gap to worst challenger: \(-0.01216\);
- active-target GP-reference IS ESS: \(84.3\%\);
- full-target GP-reference IS ESS: \(6.2\%\);
- full-target Laplace-preconditioned IS ESS: \(58.9\%\);
- active and held-out full targets selected the same BO action;
- observed held-out EI regret was zero on the action grid.

### Locked replay and structural validation — 2026-08-21

**Status:** `PASSED` for the nonlinear-PDE T2-B structural construction and
`PROVED FOR THIS FAMILY` for its T4 mapping.

The prospective implementation check failed if any accepted matrix diagnostic,
the clean-versus-notebook comparison, or the structural replay disagreed at its
recorded precision. All checks passed. The clean sparse construction uses the
same frozen \(24\times24\) model and exact active set as the archived replay:

- total/active factors: \(N=576\), \(M=40\);
- sparse acquisition gap: \(-0.0127500536\);
- theorem-backed structural bound: \(0.03874403301354687\);
- empirical inference allowance: \(0.0235285003\);
- empirical total stopping envelope: \(0.0495224797\);
- stopping tolerance: \(0.0600000000\), leaving margin \(0.0104775203\);
- active GP-reference IS ESS: \(84.3855\%\);
- full-target GP-reference IS ESS: approximately \(6.2\%\);
- full-target Laplace-IS ESS: approximately \(58.9\%\);
- active/full action: `(14, 12)`; observed grid EI regret: zero.

The clean comparison matrix agrees with a literal reproduction of the archived
notebook construction to maximum absolute difference \(1.11\times10^{-16}\).
Therefore \(A_{\rm rigorous}=A_{\rm notebook}\), the rigorous structural
correction factor is exactly one, and the structural value is unchanged.

The structural term is theorem-backed. The inference allowance is the
prototype's asymptotic Monte Carlo diagnostic, and the total envelope remains
empirical; this is not a rigorous finite-sample end-to-end action certificate.

Run record:

```text
Run ID / date: t2b_structural_validation / 2026-08-21
Git commit: fe8d994119a47b0709651f26a418c946032e90f5 (implementation base)
Configuration file: experiments/nonlinear_pde/outputs/t2b_structural_validation/frozen_config.json
Seeds: none for structural regression; archived replay seed 911 recorded as provenance
Result files: summary.json, RESULTS.md
Primary metric: clean/notebook A equality and structural value 0.03874403301354687
Certificate coverage: not applicable; no finite-sample inference certificate
Factor count M / total N: 40 / 576
Active inference metric: historical replay ESS 84.3855%
Full inference metric: historical replay ESS approximately 6.2% (GP reference), 58.9% (Laplace IS)
Wall time: recorded in summary.json
Pass/fail verdict: PASS
One-sentence interpretation: the nonlinear-PDE structural influence construction is rigorous and unchanged from the prototype.
Next action recorded at run time: human-reviewed work on the then-open
inference-theory blocker. Current status is that the separate prospective
symmetry finite-grid pilot above passed.
```

See [`T2B_NONLINEAR_PDE_AUDIT.md`](T2B_NONLINEAR_PDE_AUDIT.md) and
[`../experiments/nonlinear_pde/outputs/t2b_structural_validation/`](../experiments/nonlinear_pde/outputs/t2b_structural_validation/).

### Existing scaling result

For
\[
N=324,576,900,1296,1600,
\]

- \(M=40\) in every tested case;
- active-target GP-reference IS ESS remained approximately \(84\!\text{--}\!86\%\);
- full-target GP-reference IS ESS fell from approximately \(30.0\%\) to \(1.8\%\);
- at \(N=1600\), the factor ratio was \(N/M=40\).

**Existing notebook:** `DEC_Nonlinear_PDE_BO_Demo.ipynb`

### Final hypothesis

With a fixed BO decision region and an expanding structured-conditioning domain, the amount of conditioning required to settle the next BO action grows much more slowly than the total factor count, while inference under the full non-Gaussian target becomes substantially harder.

### Required baselines

1. full conditioning;
2. random subsets at matched \(M\);
3. fixed geometric local neighborhoods;
4. static top-sensitivity/influence factors;
5. adaptive selection without re-solving the challenger;
6. a posterior-oriented coreset/subsampling baseline where technically sensible;
7. optional diagnostic: diagonal/local influence approximation versus graph-aware influence.

### Primary figure

Three aligned scaling panels:

1. active factors \(M\) versus total factors \(N\), with a simple scaling fit and uncertainty across problem instances;
2. certified and observed full-target acquisition regret versus \(N\);
3. active-target versus full-target inference difficulty (ESS, samples required, or wall time) versus \(N\).

A small spatial panel may show which residual factors are selected around the BO decision region.

### Metrics

- \(M\) versus \(N\);
- \(M/N\);
- observed full-conditioned acquisition regret;
- certificate tightness/coverage;
- active/full ESS and wall time;
- baseline factor counts at matched action quality;
- sensitivity across source fields, BO states, and seeds.

### Pass criterion

The active factor count remains strongly sublinear over the tested expanding-domain regime, action-quality/certificate behavior remains stable, and active-target inference degrades materially more slowly than full-target inference. Adaptive selection must add value beyond a fixed local truncation in at least some problem instances.

### Failure statement

> **If this fails, we can no longer make a strong empirical claim that decision-relevant conditioning remains small as the full structured-conditioning problem grows, or that adaptive decision conditioning changes the inference regime rather than merely dropping remote PDE factors.**

### Next E2 task (after the global certificate gate)

After the finite-grid end-to-end certificate and the E1 upgrade in the global
execution order, refactor the scaling experiment to run multiple source
fields/BO states per \(N\), add the fixed-local and random baselines first, and
save a single summary CSV/JSON containing \(N,M\), regret/certificate, ESS, wall
time, and selected-factor geometry.

---

## E3 — Preference-conditioned sequential BO

**Status:** `FAILED-P2` for both the preregistered eight-factor minimal gate and
the separately preregistered 24-factor redundant-bank gate. In both, P1 and P2
performance passed while P2 sparsity failed. The full E3 baseline suite remains
planned and was intentionally deferred.

**Paper claim tested:** C1--C3 in an end-to-end BO loop; establishes relevance beyond PDE/physics examples.

### Minimal preference-BO phenomenon pilot — 2026-08-21

**Status:** `FAILED-P2` under the preregistered precedence. This was the minimal
E3 phenomenon gate, not the final E3 baseline experiment.

**Scoped pilot update:** the older planning text below called for
full/adaptive/fixed-local in the first three-way comparison. The frozen pilot
handoff explicitly replaced only that pilot-level choice with
standard/full/adaptive because scalar-only GP-EI was required to answer P1.
The eventual final-E3 baseline plan below is unchanged, and no fourth method
was run.

The run started from local `main` commit
`5f7f7cfa66f4d6ec33ebf5a5feb3d8f77ba2110a`. The exact supplied handoff is
[`E3_PREFERENCE_BO_PILOT_HANDOFF.md`](E3_PREFERENCE_BO_PILOT_HANDOFF.md), and
the frozen machine-readable configuration is
[`../experiments/preference_bo/configs/minimal_pilot.json`](../experiments/preference_bo/configs/minimal_pilot.json)
with SHA-256
`4eafe727b67864632d25d30b526cbdb5c0a83989bf31e7b0cf41f7129c8a89da`.
Seeds were exactly 0--11, with six post-initial scalar evaluations per seed and
exactly the standard, full, and adaptive methods.

**P1 — passed.** Median $T_{0.10}$ was 3 for full preference-informed BO
and 7 for standard scalar GP-EI BO, so $3\le7-1$. The full per-seed values
were `[5, 6, 2, 2, 3, 7, 2, 3, 6, 2, 6, 3]`; standard was 7 for every seed.

**P2 performance — passed; P2 sparsity — failed.** Median $T_{0.10}$ was 3
for adaptive and 3 for full, satisfying $3\le3+1$. Median
$R_{\rm factors}$ was 0.875, exceeding the frozen 0.65 threshold. Adaptive
used 500 final-active factors over the $12\times6\times8=576$ available
decision/factor slots: $M_t=7$ in 68 of 72 decisions and $M_t=6$ in four.
The per-seed ratios were
`[0.875, 0.8541667, 0.875, 0.875, 0.875, 0.875, 0.875, 0.8541667, 0.875, 0.8333333, 0.875, 0.875]`.

**Inference diagnostics:** no frozen-cap numerical-accuracy failure occurred.
Full-target ESS fractions had minimum 0.999120 and median 0.999426; adaptive
working ESS fractions had minimum 0.999121 and median 0.999423; held-out full
ESS fractions had minimum 0.999096 and median 0.999418. Full inference used
65,536 draws except for three accuracy-triggered 131,072-draw calculations;
adaptive working inference used 16,384 total draws per refinement, and every
held-out full validation used 65,536 draws. Maximum adaptive held-out
full-target acquisition regret was $9.2476\times10^{-4}$, with median zero.
Exact held-out action agreement was 59/72 and is a diagnostic, not a gate.

**Failure diagnosis:** the main evidence points to conservative structural
influence bounds rather than inference error. Structural terms exceeded
empirical inference allowances in all 500 activation rounds; their medians
were 0.387317 and 0.00103655, respectively, while held-out full-target regret
remained far below the 0.05 screening tolerance. Consecutive active-set
turnover was correspondingly small (median 0, mean 0.00714); turnover was not
a pass condition.

**One-sentence scientific interpretation:** the frozen preference bank clearly
improved scalar-observation BO in this pilot, and adaptive conditioning
preserved that optimization benefit, but the prescribed structural screen
required nearly the full bank and therefore did not establish sparse
decision-specific conditioning.

Run outputs, including all raw trajectories, refinement histories, held-out
curves, generated banks/noise, provenance, result summary, and diagnostic, are
in
[`../experiments/preference_bo/outputs/minimal_pilot/`](../experiments/preference_bo/outputs/minimal_pilot/).
The schema-complete smoke output is in
[`../experiments/preference_bo/outputs/minimal_pilot_smoke/`](../experiments/preference_bo/outputs/minimal_pilot_smoke/).
An initial run stopped before gate evaluation on an Armijo roundoff bug and is
preserved in
[`../experiments/preference_bo/outputs/minimal_pilot_failed_attempt_001/`](../experiments/preference_bo/outputs/minimal_pilot_failed_attempt_001/);
the first completed run, whose CSV omitted separate held-out sample-count/ESS
fields, is also preserved in
[`../experiments/preference_bo/outputs/minimal_pilot_schema_incomplete_attempt_002/`](../experiments/preference_bo/outputs/minimal_pilot_schema_incomplete_attempt_002/).
The implementation-only fixes did not change the frozen configuration or gate
outcome. All 60 repository tests passed before the final execution, and every
final mechanical output check passed.

**Next action:** preserve `FAIL-P2`; do not tune or rerun this pilot. In a
separately preregistered follow-up, distinguish structural-bound conservatism
from genuinely broad decision dependence before deciding how to execute the
deferred final E3 baseline suite.

### Redundant-bank preference-BO pilot — 2026-08-21

**Status:** `FAILED-P2` under its independently preregistered precedence. This
was the last small synthetic preference-bank design iteration in this sequence.
It does not alter the first minimal pilot: that run remains `N=8`, P1 `PASS`,
P2 performance `PASS`, P2 sparsity `FAIL`, and overall `FAIL-P2` under its
original 0.65 threshold.

The redundant-bank run started from `main` commit
`b68525952f58693461ac32d4658a8d611a594706`. Its preregistration is
[`E3_PREFERENCE_BO_REDUNDANT_BANK_PILOT_HANDOFF.md`](E3_PREFERENCE_BO_REDUNDANT_BANK_PILOT_HANDOFF.md),
and its frozen configuration is
[`../experiments/preference_bo/configs/redundant_bank_pilot.json`](../experiments/preference_bo/configs/redundant_bank_pilot.json)
with SHA-256
`5c53c8de7144d1d0bc3a66b1ca4233b2c17811f726e977e1a5c90f103140b260`.
Seeds were exactly 0--11, the horizon was six post-initial scalar evaluations,
and exactly standard, full, and adaptive were run.

The bank contained exactly 24 unique, index-fixed edges: eight adjacent local
comparisons, eight four-index medium-range comparisons within the two operating
regimes, and eight mirrored cross-domain comparisons. Index 8 had degree zero;
every other grid index had degree three. With one scalar block per latent grid
coordinate, the derived prior comparison matrix had minimum eigenvalue
`3.728331299274427`, minimum row-dominance margin `3.4626638901604885`, and
condition number `4.971352073782589`; every regression and later-iteration SPD
check passed.

The 27 focused preference tests and the full 70-test repository suite passed
before execution. The one-seed/two-iteration reduced-count smoke ran all three
methods and passed every mechanical, sharing, schema, refinement, and held-out
nonleakage check; its scientific gates were not evaluated.

**P1 — passed.** Median $T_{0.10}$ was 2.5 for full preference-informed BO and
7 for standard scalar GP-EI, so $2.5\le7-1$. Full per-seed values were
`[3, 2, 4, 6, 1, 6, 4, 1, 1, 1, 3, 2]`; standard was 7 for every seed.

**P2 performance — passed; P2 sparsity — failed.** Median $T_{0.10}$ was 2.5
for adaptive and 2.5 for full, satisfying $2.5\le2.5+1$. Median
$R_{\rm factors}$ was `0.8888888889`, exceeding this follow-up's independently
frozen 0.80 threshold. Across 72 adaptive decisions, $M_t=18$ occurred 4
times, 19 occurred 5 times, 20 occurred 9 times, 21 occurred 12 times, 22
occurred 22 times, and 23 occurred 20 times. The median was 22 of 24 factors;
total final-active use was 1543 of 1728 available decision/factor slots.

**Post-run factor-count audit:** the pooled 18--23 distribution is primarily a
coherent BO-iteration trend rather than unexplained across-seed instability.
All 12 first decisions used 23 factors; iteration-two counts were 22--23;
iteration three used 22 factors in 11 of 12 seeds; iteration-four counts were
21--23; iteration-five counts were 19--22; and iteration-six counts were
18--21. As direct scalar observations accumulated, the median comparison-matrix
minimum eigenvalue rose monotonically from `4.81428` at iteration one to
`6.06857` at iteration six, tightening the omitted-factor propagation bound.
Accounting was exact: every reported $M_t$ equaled both the terminal active-set
size and the number of activation rounds for that decision, all 1543
activations selected a maximum-contribution omitted factor (three rounds had
numerically tied maxima), and all 72 decisions terminated through the frozen
screening tolerance. This diagnostic does not change the sparsity failure.

**Held-out and inference diagnostics:** no frozen-cap numerical-accuracy
failure occurred. Full-target ESS fractions had minimum `0.997291` and median
`0.998526`; adaptive working ESS fractions had minimum `0.997286` and median
`0.998502`; held-out full ESS fractions had minimum `0.997197` and median
`0.998508`. All Laplace calculations converged. The largest full and held-out
split-half EI discrepancies were `0.00239821` and `0.00203253`, respectively.
Maximum adaptive held-out full-target acquisition regret was `0.00261658`, with
median zero; exact held-out action agreement was 62/72 and was not a gate. Full
inference used 65,536 draws in 70 decisions and the allowed 131,072 draws in
two; every adaptive working refinement used 16,384 total draws and every
held-out validation used 65,536. Recorded full and adaptive inference wall
times were approximately 4.21 and 25.32 seconds, respectively, within a
30.06-second complete run; wall time was not a gate.

**Failure diagnosis:** only sparsity failed. Structural terms exceeded
empirical inference allowances in all 1543 activation rounds, with medians
`0.745311` and `0.000482393`, while held-out acquisition regret stayed far
below the 0.05 screening tolerance. The main evidence again points to a
conservative structural envelope rather than importance-sampling failure.
Median consecutive active-set turnover was `0.0434783` and is diagnostic only.

**Overall verdict: `FAIL-P2`.** The redundant bank retained a clear
preference-value benefit and adaptive conditioning retained its optimization
performance, but graph redundancy did not yield the preregistered material
certified screening fraction.

Final raw outputs and provenance are in
[`../experiments/preference_bo/outputs/redundant_bank_pilot/`](../experiments/preference_bo/outputs/redundant_bank_pilot/),
with the schema-complete mechanical smoke in
[`../experiments/preference_bo/outputs/redundant_bank_pilot_smoke/`](../experiments/preference_bo/outputs/redundant_bank_pilot_smoke/).
Two reporting/provenance-incomplete attempts for each profile are preserved
alongside them and are not used for the final gates. The identical gate values
across completed attempts verify that the reporting-only fixes did not change
the scientific result.

**One-sentence scientific interpretation:** preference information robustly
improved this scalar-observation BO problem and adaptive inference preserved
that benefit, but this conservative certified screen still required nearly the
entire overlapping bank.

**Next action:** do not tune another synthetic preference bank or change the
0.80 gate. Strong certified sparsity should not be treated as the main empirical
role of this preference example; the PDE experiments remain the stronger
setting for the sparsity/scaling claim. Decide separately whether the retained
preference-value result justifies moving directly to one realistic larger
preference-informed BO case.

### Why this experiment

A preference-conditioned objective is a natural BO problem with many non-Gaussian relational factors, and the useful preferences should change as the competitive region of the objective moves. It also creates a direct empirical comparison with historical decision-theoretic GP sparsification without making that older method the paper's framing.

### Candidate factor

For an offline comparison \(a_j\succ b_j\), use a smooth pairwise factor such as

\[
e_j(f)
=
\log\!\left(1+\exp\!\left[-\frac{f(a_j)-f(b_j)}{\tau}\right]\right),
\]

or the equivalent likelihood/energy form required by the chosen preference model.

### Experimental design

Run sequential BO with ordinary noisy objective observations plus a larger fixed bank of offline preference factors. At each BO iteration, rerun the active-conditioning procedure; do not freeze one factor subset for the whole BO trajectory.

Start in 1D/2D or another setting where the fully conditioned acquisition can be validated reliably. Scale factor count only after the basic phenomenon is clear.

### Hypothesis

At each BO round, only a small and changing subset of preferences materially affects the next query. Adaptive conditioning should match the full-conditioned BO trajectory to the target tolerance while processing fewer cumulative factors and maintaining easier active-target inference than full conditioning.

### Required baselines

1. full-factor conditioned BO with the same acquisition/inference family;
2. random factor activation;
3. preference factors local to the current leader/competitive region;
4. one fixed static subset chosen at the beginning;
5. posterior-oriented coreset/subsampling at matched factor budget;
6. the closest practical analogue of Valuable Vector Machine / decision-theoretic GP sparsification, with differences in approximation object made explicit.

Approximation-Aware BO is conceptually important related work but is not automatically a fair baseline unless the experiment also introduces an approximate GP representation such as an SVGP.

### Primary figure

Plot **simple regret versus cumulative conditioning-factor evaluations** across the sequential BO trajectory.

Supporting panels:

- active factor count by BO iteration;
- certified full-target acquisition regret by iteration;
- factor-set turnover as the BO decision moves;
- active/full inference cost or ESS.

### Metrics

- simple regret versus BO evaluations;
- simple regret versus cumulative factor evaluations/wall time;
- action agreement or full-target acquisition regret each round;
- active factor count and turnover;
- inference ESS/cost;
- certificate coverage where rigorous bounds are used.

### Pass criterion

The adaptive conditioning procedure tracks the fully conditioned BO performance within the intended action tolerance while using substantially fewer cumulative factor evaluations, and the selected factor set changes meaningfully with the BO state. At least one static/local/posterior-oriented baseline should require more factors or yield worse decisions at comparable cost.

### Failure statement

> **If this fails, we can no longer argue convincingly that adaptive conditioning is a genuinely sequential BO capability rather than a one-step screening device demonstrated mainly on structured PDE examples.**

### Compute

- first implementation and debugging: **MacBook Air, 16 GB**;
- larger factor-count and repeated-seed sweeps: **Colab A100**.

### Next E3 task (after the global certificate gate)

After the finite-grid end-to-end certificate and E1 upgrade in the global
execution order, build the smallest full-target-validatable sequential
preference-BO instance and first test only three methods: full conditioning,
adaptive conditioning, and a fixed local/static subset. Do not add the full
baseline suite until the adaptive factor set demonstrably changes across BO
iterations.

---

## E4 — Linear PDE control

**Status:** `EXISTING EVIDENCE / SUPPLEMENT`.

**Paper role:** show that the phenomenon is not dependent on a nonlinear residual; provide an easier controlled setting for bound diagnostics.

### Existing result

For the residual

\[
r_{ij}(f)=f_{ij}-c\sum_{(k,\ell)\in\mathcal N(i,j)}f_{k\ell}-b_{ij}
\]

and factor

\[
e_{ij}(f)=\gamma\log\cosh(r_{ij}(f)/\tau),
\]

the current 24 x 24 experiment reports:

- \(N=576\), \(M=50\);
- total empirical EI certificate: \(0.0563\);
- structural component: \(0.0398\);
- inference component: \(0.0246\);
- active-target GP-reference IS ESS: \(85.0\%\);
- full-target GP-reference IS ESS: \(12.0\%\);
- full-target Laplace-preconditioned IS ESS: \(69.4\%\);
- held-out full posterior selected the same BO action;
- in expanding-domain tests, \(M\) stayed roughly \(50\!\to\!60\) as \(N\) grew from 324 to 1600, while full-target GP-reference IS ESS fell from \(34.1\%\) to \(0.78\%\).

**Existing notebook:** `DEC_PDE_Certified_BO_Demo.ipynb`

### Use

Retain as a supplement control and as a debugging environment for proof-aligned influence constants. It should not consume a main-paper figure unless the nonlinear case becomes unreliable.

---

# Cross-cutting ablations and validation

## A1 — Factor-selection ablations

Run only after E1/E2/E3 have stable primary pipelines.

Required candidates:

- random;
- Euclidean/geometric local;
- graph local;
- static top sensitivity/influence;
- adaptive influence without challenger reoptimization;
- full adaptive method;
- cost-aware versus cost-unaware activation where relevant.

**Main question:** is the adaptive worst-challenger logic doing useful work beyond a static locality or sensitivity heuristic?

## A2 — Inference backend comparison

Purpose: establish modularity and show where active conditioning changes the sampling regime, not to claim sampler novelty.

Candidate backends:

- GP-reference importance sampling;
- Laplace-preconditioned importance sampling;
- elliptical slice sampling;
- HMC/NUTS;
- tempered SMC;
- FlowGP/LatentFlow if a stable implementation is available and informative.

Report action error and wall time in addition to ESS/mixing diagnostics.

## A3 — Continuous conditioning / adaptive quadrature

Optional. Include only if the continuous-factor formulation becomes important in the paper narrative. A small 1D example is enough to validate the integral form of the structural bound. Do not create a second application story around it.

---

# Certification validation protocol

For any experiment marketed as certified:

1. Fix a total failure probability \(\delta\).
2. State how confidence is allocated across action candidates and adaptive rounds.
3. Record the structural, inference, and optimization components separately.
4. Evaluate the returned action against a substantially stronger held-out full-target computation.
5. Across repeated trials, record
   \[
   \mathbf 1\{R_C(\widehat x)\le U_{\mathrm{cert}}\}
   \]
   and compare empirical coverage with the advertised level.
6. Never call ESS or a Monte Carlo standard error a rigorous certificate unless the assumptions that justify it have been established.

---

# Likely main-paper figures

## Figure 1 — Why all conditioning is not needed for one BO action

E1 symmetry mechanism: leader, challenger, activated relational factors, active/full acquisition, and certificate decomposition.

## Figure 2 — The relevant conditioning changes with the BO decision

E3 sequential preference BO: simple regret versus cumulative factor evaluations, active factors by BO round, and factor-set turnover.

## Figure 3 — The conditioning problem grows; the BO decision does not

E2 nonlinear PDE scaling: \(M\) versus \(N\), certified/observed regret, and active/full inference difficulty.

## Table 1 — Quantitative summary

Rows: symmetry, preference BO, nonlinear PDE.

Columns: total factors, active factors, \(N/M\), target \(\epsilon\), certified bound, observed full-target regret, active/full inference metric, wall-time or factor-evaluation reduction.

---

# Result logging contract for Codex

Every experiment directory should save both machine-readable results and a short human-readable summary. After a run, append/update the relevant experiment entry above with:

```text
Run ID / date:
Git commit:
Configuration file:
Seeds:
Result files:
Primary metric:
Certificate coverage:
Factor count M / total N:
Active inference metric:
Full inference metric:
Wall time:
Pass/fail verdict:
One-sentence interpretation:
Next action:
```

Do not record only screenshots or notebook output. The point of this file is to make manuscript figures and claims reconstructable later.

# Immediate execution order

1. **Completed:** rigorous finite-grid end-to-end reflection-symmetry certificate.
2. **Upgrade E1** into the polished mechanism/coverage experiment.
3. **Build E3** as the first genuinely sequential non-PDE BO demonstration.
4. **Expand E2** across source fields/BO states with the smallest necessary baseline set.
5. Only then run broad supplementary sampler and factor-selection ablations.
