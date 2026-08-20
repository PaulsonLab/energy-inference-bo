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
| E1 | Nonlocal reflection symmetry | Mechanism + certificate figure | EXISTING EVIDENCE; needs rigorous repeat | MacBook Air |
| E2 | Nonlinear PDE expanding-domain scaling | Main scaling / inference consequence | EXISTING EVIDENCE; needs robustness + baselines | Colab A100 |
| E3 | Preference-conditioned sequential BO | Main end-to-end non-PDE BO test | PLANNED | MacBook Air / A100 sweeps |
| E4 | Linear PDE factor graph | Supplementary control | EXISTING EVIDENCE | MacBook Air / A100 sweeps |
| A1 | Factor-selection ablations | Supplement / supports E1--E3 | PLANNED | Mixed |
| A2 | Inference/sampler comparison | Supplement / modularity | PLANNED | A100 |
| A3 | Continuous-factor/quadrature example | Optional | NOT REQUIRED | MacBook Air |

---

## E1 — Nonlocal reflection symmetry

**Status:** `EXISTING EVIDENCE` — mechanism is established; final version needs repeated seeds, rigorous inference accounting, and baselines.

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

### Next Codex task

Convert the existing notebook into a reproducible seeded experiment that saves per-round leader/challenger, active set, bound decomposition, full-target validation, and baseline results in a tidy result table.

---

## E2 — Nonlinear PDE expanding-domain scaling

**Status:** `EXISTING EVIDENCE` — strongest current evidence for a decision-conditioning/inference separation; needs proof-aligned robustness and selection baselines.

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

### Existing result: 24 x 24 case

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

### Next Codex task

Refactor the scaling experiment to run multiple source fields/BO states per \(N\), add the fixed-local and random baselines first, and save a single summary CSV/JSON containing \(N,M\), regret/certificate, ESS, wall time, and selected-factor geometry.

---

## E3 — Preference-conditioned sequential BO

**Status:** `PLANNED` — main missing experiment.

**Paper claim tested:** C1--C3 in an end-to-end BO loop; establishes relevance beyond PDE/physics examples.

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

### Next Codex task

Build the smallest full-target-validatable sequential preference-BO instance and first test only three methods: full conditioning, adaptive conditioning, and a fixed local/static subset. Do not add the full baseline suite until the adaptive factor set demonstrably changes across BO iterations.

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

1. **Resolve structural-bound validity** for the symmetry and nonlinear-PDE constructions.
2. **Implement one rigorous end-to-end certificate** on a finite action set/grid.
3. **Upgrade E1** into the polished mechanism/coverage experiment.
4. **Build E3** as the first genuinely sequential non-PDE BO demonstration.
5. **Expand E2** across source fields/BO states with the smallest necessary baseline set.
6. Only then run broad supplementary sampler and factor-selection ablations.
