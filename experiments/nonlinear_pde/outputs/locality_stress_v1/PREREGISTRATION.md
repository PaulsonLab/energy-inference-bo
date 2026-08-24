# E2 Locality Stress V1 Preregistration

Status: **FROZEN REPLACEMENT BEFORE PROSPECTIVE EXECUTION**

This replacement explicitly supersedes preregistration
`c976ab186a730e5ddfd2270ccf1d60577ee0d6b6`, which is
`SUPERSEDED_BEFORE_EXECUTION`. No prospective E2 seed was evaluated and no
scientific result was observed under that commit. The replacement changes only
pre-execution implementation/accounting details; all scientific states,
methods, budgets, and PASS/FAIL thresholds are unchanged.

This is a paper-level de-risking experiment for E2. It tests whether the fixed
nonlinear-PDE construction demonstrates challenger-aware decision-relevant
conditioning beyond the simpler explanation that residual factors are local.
It does not change the accepted PDE family or theory.

The machine-readable source of truth is `frozen_config.json`. Its SHA-256 is
recorded by every smoke/scientific output and will be reported with the
preregistration Git SHA.

## Locked scientific question and claims

The paired experiment tests:

- H1: final adaptive factor count remains strongly sublinear in total residual
  count across the archived expanding-domain sequence;
- H2: adaptive structural influence uses materially fewer factors than a
  dynamically updated, challenger-aware geometric-shell baseline at matched
  FULL action quality;
- H3: the complete adaptive decision remains cheaper than FULL at large sizes
  and the active target remains easier under the historical GP-reference IS
  diagnostic.

H1 alone does not establish adaptivity. If decision validity, scaling, compute,
and inference pass but H2 fails, the terminal verdict is
`PASS_SCALING_NOT_ADAPTIVITY`.

## Fixed mathematics

The experiment preserves `q0=3.5`, `q_laplacian=0.6`, coupling `c=0.12`,
nonlinearity `eta=0.25`, `gamma=0.08`, `tau=0.30`, the residual and factor
energy, all derivative bounds, the Menz comparison construction, and the
structural influence formula. It never materializes `A^{-1}`. The archived
24×24 structural value `0.03874403301354687` must reproduce before smoke or
scientific execution.

## Prospective design

The five domain sizes are `n = 18, 24, 30, 36, 40`, giving
`N = 324, 576, 900, 1296, 1600` residual factors. Each uses the three literal
prospective seeds:

`4215109622, 1083605379, 4045758625`.

They are derived by interpreting the first eight SHA-256 digest bytes of
`E2_LOCALITY_STRESS_V1:<replicate>` as a big-endian integer and reducing modulo
`2^32-1`. A repository search before freeze found no prior use as E2 scientific
seeds. Development uses only seed `2026082401`.

Each source field is the archived two-peak manufactured truth plus a centered,
unit-standard-deviation draw from the fixed Gaussian reference scaled by
`0.08`. The PDE source is recomputed so that this field satisfies every fixed
residual exactly. No accepted model constant changes.

One method-independent ordinary Gaussian-reference EI trajectory is built for
each domain/replicate. It uses no structural factors. Initialization contains
four deterministic center-first farthest-point actions, followed by ordinary
reference EI until 12 total queries. The three frozen snapshots are after
queries 4, 8, and 12, labeled early, middle, and late. Observations equal the
manufactured truth; the reference conditioning model uses noise variance
`0.0025`. The core contains exactly 45 paired states. Their serialized state
fingerprints must be identical across M0–M4.

For every frozen BO state, the single EI incumbent is
`max(0.55, max(state.observed_values))`. Exactly this state-specific value is
passed to FULL, ADAPTIVE_INFLUENCE, DYNAMIC_GEOMETRIC_SHELL,
STATIC_INFLUENCE, FIXED_CHALLENGER, both FULL-shadow batches,
oracle-geometric prefixes, and random matched subsets. The ordinary reference
trajectory already uses the same policy. Paired-method audits must contain one
identical incumbent, and later checkpoints update it whenever an improved
observation has occurred.

## Methods

- `FULL`: every residual factor; computational and action reference.
- `ADAPTIVE_INFLUENCE`: the existing structural-influence procedure. It starts
  empty, re-infers, recomputes the leader and worst optimistic challenger,
  reranks omitted theorem-backed contributions, and activates a batch.
- `DYNAMIC_GEOMETRIC_SHELL`: the primary skeptical baseline. It has the same
  inference, leader, current worst challenger, stopping check, batch budget,
  and fallback. Selection ranks omitted factors only by minimum four-neighbor
  graph distance from their actual residual support to the leader/challenger
  pair. Ties use ascending factor index. It never receives influence scores.
- `STATIC_INFLUENCE`: freezes the first influence ranking but retains the
  ordinary leader and worst-challenger stopping calculation.
- `FIXED_CHALLENGER`: recomputes influence against the current leader but
  freezes the first challenger, including the no-reoptimization stopping
  ablation. This is nonredundant with `STATIC_INFLUENCE` when the leader moves.
- `RANDOM_MATCHED_M`: ten deterministic random subsets at M1's final factor
  count for all replicates, early/late snapshots, and `n=24,40` only.
- `ORACLE_GEOMETRIC_PREFIX`: a post-decision, nondeployable diagnostic using
  the FULL action and strongest FULL challenger. It reports the smallest
  ten-factor geometric prefix reproducing the FULL action or reaching FULL
  acquisition regret at most `0.01`. It cannot feed M0–M4.

All selective methods start every BO state from an empty mask. Activation is
cumulative only within one decision. The batch size is 10, the maximum number
of refinement stages is 50, and an unresolved method activates all remaining
factors and records an explicit FULL fallback.

Every stage record describes the mask under which that stage's inference and
envelope were computed. `active_count` and `active_indices` are pre-activation;
`activated_indices` are applied afterward. On fallback, the transition stage
reports its partial pre-activation mask plus the remaining activated indices.
Only the following explicit FULL stage reports the FULL active mask.

## Stopping-tolerance provenance discrepancy

The accepted 24×24 procedure uses `epsilon = 0.060`. The archived notebook's
separate expanding-domain helper defaults to the looser `epsilon = 0.075`.
This discrepancy was resolved before prospective execution: the replacement
stress test uses the primary value `0.060`. The new scaling test is therefore a
stricter prospective test rather than an exact reproduction of the historical
`0.075` curve.

## Inference and FULL reference

Routine inference uses the same mechanical strategy for every method:
GP-reference self-normalized importance sampling first, followed by the
prototype's Laplace-proposal SNIS when reference ESS fraction is below `0.10`.
The archived direct gap influence-function standard error and action-grid
Bonferroni normal quantile form the empirical inference allowance. These are
diagnostics, not a rigorous finite-sample certificate.

Scientific routine budgets are 2,048 reference samples and 2,048 Laplace
samples. The proposal inflation is `1.10`; the MAP gradient tolerance is
`1e-8` with at most 12 Newton iterations. Dense Cholesky is permitted only for
the archived Laplace proposal, with its bytes recorded. Structural matrices
remain sparse, and no dense comparison inverse is permitted.

Every scientific state receives a held-out FULL shadow calculation using two
independent Laplace-proposal batches of 8,192 samples. Reliability requires:

- the two batch actions agree;
- each ESS fraction is at least `0.20`;
- maximum absolute acquisition-vector difference is at most `0.01`.

On failure, both batches are rerun once at 16,384 samples. Continued failure
labels the state `FULL_REFERENCE_UNRELIABLE`; it is never silently excluded.
More than 10% unreliable states forces `INCONCLUSIVE_FULL_REFERENCE`.

Factor-count, timing, factor-work, and ESS statistics use all states because
they do not require accurate FULL acquisitions. Every statistic involving a
FULL action or FULL acquisition regret uses only `FULL_REFERENCE_RELIABLE`
states. This includes action agreement and regret summaries and the
matched-quality components of A3 and A4. Unreliable states remain in raw
resource/count records and are never silently excluded. The 90% reliability
gate and every numerical threshold remain unchanged.

## Work, timing, and diagnostics

Every method/state records final factor indices, M/N, stages, fallback,
leader/challenger and bound decomposition per stage, action agreement and FULL
regret, factor-energy evaluations, factor-gradient elements, factor-Hessian
elements, sparse comparison solves, inference time, challenger time, complete
method time, separate shared setup, peak RSS, reference-proposal active/FULL
ESS, and Laplace-proposal diagnostics where used.

The computational comparison is the complete selective decision from empty
mask through all refinements versus one routine FULL decision. Shadow and
oracle diagnostics are excluded from both method times and recorded separately.

## Frozen statistics

For M1 at each domain size, report all nine states plus median and 20/80
percentiles for M, M/N, FULL regret, work, wall time, and ESS. Fit
`M ~ N^alpha` with the Theil-Sen slope of log per-size median M on log N and
display all five points. Also report median-M growth from n=18 to n=40 against
the total-factor ratio `1600/324`.

For every paired state, define
`R_geom = M_ADAPTIVE_INFLUENCE / M_DYNAMIC_GEOMETRIC_SHELL`, counting fallback
as N. Report median and geometric mean, win/tie/loss counts, win fraction, and
a 10,000-resample percentile bootstrap interval for the paired mean log ratio.
The bootstrap RNG seed is `2026082402`. Apply the same analysis to static and
fixed-challenger ablations. A factor-count win is not accepted at materially
worse FULL regret; the frozen matched-quality margin is `0.01` acquisition
units.

## Terminal verdicts

### A1 — decision validity

- reliable FULL reference for at least 90% of states;
- M1 exact FULL-action agreement at least 85% of reliable states;
- M1 regret 95th percentile at most `0.01` and maximum at most `0.03`;
- M1 fallback no more than 10% of reliable states.

### A2 — scaling

- `alpha <= 0.5`;
- n=40 median M1 active fraction at most `0.10`;
- n=40 80th percentile at most `0.15`.

### A3 — beyond dynamic geometry

- M1/M2 geometric-mean count ratio at most `0.80`;
- M1 paired win fraction at least `0.60`;
- bootstrap upper ratio strictly below `1.0`;
- M1 mean FULL regret no more than `0.01` worse than M2.

### A4 — reassessment contributes

At least one of STATIC or FIXED_CHALLENGER must have M1/ablation geometric-mean
factor-count ratio at most `0.90`, or suffer regret/fallback disadvantage by
more than the frozen quality rule in at least 20% of states.

### A5 — complete compute consequence

- at n=40, median cumulative M1/FULL factor-gradient work ratio at most `0.50`;
- at n=40, median complete wall-time ratio at most `0.80`;
- at n=36, median complete wall-time ratio at most `1.00`.

### A6 — inference separation

- at n=40, median active GP-reference ESS fraction at least `0.50`;
- at n=40, median active/FULL reference ESS ratio at least `3.0`.

All A1–A6 are required for `PASS_STRONG_E2`. If A1, A2, A5, and A6 pass but
A3 fails, the verdict is `PASS_SCALING_NOT_ADAPTIVITY`. Any other material
central-mechanism failure is `FAIL_E2_MECHANISM`, unless the FULL reliability
rule forces `INCONCLUSIVE_FULL_REFERENCE`.

No threshold, seed, state, method, or sampling budget may be changed after a
prospective result is inspected. A failed gate is preserved and not tuned.

## Development-only resource gate

The earlier reduced-fidelity smoke remains an implementation diagnostic, but
its synthetic multiplicative runtime projection is superseded. The replacement
resource decision uses only development seed `2026082401` and profiles n=40
early/late states at the actual scientific routine budgets, all five core
methods, the 50-stage allowance, and the initial two 8,192-sample FULL-shadow
batches. The total-run projection is computed directly from observed n=40
state wall time plus observed setup cost, with no synthetic fidelity/stage
multiplier. Exact profile results are recorded here before the replacement
commit.

The development-only profile completed in `14.3405 s` on the recorded arm64
Mac environment. Shared n=40 state construction took `0.1247 s`; peak RSS was
`1.5701 GB`. The measured state totals were `6.8858 s` (early) and `7.3205 s`
(late). Projecting 45 states plus 15 observed setup costs gives `5.3585 min`
from the early/late mean and `5.5215 min` from the slower observed state. This
is an observed-work projection with no synthetic fidelity or stage multiplier,
and it passes the frozen local resource gate. It is operational evidence only.

| Checkpoint | Method | Actual stages | Laplace escalation | M | Inference s | Challenger s | Complete method s |
|---|---|---:|---:|---:|---:|---:|---:|
| early | FULL | 1 | yes | 1600 | 0.3036 | 0.0000 | 0.3850 |
| early | ADAPTIVE_INFLUENCE | 6 | no | 50 | 0.0181 | 0.0551 | 0.0776 |
| early | DYNAMIC_GEOMETRIC_SHELL | 6 | no | 50 | 0.0183 | 0.0775 | 0.1003 |
| early | STATIC_INFLUENCE | 6 | no | 50 | 0.0170 | 0.0342 | 0.0554 |
| early | FIXED_CHALLENGER | 5 | no | 40 | 0.0134 | 0.0445 | 0.0623 |
| late | FULL | 1 | yes | 1600 | 0.3012 | 0.0000 | 0.3878 |
| late | ADAPTIVE_INFLUENCE | 15 | no | 140 | 0.0738 | 0.1434 | 0.2219 |
| late | DYNAMIC_GEOMETRIC_SHELL | 16 | no | 150 | 0.0839 | 0.2139 | 0.3024 |
| late | STATIC_INFLUENCE | 23 | no | 220 | 0.1621 | 0.1228 | 0.2900 |
| late | FIXED_CHALLENGER | 3 | no | 20 | 0.0061 | 0.0242 | 0.0346 |

The initial `8192 x 2` FULL-shadow attempts took `2.1140 s` early and
`1.9437 s` late in summed inference time. Both development states exercised
the frozen `16384 x 2` escalation; the complete shadow paths took `5.9570 s`
and `5.8121 s`, respectively. Both remained labeled unreliable under the
unchanged FULL-reference rules. That development-only diagnostic observation
does not enter a scientific verdict and does not modify the 90% reliability
gate, sampling budgets, model, or any PASS/FAIL threshold. Exact method work
and every stage timing are frozen in
`development_full_fidelity_method_metrics.csv` and
`development_full_fidelity_stage_metrics.csv`.

## Leakage and fairness invariants

- Deployable methods see the latent field only through frozen BO observations.
- FULL shadow acquisitions never enter M1–M4 factor selection or stopping.
- Oracle geometry is post-decision and one-way.
- Random-subset seeds are independent of source seeds.
- Every refinement stage is included in method work/time.
- Common setup is separate and identical across methods.
- Omitted factor energies never enter theorem-backed screening.
- E3 seeds and outputs are untouched.

## Required outputs

The completed scientific run writes `summary.json`, `state_metrics.csv`,
`stage_metrics.csv`, `method_summary.csv`, `resource_metrics.csv`, `RESULTS.md`,
and the four frozen diagnostic figures, in addition to raw shadow, oracle, and
random-baseline records. Every result records the preregistration SHA, run SHA,
frozen-config SHA-256, machine environment, exact terminal verdict, and every
criterion as PASS/FAIL.
