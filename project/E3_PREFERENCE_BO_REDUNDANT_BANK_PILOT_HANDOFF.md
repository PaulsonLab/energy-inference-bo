# E3 Preference-BO Redundant-Bank Pilot Handoff

## Status and scope

This document prospectively freezes the final small synthetic preference-bank
follow-up. It is separate from the completed eight-factor minimal pilot.

- First minimal pilot: `N = 8`, P1 `PASS`, P2 performance `PASS`, P2 sparsity
  `FAIL`, overall `FAIL-P2`.
- This follow-up: `N = 24`, independently preregistered gates and outputs.

The first result must not be overwritten, relabeled, averaged with this run, or
used to tune this configuration. This is not a paper redesign, a sampler
contribution, or the deferred final E3 baseline suite.

## Scientific question

> If a modestly larger historical preference bank contains natural graph
> redundancy and overlap, can full preference conditioning retain its clear BO
> benefit while adaptive decision-relevant conditioning leaves a material
> fraction of the bank unused?

Run exactly standard scalar GP-EI, full preference-informed GP-EI, and adaptive
preference-informed GP-EI. No other baseline or scientific parameter sweep is
permitted.

## Frozen BO problem

All underlying settings are copied from the first pilot:

- maximize on the 17-point grid `x_i = i/16`, `i = 0,...,16`;
- true objective
  \[
  f(x)=0.55e^{-\frac12((x-0.22)/0.07)^2}
       +e^{-\frac12((x-0.76)/0.065)^2}
       +0.04\sin(4\pi x);
  \]
- initial scalar indices `[2, 8, 14]` and six post-initial evaluations;
- location-indexed scalar noise standard deviation `0.01`, pregenerated and
  shared across methods within a seed;
- zero-mean OU/Matérn-1/2 GP with amplitude `0.10`, length scale `0.05`, scalar
  noise standard deviation `0.01`, and no hyperparameter fitting;
- ordinary analytic expected improvement with the largest noisy observation as
  incumbent and exhaustive search over unobserved grid actions;
- seeds `0,...,11`;
- preference temperature `tau_pref = 0.60`;
- preference labels generated independently before any method from the same
  logistic noisy ground-truth model and shared across methods;
- adaptive active set reset to empty at every BO iteration, one factor activated
  per failed refinement, screening tolerance `0.05`;
- held-out full-target calculations are validation only.

Online scalar observations define the ordinary GP reference posterior and are
never screened. Only the fixed historical preference bank is screened.

## Frozen 24-edge preference bank

Use these exact distinct index pairs in this exact order.

Local protocol:

```text
(0,1), (2,3), (4,5), (6,7),
(9,10), (11,12), (13,14), (15,16)
```

Medium-range protocol:

```text
(0,4), (1,5), (2,6), (3,7),
(9,13), (10,14), (11,15), (12,16)
```

Cross-domain/mirrored protocol:

```text
(0,16), (1,15), (2,14), (3,13),
(4,12), (5,11), (6,10), (7,9)
```

Mechanically require 24 unique edges, degree 3 at every index except index 8,
and degree 0 at index 8. The graph is fixed from indices alone. It may not
depend on objective values, realized labels, BO state, or the optimum.

For edge `j = (a_j,b_j)`, label `s_j` satisfies

\[
\Pr(s_j=+1)=\sigma((f(a_j)-f(b_j))/\tau_{\rm pref}),
\]

and the labels are generated independently.

## Preference factor and scalar-block influence

Let `d_j = e_{a_j} - e_{b_j}` and
`z_j = s_j d_j^T Y / tau_pref`. The stable energy is

\[
e_j(Y)=\log(1+\exp(-z_j)).
\]

With `q_j = sigma(-z_j)`, use the exact derivatives

\[
\nabla e_j=-\frac{s_jq_j}{\tau_{\rm pref}}d_j,
\qquad
\nabla^2e_j=\frac{q_j(1-q_j)}{\tau_{\rm pref}^2}d_jd_j^T\succeq0.
\]

Use 17 scalar blocks, one for every latent coordinate. The factor sensitivity
is `1/tau_pref` at each endpoint and zero elsewhere. One edge contributes at
most `1/(4 tau_pref^2)` in magnitude to its endpoint cross-coordinate Hessian.

For scalar-data GP precision `Q_t`, define

\[
A_{ii}(t)=Q_{t,ii},
\qquad
A_{ik}(t)=-\left(|Q_{t,ik}|+\frac{n_{ik}}{4\tau_{\rm pref}^2}\right),
\quad i\ne k,
\]

where `n_ik` is the number of frozen edges joining coordinates `i` and `k`.
This single comparison matrix is uniform over all active sets and the
active-to-full interpolation.

The frozen prior regressions, derived rather than hard-coded, are:

```text
minimum eigenvalue(A_0)       = 3.7283312993 approximately
minimum row-dominance margin = 3.4626638902 approximately
condition number(A_0)        = 4.9713521 approximately
```

If these checks fail materially, stop before scientific execution. Direct
scalar observations add nonnegative diagonal precision and must preserve SPD.

For distinct EI leader and challenger coordinates, the footprint is 1 at each
coordinate and zero elsewhere. Self-comparison has zero footprint. For omitted
factors `U`, use

\[
h_U=\sum_{j\in U}L(e_j),
\qquad
B_{\rm struct}=L(F)^T A_t^{-1}h_U,
\]

via a linear solve. Rank an omitted factor for the current worst challenger by

\[
b_j=L(e_j)^T A_t^{-1}L(F_{\rm worst}).
\]

Never form an explicit inverse on the runtime structural path.

## Numerical inference

Reuse the existing Laplace-preconditioned self-normalized importance sampler.
The mode uses float64, gradient infinity-norm tolerance `1e-9`, at most 50
Newton iterations, and backtracking.

Adaptive working inference starts with two independent batches of 8,192 draws.
For every acquisition gap,

\[
B_{\rm infer}=\max\{2\widehat{\rm SE}_{\rm pooled},
|\widehat G^{(1)}-\widehat G^{(2)}|\}.
\]

If ESS fraction is below `0.25` or worst-challenger `B_infer` exceeds `0.01`,
double each batch to 16,384 and then 32,768 if needed.

Full-method and independent adaptive held-out calculations start at 65,536
draws. Increase only to 131,072 if ESS fraction is below `0.25` or maximum
split-half EI discrepancy exceeds `0.0025`. These are empirical numerical
diagnostics, not rigorous finite-sample certificates.

## Frozen smoke test

Before scientific execution, run seed 0 for two BO iterations and all three
methods with working batches `[512, 1024, 2048]` and full schedules
`[4096, 8192]`. The smoke test is mechanical; scientific gates are not
evaluated and its results may not motivate parameter changes.

## Frozen metrics and gates

For each seed and method,

\[
T_{0.10}=\min\{t:\text{finite-grid simple regret after }t
\text{ post-initial evaluations}\le0.10\},
\]

with value 7 if unreached during the six-step horizon.

P1 passes iff

\[
\operatorname{median}T_{0.10}^{\rm full}
\le\operatorname{median}T_{0.10}^{\rm standard}-1.
\]

P2 performance passes iff

\[
\operatorname{median}T_{0.10}^{\rm adaptive}
\le\operatorname{median}T_{0.10}^{\rm full}+1.
\]

For each adaptive seed,

\[
R_{\rm factors}=\frac{\sum_{t=1}^6M_t}{6\times24}.
\]

P2 sparsity passes iff median `R_factors <= 0.80`. This requires at least 20%
of the bank to remain unused on average in the median seed. Exact action
agreement, active-set turnover, and wall-time advantage are diagnostics only.

Verdict precedence is exactly:

```text
if P1 fails: FAIL-P1
elif P2 performance fails or P2 sparsity fails: FAIL-P2
else: PASS
```

No threshold may change after results are observed.

## Output and decision contract

Write a new immutable directory
`experiments/preference_bo/outputs/redundant_bank_pilot/` containing the frozen
configuration, banks, scalar noise, trajectories, refinement history,
acquisition validation, summary, results Markdown, one two-panel diagnostic,
and provenance. Preserve raw intermediate and negative results.

Record edge metadata/degrees, labels, action/observation histories, simple
regret, `T_0.10`, active sets, `M_t/N`, cumulative `R_factors`, refinement
leaders/challengers/contributions/bounds, inference diagnostics, likelihood
evaluation counts, wall times, held-out full-target acquisition regret, and
active-set turnover.

If `PASS`, recommend moving to one realistic larger preference-informed BO
case without more synthetic tuning. If `FAIL-P1`, preserve the loss of
preference value and do not retune. If `FAIL-P2`, identify whether performance
or sparsity failed, do not redesign another synthetic bank, and record that the
PDE experiments remain the stronger sparsity/scaling setting.
