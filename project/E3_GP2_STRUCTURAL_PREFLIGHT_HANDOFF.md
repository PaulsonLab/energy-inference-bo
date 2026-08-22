# E3 Gp2 Structural Preflight Handoff

## Purpose

This document freezes **only the first go/no-go gate** for the Gp2 realistic E3 case study in the paper *Decision-Relevant Conditioning for Bayesian Optimization*.

Project repository:

`https://github.com/PaulsonLab/energy-inference-bo`

Branch:

`main`

This handoff was designed against:

`6ca1171ae13511b081f4a9c79c112378d4119357`

External data repository:

`HackelLab-UMN/DevRep`

Pinned external commit:

`e05023a8abe7be6c2e22f42d523b20bd76cd8da5`

The previous Gp2 pairwise-preference P1 preprocessing gate is a **valid negative result and must remain unchanged**.

This new gate asks exactly one question:

> **Does a scientifically natural candidate-local proxy-conditioning formulation admit materially sparse theorem-backed decision influence on the real Gp2 sequence graph?**

This task stops after answering that question.

Do **not** implement Bayesian optimization, Laplace inference, importance sampling, adaptive conditioning, or the final scalar-only versus full-proxy comparison.

If this structural gate fails, abandon Gp2 as E3 rather than tuning or redesigning it.

---

# 1. Read first

Before implementation, read current committed versions of:

1. `project/PROJECT_HANDOFF.md`
2. `project/PAPER_STORY.md`
3. `project/THEORY.tex`
4. `project/EXPERIMENTS.md`
5. `project/E3_GP2_P1_GATE_HANDOFF.md`
6. `experiments/gp2_preference_bo/configs/p1_gate.json`
7. `experiments/gp2_preference_bo/outputs/preflight/PREPROCESSING_INVALID.json`
8. `experiments/gp2_preference_bo/outputs/preflight/preprocessing_summary.json`
9. `src/conditioned_bo/gp2_preference_bo.py`

The paper story and T1--T4 are locked.

Do not modify the previous Gp2 P1 result.

---

# 2. Scientific interpretation

One potential BO action is one held-out full-length Gp2 `Paratope` sequence for an expensive SH recombinant-yield measurement.

Scalar target:

`SH_Average_bc`

High-throughput assays are treated as **already collected proxy information** about latent SH yield.

Do not convert assays to pairwise preferences.

Historical data calibrate a proxy-to-yield relationship. A held-out candidate with the required high-throughput proxy measurements contributes one **candidate-local non-Gaussian factor** on its latent yield.

Missing proxy measurements mean **no factor for that candidate**.

Missing proxy measurements must never remove an otherwise valid BO action.

---

# 3. Frozen source split

Use exactly these pinned DevRep files:

Historical calibration:

`datasets/assay_to_yield_training_sequences.csv`

Prospective action library:

`datasets/test_sequences.csv`

Do **not** union these files for the action set.

The historical file may be used for:

- target standardization;
- proxy calibration;
- proxy residual-scale estimation.

The held-out test file may be used during this preflight only for:

- `Stop`;
- `Paratope`;
- whether `SH_Average_bc` is finite;
- proxy assay availability and magnitudes;
- graph construction;
- factor construction;
- structural calculations.

The **magnitude** of held-out `SH_Average_bc` must not be used, printed, saved, ranked, plotted, correlated with proxies, or otherwise inspected in this task.

The finite-target check is permitted only so every retained action has a retrospective oracle value for a later experiment.

---

# 4. Frozen assays

Use exactly:

- `Sort1_mean_score`
- `Sort8_mean_score`

Do not add Sort10 or another assay.

These were already selected before the previous Gp2 gate failed, so retaining them avoids post-failure assay shopping.

---

# 5. Historical target standardization

From `assay_to_yield_training_sequences.csv`, retain a historical row for target scaling iff:

1. `Stop == False`;
2. `Paratope` is present and nonempty;
3. `SH_Average_bc` is finite.

Resolve exact duplicate `Paratope` rows deterministically.

If duplicate rows disagree on gate-used values beyond:

- `rtol = 1e-12`
- `atol = 1e-12`

stop with:

`PREPROCESSING_AMBIGUITY`

Do not average inconsistent targets.

Require all retained historical paratopes to have one common length.

Define:

\[
\mu_{\mathrm{hist}}
=
\operatorname{mean}(y_{\mathrm{hist}})
\]

and

\[
s_{\mathrm{hist}}
=
\max\{
\operatorname{sd}(y_{\mathrm{hist}};\mathrm{ddof}=1),
10^{-6}
\}.
\]

Define standardized latent yield:

\[
z
=
\frac{f-\mu_{\mathrm{hist}}}{s_{\mathrm{hist}}}.
\]

This standardization is frozen for all later Gp2 work.

---

# 6. Frozen historical proxy calibration

From the valid historical target-scale rows, retain calibration rows requiring finite:

- `Sort1_mean_score`;
- `Sort8_mean_score`.

Require:

\[
N_{\mathrm{calibration}}\ge100.
\]

Otherwise:

`PREPROCESSING_INVALID`

Use exactly the scikit-learn pipeline:

```python
Pipeline([
    ("scale", StandardScaler()),
    ("ridge", Ridge(alpha=1.0, fit_intercept=True)),
])
```

Features, in exact order:

1. `Sort1_mean_score`
2. `Sort8_mean_score`

Response:

\[
z_{\mathrm{hist}}
=
\frac{y_{\mathrm{hist}}-\mu_{\mathrm{hist}}}
{s_{\mathrm{hist}}}.
\]

No hyperparameter search.

## OOF residual scale

Use exactly:

```python
KFold(
    n_splits=10,
    shuffle=True,
    random_state=0,
)
```

Fit the full scaler+ridge pipeline separately inside every fold.

Generate one OOF prediction per calibration row.

Define:

\[
\mathrm{RMSE}_{\mathrm{OOF}}
=
\sqrt{
\frac1{N_{\mathrm{calibration}}}
\sum_i
(z_i-\widehat z_i^{\mathrm{OOF}})^2
}.
\]

Define:

\[
s_{\mathrm{proxy}}
=
\frac{\sqrt3}{\pi}
\mathrm{RMSE}_{\mathrm{OOF}}.
\]

Require finite:

\[
s_{\mathrm{proxy}}>10^{-6}.
\]

Otherwise:

`PREPROCESSING_INVALID`

Finally fit the same scaler+ridge pipeline on the complete historical calibration subset.

Do not use held-out test target magnitudes for calibration.

---

# 7. Prospective action set

Start from `datasets/test_sequences.csv` only.

Retain a potential action iff:

1. `Stop == False`;
2. `Paratope` is present and nonempty;
3. `SH_Average_bc` is finite.

Do not require any assay value.

Resolve exact duplicate `Paratope` rows deterministically.

Gate-used duplicate fields are:

- `Stop`
- `Paratope`
- `SH_Average_bc`
- `Sort1_mean_score`
- `Sort8_mean_score`

If duplicate rows disagree beyond `rtol=1e-12`, `atol=1e-12`, stop:

`PREPROCESSING_AMBIGUITY`

Assert that no final held-out `Paratope` appears in the historical target-scale set.

If overlap exists:

`PREPROCESSING_AMBIGUITY`

Require held-out paratopes to have the same sequence length as the historical paratopes.

Sort final candidates lexicographically by `Paratope` before assigning canonical action indices.

---

# 8. Frozen sequence graph

Use the same target-blind graph construction as the previous Gp2 gate.

Distance:

unweighted Hamming distance over the full `Paratope` string.

Literal `X` is an ordinary symbol.

Set:

`k = 8`

For each candidate, choose exactly eight nearest other candidates.

Tie-break:

1. smaller Hamming distance;
2. lexicographically smaller neighbor `Paratope`;
3. smaller canonical candidate index.

Create the undirected graph as the union of directed 8-NN choices.

No target or assay magnitude may influence graph construction.

## Component rule

If connected, use unchanged.

If disconnected, retain the largest connected component only if it contains at least 90% of pre-component actions, then re-sort/re-index.

Otherwise:

`PREPROCESSING_INVALID`

Require final:

\[
N_{\mathrm{actions}}\ge150.
\]

Otherwise:

`PREPROCESSING_INVALID`

---

# 9. Frozen candidate-local proxy factors

For every final action \(i\) with finite values of both:

- `Sort1_mean_score`
- `Sort8_mean_score`

apply the already fitted historical calibration pipeline to obtain:

\[
\mu_i^{\mathrm{proxy}}.
\]

Create exactly one factor:

\[
e_i(z_i)
=
2\log\cosh\!\left(
\frac{
z_i-\mu_i^{\mathrm{proxy}}
}{
2s_{\mathrm{proxy}}
}
\right).
\]

No factor weight.

No candidate-specific scale.

No target-agreement filtering.

No pairwise preference factors.

Canonical factor order is increasing action index.

Require:

\[
N_{\mathrm{factors}}\ge75
\]

and:

\[
\frac{N_{\mathrm{factors}}}
{N_{\mathrm{actions}}}
\ge0.40.
\]

Otherwise:

`PREPROCESSING_INVALID`

---

# 10. Frozen graph-Gaussian reference

Let \(W\) be the binary adjacency matrix and:

\[
D_{ii}=\sum_jW_{ij}.
\]

Define normalized adjacency:

\[
S_G=D^{-1/2}WD^{-1/2}.
\]

Then:

\[
L_{\mathrm{sym}}=I-S_G
\]

and:

\[
Q_0
=
I+L_{\mathrm{sym}}
=
2I-S_G.
\]

Define:

\[
K_0=Q_0^{-1}.
\]

No graph or precision hyperparameter optimization.

A later scalar observation model will use fixed standardized noise SD \(0.05\), but no BO posterior needs to be implemented in this task.

---

# 11. Theory specialization to implement, not rediscover

Do not search for a new theorem.

Use one scalar latent yield \(z_i\) as one block in the existing Menz construction.

For a local proxy factor:

\[
e_i'(z_i)
=
\frac1{s_{\mathrm{proxy}}}
\tanh\!\left(
\frac{
z_i-\mu_i^{\mathrm{proxy}}
}{
2s_{\mathrm{proxy}}
}
\right),
\]

so:

\[
|e_i'(z_i)|
\le
\frac1{s_{\mathrm{proxy}}}.
\]

Also:

\[
e_i''(z_i)
=
\frac1{2s_{\mathrm{proxy}}^2}
\operatorname{sech}^2\!\left(
\frac{
z_i-\mu_i^{\mathrm{proxy}}
}{
2s_{\mathrm{proxy}}
}
\right)
\ge0.
\]

Thus local factors only improve one-block curvature and create no cross-coordinate Hessian terms.

For the prior/reference state:

\[
\rho_i=(Q_0)_{ii},
\]

\[
\kappa_{ij}=|(Q_0)_{ij}|,
\qquad i\ne j.
\]

Since \(Q_0\) has nonpositive off-diagonal entries:

\[
A_{ii}=(Q_0)_{ii},
\]

\[
A_{ij}
=
-|(Q_0)_{ij}|
=
(Q_0)_{ij}.
\]

Therefore:

\[
\boxed{A=Q_0}
\]

and:

\[
\boxed{C=A^{-1}=K_0}.
\]

This is the model-specific influence construction.

If implementation shows this specialization is false for the frozen graph/factors, stop:

`THEORY_INSTANTIATION_INVALID`

Do not invent a replacement construction.

---

# 12. Required theory regression checks

Verify numerically:

1. \(Q_0\) is symmetric positive definite.
2. Off-diagonal elements of \(Q_0\) are nonpositive to tolerance.
3. \(K_0\) is entrywise nonnegative to tolerance.
4. The implemented Menz comparison matrix equals \(Q_0\) to tolerance.
5. Local factor energy/gradient/Hessian agree with finite differences.

Use tolerance:

`1e-10`

for matrix identities/inequalities where appropriate.

## Graph-distance check

Because:

\[
Q_0=2I-S_G,
\]

\[
K_0
=
\frac12
\sum_{r=0}^{\infty}
\left(
\frac{S_G}{2}
\right)^r,
\]

and graph paths shorter than \(d_G(i,j)\) cannot connect \(i\) and \(j\),

\[
0\le(K_0)_{ij}
\le
2^{-d_G(i,j)}.
\]

Check this numerically for all action pairs to tolerance `1e-10`.

A material violation is:

`THEORY_INSTANTIATION_INVALID`

This is a regression check, not a tunable gate.

---

# 13. Structural contribution

For an action comparison \((x,\widehat x)\), use the conservative EI sensitivity:

\[
L_i(F_{x,\widehat x})
\le
\mathbf 1\{i=x\}
+
\mathbf 1\{i=\widehat x\}.
\]

For a factor attached to action \(a(j)\):

\[
L_i(e_j)
=
\frac1{s_{\mathrm{proxy}}}
\mathbf 1\{i=a(j)\}.
\]

Therefore define the frozen contribution:

\[
\boxed{
c_j(x,\widehat x)
=
\frac{
K_{0,x,a(j)}
+
K_{0,\widehat x,a(j)}
}{
s_{\mathrm{proxy}}
}
}
\]

for every factor \(j\).

No empirical acquisition calculations are used in this preflight.

---

# 14. Frozen structural-sparsity gate

Set:

\[
\epsilon_{\mathrm{struct}}=0.05.
\]

For every **unordered pair of distinct actions**:

\[
x<\widehat x,
\]

compute all factor contributions \(c_j(x,\widehat x)\).

Sort descending:

\[
c_{(1)}
\ge
c_{(2)}
\ge
\cdots
\ge
c_{(N_f)}.
\]

Define \(M_{0.05}(x,\widehat x)\) as the smallest \(m\in[0,N_f]\) satisfying:

\[
\sum_{k=m+1}^{N_f}c_{(k)}
\le0.05.
\]

Define:

\[
R_{0.05}(x,\widehat x)
=
\frac{
M_{0.05}(x,\widehat x)
}{
N_f
}.
\]

The structural gate passes iff:

\[
\boxed{
\frac{
\#\{
(x,\widehat x):
x<\widehat x,\,
R_{0.05}(x,\widehat x)\le0.50
\}
}{
\binom{N_{\mathrm{actions}}}{2}
}
\ge0.90.
}
\]

Interpretation:

> For at least 90% of all possible action comparisons, the theorem must permit at least half of the complete proxy bank to be omitted while leaving at most 0.05 conservative structural acquisition-gap uncertainty.

This criterion is frozen.

If it fails:

`FAIL_STRUCTURAL_SPARSITY`

and Gp2 is abandoned as E3.

Do not run BO and do not tune this construction.

---

# 15. Diagnostics to save

Save, but do not gate on:

- fraction of action pairs with \(R_{0.05}\le0.25\);
- fraction with \(R_{0.05}\le0.50\);
- median \(R_{0.05}\);
- 75th percentile;
- 90th percentile;
- 95th percentile;
- graph degree summary;
- graph diameter if cheap;
- graph-distance distribution of the highest-contribution factor per action pair;
- factor coverage fraction.

Produce one target-blind ECDF or histogram of \(R_{0.05}\) with guide lines at:

- factor fraction `0.50`;
- pair fraction `0.90`.

No target values may appear in this figure.

---

# 16. Outcome precedence

Use exactly:

```text
if preprocessing is ambiguous:
    PREPROCESSING_AMBIGUITY
elif preprocessing validity conditions fail:
    PREPROCESSING_INVALID
elif theory/regression checks fail:
    THEORY_INSTANTIATION_INVALID
elif structural sparsity criterion fails:
    FAIL_STRUCTURAL_SPARSITY
else:
    PASS_STRUCTURAL_PREFLIGHT
```

Only:

`PASS_STRUCTURAL_PREFLIGHT`

allows a later Gp2 session.

Every other outcome means Gp2 is abandoned.

---

# 17. Implementation scope

Create:

- `project/E3_GP2_STRUCTURAL_PREFLIGHT_HANDOFF.md`
- `experiments/gp2_proxy_bo/README.md`
- `experiments/gp2_proxy_bo/configs/structural_preflight.json`
- `experiments/gp2_proxy_bo/run_structural_preflight.py`
- `src/conditioned_bo/gp2_proxy_bo.py`
- `tests/test_gp2_proxy_bo.py`

Do not implement BO or inference methods.

Reuse narrowly scoped existing Gp2 utilities if helpful, but do not refactor unrelated code.

Do not alter old Gp2 result files.

---

# 18. Required tests before real preflight

At minimum test:

1. historical/test source-role separation;
2. held-out target magnitudes are absent from graph/factor/structural interfaces;
3. missing proxy does not delete an action;
4. duplicate ambiguity behavior;
5. deterministic historical scaling;
6. deterministic 10-fold calibration and \(s_{\mathrm{proxy}}\);
7. deterministic Hamming graph;
8. normalized-Laplacian precision;
9. factor gradient/Hessian finite differences;
10. \(A=Q_0\);
11. \(K_0\ge0\) entrywise to tolerance;
12. graph-distance covariance bound;
13. a hand-checkable structural-tail / \(M_{0.05}\) fixture;
14. immutable output handling.

Run focused tests and then the full repository test suite.

Implementation bugs may be fixed.

Frozen scientific choices may not be changed because of observed data.

---

# 19. Preregistration discipline

Before running the real Gp2 preflight:

1. finish implementation;
2. finish tests;
3. commit handoff, config, implementation, and tests;
4. record exact preregistration Git SHA;
5. record SHA-256 of `structural_preflight.json`.

Only then run the real data preflight.

Do not inspect the real \(R_{0.05}\) result and then change thresholds, calibration, assays, graph degree, or factor construction.

If an implementation-only bug is found after preregistration:

- preserve the failed attempt;
- fix only the implementation bug;
- rerun tests;
- create a new preregistration commit;
- rerun the real preflight from scratch;
- document the implementation-only correction.

---

# 20. Required outputs

Create immutable:

`experiments/gp2_proxy_bo/outputs/structural_preflight/`

containing at minimum:

- `frozen_config.json`
- `provenance.json`
- `preprocessing_summary.json`
- `calibration_summary.json`
- `graph_edges.csv`
- `proxy_factor_bank.csv`
- `structural_pairwise_sparsity.csv`
- `structural_sparsity_summary.json`
- `RESULTS.md`
- one target-blind structural diagnostic figure

Report:

- project preregistration SHA;
- config SHA-256;
- external source hashes;
- historical target-scale count;
- calibration count;
- \(\mu_{\mathrm{hist}}\);
- \(s_{\mathrm{hist}}\);
- OOF RMSE;
- \(s_{\mathrm{proxy}}\);
- final action count;
- graph edge count/components;
- factor count;
- factor coverage;
- fraction of pairs with \(R_{0.05}\le0.50\);
- median and 90th-percentile \(R_{0.05}\);
- exact verdict.

---

# 21. Repository bookkeeping after the verdict

After the preflight result is known:

- preserve the output immutably;
- update `project/EXPERIMENTS.md` minimally with the run record and verdict;
- update `project/PROJECT_HANDOFF.md` minimally with either:
  - Gp2 abandoned after structural preflight, or
  - Gp2 structural preflight passed; next gate not yet implemented.

Do not modify:

- `PAPER_STORY.md`
- `PROBLEM_FORMULATION.tex`
- `THEORY.tex`

in this task.

Commit the result bookkeeping.

---

# 22. Interpretation contract

## PASS

Record:

> The source-faithful candidate-local Gp2 proxy formulation admits materially sparse theorem-backed decision influence under the existing covariance construction. Proceed to a separate numerical/full-proxy BO smoke gate; no BO claim has yet been established.

## FAIL_STRUCTURAL_SPARSITY

Record:

> The source-faithful candidate-local Gp2 proxy formulation is mathematically compatible with the existing influence construction, but the real sequence graph does not exhibit the preregistered degree of structural decision sparsity. Abandon Gp2 as E3 rather than adding new theory or tuning the case study.

## PREPROCESSING_INVALID / PREPROCESSING_AMBIGUITY

Record the exact mechanical reason and abandon Gp2.

## THEORY_INSTANTIATION_INVALID

Record the failed identity/check and stop. Do not invent another theorem route.

---

# 23. Prohibitions

Do not:

- run BO;
- implement Laplace or importance-sampling inference;
- implement adaptive conditioning;
- use held-out SH target magnitude anywhere in this gate;
- change Sort1/Sort8;
- add Sort10;
- sweep Ridge alpha;
- replace Ridge with another predictor;
- change `k=8`;
- change \(\epsilon_{\mathrm{struct}}=0.05\);
- change the 90% / 50% structural criterion;
- change the 150-action threshold;
- change the 75-factor threshold;
- change the 40% factor-coverage threshold;
- weaken a theorem-backed failure using empirical target behavior;
- alter the old Gp2 P1 result;
- redesign the paper.

The task is complete when the structural preflight verdict is reproducibly committed.
