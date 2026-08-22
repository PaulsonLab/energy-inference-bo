# E3 Preference-Informed BO Phenomenon Pilot

## Status and scope

This is a prospective, minimal phenomenon pilot for E3.

It answers only:

1. **P1 — preference value:** does a fixed bank of historical pairwise preferences make ordinary scalar-observation BO materially better than scalar-only BO?
2. **P2 — sparse preservation:** if the bank is useful, can decision-specific adaptive conditioning retain essentially the same optimization benefit while conditioning on substantially fewer preference factors?

This pilot is **not** the final E3 baseline suite.

Run exactly three methods:

1. standard scalar GP-EI BO;
2. full preference-informed GP-EI BO;
3. adaptive preference-informed GP-EI BO.

Do not add random, local, static, coreset, VVM, AABO, or other baselines.

The current repository contains older E3 planning text suggesting a fixed local/static subset in the first comparison. This handoff intentionally supersedes that narrow pilot-level instruction because standard BO is required to answer P1. It does not change the final E3 baseline plan or `PAPER_STORY.md`.

---

## Scientific objective

Test:

> Can useful historical preference information improve standard BO, while a decision-selected subset preserves essentially the same benefit without conditioning on the full preference bank at every decision?

The online scalar observations always define the ordinary GP reference posterior. They are never screened.

Only the fixed offline preference bank is eligible for factor screening.

---

## Frozen model system

### True objective

Maximize on \(x\in[0,1]\):

\[
f_{\rm true}(x)=
0.55\exp\!\left[
-\frac12\left(\frac{x-0.22}{0.07}\right)^2
\right]
+
1.00\exp\!\left[
-\frac12\left(\frac{x-0.76}{0.065}\right)^2
\right]
+
0.04\sin(4\pi x).
\]

Use the finite action grid

\[
x_i=i/16,\qquad i=0,\ldots,16.
\]

The expected grid maximum from the formula is at \(x=0.75\), with value approximately \(0.9882354306\). Verify this mechanically in a unit test rather than hard-coding optimization behavior around it.

This objective has a broad inferior left mode and a narrower superior right mode. The purpose is to create ordinary multimodal BO ambiguity, not a rare-event construction.

### Scalar BO observations

Initial action indices:

```text
[2, 8, 14]
```

corresponding to

```text
[0.125, 0.5, 0.875]
```

Then execute exactly six post-initial BO evaluations.

Observation model:

\[
y=f_{\rm true}(x)+\epsilon,
\qquad
\epsilon\sim N(0,0.01^2).
\]

Do not repeat previously observed actions.

For each experimental seed, pre-generate one noise realization for every grid point and reuse that location-indexed noise table across all three methods. Thus methods that query the same point under the same seed observe the same scalar outcome.

### GP reference model

Use a zero-mean OU/Matérn-\(1/2\) GP:

\[
k(x,x')
=
0.10\exp(-|x-x'|/0.05).
\]

Use observation-noise standard deviation \(0.01\).

Do not optimize or fit GP hyperparameters.

At BO iteration \(t\), ordinary scalar observations alone define

\[
P_{0,t}=N(m_t,K_t).
\]

### Acquisition

Use ordinary expected improvement:

\[
u_x(f)=(f(x)-y_t^\star)_+,
\]

where \(y_t^\star\) is the largest observed noisy scalar outcome.

The finite action grid is searched exhaustively, so

\[
\eta_{\rm opt}=0.
\]

---

## Frozen historical preference bank

Use exactly \(N=8\) endpoint pairs

\[
(a_j,b_j)=(x_j,x_{16-j}),
\qquad
j=0,\ldots,7.
\]

These form a perfect matching of all noncentral grid points and have maximum preference-graph degree one.

The interpretation is a historical campaign in which low-\(x\) and high-\(x\) operating conditions at matched distance from the domain center were compared.

Use preference temperature

\[
\tau_{\rm pref}=0.60.
\]

For each seed independently, generate

\[
\Pr(s_j=+1)
=
\sigma\!\left(
\frac{f_{\rm true}(a_j)-f_{\rm true}(b_j)}
{\tau_{\rm pref}}
\right),
\]

with \(s_j=+1\) denoting \(a_j\succ b_j\).

Generate the complete preference bank for every pilot seed **before executing any BO method**.

Use exactly seeds

```text
0, 1, 2, ..., 11
```

The realized bank for a seed must be reused unchanged by standard, full, and adaptive runs.

Save the generated banks to machine-readable output.

---

## Preference factor

For latent grid vector \(Y\), define

\[
d_j=e_{a_j}-e_{b_j},
\qquad
z_j=\frac{s_jd_j^\top Y}{\tau_{\rm pref}}.
\]

Use

\[
e_j(Y)=\log(1+\exp(-z_j)).
\]

Let \(q_j=\sigma(-z_j)\). Then

\[
\nabla e_j
=
-\frac{s_jq_j}{\tau_{\rm pref}}d_j,
\]

and

\[
\nabla^2e_j
=
\frac{q_j(1-q_j)}{\tau_{\rm pref}^2}
d_jd_j^\top
\succeq0.
\]

Therefore the factor is convex and

\[
\|\nabla e_j\|_2
\le
\frac{\sqrt2}{\tau_{\rm pref}}.
\]

Implement the energy stably with `logaddexp`.

---

## Preference influence construction

Partition the 17 latent coordinates into

\[
B_0=\{8\},
\qquad
B_r=\{8-r,8+r\},
\quad r=1,\ldots,8.
\]

Each preference factor lies wholly in one two-coordinate block.

Its sensitivity vector is therefore

\[
L_r(e_j)=
\begin{cases}
\sqrt2/\tau_{\rm pref}, & r=B(j),\\
0, & \text{otherwise}.
\end{cases}
\]

A preference factor contributes zero cross-block Hessian coupling under this partition.

For the scalar-data-conditioned Gaussian reference with precision \(Q_t\), define

\[
\rho_r(t)
=
\lambda_{\min}((Q_t)_{B_rB_r}),
\]

\[
\kappa_{rs}(t)
=
\|(Q_t)_{B_rB_s}\|_{\rm op},
\qquad r\ne s,
\]

and

\[
A_{rr}(t)=\rho_r(t),
\qquad
A_{rs}(t)=-\kappa_{rs}(t).
\]

For the frozen prior configuration, the expected numerical check is

```text
lambda_min(A_0) = 5.63301154 approximately
condition_number(A_0) = 3.18 approximately
```

The code must recompute these values from the configuration.

Direct scalar observations add only nonnegative diagonal precision, so later \(A_t\) must remain positive definite.

For an EI leader/challenger pair, use block decision sensitivity:

- self-comparison: all zero;
- distinct actions in distinct blocks: sensitivity 1 in each of the two involved blocks;
- distinct actions within the same two-coordinate block: sensitivity \(\sqrt2\) in that block.

For omitted set \(U\),

\[
h_U=\sum_{j\in U}L(e_j).
\]

Compute

\[
B_{\rm struct}
=
L(F)^\top A_t^{-1}h_U
\]

using a linear solve rather than forming \(A_t^{-1}\).

For the current worst challenger, rank omitted factors by

\[
b_j
=
L(e_j)^\top A_t^{-1}L(F_{\rm worst}).
\]

Implement this interface consistently with the existing symmetry and nonlinear-PDE influence modules.

---

## Three methods

### 1. Standard BO

Use only the scalar-data GP posterior.

No preference factors are conditioned upon.

Compute analytic GP EI on the unobserved finite action grid.

### 2. Full preference-informed BO

At every BO iteration use all eight preference factors:

\[
\pi_{C,t}(dY)
\propto
P_{0,t}(dY)
\exp[-\sum_{j=1}^{8}e_j(Y)].
\]

Estimate EI using the full-target inference procedure below and select its exhaustive-grid maximizer.

### 3. Adaptive preference-informed BO

At the beginning of **every BO iteration**, reset

```text
S_t = empty
```

Do not carry the previous iteration's active factors forward automatically.

For each refinement:

1. infer under the current active target;
2. estimate active EI over the unobserved grid;
3. choose its leader;
4. compute structural bounds against every challenger;
5. combine active gap, empirical inference allowance, and structural bound;
6. identify the worst optimistic challenger;
7. stop if its envelope is at most
   \[
   \epsilon_{\rm screen}=0.05;
   \]
8. otherwise activate the single omitted factor with largest \(b_j\) for that challenger and repeat.

If all eight factors are active, the active target is the full target and the method must stop without numerical inconsistency.

---

## Numerical inference

Use Laplace-preconditioned self-normalized importance sampling.

For active factor set \(S\), the target is

\[
\pi_S(Y)
\propto
N(Y;m_t,K_t)
\prod_{j\in S}\sigma(z_j).
\]

Find the unique mode using the exact preference gradient and Hessian.

Frozen mode settings:

```text
float64
gradient infinity-norm tolerance: 1e-9
maximum Newton iterations: 50
backtracking line search: enabled
```

Use as proposal

\[
q_S=N(Y_{\rm mode},H_{\rm mode}^{-1}).
\]

Normalize importance weights in log space.

### Adaptive working inference

Start with two independent batches of 8,192 proposal draws each.

For each acquisition gap use the empirical allowance

\[
B_{\rm infer}
=
\max\{
2\widehat{\rm SE}_{\rm pooled},
|\widehat G^{(1)}-\widehat G^{(2)}|
\}.
\]

If proposal ESS fraction is below 0.25 or the worst-challenger \(B_{\rm infer}\) exceeds 0.01, double each working batch to 16,384 and, if necessary, 32,768.

These are numerical-accuracy refinements, not scientific tuning.

### Full target and held-out validation

For full preference-informed BO, use 65,536 independent Laplace-IS draws per BO iteration.

After the adaptive method stops at each BO iteration, independently evaluate the full target with 65,536 draws and record the adaptive action's full-target acquisition regret.

If full-target ESS fraction is below 0.25 or the maximum split-half EI discrepancy exceeds 0.0025, increase that full computation to 131,072 draws.

Do not claim this empirical \(B_{\rm infer}\) is a rigorous finite-sample T3 certificate. This is a phenomenon pilot with strong held-out full-target validation.

---

## Frozen scientific configuration

```text
action grid: 17 equally spaced points on [0,1]
initial indices: [2, 8, 14]
post-initial horizon: 6
observation noise sd: 0.01
GP kernel: 0.10 * exp(-|x-x'| / 0.05)
GP hyperparameter fitting: disabled
preference factors: 8
preference endpoints: (j, 16-j), j=0,...,7
preference temperature: 0.60
seeds: 0,...,11
adaptive active set at each BO iteration: empty
activation batch size: 1
screening tolerance: 0.05
action optimization: exhaustive finite grid
repeated scalar actions: prohibited
methods: standard, full, adaptive only
```

These values must not change after results are observed.

---

## Primary optimization metric

For every seed and method define

\[
T_{0.10}
=
\min\{
t:
\text{simple regret after }t
\text{ post-initial scalar evaluations}\le0.10
\}.
\]

Simple regret uses the noise-free true objective on the finite action grid.

If the target is not reached during the six post-initial evaluations, record

```text
T_0.10 = 7
```

All methods still run the complete six-iteration horizon.

---

## Prospective gates

### P1 — preference value

P1 passes iff

\[
\operatorname{median}T_{0.10}^{\rm full}
\le
\operatorname{median}T_{0.10}^{\rm standard}-1.
\]

### P2 — sparse preservation

For adaptive conditioning define, for each seed,

\[
R_{\rm factors}
=
\frac{\sum_{t=1}^6M_t}{6N}.
\]

P2 passes iff both

\[
\operatorname{median}T_{0.10}^{\rm adaptive}
\le
\operatorname{median}T_{0.10}^{\rm full}+1
\]

and

\[
\operatorname{median}R_{\rm factors}\le0.65.
\]

Do not require exact action agreement.

### Overall verdict

Evaluate in this order:

```text
if P1 fails:
    FAIL-P1
elif either P2 condition fails:
    FAIL-P2
else:
    PASS
```

Do not change thresholds after the run.

---

## Result schema

Save, for every seed, BO iteration, and method:

- method;
- seed;
- BO iteration;
- selected action index and \(x\);
- observed scalar outcome;
- incumbent;
- noise-free best value observed;
- simple regret;
- \(T_{0.10}\) status;
- active factor indices;
- \(M_t\);
- \(N\);
- cumulative final-active-factor use \(\sum_tM_t\);
- raw factor-likelihood evaluation count;
- inference ESS and ESS fraction;
- inference wall time;
- selected-action acquisition estimate;
- active acquisition curve where applicable;
- full acquisition curve where applicable;
- structural bound at the worst challenger;
- empirical inference allowance;
- adaptive action's held-out full-target acquisition regret;
- exact configuration identifier/hash.

For every adaptive refinement additionally save:

- refinement index;
- leader;
- worst optimistic challenger;
- active set before refinement;
- newly activated factor;
- per-factor contribution scores;
- active acquisition gap;
- \(B_{\rm struct}\);
- \(B_{\rm infer}\);
- stopping envelope;
- stopping reason;
- inference sample count and ESS.

Also save the generated preference bank and scalar-noise lookup table for every seed.

Report consecutive active-set Jaccard overlap/turnover as a diagnostic only. It is not a pass criterion.

---

## Raw computational factor accounting

Keep two notions distinct:

1. **scientific factor-use ratio:** final \(M_t/N\) at each decision and
   \[
   \sum_tM_t/(6N),
   \]
   which is the P2 sparsity metric;
2. **executed factor-likelihood evaluations:** actual number of factor-energy evaluations incurred by optimization and importance-sampling batches, which is only a computational diagnostic.

Do not substitute wall time or ESS for the scientific P2 factor-use metric.

---

## Focused tests required before the pilot

At minimum test:

1. stable logistic-energy evaluation;
2. analytic gradient versus finite differences;
3. analytic Hessian versus finite differences;
4. Hessian positive semidefiniteness;
5. gradient sensitivity bound \(\sqrt2/\tau_{\rm pref}\);
6. every preference factor is contained in exactly one proposed block;
7. preference cross-block Hessian contribution is zero;
8. frozen \(A_0\) has minimum eigenvalue approximately 5.63301154 and is SPD;
9. scalar observations cannot decrease the comparison-matrix diagonal curvature or destroy SPD;
10. all-active structural bound is exactly zero numerically;
11. structural bound equals the sum of the omitted per-factor contribution scores up to numerical tolerance;
12. structural bound is nonincreasing when factors are removed from the omitted set with \(A_t\) fixed;
13. zero-factor conditioned target reproduces analytic GP EI within Monte Carlo tolerance;
14. all-eight-factor active target agrees with the full-target implementation;
15. preference banks are generated before method execution and are identical across methods for a seed;
16. location-indexed scalar noise is identical across methods for a seed;
17. scalar BO observations never enter the screenable-factor list;
18. exhaustive action search never selects an already observed point;
19. result rows contain all required provenance fields;
20. gate evaluator returns known synthetic PASS, FAIL-P1, and FAIL-P2 cases correctly.

Run the repository's complete existing test suite as well.

---

## Smoke test

Before the frozen run, execute one seed for two BO iterations using substantially reduced **numerical** sample counts.

The smoke test checks:

- target construction;
- factor activation;
- inference;
- output schema;
- seed sharing/separation;
- full-target validation;
- file writing;
- gate evaluator mechanics.

Do not evaluate P1/P2 scientifically from the smoke test.

Do not change a scientific parameter because of smoke-test behavior unless an actual implementation conflict makes the handoff impossible; report such a conflict instead.

---

## Expected repository changes

Create:

```text
project/E3_PREFERENCE_BO_PILOT_HANDOFF.md

src/conditioned_bo/preference_influence.py
src/conditioned_bo/preference_bo.py

tests/test_preference_influence.py
tests/test_preference_bo.py

experiments/preference_bo/configs/minimal_pilot.json
experiments/preference_bo/run_minimal_pilot.py

experiments/preference_bo/outputs/minimal_pilot/
    frozen_config.json
    preference_banks.csv
    scalar_noise.csv
    trajectory.csv
    refinement_history.csv
    acquisition_validation.csv
    summary.json
    RESULTS.md
    diagnostic.png
    provenance.json
```

Modify as needed:

```text
experiments/preference_bo/README.md
project/EXPERIMENTS.md
src/conditioned_bo/__init__.py
```

Do not modify:

```text
project/PAPER_STORY.md
```

Do not modify theory files merely to accommodate the experiment. If the implementation exposes a genuine mathematical conflict with the formulas above or the current committed theory, stop and report it rather than silently changing theory.

No notebook is required.

No new Python dependency should be necessary.

---

## Diagnostic figure

Create one simple diagnostic figure, not publication-ready, containing:

1. simple regret versus post-initial scalar BO iteration for standard/full/adaptive;
2. adaptive \(M_t/N\) versus BO iteration.

Optionally overlay a few selected preference edges if doing so is trivial.

---

## Definition of done

This pilot is done when:

- the exact starting Git commit is recorded;
- this handoff and an immutable frozen JSON configuration are stored;
- the preference influence code mirrors the existing reusable influence interface;
- all focused and pre-existing tests pass;
- the smoke test passes;
- all 12 frozen seeds run for all three and only three methods;
- no scientific parameter or gate is altered after viewing results;
- all raw machine-readable trajectories, refinement histories, banks, noise, inference diagnostics, and provenance are saved;
- P1 and P2 are evaluated mechanically from the preregistered thresholds;
- the result is labeled exactly `PASS`, `FAIL-P1`, or `FAIL-P2`;
- negative results are retained rather than overwritten;
- `project/EXPERIMENTS.md` records the run, configuration, provenance, verdict, interpretation, and next action;
- `project/PAPER_STORY.md` is unchanged;
- any repository-versus-handoff conflict is explicitly reported.