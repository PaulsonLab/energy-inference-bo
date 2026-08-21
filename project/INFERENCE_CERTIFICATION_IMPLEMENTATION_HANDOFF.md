# Inference Certification Implementation Handoff

## Role of this file

This is the single implementation and preregistration handoff for the final finite-sample certification blocker.

It is intentionally **not** a new source of truth for the paper story or general theory.

Canonical roles remain:

1. `PAPER_STORY.md` — narrative and paper claims.
2. `PROBLEM_FORMULATION.tex` — notation and problem definition.
3. `THEORY.tex` — theorem statements, proofs, and mathematical status.
4. `EXPERIMENTS.md` — empirical plans, frozen gates, run results, and pass/fail status.
5. This file — exact implementation contract for the flagship inference-certification pilot.

After the pilot, this file should remain as the reproducibility/preregistration record rather than becoming another living project-status document.

## Source state

Implementation must start from repository:

`https://github.com/PaulsonLab/energy-inference-bo`

Session-4 theory audit base commit:

`653d7a0b59e5e8943a810e0043d183b09b763030`

Before changing code, verify that the current repository still has:

- paper story locked;
- reflection-symmetry structural construction proved;
- nonlinear-PDE structural construction proved;
- nonlinear-PDE T4 family mapping proved;
- structural-influence blocker closed;
- end-to-end finite-sample certification as the only remaining technical blocker.

Do not reopen T1/T2/T3/T4.

---

# 1. What this pilot is testing

The pilot is **not** testing whether the structural theorem is correct, whether rejection sampling is a novel or generally superior inference method, whether the preference-conditioned sequential BO experiment works, or whether the nonlinear-PDE scaling story works.

Those are separate questions.

The pilot tests exactly this:

> In the already validated reflection-symmetry setting, does the mathematically rigorous finite-sample inference allowance remain small enough that the complete T3 action certificate can stop while a materially sparse subset of conditioning factors is active, using exact active-target samples at reasonable computational cost?

The existing symmetry evidence already showed that the **structural** omitted-factor term is non-vacuous with a sparse active set. The missing piece is replacing empirical/heuristic Monte Carlo uncertainty with a genuine finite-sample bound.

A successful pilot therefore demonstrates one complete empirical instance of

\[
\widehat G_S
+
B_{\rm infer}
+
B_{\rm struct}
\]

being small enough to certify a BO action relative to the fully conditioned acquisition.

That is the rigorous certificate role of the symmetry experiment in the paper:

- **symmetry:** mechanism + rigorous finite-sample action certificate;
- **preference-conditioned BO:** sequential BO relevance and changing factor relevance;
- **nonlinear PDE:** scaling and separation between decision complexity and full-conditioning inference difficulty.

Do not change these experiment roles.

---

# 2. Locked mathematical construction

## 2.1 Active target

Let

\[
Y=(Y_1,\ldots,Y_N), \qquad
Y_j=(f(-r_j),f(r_j)),
\]

with Gaussian reference

\[
\gamma=\mathcal N(m,Q^{-1}).
\]

For active set \(S\),

\[
E_S(Y)
=
\sum_{j\in S}
\gamma_j
\log\cosh\!\left(
\frac{f(r_j)-f(-r_j)}{\tau_j}
\right),
\]

and

\[
\mu_S(dY)
=
Z_S^{-1}e^{-E_S(Y)}\gamma(dY).
\]

Because \(\gamma_j\ge0\), \(\tau_j>0\), and \(\log\cosh z\ge0\),

\[
E_S(Y)\ge0,
\qquad
0<e^{-E_S(Y)}\le1.
\]

## 2.2 Exact active-target sampler

Use Gaussian-reference rejection sampling:

1. draw \(Y\sim\gamma\);
2. independently draw \(U\sim{\rm Uniform}(0,1)\);
3. accept iff
   \[
   \log U\le -E_S(Y).
   \]

The accepted samples are exact i.i.d. draws from \(\mu_S\).

The acceptance probability is exactly

\[
Z_S=\mathbb E_\gamma[e^{-E_S(Y)}].
\]

This is standard rejection sampling and is **not** a paper contribution.

## 2.3 Rao--Blackwellized EI gap

Use the already proved symmetry construction.

For each grid action \(x\),

\[
a_x=Qk_{Yx}.
\]

Conditional on \(Y=y\),

\[
f(x)\mid Y=y
\sim
\mathcal N(\mu_x(y),\sigma_x^2),
\]

with

\[
\mu_x(y)=m(x)+a_x^\top(y-m).
\]

Let \(\bar u_x(y)\) be the closed-form conditional expected improvement and define

\[
\bar F_{x,\widehat x}(y)
=
\bar u_x(y)-\bar u_{\widehat x}(y).
\]

Then

\[
G_S(x,\widehat x)
=
\mathbb E_{\mu_S}
[\bar F_{x,\widehat x}(Y)].
\]

For a fresh certification batch \(Y^{(1)},\ldots,Y^{(n)}\),

\[
\widehat G_S(x,\widehat x)
=
\frac1n
\sum_{\ell=1}^n
\bar F_{x,\widehat x}(Y^{(\ell)}).
\]

Use the **same accepted batch** for every challenger and for the leader term.

## 2.4 Locked inference confidence radius

The active target has potential

\[
V_S(y)
=
\frac12(y-m)^\top Q(y-m)+E_S(y).
\]

Since the symmetry factors are convex,

\[
\nabla^2E_S(y)\succeq0.
\]

After whitening \(y=m+Q^{-1/2}z\),

\[
\nabla_z^2V_S\succeq I.
\]

Use the resulting Bakry--Émery/log-Sobolev + Herbst concentration bound.

For leader \(\widehat x\), define

\[
\Lambda_Q(x,\widehat x)
=
\begin{cases}
0, & x=\widehat x,\\[1mm]
\max\left\{
\sqrt{a_x^\top Q^{-1}a_x},
\sqrt{a_{\widehat x}^\top Q^{-1}a_{\widehat x}},
\sqrt{(a_x-a_{\widehat x})^\top
Q^{-1}(a_x-a_{\widehat x})}
\right\},
&x\ne\widehat x.
\end{cases}
\]

Compute \(v^\top Q^{-1}v\) by solving \(Qz=v\), never by explicitly constructing \(Q^{-1}\).

Let

\[
K=|\mathcal X_{\rm grid}|
\]

and let \(R_{\max}\) be the known maximum number of certification rounds.

The locked two-sided confidence radius is

\[
\boxed{
B_{\rm infer}^{(r)}(x,\widehat x)
=
\Lambda_Q(x,\widehat x)
\sqrt{
\frac{2}{n_r}
\log\!\left(
\frac{2KR_{\max}}{\delta}
\right)
}.
}
\]

For the locked flagship configuration,

\[
K=401,\qquad
R_{\max}=15,\qquad
\delta=0.05,
\]

so code must reproduce

\[
\log(2KR_{\max}/\delta)
=
12.390891082522716.
\]

## 2.5 Structural term

Do not redesign or rederive the structural term.

Use the committed symmetry construction unchanged:

\[
B_{\rm struct}(x,\widehat x)
=
d(x,\widehat x)^\top A^{-1}h_U.
\]

Important:

- \(Q\) is the Gaussian precision used in \(B_{\rm infer}\).
- \(A\) is the Menz comparison matrix used in \(B_{\rm struct}\).
- They are not interchangeable.

## 2.6 Complete finite-grid certificate

For every challenger,

\[
\Psi_S(x;\widehat x)
=
\widehat G_S(x,\widehat x)
+
B_{\rm infer}(x,\widehat x)
+
B_{\rm struct}(x,\widehat x).
\]

On the exhaustive finite action grid,

\[
U_{\rm cert}
=
\max_{x\in\mathcal X_{\rm grid}}
\Psi_S(x;\widehat x),
\qquad
\eta_{\rm opt}=0.
\]

The round certifies when

\[
U_{\rm cert}\le\epsilon.
\]

The locked pilot uses

\[
\epsilon=0.01.
\]

---

# 3. Adaptivity and sample-separation rule

At every refinement round:

### Working stage

Use working samples/estimates to fix:

- active set \(S_r\);
- current leader \(\widehat x_r\).

Working-stage uncertainty is not used as the final finite-sample certificate.

### Certification stage

Only after \(S_r\) and \(\widehat x_r\) are fixed:

1. open a fresh independent certification RNG stream;
2. draw a fresh exact rejection-sampling batch from \(\mu_{S_r}\);
3. estimate all 401 challenger gaps using that batch;
4. add the simultaneous \(B_{\rm infer}\) and committed \(B_{\rm struct}\);
5. compute the exact finite-grid maximum \(U_{\rm cert}\).

If the certificate fails and the batch affects which factors are activated next, the batch becomes part of the history and is **expended**.

After the active set or leader changes, never reuse its confidence statement. Draw a new independent certification batch.

---

# 4. Frozen prospective pilot

These values must be written to a machine-readable frozen configuration and committed **before any prospective certification samples are generated**.

```text
N_FACTORS = 40

ACTION_COUNT = 401
ACTION_MIN = -0.58
ACTION_MAX = -0.06

INCUMBENT = 0.50
EPSILON = 0.01
DELTA = 0.05

ACTIVATION_BATCH_SIZE = 3
R_MAX = 15

WORKING_REFERENCE_SAMPLES = 80_000
WORKING_SEED = 123

CERTIFICATION_ROOT_SEED = 314159265
PROPOSAL_CHUNK_SIZE = 25_000

N_CERT_IF_ACTIVE_LT_15 = 100_000
N_CERT_IF_ACTIVE_GE_15 = 1_500_000
```

Use

`numpy.random.SeedSequence(CERTIFICATION_ROOT_SEED).spawn(R_MAX)`

for independent round-specific certification streams.

Record:

- Git commit;
- frozen-config SHA-256;
- Python version;
- NumPy/SciPy versions;
- hardware;
- per-round child-seed metadata;
- active-set hash;
- leader;
- certification batch ID.

---

# 5. Prospective PASS/FAIL criterion

The pilot is **PASS if and only if all conditions below hold**:

1. A reached round satisfies
   \[
   U_{\rm cert}\le0.01.
   \]

2. Certification occurs with
   \[
   M\le18,
   \]
   meaning at least 22 of 40 factors (55%) remain omitted.

3. At the maximizing optimistic challenger,
   \[
   B_{\rm infer}\le0.0045.
   \]

4. Cumulative Gaussian proposals across all certification rounds are at most
   \[
   20{,}000{,}000.
   \]

5. Final-round rejection-sampling acceptance rate is at least
   \[
   0.20.
   \]

6. All mathematical, unit/regression, seed-separation, and no-batch-reuse tests pass.

The pilot is **FAIL** if any condition fails.

Do not alter any threshold after observing prospective pilot results.

Planning calculations from Session 4 predict approximately:

- stopping near \(M=15\), with allowance through \(M=18\);
- final rejection acceptance roughly \(0.27\)--\(0.34\);
- final inference radius roughly \(0.0038\)--\(0.0040\);
- final \(U_{\rm cert}\) roughly \(0.0089\)--\(0.0091\);
- about 5.3 million cumulative proposals if stopping at \(M=15\);
- about 11 million if an \(M=18\) high-precision round is required.

These are **planning estimates only**, not results.

---

# 6. Implementation files

## Add

### `src/conditioned_bo/inference_certification.py`

Implement at least:

- `symmetry_active_energy_batch`
- `rejection_sample_symmetry_target`
- `q_inverse_norm`
- `ei_gap_q_lipschitz`
- `inference_confidence_radius`
- `certify_symmetry_grid_round`

Requirements:

- exact no-jitter Gaussian reference;
- active factors only in \(E_S\);
- stable `logcosh`;
- log-space rejection condition;
- streaming proposal chunks;
- exact requested number of accepted samples;
- proposal counts;
- acceptance diagnostics;
- batch manifest.

### `tests/test_inference_certification.py`

Add tests in Section 7.

### `experiments/symmetry/run_inference_certification_pilot.py`

Implement the frozen pilot while reusing existing working-stage and structural-bound code.

### `experiments/symmetry/configs/inference_certification_pilot.json`

Store the frozen prospective configuration in machine-readable form.

## Modify before pilot

### `project/THEORY.tex`

Replace the stale recommendation of GP-reference importance sampling + ratio confidence intervals with the locked exact rejection-sampling + whitened log-Sobolev construction.

Add:

- exact accepted-sample law;
- strong-log-concavity/whitening argument;
- EI-gap \(Q^{-1}\)-metric Lipschitz constant;
- finite-\(n\) confidence radius;
- fresh-batch conditioning/adaptivity proof;
- challenger/round confidence allocation;
- finite-grid T3 interface proposition.

Do **not** modify T1/T2/T3/T4 themselves.

The inference theory status should become:

**THEORY READY — PROSPECTIVE PILOT REQUIRED**

Do not mark the empirical/end-to-end blocker closed yet.

### `project/PROJECT_HANDOFF.md`

Synchronize the inference section with Session 4:

- remove the stale importance-sampling/ratio-CI recommendation;
- state that the canonical flagship rigorous instantiation is exact Gaussian-reference rejection sampling + the whitened strong-log-concavity sample-mean radius;
- state that the mathematical construction is ready;
- state that the prospective locked symmetry pilot is the remaining gate.

Do not alter the paper thesis or contribution structure.

### `project/EXPERIMENTS.md`

Before running:

- add the frozen E1 inference-certification pilot configuration;
- record the prospective PASS/FAIL criterion;
- distinguish planning predictions from prospective results;
- keep E1 status as not yet fully certified.

After running:

- append the actual run metadata, outputs, component decomposition, proposal counts, acceptance rate, active count, and mechanical PASS/FAIL result.

### `project/PAPER_STORY.md`

Do not add implementation details.

Only fix a statement if it is genuinely inconsistent with the current project state. Before the pilot, it is still correct to say that finite-sample end-to-end certification is the sole remaining technical blocker.

### `project/T2B_SYMMETRY_AUDIT.md`

Treat as the closed structural audit. Do not rewrite it into an inference audit. Add only a short cross-reference if useful.

---

# 7. Required tests

1. **Energy sign and envelope**
   - \(E_S\ge0\);
   - weights are in \((0,1]\);
   - empty active set has acceptance probability one.

2. **Stable log-cosh**
   - vectorized implementation agrees with the existing scalar implementation;
   - extreme finite inputs do not overflow.

3. **Low-dimensional sampler-law regression**
   - compare exact/quadrature moments for a one- or two-block target with rejection-sample moments using a predeclared tolerance.

4. **Proposal-order / reproducibility**
   - fixed seed and frozen chunk size reproduce samples and proposal count.

5. **Fresh-batch bookkeeping**
   - every reached round has a unique child seed and batch ID;
   - a batch cannot certify two different `(active_set, leader)` pairs.

6. **\(Q^{-1}\)-norm orientation**
   - sparse/dense solve agreement;
   - no production explicit inverse.

7. **EI Lipschitz bound**
   - analytic/finite-difference checks;
   - self-comparison is exactly zero.

8. **Confidence constant**
   - test
     \[
     \log(2\cdot401\cdot15/0.05)
     =
     12.390891082522716;
     \]
   - explicitly test the two-sided factor two.

9. **Structural regression**
   - inference additions must not alter the existing T2-B structural result.

10. **Synthetic end-to-end certificate**
    - low-dimensional problem with essentially exact full-target grid regret;
    - every reported passing certificate must upper-bound exact regret.

11. **Output integrity**
    - all 401 action rows saved;
    - maximizing row equals `U_cert`;
    - component sum equals `Psi`.

---

# 8. Required outputs

Write under:

`experiments/symmetry/outputs/inference_certification_pilot/`

At minimum:

- `frozen_config.json`
- `batch_manifest.json`
- `round_history.csv`
- `sampler_diagnostics.csv`
- one challenger-bound file per reached round
- `summary.json`
- `RESULTS.md`
- `certificate_decomposition.png`

For every round, record:

- active count and active factors;
- working leader;
- worst optimistic challenger;
- accepted sample count;
- Gaussian proposal count;
- acceptance rate;
- estimated active-target gap;
- \(B_{\rm infer}\);
- \(B_{\rm struct}\);
- \(U_{\rm cert}\);
- activated factors if failing;
- batch ID;
- mechanical pass/fail state.

---

# 9. Two-commit execution discipline

## Commit A — preregistration / implementation

Before prospective pilot sampling:

1. add this handoff;
2. implement sampler, bound, tests, and pilot runner;
3. synchronize `THEORY.tex`, `PROJECT_HANDOFF.md`, and the planned E1 entry in `EXPERIMENTS.md`;
4. add and freeze the machine-readable pilot config;
5. run all unit/regression tests;
6. commit and record the exact SHA.

**No prospective certification samples may be drawn before Commit A exists.**

## Commit B — prospective result

Starting from Commit A:

1. run the locked prospective pilot exactly once;
2. do not change thresholds or sample counts in response to the output;
3. save all outputs;
4. mechanically evaluate the PASS/FAIL criterion;
5. update `EXPERIMENTS.md` with the result;
6. if PASS, update project status files to say the final technical blocker is closed;
7. if FAIL, keep the blocker open and state exactly which predeclared condition failed;
8. commit the result and report the exact SHA.

---

# 10. Definition of done

Before pilot:

- [ ] Locked theory implemented.
- [ ] Stale importance-sampling recommendation removed from canonical current-status files.
- [ ] Exact confidence constants tested.
- [ ] Fresh-batch lifecycle enforced in code.
- [ ] Frozen machine-readable pilot config committed.
- [ ] PASS/FAIL thresholds committed before prospective sampling.
- [ ] Existing structural tests/results unchanged.

After pilot:

- [ ] Pilot run uses the committed frozen config.
- [ ] All per-round outputs are saved.
- [ ] PASS/FAIL is evaluated mechanically.
- [ ] `EXPERIMENTS.md` records the prospective result.
- [ ] Project blocker status changes only if the prospective pilot passes.
- [ ] No sampler novelty is claimed.
- [ ] No finite-sample guarantee is implied for HMC/SMC/FlowGP or other backends.
