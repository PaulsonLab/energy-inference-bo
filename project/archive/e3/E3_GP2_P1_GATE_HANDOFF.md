# E3 Gp2 Prospective P1 Gate Handoff

## 1. Scope

This prospectively freezes the next **cheap E3 gate**.

The paper direction is locked. Do not redesign the paper or reopen T1--T4.

This gate answers exactly one scientific question:

> **Does authentic, yield-blind pairwise information from independent high-throughput Gp2 developability assays materially improve ordinary scalar-yield BO?**

This is **P1 only**. Do not test adaptive screening, empirical decision sparsity, certified sparsity, factor-selection baselines, VVM, coresets, AABO, or the final E3 suite.

A PASS justifies a later realistic-E3 screening/saturation gate. A FAIL rejects this frozen Gp2 construction without tuning it.

---

## 2. Project grounding

Project repository:

`https://github.com/PaulsonLab/energy-inference-bo`

Branch:

`main`

This gate was designed against:

`4f9b4c1f5349d52dc7799dd4b5229aae9f03d1c7`

Before implementation, report the current exact `main` SHA.

If `main` has moved, inspect the diff. Proceed only if the changes do not materially alter the locked story, completed E3-pilot interpretation, or preference-model/inference interface.

Read:

1. `project/PROJECT_HANDOFF.md`
2. `project/PAPER_STORY.md`
3. `project/EXPERIMENTS.md`
4. `project/E3_PREFERENCE_BO_PILOT_HANDOFF.md`
5. `project/E3_PREFERENCE_BO_REDUNDANT_BANK_PILOT_HANDOFF.md`
6. both final synthetic E3 `RESULTS.md` files
7. `src/conditioned_bo/preference_bo.py`
8. `src/conditioned_bo/preference_influence.py`

Inherited fact: both synthetic E3 pilots showed strong historical-information value and full/adaptive performance agreement, but failed their preregistered certified-sparsity gates. Do not retune or reinterpret those pilots here.

---

## 3. External data provenance

Use public repository:

`HackelLab-UMN/DevRep`

Pinned external commit:

`e05023a8abe7be6c2e22f42d523b20bd76cd8da5`

Use exactly:

- `datasets/assay_to_yield_training_sequences.csv`
- `datasets/test_sequences.csv`

Record the external commit, paths, downloaded SHA-256 hashes, and preprocessing row counts.

Prefer downloading/caching the pinned files rather than committing duplicate copies.

### Frozen assay mapping

The original source maps:

- `Sort1` = on-yeast protease-resistance;
- `Sort8` = split-GFP;
- `Sort10` = split beta-lactamase.

Use **Sort1 and Sort8 only**.

**Sort10 is excluded.**

Do not choose assays based on SH-yield performance.

---

## 4. BO semantics

### Action

One action is one unique full-length Gp2 `Paratope` sequence selected for conventional recombinant-yield measurement in the SH strain.

### Scalar target

Maximize the released:

`SH_Average_bc`

Use it as the retrospective scalar oracle.

Do not add synthetic observation noise.

### Historical factors

Sort1 and Sort8 are independent high-throughput proxy assays, not calibrated scalar observations of SH recombinant yield.

They are used only to create local ordinal statements.

For factor `j`:

\[
e_j(z)=
\log\!\left(
1+\exp\!\left[
-\frac{s_j(z_{a_j}-z_{b_j})}{\tau_{\rm pref}}
\right]
\right),
\]

with fixed:

\[
\tau_{\rm pref}=1.0.
\]

No factor weights, assay-specific temperatures, or learned preference calibration.

---

## 5. Deterministic candidate set

Take the union of the two pinned CSV files.

Keep a row iff:

1. `Stop == False`;
2. `Paratope` is present and nonempty;
3. `SH_Average_bc` is finite;
4. all of these are finite:
   - `Sort1_1_score`
   - `Sort1_2_score`
   - `Sort1_3_score`
   - `Sort8_1_score`
   - `Sort8_2_score`
   - `Sort8_3_score`

Do **not** filter on target magnitude, optimum proximity, proxy/target correlation, model predictions, or preference agreement with yield.

Require all retained `Paratope` strings to have identical length. Treat every character, including literal `X`, as an ordinary Hamming-distance symbol.

### Duplicates

Final action space must contain one row per exact `Paratope`.

If duplicate `Paratope` rows exist:

- if all fields used by this gate agree to ordinary serialization tolerance, keep one deterministic copy;
- otherwise stop with `PREPROCESSING_AMBIGUITY`.

Do not average inconsistent targets.

Sort final candidates lexicographically by `Paratope` before assigning canonical integer indices.

---

## 6. Frozen sequence graph

Use unweighted Hamming distance.

Set:

`k = 8`

For each candidate, choose exactly eight nearest other candidates.

Tie-break by:

1. smaller Hamming distance;
2. lexicographically smaller neighbor `Paratope`;
3. canonical candidate index.

Create one undirected graph from the **union** of the directed 8-NN choices.

Do not sweep `k`.

No target or assay values may influence graph construction.

### Connected-component rule

If connected, use the graph unchanged.

If disconnected, retain the largest connected component only if it contains at least 90% of candidates, then re-sort/re-index by `Paratope`.

Otherwise preprocessing fails.

---

## 7. Frozen preference bank

For every undirected graph edge `{a,b}`, evaluate Sort1 and Sort8 separately.

For assay `q`, compute:

\[
\Delta_r^{(q)}
=
{\rm score}^{(q)}_r(a)-{\rm score}^{(q)}_r(b),
\quad r=1,2,3.
\]

Create a factor only under strict replicate majority:

- `a > b` if at least 2 of 3 differences are strictly positive;
- `b > a` if at least 2 are strictly negative;
- otherwise no factor.

A zero difference is an abstention.

Do not threshold magnitude and do not compare assay means.

Sort1 and Sort8 may create two factors on one graph edge. If they disagree, retain both.

Canonical factor order:

1. smaller endpoint index;
2. larger endpoint index;
3. Sort1 before Sort8.

Generate this complete bank once before any BO seed/method and reuse it unchanged.

---

## 8. Preprocessing validity

Run this **before inspecting BO performance**.

Save only mechanical/provenance information:

- source row counts;
- filtering counts;
- duplicate count;
- final candidate count;
- sequence-length check;
- graph edges/components;
- factor count by assay;
- total factor count;
- endpoint-degree summary;
- file hashes.

Do not print/save target optimum, target histogram, proxy/target correlation, or BO performance during preflight.

Preprocessing passes iff:

1. `N_actions >= 250`;
2. largest sequence-graph component contains at least 90% of retained candidates;
3. final preference bank has at least 1,000 factors;
4. every factor is traceable only to Sort1/Sort8 replicate comparisons;
5. graph/factor generation has no target-value dependency.

Otherwise outcome:

`PREPROCESSING_INVALID`

and do not run the scientific gate.

---

## 9. Frozen graph-Gaussian reference

Let `W` be the binary adjacency matrix of the frozen undirected graph and

\[
L_{\rm sym}=I-D^{-1/2}WD^{-1/2}.
\]

Use standardized-latent prior precision:

\[
Q_0=I+L_{\rm sym},
\qquad
z\sim N(0,Q_0^{-1}).
\]

Do not optimize graph or precision hyperparameters.

### Per-seed target scaling

Use only the five initial scalar observations.

Freeze for the whole trajectory:

\[
\mu_0={\rm mean}(y_{\rm init}),
\]

\[
s_0=\max\{{\rm sd}(y_{\rm init};{\rm ddof}=1),10^{-6}\},
\]

and model:

\[
z(x)=\frac{f(x)-\mu_0}{s_0}.
\]

Do not recompute this scaling later.

Use fixed Gaussian observation-noise SD:

\[
\sigma_{\rm obs}=0.05
\]

in standardized units.

The public scalar value returned by the retrospective oracle remains exactly `SH_Average_bc`; this noise term is only the fixed reference-model likelihood regularizer.

---

## 10. BO protocol

Run exactly:

1. `scalar_only`
2. `full_preference`

**Do not implement adaptive conditioning in this gate.**

### Seeds and initialization

Use exactly seeds:

`0, 1, ..., 19`

For seed `s`, choose five initial actions using:

`np.random.default_rng(s).choice(N_actions, size=5, replace=False)`

after final canonical indexing.

Same initial actions for both methods.

Do not stratify using target or proxy data.

### Horizon

- initial scalar observations: `5`
- post-initial BO evaluations: `5`

No repeated actions.

### Acquisition

Use ordinary expected improvement on standardized latent yield.

- scalar-only: analytic Gaussian EI;
- full-preference: EI estimated by the frozen Laplace-preconditioned importance sampler below.

Search all currently unobserved finite actions exhaustively.

---

## 11. Full-preference inference

At BO iteration `t`:

\[
\pi_{C,t}(dz)
\propto
N(dz;m_t,K_t)
\prod_j
\sigma\!\left(
\frac{s_j(z_{a_j}-z_{b_j})}{\tau_{\rm pref}}
\right).
\]

Reuse the existing stable preference energy, exact derivatives, Laplace mode, and Laplace-preconditioned self-normalized importance-sampling machinery where practical.

Do not introduce HMC, SMC, FlowGP, or a neural surrogate.

Use float64.

Laplace:

- gradient infinity tolerance: `1e-8`
- max Newton iterations: `60`
- backtracking: enabled

Scientific draw schedule:

`[8192, 16384, 32768, 65536]`

A decision is numerically reliable iff:

1. proposal ESS fraction `>= 0.20`; and
2. maximum absolute split-half EI discrepancy over unobserved actions `<= 0.01`.

If either fails, advance to the next frozen draw count.

If still unreliable at 65,536:

`INCONCLUSIVE_NUMERICAL`

Implement chunked evaluation so the maximum draw count is safe on a 16 GB laptop.

Random streams must be deterministic and independent across split halves, seeds, BO iterations, and draw stages.

---

## 12. Mechanical smoke

Before scientific execution run:

- seed `0`;
- same frozen five-action initialization;
- post-initial horizon `2`;
- both methods;
- full-preference draw schedule `[1024, 2048, 4096]`.

Smoke is mechanical only; do not evaluate the 25% gate.

Verify:

- pinned downloads/hashes;
- deterministic preprocessing;
- no target leakage;
- candidate indexing;
- graph/component rule;
- factor provenance;
- Gaussian posterior against a tiny direct regression fixture;
- preference derivative finite differences;
- Laplace convergence;
- finite EI;
- no repeated actions;
- shared initialization;
- deterministic rerun;
- output schema.

Implementation bugs may be fixed.

Scientific choices above may not be changed because of smoke performance.

---

## 13. Preregistration

The scientific gate may run only after:

1. focused tests pass;
2. full repository tests pass;
3. preflight passes;
4. smoke passes;
5. implementation + handoff + `p1_gate.json` are committed;
6. exact preregistration Git SHA and config SHA-256 are recorded.

Run the scientific result from that exact commit.

Do not alter scientific code/config after preregistration.

If an implementation-only bug is discovered before a valid result:

- preserve the failed attempt;
- fix only the bug;
- create a new preregistration commit;
- rerun from scratch;
- document the implementation-only nature of the change.

Never overwrite a completed scientific result.

---

## 14. Primary metric and gate

Using raw `SH_Average_bc` only for retrospective evaluation, define:

\[
f^\star=\max_x f(x),
\qquad
f_{\min}=\min_x f(x).
\]

After five initial plus five BO-selected actions:

\[
r_5
=
\frac{
f^\star-\max_{x\in D_{10}}f(x)
}{
f^\star-f_{\min}
}.
\]

`f_star` and `f_min` must never enter model fitting, graph construction, preference generation, initialization, or acquisition.

Let:

\[
m_{\rm scalar}
=
{\rm median}_s\,r_{5,s}^{\rm scalar\_only},
\]

\[
m_{\rm pref}
=
{\rm median}_s\,r_{5,s}^{\rm full\_preference}.
\]

Define:

\[
I=1-\frac{m_{\rm pref}}{m_{\rm scalar}}.
\]

### PASS

PASS iff:

\[
I\ge0.25,
\]

equivalently:

\[
m_{\rm pref}\le0.75\,m_{\rm scalar}.
\]

If `m_scalar == 0`, report:

`GATE_UNINFORMATIVE_SCALAR_CEILING`

Do not switch metrics/horizons post hoc.

### Outcome precedence

```text
if preprocessing fails:
    PREPROCESSING_INVALID
elif smoke cannot be fixed without changing frozen science:
    IMPLEMENTATION_BLOCKED
elif any scientific decision exhausts inference schedule unreliably:
    INCONCLUSIVE_NUMERICAL
elif median scalar-only r_5 == 0:
    GATE_UNINFORMATIVE_SCALAR_CEILING
elif relative median improvement >= 0.25:
    PASS
else:
    FAIL_P1
```

---

## 15. Secondary diagnostics

Record but do not gate on:

- simple regret after each BO step;
- top-10% hit discovery time;
- action overlap;
- ESS and split-half discrepancy;
- Laplace iterations;
- factor counts/degrees;
- fraction of Sort1/Sort8 conflicts;
- wall time;
- factor-likelihood evaluation count.

Do not add another success criterion.

---

## 16. Repository layout

Keep this compact.

Create:

- `project/E3_GP2_P1_GATE_HANDOFF.md`
- `experiments/gp2_preference_bo/README.md`
- `experiments/gp2_preference_bo/configs/p1_gate.json`
- `experiments/gp2_preference_bo/run_p1_gate.py`
- one reusable source module under `src/conditioned_bo/` if needed
- one focused test file under `tests/`

Do not add more planning Markdown files.

Runner must support:

```text
--mode preflight
--mode smoke
--mode scientific
```

Outputs:

- `experiments/gp2_preference_bo/outputs/preflight/`
- `experiments/gp2_preference_bo/outputs/p1_gate_smoke/`
- `experiments/gp2_preference_bo/outputs/p1_gate/`

Final `p1_gate/` is immutable and must not be overwritten.

---

## 17. Required final outputs

Final `p1_gate/` must contain:

- `frozen_config.json`
- `provenance.json`
- `preprocessing_summary.json`
- `preference_bank.csv`
- `trajectory.csv`
- `inference_diagnostics.csv`
- `summary.json`
- `RESULTS.md`
- one two-panel diagnostic figure

Report:

- preregistration SHA;
- config hash;
- external source/hash provenance;
- candidate/graph/factor counts;
- per-seed scalar-only and full-preference `r_5`;
- both medians;
- relative median improvement;
- frozen verdict;
- numerical caveats;
- one-sentence interpretation.

Figure:

1. median/IQR normalized simple regret versus post-initial scalar evaluations;
2. paired per-seed scalar-only vs full-preference `r_5`.

---

## 18. Interpretation contract

### PASS

Record:

> Authentic ordinal information from independent high-throughput Gp2 developability assays materially improves retrospective scalar-yield BO under the frozen graph-Gaussian preference model. Proceed to a separate realistic-E3 gate for empirical decision saturation and current certified sparsity.

Do not add the final baseline suite in this task.

### FAIL_P1

Record:

> Under the frozen, scientifically defensible ordinal construction, the independent Gp2 high-throughput assays did not improve scalar-yield BO by the preregistered amount. Do not tune assay choice, graph degree, temperature, initialization, horizon, or metric. Preserve the result and move to the preselected oxide backup.

### INCONCLUSIVE_NUMERICAL

Do not change scientific settings.

The exact preregistered commit may be rerun on a stronger machine, including Colab/A100, with identical scientific configuration and seeds.

---

## 19. Prohibitions

Do not:

- reopen the paper story or T1--T4;
- change completed synthetic E3 results;
- inspect SH yield to choose assays/edges;
- tune `k`, `tau_pref`, or graph precision;
- retain only target-agreeing preferences;
- use Sort10;
- use proxy magnitudes as scalar GP observations;
- use IQ yield as another objective;
- add batch/multiobjective BO;
- train a protein language model/neural surrogate;
- implement adaptive conditioning here;
- add final E3 baselines;
- change the 25% threshold or five-step horizon;
- switch scientific seeds after a valid run;
- overwrite a negative/inconclusive result.

The task is complete when the prospective P1 verdict is known and reproducibly committed.
