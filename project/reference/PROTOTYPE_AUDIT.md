# Prototype Notebook Evidence Audit

**Audit date:** 2026-08-20
**Scope:** provenance and consistency audit only
**Source files changed by this audit:** none
**Prototype notebooks changed by this audit:** none

> **Post-audit provenance note (2026-08-20):** `PAPER_STORY.md` and `PROJECT_HANDOFF.md` were subsequently updated at the human author's request. The source hashes recorded below identify the exact inputs reviewed during the audit; they are not hashes of the later approved revisions.
>
> **Post-audit structural note (2026-08-21):** the nonlinear-PDE T2-B
> construction was subsequently proved and regression-tested; see
> `T2B_NONLINEAR_PDE_AUDIT.md`. That result supersedes only the structural
> blocker statements in this historical provenance audit. Its conclusions about
> empirical inference allowances remain current.
>
> **Post-audit finite-sample note (2026-08-21):** the exact-rejection inference
> construction was subsequently proved and its locked prospective
> reflection-symmetry finite-grid pilot passed; see
> `INFERENCE_CERTIFICATION_IMPLEMENTATION_HANDOFF.md`, `THEORY.tex`, and
> `EXPERIMENTS.md`. References below to a missing end-to-end certificate or
> current certification blockers are preserved as dated audit provenance. The
> later guarantee is limited to the exhaustive 401-action symmetry grid and
> exact-rejection backend; it does not upgrade the prototype notebooks or
> establish a continuous-action certificate.

## Verdict vocabulary

- `SAVED_OUTPUT_VERIFIED`: the original notebook contains the result as saved output and an isolated replay reproduced it.
- `REPLAY_VERIFIED`: an unchanged isolated replay produced the result, but the original notebook did not contain saved output for it.
- `CODE_EXPECTATION_ONLY`: the notebook states or configures the claim but supplies no saved or replayed result.
- `NOT_PRESENT`: the claimed evidence is absent from the supplied notebook.
- `DISCREPANCY`: the replay conflicts with the value or behavior stated in a project source file at its reported precision.
- `THEORY_ALIGNMENT_RISK`: the notebook calculation is not yet covered by the current proved theory or rigorous inference contract.

These verdicts concern evidence provenance. They do not upgrade a prototype replay into clean algorithmic evidence or resolve any `BLOCKER` in `THEORY.tex`.

## Executive finding

All three supplied notebooks are valid notebook files and replay successfully without editing under the locked local environment. The symmetry notebook is the only artifact that arrived with execution counts and saved outputs. Its saved numerical evidence reproduces exactly. The two PDE notebooks arrived completely unexecuted; their numerical evidence is recoverable from deterministic replay, but several exact values in `EXPERIMENTS.md` do not match that replay.

The broad mechanism claims are supported as historical prototype evidence:

- symmetry stops with 12 of 40 factors and chooses the same grid action under post-hoc full conditioning;
- the linear PDE replay stops with 50 of 576 factors and shows the reported active/full importance-sampling separation;
- the nonlinear PDE replay stops with 40 of 576 factors and shows the reported active/full importance-sampling separation;
- both PDE scaling functions reproduce the reported factor-count and ESS trends.

The following claims are not supported by the supplied artifacts:

- the symmetry result with approximately 15 factors after adding a Monte Carlo allowance;
- the earlier symmetry scaling experiment with $N=20,40,80,160$ and approximately 10 active factors.

None of the notebooks establishes a rigorous end-to-end full-target action certificate under the current theory. This agrees with the blockers already recorded in `PROJECT_HANDOFF.md` and `THEORY.tex`.

## Artifact provenance

The source hashes below were recorded before replay and rechecked afterward.

| Artifact | SHA-256 | Original execution state |
|---|---|---|
| `DEC_Symmetry_Continuous_BO_Demo.ipynb` | `74631161db99d37b386a49bcdffcb9a1e364a8330c3eb7eb1ae76a7569705636` | 12 executed code cells; saved outputs in 9 cells; metadata reports Python 3.13.5 |
| `DEC_PDE_Certified_BO_Demo.ipynb` | `b3fb6fcb6d937e592edcfedc19a1993d0812a2c493a53ba5fcee007e3fb62ee9` | 9 code cells; all execution counts null; no outputs |
| `DEC_Nonlinear_PDE_BO_Demo.ipynb` | `73459edc0545ea2470a0cba5ab3cf60d18508b78b0acae08068790905b48e6fd` | 11 code cells; all execution counts null; no outputs |

Source-of-truth hashes:

| Artifact | SHA-256 |
|---|---|
| `PROJECT_HANDOFF.md` | `c3012ea505afa9aed5db08a54c24d9b249da98134c536455e1c4d0a6748ef14a` |
| `PAPER_STORY.md` | `d866947eb2ce027c3dc3300d377155f9a94d13f3a5714a93b29b6ce9db2f3e75` |
| `PROBLEM_FORMULATION.tex` | `6720ce470fee3fa99659b04b06c7a2336a486e7a395fdc0b2b7475fabc2e05ec` |
| `THEORY.tex` | `4c67b723a0421e8937748fba320b16f64d728d8504d8a0ad5e455f05b042996e` |
| `EXPERIMENTS.md` | `261a740e9ecbe56274063ebcd557656097a7239df34db48a6b76288af1df907d` |
| `RELATED_WORK.md` | `4885405fb780ce995c2570a5320734c0733f84c86d24c40d9c235e1a1f5a2e02` |

## Replay protocol

Each notebook was copied byte-for-byte to its own directory under a temporary root outside the repository. The copy was executed with `jupyter nbconvert --execute`, a 3,600-second per-cell timeout, and `MPLBACKEND=Agg`. Generated notebooks, CSV files, plots, Jupyter state, IPython state, and Matplotlib caches remained in that temporary root.

Environment:

- Darwin 24.6.0, arm64;
- Python 3.12.13;
- NumPy 2.3.5;
- SciPy 1.16.3;
- pandas 3.0.1;
- Matplotlib 3.11.1;
- Jupyter Core 5.9.1;
- nbconvert 7.17.1.

The replay used the authored seeds and configurations without tuning.

| Notebook | Wall time | Maximum resident set size | Result |
|---|---:|---:|---|
| Symmetry | 14.40 s | 297.7 MiB | completed |
| Linear PDE | 11.57 s | 654.3 MiB | completed |
| Nonlinear PDE | 10.50 s | 620.8 MiB | completed |

## Frozen notebook configurations

| Setting | Symmetry | Linear PDE | Nonlinear PDE |
|---|---:|---:|---:|
| Seed | 123 | 702 | 911 |
| Primary latent dimension | 80 | 576 | 576 |
| Total factors | 40 | 576 | 576 |
| Active/reference samples | 80,000 | 6,000 | 6,000 |
| Full-validation samples | same reference sample | 6,000 independent Laplace-proposal samples | 6,000 independent Laplace-proposal samples |
| Action grid | 401 points on $[-0.58,-0.06]$ | 117 lattice sites | 117 lattice sites |
| Batch size | 3 | 10 | 10 |
| Main stopping tolerance | 0.03 log-acquisition units | 0.06 EI units | 0.06 EI units |
| Scaling tolerance | not implemented | 0.075 EI units | 0.075 EI units |

The symmetry notebook uses $u_x(f)=\exp(0.7 f(x))$ and optimizes log acquisition. The PDE notebooks use expected improvement with incumbent $0.55$.

## Claim matrix: reflection symmetry

Primary evidence locations are cells 2, 6, 14, 16, 18, and 23 of `DEC_Symmetry_Continuous_BO_Demo.ipynb`.

| Project claim | Notebook/replay evidence | Verdict |
|---|---|---|
| $N=40$ factors | Saved and replayed configuration: 40 | `SAVED_OUTPUT_VERIFIED` |
| Structural loop stops with $M=12$ | Saved and replayed stopping state: 12 | `SAVED_OUTPUT_VERIFIED` |
| Sparse leader $\widehat x\approx-0.2082$ | Saved and replayed: `-0.2082` | `SAVED_OUTPUT_VERIFIED` |
| Worst challenger $x_c\approx-0.359$ | Saved and replayed: `-0.3590` | `SAVED_OUTPUT_VERIFIED` |
| Structural log-acquisition certificate $\epsilon\approx0.0261$ | Saved and replayed: `0.02614054507118245` | `SAVED_OUTPUT_VERIFIED` |
| Fully conditioned acquisition selects the same action | Saved and replayed full-grid maximizer: `-0.2082`; observed log regret: `0.0` | `SAVED_OUTPUT_VERIFIED` |
| Approximately $M=15$ after a conservative Monte Carlo allowance | No such calculation, configuration, output, or result appears in the notebook | `NOT_PRESENT` |
| Scaling over $N=20,40,80,160$ with $M\approx10$ | No scaling code or output appears in the notebook | `NOT_PRESENT` |
| Result instantiates the current T1--T3 acquisition-gap certificate | Notebook bounds a difference of log acquisitions for exponential utility, not the raw acquisition gap defined in the current formulation | `THEORY_ALIGNMENT_RISK` |
| Continuous-action certificate | The notebook uses a 401-point grid and explicitly omits an optimization/discretization remainder | `THEORY_ALIGNMENT_RISK` |

The full-conditioning validation uses the same 80,000 Gaussian reference draws as the active calculation. It holds factor energies out until validation, but it is not an independent Monte Carlo sample. Calling it “held-out full conditioning” is accurate only in the factor-evaluation sense.

## Claim matrix: linear PDE

Primary evidence locations are cells 0, 2, 6, 8, 10, 13, and 15 of `DEC_PDE_Certified_BO_Demo.ipynb`. The original notebook contains no saved execution output, so all numerical verdicts below come from the unchanged replay.

| Project claim | Unchanged replay | Verdict |
|---|---:|---|
| $N=576$, $M=50$ | $576,50$ | `REPLAY_VERIFIED` |
| Total empirical EI certificate $0.0563$ | `0.05618603416306421` (prints `0.0562`) | `DISCREPANCY` |
| Structural component $0.0398$ | `0.038970375784585884` (prints `0.0390`) | `DISCREPANCY` |
| Inference component $0.0246$ | `0.024681355146455822` (prints `0.0247`) | `DISCREPANCY` |
| Active-target GP-reference ESS $85.0\%$ | `85.06204637962801%` (prints `85.1%`) | `DISCREPANCY` at the stated one-decimal precision; “about 85%” is supported |
| Full-target GP-reference ESS $12.0\%$ | prints `12.0%` | `REPLAY_VERIFIED` |
| Full-target Laplace-preconditioned ESS $69.4\%$ | prints `69.4%` | `REPLAY_VERIFIED` |
| Active and held-out full targets select the same action | both select lattice site `(14, 11)` | `REPLAY_VERIFIED` |
| Observed held-out EI regret is zero | prints `0.0000` on the action grid | `REPLAY_VERIFIED` |
| Scaling $M\approx50\to60$ for $N=324\to1600$ | $M=(50,50,50,50,60)$ | `REPLAY_VERIFIED` |
| Scaling active ESS remains approximately $81\%$--$86\%$ | $85.32,84.77,84.72,85.79,80.96\%$ | `REPLAY_VERIFIED` |
| Scaling full ESS falls from $34.1\%$ to $0.78\%$ | $34.111\%$ to $0.77997\%$ | `REPLAY_VERIFIED` |

The total is internally consistent with the replayed decomposition:


\[
-0.0074656968 + 0.0389703758 + 0.0246813551
= 0.0561860342.
\]

## Claim matrix: nonlinear PDE

Primary evidence locations are cells 0, 2, 6, 8, 10--14, and 16 of `DEC_Nonlinear_PDE_BO_Demo.ipynb`. The original notebook contains no saved execution output, so all numerical verdicts below come from the unchanged replay.

| Project claim | Unchanged replay | Verdict |
|---|---:|---|
| $N=576$, $M=40$ | $576,40$ | `REPLAY_VERIFIED` |
| Total empirical EI certificate $0.04758$ | `0.049522479694193114` | `DISCREPANCY` |
| Structural component $0.03626$ | `0.03874403301354687` | `DISCREPANCY` |
| Inference component $0.02347$ | `0.023528500294517668` | `DISCREPANCY` |
| Sparse gap $-0.01216$ | `-0.01275005361387142` | `DISCREPANCY` |
| Active-target GP-reference ESS $84.3\%$ | `84.3854775843621%` (prints `84.4%`) | `DISCREPANCY` at the stated one-decimal precision; “about 84%” is supported |
| Full-target GP-reference ESS $6.2\%$ | prints `6.2%` | `REPLAY_VERIFIED` |
| Full-target Laplace-preconditioned ESS $58.9\%$ | prints `58.9%` | `REPLAY_VERIFIED` |
| Active and held-out full targets select the same action | both select lattice site `(14, 12)` | `REPLAY_VERIFIED` |
| Observed held-out EI regret is zero | prints `0.000000` on the action grid | `REPLAY_VERIFIED` |
| Scaling uses $N=324,576,900,1296,1600$ | replay uses exactly those sizes | `REPLAY_VERIFIED` |
| Scaling keeps $M=40$ throughout | $M=(40,40,40,40,40)$ | `REPLAY_VERIFIED` |
| Scaling active ESS remains approximately $84\%$--$86\%$ | $84.54,84.90,84.69,84.53,85.52\%$ | `REPLAY_VERIFIED` |
| Scaling full ESS falls from $30.0\%$ to $1.8\%$ | $29.953\%$ to $1.8048\%$ | `REPLAY_VERIFIED` |
| At $N=1600$, $N/M=40$ | $1600/40=40$ | `REPLAY_VERIFIED` |

The total is internally consistent with the replayed decomposition:

\[
-0.0127500536 + 0.0387440330 + 0.0235285003
= 0.0495224797.
\]

## Mathematical and methodological alignment

### Acquisition object

The current formulation defines the decision object as the raw acquisition gap

\[
G_Q(x,\widehat x)=\alpha_Q(x)-\alpha_Q(\widehat x).
\]

The symmetry notebook instead bounds

\[
\log\alpha_Q(x)-\log\alpha_Q(\widehat x)
\]

for exponential utility. Its argument relies on utility-tilted measures whose added term is linear in the latent field. That may admit a separate valid derivation, but it is not the theorem currently stated in `PROBLEM_FORMULATION.tex` and `THEORY.tex`. The exact replay therefore verifies the historical calculation, not alignment with T1--T3.

### Concrete influence operator

All notebooks construct a comparison matrix $A$, check $A\succ0$, and use $A^{-1}$ for structural influence. The code provides plausible global derivative and curvature bounds for the authored factors. It does not prove or cite the block covariance comparison result needed to establish the required inequality uniformly along every active-to-full path. Positive definiteness alone does not close that proof obligation. This is exactly the concrete T2 blocker already recorded in the theory ledger.

### Inference error

The PDE notebooks use a normal critical value multiplied by an estimated self-normalized importance-sampling standard error. The Bonferroni factor covers the finite action grid within one calculation, but the code does not provide a finite-sample ratio bound or simultaneous coverage over the data-dependent leader, challenger, and adaptive refinement rounds. These terms are empirical/asymptotic error allowances, not rigorous $B_{\mathrm{infer}}$ certificates.

ESS is an overlap diagnostic. The Laplace-importance calculations use exact target/proposal weights for the sampled proposal, but 6,000 weighted samples and action agreement do not by themselves certify the fully conditioned optimum.

### Optimization and scaling

- Symmetry searches a dense one-dimensional grid but does not include a continuous optimization/discretization remainder.
- The PDE action spaces are finite 117-site grids, so their reported zero regret is grid-relative.
- The PDE scaling runs use tolerance $0.075$, not the main-case tolerance $0.06$.
- Scaling uses one seed per size and fewer particles as the domain grows: 3,000, 3,000, 2,500, 2,200, and 2,000.
- The scaling helpers report factor counts, empirical envelope values, and ESS. They do not perform a strong full-target action/regret validation for every size.
- No random, local, static, or posterior-oriented factor-selection baseline appears in these prototypes.

### Framing and terminology

The notebooks use “Decision-Equivalent Conditioning,” “Decision-Certified Conditioning,” and `DEC`. The current project deliberately has no final method name. Those labels should remain historical notebook terminology and should not propagate into new code or paper claims without a human naming decision.

The notebook titles and prose sometimes use “certified” or “rigorous structural” more strongly than the current proof ledger permits. The source-of-truth files correctly retain `BLOCKER` status for the concrete influence construction and rigorous inference instantiation.

## Proposed source corrections — not applied

The source-of-truth files should not silently replace legacy values with one environment-specific replay. The preferred correction is to preserve both provenance layers.

1. In `PROJECT_HANDOFF.md` and `PAPER_STORY.md`, mark the symmetry $M\approx15$ Monte Carlo-adjusted result and $N=20,40,80,160$ scaling result as **untraced legacy claims not present in the archived notebook**. Remove them from active evidence if no source artifact can be located.
2. In the linear-PDE entry, retain the qualitative “about 85% versus 12%” statement, but add a replay note: `2026-08-20 locked-environment replay: active ESS 85.062%, total 0.056186, structural 0.038970, inference 0.024681.`
3. In the nonlinear-PDE entry, add a replay note: `2026-08-20 locked-environment replay: active ESS 84.385%, total 0.049522, structural 0.038744, inference 0.023529, sparse gap -0.012750.`
4. In `EXPERIMENTS.md`, label the currently listed exact PDE numbers as legacy reported values and place the replayed values beside them. Do not choose one set as canonical until clean reproduction code fixes the environment and output schema.
5. In E1, state explicitly that the archived prototype uses exponential utility and a log-acquisition certificate. Do not cite it as a direct T1--T3 instantiation unless a separate log-acquisition result is proved or the clean experiment is reformulated around the raw acquisition gap.
6. Continue describing all ESS, asymptotic SNIS allowances, and held-out action agreements as empirical diagnostics rather than rigorous certificates.

## Recommended clean E1 reproduction path

1. Preserve the archived symmetry notebook unchanged and use its exact replay values as a regression target for a historical-mechanism mode.
2. Extract deterministic model construction, factor metadata, influence construction, active-factor selection, acquisition evaluation, and result serialization into clean modules outside `notebooks/prototypes/`.
3. Add unit tests for the OU conditional representation, factor-gradient bound, comparison-matrix construction, factor-cache laziness, envelope decomposition, and deterministic replay target.
4. Resolve the concrete T2 comparison-theorem assumptions before calling the structural quantity a certificate.
5. Prefer a theory-aligned E1 variant using ordinary EI and the raw acquisition gap. If the log-acquisition/exponential-utility formulation is retained instead, treat the required theorem as a separate proof task rather than assuming T1 covers it.
6. Add an explicit finite-grid inference-error construction and confidence allocation across adaptive rounds before making an end-to-end certification claim.
7. Only after the clean single-seed mechanism reproduces should E1 add repeated coverage and the prospective random, Euclidean-local, graph-local, and static-influence baselines specified in `EXPERIMENTS.md`.

## Audit conclusion

The prototype package is recoverable and internally coherent enough to guide clean reproduction. It does not fully substantiate every historical number in the project documents, and it does not resolve the paper's two declared certification blockers. Human review should decide whether to annotate the source files with the provenance distinctions above before clean E1 implementation begins.
