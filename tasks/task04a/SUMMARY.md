# Task 04A — implementation and revised preflight status

Task 04A implementation and the single prescribed CPU smoke are complete. The local
smoke used seed 0, D=6, n=64, all three regimes, 128 predictive points, and 256 EI
candidates. It is wiring evidence only and cannot authorize Task 04B.

The eventual full summary must answer:

1. Did exact GP/Vecchia nesting and every normalization identity pass?
2. Were U/P safe under Gaussian truth?
3. Did P beat U under explicit interaction truth at n=128–256?
4. Did pairwise structure improve W, or was a unary correction sufficient?
5. Did P improve q=1 EI decisions by medium data sizes?
6. Did the fixed interaction remain identifiable across realized contexts?
7. Did CUDA batching scale locally without hidden global factorization?
8. Do the frozen gates imply GO, NO-GO, or strong NO-GO?

## Quantitative smoke evidence

- All three cases were finite; U/P converged in 8–9 closure calls. Fits took
  `0.0034–0.0188 s` for U and `0.0063–0.0067 s` for P on CPU.
- In G, G0 had zero numerical KL/regret. U/P correction KL was
  `0.000328/0.000426`, and neither changed the selected decision.
- In W, G0/U/P conditional KL was `0.7758/0.7764/0.7773`. All happened to select an
  oracle-optimal candidate in this one smoke realization, despite imperfect EI ranks.
- In I, G0/U/P conditional KL was `0.002604/0.002530/0.002440`; U/P EI Spearman was
  about `0.99997`, and all selected an oracle-optimal candidate.

The I result is an important warning: with the frozen interaction coefficient `2`,
the positive control is weak on this realized local geometry.

## Bounded interaction diagnostic

At the maintainer's request, a second local diagnostic isolated regime I at seeds
`0–2`, `D=6`, `n={64,128,256}`, 256 predictive points, and 512 EI candidates. This
was deliberately smaller than the full experiment and did not alter the frozen
model or gates.

| n | mean KL(I || G0) | mean KL(I || U) | mean KL(I || P) | mean P gain over U | P KL wins |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 0.002510 | 0.002180 | 0.001907 | 0.000273 | 3/3 |
| 128 | 0.000909 | 0.000801 | 0.000722 | 0.000078 | 3/3 |
| 256 | 0.000301 | 0.000246 | 0.000214 | 0.000032 | 3/3 |

The absolute Gate-B requirement is a P-over-U KL gain of at least `0.01` nat at
`n=128` and `n=256`. In this diagnostic, even an exact P oracle could improve by at
most U's remaining KL, only `0.000801` and `0.000246` nat on average. The gate is
therefore unreachable on these cases, rather than merely difficult.

The failure mechanism is local-reference contraction. Median candidate conditional
standard deviation fell from about `0.144` at `n=64`, to `0.068` at `n=128`, to
`0.033` at `n=256`. The smooth raw-output RBF interaction consequently becomes
nearly constant over each conditional support, and its constant component cancels
under normalization. The optimizer itself showed no obvious failure: all fits
converged, P beat U in conditional KL in all nine cases, and the learned pair matrix
had cosine `0.42–0.67` with the known oracle direction. However, shrinkage dominated
the tiny likelihood signal: the learned projection onto the true coefficient was
only `0.024–0.108` of its amplitude. The true coefficient incurs a `20`-nat L2
penalty, whereas the available summed likelihood signal was only about `0.08–0.18`
nat per case.

All G0/U/P decisions still had zero finite-set oracle regret in all nine cases. P's
small density improvement therefore provided no evidence of decision value.

As a non-gating mechanism check, applying the child RBFs to the local standardized
residual `z=(y-mu)/sigma` while retaining the same realized neighbor contexts gave a
mean oracle-to-Gaussian conditional KL of `0.021–0.037` at `n=64`, `0.032–0.047` at
`n=128`, and `0.043–0.056` at `n=256`. This does not establish learnability, because
the data were not regenerated or refit under that alternative. It does show that a
reference-standardized child energy is the smallest plausible way to preserve a
meaningful positive control as local uncertainty contracts.

## Standardized-child revision and preflight

The approved smallest correction was then implemented: U, P, and interaction truth
now evaluate only the child RBFs at `z=(y-mu_i)/sigma_i`; causal neighbor summaries
remain functions of earlier raw observations. The exact G0 limit, convexity, scalar
normalization, and directed factorization are unchanged. A new invariance test checks
that equal local z values give equal corrections under different means and scales.

The frozen CPU preflight used seeds `0–2`, D=6, G/I, `n={64,128,256}`, 256 predictive
points, and 512 candidates. All fits converged and Gaussian safety passed.

| n | mean KL(I || G0) | mean KL(I || U) | mean KL(I || P) | P gain | relative gain | P KL wins |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 0.029195 | 0.026745 | 0.023649 | 0.003096 | 11.6% | 3/3 |
| 128 | 0.039361 | 0.040416 | 0.034127 | 0.006290 | 15.6% | 3/3 |
| 256 | 0.048663 | 0.047074 | 0.030598 | 0.016476 | 35.0% | 3/3 |

The revision successfully fixed the vanishing positive control. At n=256, P passed
the density preflight with a `0.0165`-nat, 35% improvement over U. At n=128 it missed
the prespecified 20% relative requirement despite winning all three seeds. Gaussian
median correction KL was `0.000894` for U and `0.001967` for P, with no material
median regret increase.

The decision-relevance preflight failed decisively. U selected an oracle-optimal
candidate in all six I cells at n=128/256. P had no decision wins and introduced
`0.0347` normalized regret in one n=128 case. Thus the revised construction provides
credible medium-data conditional-density evidence, but no evidence that pairwise
structure improves q=1 decisions beyond U.

The frozen result is `PAUSE_FULL_STUDY`: safety and compute passed; density learning
failed at n=128 and decision relevance failed. The A100 profile must not be launched
from the present contract, and its thresholds must not be relaxed retrospectively.

## Withheld-seed decision diagnostic

Task 04A-D then tested seeds 100–107 locally without changing the model or truth. It
used natural active-space candidate sets, teacher-free G0-matched real pairs, and
same-mean/scale context swaps. All 24 cases completed and all fits converged.

The official classification is **`INVALID`**: all 16 I cases had zero qualifying real
pairs under the frozen criteria, versus at least 32 required. Only `6.64%` of n=256
counterfactual oracle pairs had normalized EI contrast at least 1%, versus 75%
required. U had no natural decision opportunity in any n=256 seed; U/P median regret
was zero. The absence of pairs was genuine—in the seed-100 wiring case, the closest
low/high-context G0-EI mismatch was `2.10%`, exceeding the frozen `0.5%` limit.

Density learning nevertheless replicated: at I n=256, U/P mean KL was
`0.045686/0.022366`, a `0.023321`-nat or `51.0%` P gain with 8/8 wins. Gaussian
safety passed. The evidence supports pairwise conditional-density learnability but
does not establish, or even furnish a valid natural opportunity to test, q=1 decision
value. Do not reinterpret `INVALID` as `LOCAL_GO` or relax the panel after observation.

## Current answers

1. GP/Vecchia nesting, scalar normalization, convexity, zero-correction recovery, and
   Gaussian/W/I EI checks pass the local test suite.
2. U/P were safe in the one Gaussian smoke case; five-seed evidence is pending.
3. Standardization made the I signal meaningful. P beat U in all revised preflight
   cells and passed the density criterion at n=256, but missed it at n=128.
4. P did not beat U in W smoke KL; this is not a gate evaluation.
5. U retained zero regret in all six medium-data I cells; P did not improve a decision
   and worsened one. Pairwise decision value is therefore unsupported.
6. Controlled contexts have a nonconstant density ratio and KL above `0.05`, but the
   realized I test distribution was much closer to G0 (`0.002604` mean KL).
7. CPU timing/output plumbing passes; CUDA parity and scaling remain Colab-only.
8. **Task 04B remains unauthorized and the A100 study stays disabled.** The withheld
   diagnostic was invalid for decision comparison because the frozen oracle/candidate
   construction supplied no qualifying natural pairs and almost no counterfactual EI
   contrast. Further work requires a new oracle contract designed prospectively around
   decision-relevant tail variation, not threshold relaxation.
