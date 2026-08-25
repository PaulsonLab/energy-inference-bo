# E2 FULL-Shadow Backend Rescue

Terminal classification: **`FULL_REFERENCE_BACKEND_UNRESOLVED`**

Status: **DEVELOPMENT ONLY — NO PROSPECTIVE E2 SOURCE SEED ACCESSED**

This task started from fetched `main` SHA
`96a36c32cdff64fc01159d49cd3aa771b9c2e613`. It used calibration source seed
`2026082401` and the permanently development-only held-out source seed
`3321078991`. The latter is the big-endian first eight bytes of
`SHA256("E2_FULL_SHADOW_BACKEND_VALIDATE_V1")`, reduced modulo `2^32-1`;
the complete digest is
`467842fc7f7b61d3b6424f6978763c87f5f67c8cf3fddbe8acef67de7955d1f0`.
It is distinct from prospective seeds `4215109622`, `1083605379`, and
`4045758625`, and repository search found no prior E2 scientific use.

Preregistration `0dcb76f5f2d053e098b472ac9984182b837295b5` remains superseded/unrun.
No new scientific preregistration was created. The BO problem, PDE target,
factor selection, stopping tolerance `0.060`, E2 A1--A6 thresholds, and E3
were not changed. The implementations here are held-out FULL shadows only;
they do not feed deployable selection or the timed `FULL` baseline.

## Starting blocker

The preserved diagnostic showed healthy n=24 baseline Laplace-SNIS ESS
fractions (`0.510`--`0.559`) and successful frozen 16,384-sample escalation.
At n=40, every early/middle/late 8,192- and 16,384-sample pair failed relative
ESS; mean fractions through 32,768 were `0.19925`, `0.20196`, and `0.17850`.
Worst cross-batch action regret was `0.02444192166936543`. Every Laplace mode
converged in three accepted Newton steps with gradient infinity norm below
`5e-14`. Increasing sample count grows absolute ESS but does not repair the
proposal/target-overlap fraction.

The prior record was hash-locked and was not overwritten.

## Curvature-tempered proposal pilots

Every precision was mechanically SPD by dense Cholesky. Entries are pilot ESS
fractions from 2,048 calibration-only samples. Exact ties would favor the
existing baseline, then the listed lambda order.

| n | checkpoint | baseline 1.1 H^-1 | lambda 1.00 | lambda 0.80 | lambda 0.60 | lambda 0.40 | lambda 0.20 | selected |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 24 | early | 0.521180 | 0.015398 | 0.204023 | 0.485270 | 0.695020 | 0.288071 | lambda 0.40 |
| 24 | middle | 0.468044 | 0.027899 | 0.179161 | 0.468733 | 0.646743 | 0.370172 | lambda 0.40 |
| 24 | late | 0.555807 | 0.080088 | 0.161368 | 0.486442 | 0.680412 | 0.364420 | lambda 0.40 |
| 40 | early | 0.227653 | 0.004321 | 0.028645 | 0.068014 | 0.146192 | 0.082347 | baseline |
| 40 | middle | 0.113746 | 0.001059 | 0.018710 | 0.079632 | 0.389601 | 0.089934 | lambda 0.40 |
| 40 | late | 0.198859 | 0.001621 | 0.016717 | 0.098795 | 0.346378 | 0.138638 | lambda 0.40 |

Curvature tempering materially improved pilot ESS in five of six states,
including about `3.43x` at n=40 middle and `1.74x` at n=40 late. It did not
repair n=40 early: every tempered candidate was worse than the baseline.

The resulting `(n, checkpoint)` proposal mapping was saved and hash-locked at
`932e757d3a710e93e591a070eb45690970f0205b0789b1ba79d421cb7e612078`
before held-out validation. Pilot samples were discarded; production used two
fresh independent 8,192-sample batches.

## Calibrated independent SNIS

| n | held-out checkpoint | ESS A/B | actions A/B | max vector difference | max reciprocal regret | gate |
|---:|---|---|---|---:|---:|---|
| 24 | early | 0.670433 / 0.674002 | 348 / 348 | 0.009598 | 0 | PASS |
| 24 | middle | 0.669784 / 0.685404 | 324 / 323 | 0.007519 | 0.008820 | FAIL |
| 24 | late | 0.675584 / 0.687274 | 203 / 203 | 0.006735 | 0 | PASS |
| 40 | early | 0.194438 / 0.173395 | 699 / 699 | 0.020219 | 0 | FAIL |
| 40 | middle | 0.334872 / 0.270845 | 739 / 940 | 0.003849 | 0.007061 | FAIL |
| 40 | late | 0.323849 / 0.315913 | 939 / 939 | 0.004117 | 0 | PASS |

The held-out n=40 gate requires each ESS fraction at least `0.30`, exact batch
action agreement, maximum vector difference at most `0.01`, and maximum
reciprocal regret at most `0.005`. It failed early and middle. The n=24 health
gate also failed at held-out middle. On the calibration seed, every n=24 fresh
batch remained decision-consistent with the reconstructed prior pooled
32,768x2 baseline: maximum vector differences were at most `0.008370` and
maximum reciprocal regret at most `0.004791`; the fresh early pair itself
nonetheless missed the `0.01` vector gate at `0.010465`.

Classification: `CALIBRATED_SNIS_BACKEND_FAIL_REQUIRES_INDEPENDENCE_MH`.

## Laplace independence MH

The implementation used the exact target/proposal log-weight acceptance ratio,
eight deterministic independent chains, one chain initialized at the Laplace
mode, seven initialized from independent selected-proposal draws, no thinning,
and two calibration-only length schedules. Required scalar diagnostics covered
target energy, full factor energy, candidate-top-action utilities, and
leader/challenger gaps. MCMC ESS is autocorrelation ESS, not importance-weight
ESS.

Both schedules failed calibration. Under the longer schedule (2,048 burn-in,
8,192 retained per chain), median chain acceptance was `0.374`--`0.661` and
minimum gap ESS was `6,426`--`25,682`, but the Laplace-mode chain accepted
exactly zero proposals in all six states. Consequently maximum split R-hat was
`7.87`--`7.93` at n=24 and `13.03`--`13.19` at n=40. Independent group vector
differences were `0.0152`--`0.0460`, also above `0.01` in every state. This is
a genuine sticky independence-chain failure, not an optimizer or R-hat coding
failure.

No independence-MH setting passed calibration, so no independence-MH held-out
validation was run.

Classification: `INDEPENDENCE_MH_BACKEND_FAIL_REQUIRES_ELLIPTICAL_SLICE`.

## Elliptical slice FULL

The implementation used the exact current Gaussian BO reference for fresh
ellipse directions, `-E_FULL` as the log likelihood, and a randomized initial
bracket `[theta-2*pi, theta]`. Exact vectorized factor energy agrees with the
existing factor loop. Eight independent chains used no thinning and the same
decision/convergence diagnostics.

The longer calibration schedule used 1,024 burn-in and 4,096 retained draws
per chain. Results were:

| n | checkpoint | max R-hat | min gap ESS | group actions | max vector difference | max regret | result |
|---:|---|---:|---:|---|---:|---:|---|
| 24 | early | 1.002653 | 8,480 | 347 / 348 | 0.012283 | 0.011532 | FAIL |
| 24 | middle | 1.003138 | 6,084 | 371 / 371 | 0.005807 | 0 | PASS |
| 24 | late | 1.003096 | 5,881 | 371 / 371 | 0.003618 | 0 | PASS |
| 40 | early | 1.005249 | 2,780 | 900 / 900 | 0.023667 | 0 | FAIL |
| 40 | middle | 1.010497 | 2,545 | 659 / 859 | 0.012037 | 0.006423 | FAIL |
| 40 | late | 1.007923 | 1,849 | 939 / 740 | 0.005969 | 0.007712 | FAIL |

All n=24 pooled estimates remained consistent with the prior reliable reference:
maximum prior-reference vector difference was `0.007927` and maximum reciprocal
regret was `0.004791`. Elliptical slice used about `2.73`--`2.76` factor-energy
evaluations per transition at n=24 and `3.91`--`3.97` at n=40. However, the
frozen calibration gate is conjunctive; exact group action, vector, regret,
and R-hat failures above cannot be waived.

No elliptical-slice setting passed calibration, so no elliptical-slice
held-out validation was run.

## Resources

| backend phase | wall time | isolated worker range | peak RSS | scope |
|---|---:|---:|---:|---|
| calibrated SNIS | 28.08 s | 0.48--2.97 s/state | 0.961 GB | calibration + held-out validation + n=24 prior reconstruction |
| independence MH | 46.71 s | 1.23--7.24 s/state | 0.373 GB | two calibration schedules only |
| elliptical slice | 63.08 s | 2.10--8.55 s/state | 0.297 GB | two calibration schedules only |
| complete rescue | 137.87 s | — | 0.961 GB | all allowed stages |

For the longer n=40 schedules, factor-energy work was `131,084,800` evaluations
per state for independence MH and `256,131,200`--`260,244,800` per state for
elliptical slice. Detailed per-chain/stage timing, work, R-hat, ESS, MCSE,
actions, and top-five acquisitions are in the CSV and JSON records.

## Decision and recommendation

**Do not create a replacement E2 preregistration and do not spend a prospective
source seed.** None of the three permitted standard backends passed its
calibration/held-out protocol in the mandated order. The simplest trustworthy
FULL-shadow backend for n=40 therefore remains unidentified.

No backend-specific FULL-shadow reliability rule should be frozen for a future
E2 run from these results. In particular, do not weaken exact action agreement,
the `0.01` acquisition-vector threshold, the `0.005` reciprocal-regret margin,
the `0.30` SNIS ESS margin, or the MCMC `R-hat <= 1.01` / gap-ESS `>= 1000`
requirements merely to make these development cases pass. The current
scientific FULL-reference rule is unchanged; it remains unusable at n=40 with
the current backend. Per the task boundary, no fourth sampler is proposed or
implemented here.
