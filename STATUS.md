# Project Status

## Thesis
Decision-adapted integration may reduce Monte Carlo acquisition cost when utility-tilted worlds differ sharply from ordinary posterior worlds.

## Phase
WELDED_BEAM_SHIFT_NEGATIVE_REVIEW_REQUIRED

## Claims

| Claim | Status | Evidence |
| --- | --- | --- |
| C1 — Standard-MC relative variance equals decision-shift chi-square divergence divided by sample count | `VERIFIED_IN_CODE` | Exact identity; empirical iid relative variance agrees within 3.16% across the frozen sample counts ([variance table](experiments/rare_mode_mechanism/outputs/variance_results.csv)) |
| C2 — Importance-proposal relative variance is governed by divergence from the decision tilt | `TO_VERIFY_IN_CODE` | No importance proposal is authorized or implemented |
| C3 — Free-energy and variational forms recover the same acquisition | `TO_VERIFY_IN_CODE` | Outside this mechanism experiment |
| C4 — High-value BO decisions can exhibit large posterior-to-decision shift | `SHIFT_OBSERVED; PRACTICAL_QMC_FAILURE_NOT_OBSERVED` | Rare-mode Figure 1 is positive; constrained Hartmann was invalid; the valid q=1 Welded Beam gate found low competitive ESS but zero QMC selection regret |
| C5 — Decision-adapted integration materially reduces acquisition computation | `UNTESTED` | Requires a separately authorized method comparison; QMC is a strong control |
| C6 — Reduced integration error improves sequential BO with a complex belief | `UNTESTED` | Future flagship; unauthorized |

## Current result

The prospectively frozen `rare_mode_mechanism` experiment met every GO expectation. Candidate B's exact EI is `0.0350000`, 25.95% larger than candidate A's `0.0277885`, while 99.999999995% of B's raw-EI decision tilt lies in a component with posterior probability 0.5%. With 512 samples, iid MC ranks B correctly only 56.27% of the time and scrambled Sobol QMC only 59.47%; the strictly positive softplus control gives 48.99% iid accuracy. Iid accuracy reaches 91.02% at 8,192 samples. The complete numerical record is [here](experiments/rare_mode_mechanism/outputs/summary.json).

The result establishes the intended synthetic failure mechanism. It does not yet show that a decision-adapted method is faster than strong QMC: in this example scrambled QMC reaches 100% ranking accuracy at 1,024 samples, whereas iid MC needs substantially more.

The frozen constrained-batch A100 campaign ran from Git SHA `0db7ea2` on an NVIDIA A100-SXM4-40GB. States 3101 and 3102 completed; State 3103 failed the frozen high-budget Gaussian-reference convergence check after escalation to $2^{18}$ samples per replicate, so States 3104–3108 were not run. The audited campaign status is `INVALID`, not GO or NO-GO. The compact evidence and integrity record are [here](experiments/constrained_batch_shift/outputs/invalid_reference_v2/README.md).

The two completed states are informative but not gate evidence. Across 180 top-decile batch/belief evaluations, zero had ESS fraction at or below `0.05`; their median ESS fractions were `0.411–0.452`, and every low-ESS candidate had at most `7.21%` of maximum acquisition quality. At 512 QMC samples, median value error was below `0.49%` and ranking disagreement below `5.36%`. State 2 nevertheless showed poor gradient cosine (`0.859–0.873`) and material outer-optimizer regret, suggesting an optimization/gradient issue not explained by the prespecified severe decision shift. Gaussian and Student-t behavior was very similar.

The prospectively frozen q=1 Welded Beam experiment is complete and valid.
Its 16,384-point candidate set had a true feasible fraction of `0.0010376`,
and all 21 independent Gaussian GP fits converged. Top-32 median population ESS
fractions were `0.1133`, `0.0745`, and `0.0874`, so multiple ordinary
constraints did create substantial decision shift among nominally competitive
candidates. However, the exact maximizers themselves had healthier ESS
fractions of `0.4089`, `0.4292`, and `0.1360`. Every one of 64 QMC scrambles at
every tested sample count from 64 through 1,024 selected the exact finite-pool
best candidate in all three states. Selection regret was identically zero; at
512 samples, mean top-32 pairwise disagreement was only `1.15%`, `1.94%`, and
`4.14%`. The prospective classification is therefore
`WELDED_BEAM_SHIFT_NEGATIVE_REVIEW_REQUIRED`. Full evidence is
[`here`](experiments/welded_beam_shift/outputs/RESULTS.md).

## Next authorized action

Human review of the negative Welded Beam result and the paper stop conditions.
No further experiment or method is automatically authorized. The constrained
Hartmann campaign remains permanently invalid and must not be rerun or modified.

## Not authorized

- any decision-adapted inference method
- complex-posterior flagship
- a constrained-batch rerun with relaxed thresholds, altered seeds, or larger reference budgets
- any constrained-batch GO/NO-GO claim from the two completed states
- q=4 or non-Gaussian Welded Beam experiments

## Unresolved concerns

- The construction is deliberately synthetic: a 0.5% mode at outcome 7 dominates a common mode near zero. Its value is mechanistic clarity, not naturalistic evidence.
- Scrambled QMC is already fully reliable by 1,024 samples here, so the experiment does not demonstrate an orders-of-magnitude wall-clock advantage for a new method.
- The QMC ranking curve is not monotone at small sample counts because inverse-mixture sampling exposes a discontinuous component threshold to randomized stratification; this is a property of the correctly applied transform, not numerical LogEI failure.
- The softplus temperature `0.01` is a close smooth approximation to EI. It rules out dependence on exact zero utility but does not establish robustness to substantially broader smoothing.
- State 3's exact failed convergence components were not serialized before the exception. The package proves the partial-state boundary, and the Colab transcript records failure at the cap, but value-versus-gradient diagnosis requires a new approved diagnostic contract.
- State generation emitted GPyTorch noise/jitter and SciPy optimization warnings. The completed diagnostic fits converged and their saved arrays are finite, but these warnings weaken claims that the generated states are ideal late-stage BO states.
- The first two states lean toward high-value regions having healthy ESS, while State 2 exposes gradient and outer-optimizer difficulty. With only two valid states, neither observation can be generalized.
- The Welded Beam study uses three anchored initial-design states and a finite
  Sobol candidate pool rather than sequential late-stage BO states or continuous
  acquisition optimization.
- Its low-ESS candidates were not generally the state maximizers. Exact-best
  selection remained perfect under common-random-number QMC, so low ESS alone
  did not imply a meaningful decision error.
