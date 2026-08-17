# Project Status

## Thesis
Decision-adapted integration may reduce Monte Carlo acquisition cost when utility-tilted worlds differ sharply from ordinary posterior worlds.

## Phase
RARE_MODE_MECHANISM_COMPLETE_AWAITING_HUMAN_REVIEW

## Claims

| Claim | Status | Evidence |
| --- | --- | --- |
| C1 — Standard-MC relative variance equals decision-shift chi-square divergence divided by sample count | `VERIFIED_IN_CODE` | Exact identity; empirical iid relative variance agrees within 3.16% across the frozen sample counts ([variance table](experiments/rare_mode_mechanism/outputs/variance_results.csv)) |
| C2 — Importance-proposal relative variance is governed by divergence from the decision tilt | `TO_VERIFY_IN_CODE` | No importance proposal is authorized or implemented |
| C3 — Free-energy and variational forms recover the same acquisition | `TO_VERIFY_IN_CODE` | Outside this mechanism experiment |
| C4 — High-value BO decisions can exhibit large posterior-to-decision shift | `SUPPORTED_IN_FROZEN_SYNTHETIC_MECHANISM` | Candidate B has chi-square shift 199.163 and ESS fraction 0.004996; see [Figure 1](experiments/rare_mode_mechanism/outputs/figure1_rare_mode_mechanism.png) |
| C5 — Decision-adapted integration materially reduces acquisition computation | `UNTESTED` | Requires a separately authorized method comparison; QMC is a strong control |
| C6 — Reduced integration error improves sequential BO with a complex belief | `UNTESTED` | Future flagship; unauthorized |

## Current result

The prospectively frozen `rare_mode_mechanism` experiment met every GO expectation. Candidate B's exact EI is `0.0350000`, 25.95% larger than candidate A's `0.0277885`, while 99.999999995% of B's raw-EI decision tilt lies in a component with posterior probability 0.5%. With 512 samples, iid MC ranks B correctly only 56.27% of the time and scrambled Sobol QMC only 59.47%; the strictly positive softplus control gives 48.99% iid accuracy. Iid accuracy reaches 91.02% at 8,192 samples. The complete numerical record is [here](experiments/rare_mode_mechanism/outputs/summary.json).

The result establishes the intended synthetic failure mechanism. It does not yet show that a decision-adapted method is faster than strong QMC: in this example scrambled QMC reaches 100% ranking accuracy at 1,024 samples, whereas iid MC needs substantially more.

## Next authorized action

Human review of the frozen evidence and candidate paper Figure 1. Decide whether the mechanism is sufficiently meaningful and visually clear to authorize the separately specified `constrained_batch_shift` experiment. No downstream experiment is authorized by this status update alone.

## Not authorized

- `constrained_batch_shift` without an explicit human decision
- decision-adapted sampler implementation
- complex-posterior flagship

## Unresolved concerns

- The construction is deliberately synthetic: a 0.5% mode at outcome 7 dominates a common mode near zero. Its value is mechanistic clarity, not naturalistic evidence.
- Scrambled QMC is already fully reliable by 1,024 samples here, so the experiment does not demonstrate an orders-of-magnitude wall-clock advantage for a new method.
- The QMC ranking curve is not monotone at small sample counts because inverse-mixture sampling exposes a discontinuous component threshold to randomized stratification; this is a property of the correctly applied transform, not numerical LogEI failure.
- The softplus temperature `0.01` is a close smooth approximation to EI. It rules out dependence on exact zero utility but does not establish robustness to substantially broader smoothing.
