# Project Status

## Thesis
Decision-adapted integration may reduce Monte Carlo acquisition cost when utility-tilted worlds differ sharply from ordinary posterior worlds.

## Phase
READY_FOR_RARE_MODE_MECHANISM

## Claims

| Claim | Status | Evidence |
| --- | --- | --- |
| C1 — Standard-MC relative variance equals decision-shift chi-square divergence divided by sample count | `TO_VERIFY_IN_CODE` | Mathematical contract only |
| C2 — Importance-proposal relative variance is governed by divergence from the decision tilt | `TO_VERIFY_IN_CODE` | Mathematical contract only |
| C3 — Free-energy and variational forms recover the same acquisition | `TO_VERIFY_IN_CODE` | Mathematical contract only |
| C4 — High-value BO decisions can exhibit large posterior-to-decision shift | `UNTESTED` | No paper experiment run |
| C5 — Decision-adapted integration materially reduces acquisition computation | `UNTESTED` | Blocked pending C4 |
| C6 — Reduced integration error improves sequential BO with a complex belief | `UNTESTED` | Future flagship; unauthorized |

## Current result
Repository reset complete.

## Next authorized action
Implement `rare_mode_mechanism` only.

## Not authorized

- `constrained_batch_shift`
- decision-adapted sampler implementation
- complex-posterior flagship
