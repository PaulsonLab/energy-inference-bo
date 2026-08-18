# Project Status

## Thesis
Optimize the causal policy rather than explicitly enumerating the fantasy tree; test whether causal path-KL transport enables efficient critic-free nonmyopic BO.

## Phase
POLICY_KILL_NEGATIVE_REVIEW_REQUIRED

## Current evidence
The prospectively frozen 20-candidate two-regime scan failed to create a meaningful nonmyopic planning opportunity. The closest candidate changed the converged first action from `0.250` at H=1 and H=2 to the diagnostic action `0.500` at H=3, but forcing the H=1 action reduced the H=3 value by only `0.0180%`, far below the frozen `2%` gate. Medium-to-fine reference value drift was at most `0.0292%`.

## Scientific answers
1. Nonmyopia: **NO** — an action-location change existed, but its value was not meaningful.
2. Policy representation: **NOT TESTED** — prohibited by the failed reference gate.
3. Energy/transport value: **NOT TESTED** — prohibited by the failed reference gate.

## Next action
Human review only. Decide whether to stop the paper direction or authorize a new scientific hypothesis; no benchmark redesign or learned-policy experiment is automatically authorized.

## Not authorized
- learned-policy optimization on the failed `policy_kill` benchmark
- benchmark retuning after this negative result
- long-horizon benchmark suite
- EARL-BO implementation
- LookaHES implementation
- complex non-Gaussian flagship
- protein/molecule experiments

POLICY_KILL_NEGATIVE_REVIEW_REQUIRED
