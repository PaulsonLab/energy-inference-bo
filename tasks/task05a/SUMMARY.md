# Task 05A summary

## Question

Is there a credible structured belief for low-data measured protein design, and does
biological sequence structure materially improve finite-pool BO decisions?

## Protocol actually run

The frozen `task05a-v1` protocol ran on an A100 at Git `56e8bc1`: TrpB and CreiLOV,
seeds 0–9, S0/S1/S2, offline sizes 48/96/192, and 32 measured-pool LogEI selections
from 48 initial observations. All 20 shards, 180 offline fits, 60 trajectories, and
1,920 BO fits completed and converged. All 374 source-package checksums passed.

## Result

No model met the prespecified credibility requirement at both n=48 and 96. At n=48,
median high-utility interval error was 0.101/0.106/0.176 for S0/S1/S2 on TrpB and
0.159/0.157/0.115 on CreiLOV, versus the 0.10 limit. S2 became catastrophically
miscalibrated on TrpB at n=96 (interval error 0.276; NLL gap +0.634 nat).

Structured kernels also missed the decision threshold. On TrpB, paired median AUC
gain versus S0 was -56.0% for S1 (0/10 wins) and -29.4% for S2 (2/10). On CreiLOV,
it was -1.4% for S1 (4/10) and +1.4% for S2 (7/10), far below the required 20%.
Final top-1% discovery was already 100% for S0 on both datasets; S1 reduced it to 70%
on TrpB.

## Gate decision

**FAIL.** The independently recomputed gate exactly matches the generated
`gate_result.json`. PASS-A fails because no structured belief is credible and gains
are insufficient; PASS-B fails because S0 is not credible at n=48. Task 05B is not
authorized.

## What we learned

The measured-pool pipeline is numerically and computationally healthy, but the tested
exact-GP ladder is not a sufficiently credible low-data belief. Simple sequence
structure did not reliably improve BO: S2 showed a small CreiLOV tendency but failed
calibration and transferred poorly to TrpB, while S1 was actively harmful on TrpB.
The result supports the control plan's premise that downstream composition or planning
should not be judged on top of an untrusted belief.

## Known limitations

This tests two measured landscapes, ten random nested starts, one 32-step finite-pool
policy, and an independently implemented integer-coded LOCK-GP adaptation. Top-1%
discovery saturated for S0, limiting that endpoint, although regret AUC remained
informative. A non-gating offline one-step-regret bug was found: 118/180 historical
rows failed to retain the incumbent and could exceed one. The source is fixed;
sequential metrics and the gate are unaffected. Resumed-shard wall time is also
undercounted, but median uninterrupted times were 66.6 s (TrpB) and 357.6 s (CreiLOV).

## Next authorized action

`HUMAN_REVIEW_REQUIRED`. Do not implement Task 05B or relax Task 05A retrospectively.
The narrow decision is whether to stop the compositional program or approve a new,
separately frozen belief-stage contract before any downstream composition work.
