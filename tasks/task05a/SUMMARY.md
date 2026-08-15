# Task 05A summary

## Question

Is there a credible structured belief for low-data measured protein design, and
is belief quality decision-limiting?

## Protocol actually run

The full repository suite passed: 101 tests passed and 5 CUDA-only tests skipped. One
CPU smoke ran both measured datasets, seed 0, n=48, S0/S1/S2, 2,048-point pools, and
two sequential BO steps. It used the pinned checksummed files, finished in 7.73
seconds, and peaked at 738.27 MB resident memory.

## Result

All 18 smoke fits (six offline and twelve sequential) were finite and converged. The
six offline rows and six trajectory summaries serialized with configuration hash
`6b6fb968015c52be1c4abb4c40b8a9b35a5ed9965fe9cbe31033ef97bc243c0f`.
Smoke interval errors ranged from 0.1216 to 0.2887; these single-split values are
wiring diagnostics, not calibration evidence.

## Gate decision

`READY_FOR_A100_RUN`. The automatic smoke result is correctly `INCONCLUSIVE`; no
Task 05A PASS/FAIL decision exists until all twenty frozen shards are aggregated.

## What we learned

Pinned measured data, train-only scaling, the three exact-GP kernels, observed/latent
variance separation, finite-pool LogEI, sequential exclusion, checkpointing, metrics,
and the gate evaluator operate end to end on CPU.

## Known limitations

The smoke is one small split and two BO steps. LOCK-GP is an integer-coded adaptation,
formula-audited against official commit `df384fe2`, not an official wrapper.
Hardware/runtime and scientific credibility remain unknown until the complete
measured-pool A100 protocol is returned.

## Next authorized action

Use the guarded notebook to profile `trpb_seed0`, then run the Drive-backed resumable
campaign. Submit its single final ZIP for checksum audit and frozen aggregation review.
Do not implement Task 05B.
