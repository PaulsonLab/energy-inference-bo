# Task 03A — full result

## Scope and conclusion

Task 03A is complete. The full Colab oracle study used five seeds, D=20 with two true
active dimensions, Gaussian and invertibly warped regimes, n=16/32/64, 1,024 test
points, 2,048 EI candidates, genuine four-fold cross-fitting, four-component
Ensemble MAP-SAAS, and 256 retained fresh SAAS-NUTS particles per case.

The prespecified decision is **NO-GO for Task 03B**. All four gates failed, and the
structural-only fallback also failed. This does not invalidate the normalized
reference-PIT mathematics or convex energy fit; it says the tested residual energies
did not add capability beyond strong calibration baselines, while honest cross-fit
reference construction was not cheaper than NUTS on this A100 run.

## Eight completion questions

1. **Was the run complete, finite, and leak-free?** The package contains all expected
   240 primary method rows, 30 timing rows, 1,200 structural rows, 30 case archives,
   and both sensitivity slices. All numeric values are finite, case/aggregate keys
   agree, and final CSVs equal their partial checkpoints byte-for-byte. Cross-fit code
   and tests enforce disjoint held-out/reference/preprocessing indices; 14 PIT values
   were clamped at the documented inversion bound, at most two in any case. Fold
   indices were not serialized, so this last provenance check rests on tested runtime
   assertions rather than reconstruction from the ZIP.

2. **Were E0/E1 safe under Gaussian truth at n=16/32?** Not under the frozen gate.
   Relative to B1, mean excess NLL was favorable (`-0.2571` E0, `-0.3269` E1), median
   regret increase was zero, and each had only one of ten cells degrade by more than
   0.10. But median correction KL was `0.0443` and `0.0658`, both above the 0.02
   safety cap. Gate A therefore failed: the corrections helped density fit but did not
   remain sufficiently close to a correctly specified reference.

3. **Did an energy beat all strong baselines under warped truth?** No. B4 was the
   strongest non-NUTS baseline by mean true NLL at n=32 and n=64. At n=32, B4/E0/E1
   NLLs were `1.7417/2.9087/2.8526`; at n=64 they were
   `0.2837/0.3437/0.3557`. Both energy variants therefore lost the required NLL
   comparison at both sizes. Gate B failed.

4. **Did the energies improve the n=64 warped EI decision?** Only weakly and not
   consistently. Their median normalized regret was `0.2744` versus B4's `0.3003`, an
   absolute gain of `0.0259`, but each beat B4 in only `1/5` paired seeds rather than
   the required `4/5`. E0 selected the same candidate as raw B1 in 28/30 primary
   cases, indicating that its density correction rarely changed the decision.

5. **How did decision quality compare with NUTS?** At Gaussian n=64, median regret
   was `0.0275` for B1/E0 and `0.0468` for NUTS. At warped n=64, however, NUTS reached
   `0.0349` while B1/E0/E1 remained at `0.2744`, a gap of `0.2395`. Raw MAP-SAAS
   therefore failed the structural-only fallback despite recovering both active
   dimensions in all five seeds at n=64. Structural relevance recovery alone did not
   guarantee a good EI decision.

6. **Was the corrected approach cheaper?** Energy optimization itself was negligible:
   medians `0.0237 s` (E0) and `0.0334 s` (E1), only about `0.19%` and `0.18%` of
   reference-construction time. Honest four-fold cross-fit plus the final MAP fit was
   not cheap: median fully charged time was `18.08 s` versus `14.62 s` for NUTS, a
   ratio of about `1.26`. A single final MAP fit was `0.236` times NUTS, but the four
   additional reference fits erased that advantage.

7. **Did correction unlock with more warped evidence, and did sensitivities rescue
   it?** No. The specified paired unlocking condition held in `0/5` seeds for both E0
   and E1. Eight MAP components improved W-n32 NLL in only 2/5 seeds and regret in
   only 1/5 (three ties, one loss). On the non-gating seed-0 W-n64 shrinkage slice,
   precisions 3 and 30 produced the same `0.1780` regret for E0/E1. No sensitivity
   result reverses the primary gate.

8. **Did the numerical and computational validation pass?** The notebook preflight,
   59-test suite, JAX float64/device checks, finite-output audit, and CPU/CUDA MAP
   comparison passed. The MAP device probe found only a `1.071x` CUDA speedup, below
   the prespecified `1.2x` threshold, with maximum prediction/MLL discrepancy
   `3.112e-9`; MAP fitting correctly remained on CPU. One instrumentation defect is
   recorded: all MAP forward-evaluation counters are zero despite nonzero optimizer
   iterations, so those counters are invalid. Wall-time and optimizer-iteration
   records remain usable.

## Hardware interpretation

The run manifest confirms a CUDA JAX backend and about 1.27–1.30 GB peak Torch CUDA
allocation; the maintainer reports an A100, although the manifest records only
`cuda:0`. The A100 plausibly explains the unusually short 13.0–24.1 s NUTS fits and
therefore materially affects the wall-time ratios. MAP fitting remained CPU-based,
while NUTS and batched prediction used the GPU, so the A100 specifically favors the
NUTS comparator. These times should not be generalized to T4/L4/CPU systems.

That hardware caveat does not alter the scientific NO-GO: predictive NLL,
calibration, EI ranking, selected candidate, and regret are not speed measurements,
and Gate B already fails decisively on those metrics.

## Recommendation

Do not implement Task 03B from this result, and do not tune the energy or cross-fit
scheme retrospectively on these five seeds. The narrow finding is that this small
PIT-space residual energy did not earn its complexity over strong parametric
calibration, and the proposed honest cross-fitting workflow lost its A100 cost case.

No implementation task is active. A future planning brief should start from the
observed bottleneck—structural/mean quality and high-variance reduced-data cross-fit
PITs, especially before n=64—rather than assuming a richer residual density is the
next fix. Candidate bounded directions include diagnosing transfer from 0.75n
cross-fit references to the full reference, or testing whether fast MAP-SAAS is useful
with a cheaper calibration protocol, but either requires a new approved contract.

## Evidence and reproducibility

The reviewed package and independent audit are in
[`results/task03a/full/`](../../results/task03a/full/). Raw per-case archives remain
ignored under `artifacts/task03a/full/` and are identified by committed SHA-256
checksums. The run used source commit `a671671`; all changes since the Task 03A
implementation commit were Colab environment/handoff fixes, not scientific-code
changes.
