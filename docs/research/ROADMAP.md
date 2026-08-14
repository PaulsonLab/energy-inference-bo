# Current research roadmap

This is the concise human overview. It does not replace the global mathematical
invariants, task contracts, task-specific derivations, or reviewed evidence.

## Completed evidence

| Stage | What the evidence established | Canonical record |
| --- | --- | --- |
| Task 01 | Non-Gaussian predictive shape can alter q=1 EI even with matched moments; augmented q=1 inference reproduces EI/LogEI. | [task](../../tasks/task01/README.md), [evidence](../../results/task01/README.md) |
| Task 02A | Fixed-support SAAS posterior reuse collapses quickly, although some EI decisions remain more robust than global posterior diagnostics. | [task](../../tasks/task02a/README.md), [evidence](../../results/task02a/README.md) |
| Task 02B | Acquisition signatures are moderately compressible, oracle coresets preserve decisions, and the M=1/M=2 structural-decision identities hold exactly. | [task](../../tasks/task02b/README.md), [evidence](../../results/task02b/full/README.md) |
| Task 02C | Exact decision-energy mathematics passed, but decision-tilted SVGD failed to improve q=1 decisions at matched compute. | [task](../../tasks/task02c/README.md), [evidence](../../results/task02c/full/README.md) |
| Task 03A | Normalized PIT residual energies did not beat strong calibration baselines; honest cross-fit construction was slower than A100 NUTS, and Task 03B was not opened. | [task](../../tasks/task03a/README.md), [evidence](../../results/task03a/full/README.md) |

Task 04A is paused after preflight and therefore is not listed as completed full-run
evidence.

## Task 02C gate result

The full D=10 study contained 108 paired P-SVGD/DT-SVGD comparisons. DT-SVGD won only
20 (`18.5%`); median normalized regret was `0.5330` versus `0.4139` for P-SVGD and
`0.3543` for MAP-SAAS. Neither transport method produced a run below 10% regret.
DT-SVGD was also slower: median charged time `33.1 s`, compared with `22.9 s` for
P-SVGD and `23.4 s` for fresh NUTS.

The failure is not attributable to incorrect energy mathematics or intrinsic teacher
tilt degeneracy. Envelope and exact-potential checks passed, beta-one teacher ESS/P
was `0.612–0.986`, and no final particle-collapse criterion was triggered. However,
initialization forced its tempering constraint in 625/864 steps, velocity clipping was
common, and kernel repulsion was weak relative to attraction. The result is therefore
a **NO-GO for this SVGD configuration**, not a disproof of the decision-energy
identity or every possible transport algorithm.

## Task 03A gate result

The full five-seed D=20 oracle study was complete and numerically valid, but all four
prespecified gates failed. E0/E1 improved mean NLL over raw B1 under Gaussian truth,
yet exceeded the correction-KL safety cap (`0.0443/0.0658` versus `0.02`). Under
warped truth, B4 beat both energies in mean NLL at n=32 and n=64. At warped n=64,
median normalized regret was `0.0349` for NUTS but `0.2744` for B1/E0/E1, so the
structural-only fallback also failed.

The energy optimizer was genuinely cheap—roughly 0.2% of reference-construction
time—but genuine four-fold cross-fitting plus the final MAP ensemble cost a median
`18.08 s`, compared with `14.62 s` for NUTS on the reported A100. That ratio is
hardware-specific because MAP fitting remained on CPU while NUTS used the GPU; the
predictive and decision failures are not hardware-timing artifacts.

Task 03B is therefore **not authorized**. Task 04A subsequently opened as a separate
contracted pivot and is now paused after its preflight. The Task 03A finding implies
that a future calibration-focused
planning brief should address the observed structural/mean and reduced-data
cross-fit-transfer bottlenecks rather than presuming a richer residual density is the
next step.

## Active Task 04A pivot

Task 04A tests a distinct process-level hypothesis: fixed oracle geometry and
Vecchia-style Gaussian conditionals are augmented by small, exactly normalized unary
and pairwise local energies. It is limited to q=1 oracle evaluation with no learned
geometry or sequential BO. [Contract](../../tasks/task04a/SPEC.md),
[mathematics](../../tasks/task04a/MATH.md), and [status](../../tasks/task04a/SUMMARY.md)
are the implementation authority. A standardized-child revision repaired the weak
interaction density signal. The subsequent withheld-seed diagnostic strengthened the
density result to a 51.0% P-over-U KL gain with 8/8 wins, but was officially `INVALID`
for decisions: all 16 I cases had zero qualifying natural near-tie pairs, only 6.64%
of counterfactual oracle pairs had 1% EI contrast, and U had no natural decision
opportunity. The A100 study remains disabled and Task 04B is not authorized. Further
work requires a prospective oracle contract that creates decision-relevant tail
variation, not weaker post-observation thresholds.

Revisiting structural transport is lower priority and requires a separate diagnostic
contract addressing forced likelihood tempering, frequent clipping, and weak SVGD
repulsion without tuning retrospectively on the six Task 02C cases.

## Scope boundary

Task 04A authorizes only its fixed-geometry, local Vecchia-style oracle experiment.
No later method is authorized automatically. Do not add another sampler, learned
geometry, q>1 BO, molecular optimization, or an end-to-end BO loop without a new
approved task contract.

## How to read the repository

- [AGENTS.md](../../AGENTS.md) and [MATH_AND_SCOPE.md](../../MATH_AND_SCOPE.md) define
  global invariants.
- [ACTIVE_TASK.md](../../tasks/ACTIVE_TASK.md) records that no implementation task is active.
- Each task's `SPEC.md` defines its bounded contract; its math note derives identities;
  its `SUMMARY.md` and `results/` package record what happened.
- [Historical research documents](history/README.md) are provenance only.
