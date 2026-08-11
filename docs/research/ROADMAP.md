# Current research roadmap

This is the concise human overview. It does not replace the global mathematical
invariants, task contracts, or task-specific derivations linked below.

## Evidence so far

| Stage | What the evidence established | Canonical record |
| --- | --- | --- |
| Task 01 | Non-Gaussian predictive shape can alter q=1 EI even with matched first two moments; the augmented q=1 marginal reproduces EI/LogEI. | [task](../../tasks/task01/README.md), [evidence](../../results/task01/README.md) |
| Task 02A | Fixed-support SAAS posterior reuse collapses quickly, so it is not a reliable replacement for fresh posterior inference; some EI decisions remained more robust. | [task](../../tasks/task02a/README.md), [full results](../../results/task02a/README.md) |
| Task 02B | Full evidence shows modest acquisition-space rank, strong oracle coreset decision preservation, and exact M=1/M=2 structural-decision identities. The prespecified Task 02C gate passed 5/5. | [task](../../tasks/task02b/README.md), [full evidence](../../results/task02b/full/README.md), [summary](../../tasks/task02b/SUMMARY.md) |

## Task 02B gate result

The full D=10, three-seed, 18-checkpoint Colab study passed all five prespecified
conditions:

1. At low ESS/P, 6/10 checkpoints still had normalized decision regret below 5%.
2. Mean acquisition-signature entropy rank was `8.206`, below the threshold of 32.
3. Acquisition Frank–Wolfe with K=8 stayed below 5% regret at all 18 checkpoints; K=16
   reproduced every teacher candidate.
4. M=1 and independent-replica M=2 marginal errors were at most `2.78e-17` and `0` on
   the representative full checkpoint.
5. Mean oracle acquisition-coreset regret (`0.004445`) was lower than posterior medoids
   (`0.01028`) and random equal thinning (`0.02586`).

The evidence also contains important limits: 4/15 reuse checkpoints exceeded 10%
regret, acquisition rank reached `15.388`, the 99%-energy rank reached 83, and fresh
teacher reruns differed from saved Task 02A curves by up to `0.04298` absolute EI.
Frank–Wolfe is an oracle compression diagnostic, not a deployable inference method.

## Next eligible stage: specify Task 02C

Task 02C may now be designed as a bounded falsification experiment for direct inference
on the joint structural-decision energy

\[
E(x,\theta) = -\log p_0(x) - \log p(\theta) - \log p(D\mid\theta) - \log EI_\theta(x).
\]

The implementation method remains undecided. Passing Task 02B authorizes writing a
Task 02C specification; it does not authorize implementation without that contract and
does not claim transport or BO performance. SVGD, annealed Langevin/MALA, and
resample-move SMC are possible comparisons to evaluate during scoping, not implemented
methods.

## Scope boundary

Do not add SVGD, MALA, annealed Langevin, new SMC rejuvenation, Vecchia,
residual-output EBMs, q>1 BO, molecular optimization, or an end-to-end BO loop until a
new task explicitly selects and bounds the method.

## How to read the repository

- [AGENTS.md](../../AGENTS.md) and [MATH_AND_SCOPE.md](../../MATH_AND_SCOPE.md) define
  global non-negotiable invariants.
- [ACTIVE_TASK.md](../../tasks/ACTIVE_TASK.md) gives coding agents the required reading
  order and names the authoritative task contract.
- Each task's `SPEC.md` defines its bounded implementation and validation plan; its math
  note derives task-specific identities; its `SUMMARY.md` records results and the next
  gate.
- [Historical research documents](history/README.md) are retained for provenance only.
