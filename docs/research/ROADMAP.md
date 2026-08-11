# Current research roadmap

This is the concise human overview. It does not replace the global mathematical
invariants, task contracts, or task-specific derivations linked below.

## Evidence so far

| Stage | What the evidence established | Canonical record |
| --- | --- | --- |
| Task 01 | Non-Gaussian predictive shape can alter q=1 EI even with matched first two moments; the augmented q=1 marginal reproduces EI/LogEI. | [task](../../tasks/task01/README.md), [evidence](../../results/task01/README.md) |
| Task 02A | Fixed-support SAAS posterior reuse collapses quickly, so it is not a reliable replacement for fresh posterior inference; some EI decisions remained more robust. | [task](../../tasks/task02a/README.md), [full results](../../results/task02a/README.md) |
| Task 02B | Active diagnostic: test whether fresh SAAS uncertainty is compressible in acquisition space and verify the exact structural-decision marginal with NUTS as teacher. | [task](../../tasks/task02b/README.md), [math](../../tasks/task02b/MATH.md), [summary](../../tasks/task02b/SUMMARY.md) |

## Current gate: Task 02B

The local smoke result is not enough to justify a transport method. The full Colab
study must establish most of the following before Task 02C is considered:

1. Decision regret is materially more robust than posterior-fidelity diagnostics.
2. Fresh-NUTS acquisition signatures have modest effective rank.
3. A small oracle acquisition-space coreset, typically K <= 16 or 32, preserves the
   fully Bayesian EI decision more reliably than random and posterior-space baselines.
4. The M=1 and independent-replica M=2 structural-decision marginal identities pass
   at strict double-precision tolerance.

The exact full extraction procedure is in [Task 02B Colab](../../tasks/task02b/COLAB.md).

## Conditional next stage

Only if the Task 02B gate passes, Task 02C may evaluate direct inference on the joint
structural-decision energy

\[
E(x,\theta) = -\log p_0(x) - \log p(\theta) - \log p(D\mid\theta) - \log EI_\theta(x).
\]

The implementation method is deliberately undecided. SVGD, annealed Langevin/MALA,
and resample-move SMC are possible comparisons, not planned Task 02B code or claims.

## Scope boundary

Do not add SVGD, MALA, annealed Langevin, new SMC rejuvenation, Vecchia,
residual-output EBMs, q>1 BO, molecular optimization, or an end-to-end BO loop before
the relevant task gate is met.

## How to read the repository

- [AGENTS.md](../../AGENTS.md) and [MATH_AND_SCOPE.md](../../MATH_AND_SCOPE.md) define
  global non-negotiable invariants.
- [ACTIVE_TASK.md](../../tasks/ACTIVE_TASK.md) gives coding agents the required reading
  order and names the authoritative task contract.
- Each task's `SPEC.md` defines its bounded implementation and validation plan; its math
  note derives task-specific identities; its `SUMMARY.md` records results and the next
  gate.
- [Historical research documents](history/README.md) are retained for provenance only.
