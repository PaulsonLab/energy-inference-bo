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

## Next planning gate

There is no active implementation task. The preferred next planning direction is a
return to the modeling hypothesis left open after Task 01: exact-GP prequential
residual-energy correction, compared against strong Gaussian location/scale and
calibration baselines. A new task must define the sequential likelihood, reference GP
fitting protocol, nested baselines, falsification thresholds, and local/Colab compute
boundary before code is added.

Revisiting structural transport is lower priority and requires a separate diagnostic
contract addressing forced likelihood tempering, frequent clipping, and weak SVGD
repulsion without tuning retrospectively on the six Task 02C cases.

## Scope boundary

No later method is authorized automatically. Do not add another sampler, SMC/MALA,
Vecchia, q>1 BO, molecular optimization, or an end-to-end BO loop without a new
approved task contract.

## How to read the repository

- [AGENTS.md](../../AGENTS.md) and [MATH_AND_SCOPE.md](../../MATH_AND_SCOPE.md) define
  global invariants.
- [ACTIVE_TASK.md](../../tasks/ACTIVE_TASK.md) records that no implementation stage is
  active and gives the current review path.
- Each task's `SPEC.md` defines its bounded contract; its math note derives identities;
  its `SUMMARY.md` and `results/` package record what happened.
- [Historical research documents](history/README.md) are provenance only.
