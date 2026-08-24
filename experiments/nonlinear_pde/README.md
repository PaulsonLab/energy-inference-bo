# E2 — Nonlinear PDE Expanding-Domain Scaling

Status: nonlinear-PDE T2-B **PASS** and T4 **PROVED FOR THIS FAMILY**;
the replacement stress-test preregistration remains unexecuted, and a
development-only FULL-shadow backend blocker must be resolved before any
prospective E2 source seed is used. See the
[E2 specification](../../project/EXPERIMENTS.md#e2--nonlinear-pde-expanding-domain-scaling)
and the permanent
[proof audit](../../project/reference/T2B_NONLINEAR_PDE_AUDIT.md).

## Current execution blocker and recommendation

The development-only
[`full_shadow_reliability_diagnostic`](outputs/full_shadow_reliability_diagnostic/RESULTS.md)
used only seed `2026082401`. Its n=24 FULL shadows pass the unchanged frozen
rule after escalation, but all n=40 early/middle/late states fail the frozen
ESS-fraction requirement at both 8,192 and 16,384 samples per batch. The
Laplace optimizer converges cleanly; relative ESS remains near or below `0.20`
as absolute ESS grows, identifying proposal/target overlap as the primary
problem. One early 16,384 pair also shows material cross-batch regret
`0.024442`.

**Recommendation:** do not run preregistration
`0dcb76f5f2d053e098b472ac9984182b837295b5` prospectively yet. First improve
and independently development-validate the FULL-reference numerical backend.
Do not treat a larger sample count or a post hoc decision-aligned reliability
rule as the sole correction. No scientific method, model constant, source
seed, factor-selection procedure, or success threshold has been changed.

The lightweight structural regression is
[`run_structural_validation.py`](run_structural_validation.py). Its frozen
configuration, matrix diagnostics, notebook comparison, and locked structural
replay are under [`outputs/t2b_structural_validation/`](outputs/t2b_structural_validation/).
It evaluates no factor energies and performs no posterior inference.

The theorem-backed structural value is `0.03874403301354687`. The archived
inference allowance and total stopping envelope remain empirical and are not a
finite-sample end-to-end certificate.
