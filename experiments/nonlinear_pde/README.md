# E2 — Nonlinear PDE Expanding-Domain Scaling

Status: nonlinear-PDE T2-B **PASS** and T4 **PROVED FOR THIS FAMILY**;
the replacement stress-test preregistration remains unexecuted. The historical
development-only backend rescue ended `FULL_REFERENCE_BACKEND_UNRESOLVED`, and
the final fixed-transition standard-ESS resolution gate now ends
**`BACKEND_HEALTHY_REFERENCE_RULE_TOO_BRITTLE`**. No prospective E2 source seed
may be used. See the
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

The subsequent development-only
[`full_shadow_backend_rescue`](outputs/full_shadow_backend_rescue/RESULTS.md)
used calibration seed `2026082401` and separately derived held-out development
seed `3321078991`. Curvature-tempered SNIS failed held-out n=40 early/middle;
independence MH failed calibration because the Laplace-mode chain accepted no
proposals; and elliptical slice failed the conjunctive calibration decision
and convergence gate. The exact terminal classification is
`FULL_REFERENCE_BACKEND_UNRESOLVED`.

The final development-only
[`ess_resolution_gate`](outputs/ess_resolution_gate/RESULTS.md) changed only
chain length for that exact standard ESS transition. S1 and S2 failed the
unchanged conjunctive strict gate. At S3 (8 chains, 4,096 burn-in and 32,768
retained draws per chain), all n=24 states and n=40 early passed. At n=40,
maximum required split R-hat was below `1.0014` and minimum required gap ESS was
above `19,800`. Middle and late failed only exact group-action agreement across
near ties: reciprocal regrets were `0.001845` and `0.001420`, vector differences
were below `0.004`, and pooled leader/challenger gaps were only about `0.57` and
`0.05` MCSE. No schedule passed globally, so the fresh validation source was
not derived or accessed and no schedule was frozen.

**Recommendation:** do not run preregistration
`0dcb76f5f2d053e098b472ac9984182b837295b5`, create a replacement
preregistration, or spend a prospective source seed. Do not develop another
sampler next. Formulate and independently audit a prospective decision-aligned
FULL-reference rule based on acquisition-gap Monte Carlo uncertainty. No
replacement rule has yet earned a freeze. No scientific method, model constant,
source seed, factor-selection procedure, or success threshold has been changed.

The lightweight structural regression is
[`run_structural_validation.py`](run_structural_validation.py). Its frozen
configuration, matrix diagnostics, notebook comparison, and locked structural
replay are under [`outputs/t2b_structural_validation/`](outputs/t2b_structural_validation/).
It evaluates no factor energies and performs no posterior inference.

The theorem-backed structural value is `0.03874403301354687`. The archived
inference allowance and total stopping envelope remain empirical and are not a
finite-sample end-to-end certificate.
