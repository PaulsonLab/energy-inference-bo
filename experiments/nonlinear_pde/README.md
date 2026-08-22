# E2 — Nonlinear PDE Expanding-Domain Scaling

Status: nonlinear-PDE T2-B **PASS** and T4 **PROVED FOR THIS FAMILY**;
existing scaling evidence still requires clean reproduction, robustness checks,
and baselines. See the
[E2 specification](../../project/EXPERIMENTS.md#e2--nonlinear-pde-expanding-domain-scaling)
and the permanent
[proof audit](../../project/reference/T2B_NONLINEAR_PDE_AUDIT.md).

The lightweight structural regression is
[`run_structural_validation.py`](run_structural_validation.py). Its frozen
configuration, matrix diagnostics, notebook comparison, and locked structural
replay are under [`outputs/t2b_structural_validation/`](outputs/t2b_structural_validation/).
It evaluates no factor energies and performs no posterior inference.

The theorem-backed structural value is `0.03874403301354687`. The archived
inference allowance and total stopping envelope remain empirical and are not a
finite-sample end-to-end certificate.
