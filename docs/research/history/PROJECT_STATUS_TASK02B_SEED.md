# Project Status — after Task 02A

This planning seed preceded the active Task 02B implementation. It is retained as
research history; use [`../PROJECT_STATUS.md`](../PROJECT_STATUS.md) for current status.

## Completed

### Task 01

Validated:

- non-Gaussian predictive shape can alter q=1 EI decisions even with matched first two moments;
- augmented expected-utility inference reproduces q=1 EI / LogEI;
- naive replicated importance sampling degenerates rapidly.

### Task 02A

Tested fixed-support reuse of a NUTS-seeded SAAS structural posterior.

Main conclusion:

- full structural posterior reuse degrades rapidly under cumulative reweighting;
- therefore simple fixed-support reuse is not an adequate SAAS replacement;
- however, acquisition ranking and BO decisions appear materially more robust than global posterior diagnostics.

This motivates a pivot from **posterior-faithful structural inference** toward **decision-relevant structural inference**.

## Active hypothesis at the time

For BO, it may be unnecessary to approximate the entire structural posterior accurately. It may be sufficient to preserve

\[
A(x)=E_{\theta\mid D}[a_\theta(x)]
\]

over the decision-relevant candidate/design region.

Task 02B tests whether fresh SAAS structural uncertainty is compressible in this acquisition space and validates the exact joint structural-decision energy marginal.

## Not yet claimed

We do not yet claim a faster SAAS method, a novel EBM BO algorithm, that a particular
particle method will outperform NUTS, a batch-BO advantage, or a residual-output EBM
advantage in a real GP BO loop.
