# E3 Preference-BO Minimal Pilot Results

Profile: `frozen_pilot`
Configuration SHA-256: `4eafe727b67864632d25d30b526cbdb5c0a83989bf31e7b0cf41f7129c8a89da`
Starting commit: `5f7f7cfa66f4d6ec33ebf5a5feb3d8f77ba2110a`

## Preregistered gates

- P1: median T_0.10 full = 3; standard = 7; pass = True.
- P2 performance: median T_0.10 adaptive = 3; full = 3; pass = True.
- P2 sparsity: median R_factors = 0.875; pass = False.
- Overall verdict: **FAIL-P2**.

## Mechanical status

All required mechanical checks passed: `True`.

## Numerical caveat

None under the preregistered empirical ESS/split diagnostics.

## Interpretation

The preference bank passed the value gate, but adaptive conditioning did not satisfy both preregistered preservation and sparsity conditions.

## Failure diagnosis

Main evidence: **conservative structural influence bounds**. Adaptive factor use failed the sparsity gate while structural terms dominated inference allowances in at least 90% of activation rounds and held-out full-target regret stayed within the screening tolerance.


## Next action

Preserve the failed sparse-preservation gate and separately analyze structural-bound conservatism versus broad decision dependence; do not tune or rerun this pilot.
