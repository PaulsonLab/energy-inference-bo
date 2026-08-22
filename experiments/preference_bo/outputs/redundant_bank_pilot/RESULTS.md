# E3 Preference-BO Redundant-Bank Pilot Results

Profile: `frozen_pilot`

Configuration SHA-256: `5c53c8de7144d1d0bc3a66b1ca4233b2c17811f726e977e1a5c90f103140b260`

Starting commit: `b68525952f58693461ac32d4658a8d611a594706`

## Preregistered gates

- P1: median T_0.10 full = 2.5; standard = 7; pass = True.
- P2 performance: median T_0.10 adaptive = 2.5; full = 2.5; pass = True.
- P2 sparsity: median R_factors = 0.888889; pass = False.
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

Preserve the failed redundant-bank gate and do not design another synthetic bank in this task; the PDE experiments remain the stronger setting for the sparsity/scaling claim.
