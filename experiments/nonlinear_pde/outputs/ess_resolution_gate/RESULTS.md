# E2 Standard-ESS Resolution Gate

Terminal classification: **`BACKEND_HEALTHY_REFERENCE_RULE_TOO_BRITTLE`**

Status: **DEVELOPMENT ONLY — ZERO PROSPECTIVE E2 SOURCE ACCESS**

Starting fetched `main` SHA: `4219f5082063e8c714cf7f998221aa5125faa7e2`.
The historical backend-rescue classification
`FULL_REFERENCE_BACKEND_UNRESOLVED` remains unchanged for that prior protocol.
No scientific preregistration was created or executed.

## Schedules reached

| schedule | burn-in | retained/chain | global calibration pass |
|---|---:|---:|---|
| S1 | 2048 | 8192 | False |
| S2 | 2048 | 16384 | False |
| S3 | 4096 | 32768 | False |

## Six-state calibration diagnostics

| schedule | n | checkpoint | Rhat | min gap ESS | groups | vector diff | max regret | prior n=24 diff/regret | gate | mechanism |
|---|---:|---|---:|---:|---|---:|---:|---|---|---|
| S1 | 24 | early | 1.003402 | 16196.6 | 227/227 | 0.009677 | 0.000000 | 0.009002/0.000000 | True | — |
| S1 | 24 | middle | 1.001936 | 12104.4 | 371/371 | 0.005928 | 0.000000 | 0.002469/0.000000 | True | — |
| S1 | 24 | late | 1.001229 | 12318.8 | 324/371 | 0.005090 | 0.006363 | 0.001900/0.000000 | False | — |
| S1 | 40 | early | 1.004094 | 5572.7 | 900/900 | 0.021795 | 0.000000 | — | False | — |
| S1 | 40 | middle | 1.004194 | 5192.6 | 740/939 | 0.005673 | 0.005528 | — | False | — |
| S1 | 40 | late | 1.004370 | 4747.4 | 660/740 | 0.005741 | 0.002215 | — | False | — |
| S2 | 24 | early | 1.001007 | 34623.8 | 347/347 | 0.006834 | 0.000000 | 0.003634/0.004791 | True | — |
| S2 | 24 | middle | 1.001074 | 25331.4 | 203/203 | 0.002474 | 0.000000 | 0.002672/0.003028 | True | — |
| S2 | 24 | late | 1.000999 | 24487.2 | 371/371 | 0.002439 | 0.000000 | 0.001307/0.000000 | True | — |
| S2 | 40 | early | 1.003027 | 10778.9 | 899/899 | 0.010166 | 0.000000 | — | False | — |
| S2 | 40 | middle | 1.001634 | 9512.6 | 659/860 | 0.007690 | 0.003677 | — | False | — |
| S2 | 40 | late | 1.001710 | 10286.4 | 660/939 | 0.004635 | 0.004259 | — | False | — |
| S3 | 24 | early | 1.000354 | 71088.9 | 227/227 | 0.005107 | 0.000000 | 0.003011/0.000000 | True | PASS |
| S3 | 24 | middle | 1.000337 | 50641.0 | 203/203 | 0.002181 | 0.000000 | 0.002723/0.001143 | True | PASS |
| S3 | 24 | late | 1.000379 | 48416.1 | 371/371 | 0.003473 | 0.000000 | 0.001041/0.000000 | True | PASS |
| S3 | 40 | early | 1.001385 | 22980.7 | 900/900 | 0.009790 | 0.000000 | — | True | PASS |
| S3 | 40 | middle | 1.001282 | 19841.3 | 859/939 | 0.003621 | 0.001845 | — | False | FINITE_MC_NEAR_TIE |
| S3 | 40 | late | 1.001302 | 20365.0 | 660/740 | 0.003154 | 0.001420 | — | False | FINITE_MC_NEAR_TIE |

## Schedule freeze and held-out validation

No calibration schedule passed globally, so no schedule was frozen and the fresh final development-validation seed was not derived or accessed.

## MCSE interpretation

At the decision schedule, n=40 Rhat and required gap ESS were healthy and all reciprocal regrets were at most 0.01; the remaining strict failures are therefore decision-near-tie failures rather than evidence of an unmixed standard-ESS chain.
Full group-A, group-B, and pooled acquisition vectors, pooled top-ten actions, per-action MCSE, and leader/top-ten gap MCSE are in the CSV and JSON records. These diagnostics did not alter the strict gate.

## Runtime and memory

| n | checkpoint | likelihood eval/transition | factor eval/transition | total factor eval | wall s | peak RSS GB |
|---:|---|---:|---:|---:|---:|---:|
| 24 | early | 2.7383 | 1577.3 | 465152832 | 25.145 | 0.350 |
| 24 | middle | 2.7390 | 1577.7 | 465276096 | 25.224 | 0.374 |
| 24 | late | 2.7418 | 1579.3 | 465746688 | 25.231 | 0.348 |
| 40 | early | 3.9030 | 6244.7 | 1841651200 | 55.229 | 0.361 |
| 40 | middle | 3.9148 | 6263.7 | 1847235200 | 55.675 | 0.355 |
| 40 | late | 3.8954 | 6232.6 | 1838072000 | 55.462 | 0.358 |

Observed median state time was `25.224` s at n=24 and `55.462` s at n=40.
The observed-work projection for 45 shadows is `1616.4` s (`0.449` h).
Peak isolated-worker RSS was `0.374` GB; 16 GB local suitability: `True`.
High-fidelity shadow time remains excluded from the paper's adaptive-versus-FULL routine timing comparison.

## Decision

Strategy case: **`CASE_B_HEALTHY_NEAR_TIE_DECISIONS`**.

Do not develop another sampler next. Formulate and prospectively audit a decision-aligned FULL-reference rule based on acquisition-gap Monte Carlo uncertainty.

Recommended backend for a future replacement E2 preregistration: `STANDARD_ELLIPTICAL_SLICE_FULL_PENDING_DECISION_ALIGNED_RULE_AUDIT`.
Recommended reliability rule to freeze now: none from this diagnostic; retain the current strict rule historically and complete the recommended separate prospective rule/backend task

No prospective source field, trajectory, or posterior state was constructed. The PDE model, BO problem, factor selection, scientific thresholds, E3, and superseded/unrun preregistration remain unchanged.
