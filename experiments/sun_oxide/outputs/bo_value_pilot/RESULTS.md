# Sun oxide GW BO value pilot

Terminal state: `PASS_PBE_VALUE_COLAB`.

- RUN_SHA: `44f58f100f41247afe0937e42eebe58055104225`
- Frozen config SHA-256: `6cc47d41dfbdbf88187d535d405ca6afd971e4b07f91932d55dbbbf5c101ef0f`
- Methods: `NO_PBE`, `FULL_PBE`
- FULL_PBE routine posterior: Laplace approximation, not the exact conditioned posterior.
- Oracle values were isolated from both acquisitions and used globally only after all rollouts for evaluation.

## Laplace importance validation

- `seed_0_initial`: passed=True, ESS/N=0.905102, IS regret=0.00289358, gap MC SE=0.00137015.
- `seed_0_after_6_queries`: passed=True, ESS/N=0.90534, IS regret=3.53229e-05, gap MC SE=3.53153e-05.
- `seed_0_after_12_queries`: passed=True, ESS/N=0.906766, IS regret=8.74349e-05, gap MC SE=8.742e-05.

## Frozen scientific metrics

- Median AURC, NO_PBE / FULL_PBE: `19.444499999999998` / `1.0169999999999995` eV.
- Median final regret, NO_PBE / FULL_PBE: `0.8309999999999995` / `0.0` eV.
- Fraction of seeds FULL_PBE wins: `0.8333333333333334`.
- Global-optimum discovery count, NO_PBE / FULL_PBE: `4` / `10`.
- PBE-vs-GW Spearman diagnostic: `0.833265591875016`.
