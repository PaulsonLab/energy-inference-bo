# CURRENT_NLR_PBE_GW_V1 descriptor/graph Colab result

Colab verdict: `PASS_DESCRIPTOR_GRAPH_COLAB`

This is a pending compatibility-gate result awaiting independent ZIP
verification. It is not a final `PASS_DESCRIPTOR_GRAPH` record and contains no
preference factors, influence calculation, inference, or BO result.

## Provenance and scope

- RUN_SHA: `843b2173454b70cf12a6199b4d1a32740e60315e`
- Frozen benchmark: `CURRENT_NLR_PBE_GW_V1`
- Legacy/action counts: `2142` / `191`
- GW values read: `False`

## Descriptor compatibility

- Raw descriptor shape: `[2142, 132]`
- Non-finite values: `0`
- Zero-variance features: `15`
- Matrix SHA-256: `c85018bc9be1469f76d741e12ba23491b86f7793094ce9340d547b860ebd2148`
- Row-key SHA-256: `bc15fb684f09e1568a0cebb85909618162c3ccf6d01b40c7d1fca5746a79a625`

## Graph compatibility

- 10-NN union edges: `14063`
- MST edges: `2141`
- Final unique edges: `14072`
- Connected components / isolated nodes: `1` / `0`
- Degree min / median / mean / max: `10` / `13.0` / `13.139122315592903` / `26`
- Exact distance ties encountered: `20`

## Graph-Gaussian reference

- Q0 smallest / largest eigenvalue: `0.999999999999998` / `2.4241838462755485`
- Q0 symmetric, nonpositive off-diagonals, positive definite: `True`, `True`, `True`

Sparse-solve diagnostics:

- `ones`: residual `9.060869410829694e-16`, wall time `0.02881398799991075` s.
- `linear_minus_one_to_one`: residual `7.489941091826488e-16`, wall time `0.026898291999941648` s.
- `sinusoid_index_times_sqrt_two`: residual `5.888006486024402e-16`, wall time `0.026922048999949766` s.
