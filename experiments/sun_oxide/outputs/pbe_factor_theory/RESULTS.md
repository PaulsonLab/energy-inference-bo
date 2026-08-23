# Frozen PBE-order factor and existing-theory compatibility result

Terminal verdict: `PASS_PBE_FACTOR_THEORY`.

This is the model-specific E3 construction for `CURRENT_NLR_PBE_GW_V1`.
It constructs no posterior, runs no Bayesian optimization, and changes no
project-level covariance theorem.

## Frozen factor bank

- Starting SHA: `903f61930e8e46f5e124083fb728084b62912d51`
- Factor model: `ADJACENT_STRICT_PBE_ORDER_V1`
- Temperature: `1.0`
- Legacy nodes / strict adjacent factors: `2142` / `1681`
- Exact PBE tie groups / skipped adjacent ties: `369` / `460`
- Tie-size histogram: `{'3': 57, '2': 296, '5': 2, '4': 14}`
- Incident nodes / fraction: `2050` / `0.9570494864612512`
- Endpoint degree distribution / maximum: `{'0': 92, '1': 738, '2': 1312}` / `2`

This is not Sun et al.'s all-pairs likelihood. It is a sparse, transparent
ordinal legacy-information model. Every retained relation is a true strict
ordering in the frozen PBE data; no arbitrary preference is inserted inside
an exact PBE tie. No comparison across PBE and GW numerical values is made.

## Factor calculus and existing Menz construction

For `d = z_a-z_b`, `e=softplus(d)`. Therefore
`gradient(e)=[sigmoid(d),-sigmoid(d)]` and
`Hessian(e)=sigmoid(d)(1-sigmoid(d))[[1,-1],[-1,1]]`. The endpoint gradient
magnitudes are at most one, the Hessian is positive semidefinite, and the
mixed-Hessian magnitude is at most `1/4`. Stable softplus/sigmoid evaluations
and finite-difference regressions cover these identities.

With one scalar Menz block per node, Gaussian conditional curvature is
`Q0_ii`. Convex preference factors cannot reduce it, while a factor edge can
increase cross-coordinate Hessian magnitude by at most `1/4`. Thus the
existing Menz construction is bounded by `A0 = Q0 - 0.25 R`; this is a
model-specific use of the paper's existing theory, not a new theorem.

- `max degree(R)`: `2`
- `||R||_2` (numerical): `1.988275914308721`
- Analytic `lambda_min(A0)` lower bound: `0.5`
- Numerical `lambda_min(A0)`: `0.5896249844278044`
- Symmetric / nonpositive off-diagonals / SPD: `True` / `True` / `True`

Nonnegative diagonal observation precision gives `At=A0+Dt`. The committed
full-size synthetic patterns remain SPD and satisfy the sampled sparse-solve
resolvent identity `A0^-1-At^-1=A0^-1 Dt At^-1`; the factor Hessian bound is
unchanged. Hence the single fixed operator `C=A0^-1` is conservative at later
scalar-observation BO iterations. No dense inverse was formed.

## Target-blind influence diagnostic

- Sparse A0 factorization / 191-RHS solve time: `0.22124691610224545` / `0.12354254117235541` seconds.
- Unordered action pairs: `18145`

Fractions of factors required to account for structural influence:

- 90 percent: 25th / median / 75th / 90th percentile `0.30517549077929806` / `0.32837596668649616` / `0.3509815585960738` / `0.37180249851279`.
- 95 percent: 25th / median / 75th / 90th percentile `0.535990481856038` / `0.5597858417608567` / `0.5812016656751934` / `0.5990481856038072`.
- 99 percent: 25th / median / 75th / 90th percentile `0.8530636525877454` / `0.8685306365258775` / `0.8786436644854253` / `0.8857822724568709`.

These quantities are diagnostics only and have no pass/fail sparsity
threshold. Factor-set variation and reference-graph distance summaries are in
`influence_summary.json`; every action-pair diagnostic is in
`influence_pair_summary.csv`.

## Isolation

GW oracle read: `False`. No GW target statistic was computed. The scientific
input interface admitted only the frozen legacy PBE table, sparse `Q0`, action
mapping, and NLR data-use notice.
