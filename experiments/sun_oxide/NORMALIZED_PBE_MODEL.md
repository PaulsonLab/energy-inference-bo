# Frozen normalized PBE model for E3

`PBE_SUPPORT_500_V1` contains all 191 BO action nodes and 309 PBE-only nodes.
The latter are chosen by deterministic farthest-point sampling in the already
validated standardized 132-dimensional Magpie descriptor space. Exact distance
ties use the lexicographically smaller `composition_key`. Neither PBE gaps nor
GW values enter support selection.

`NORMALIZED_ALL_PAIRS_PBE_500_V1` contains every strict PBE pair on that frozen
support and omits every exact PBE tie. With `omega=1/(500-1)=1/499`, factor
`(i,j)` has energy

```text
omega * softplus(-s_ij * (Y_i-Y_j)),  s_ij = sign(PBE_i-PBE_j).
```

This is a normalized composite/generalized-Bayes ranking energy, not an
independent all-pairs likelihood. It makes no numerical comparison between PBE
and GW values.

For each factor, endpoint gradient magnitudes are at most `omega`, its Hessian
is positive semidefinite, and its mixed-Hessian magnitude is at most
`omega/4`. If `W_pbe` is the weighted strict-pair adjacency, then its maximum
row sum and spectral norm are at most one. The existing scalar-block Menz
construction therefore uses

```text
A0 = Q0 - 0.25 * W_pbe,
lambda_min(A0) >= 1 - 0.25 = 0.75.
```

Later scalar observations add nonnegative diagonal precision: `At=A0+Dt`.
The existing symmetric nonsingular M-matrix/resolvent argument makes
`C=A0^-1` a fixed conservative operator. This experiment is a model-specific
application of the committed theory; it introduces no covariance theorem and
does not edit the theory ledger.

For factor `(a,b)` and action comparison `(x,xhat)`, the structural influence
diagnostic is

```text
omega * (C[x,a] + C[x,b] + C[xhat,a] + C[xhat,b]).
```

The PBE-only MAP signal and 256 stratified action-pair influence calculations
are target-blind diagnostics, not pass/fail sparsity or performance gates.
