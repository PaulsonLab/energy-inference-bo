# Task 04A-E contract

## Question and fixed oracle

Use standard-normal pair marginals and Pearson correlation `0.5`. The Gaussian
endpoint is a bivariate normal. The non-Gaussian endpoint is a `df=3` Student-t
copula whose latent correlation is deterministically calibrated so its normal-score
Pearson correlation is also `0.5`. Context `r` mixes the endpoint densities.

The oracle must pass marginal mean `<1e-3`, variance `<2e-3`, covariance mismatch
`<5e-3`, identical analytic q=1 EI, and at least 5% endpoint qEI contrast at incumbent
`1.5` before fitting begins.

## Models and data

- G0 is the matched bivariate Gaussian.
- U adds one shared seven-RBF unary correction.
- P adds 28 unique symmetric pair coordinates multiplied by `r`.
- Basis centers, bandwidth, zero initialization, and L2 precision `10` are inherited
  from Task 04A. The pair feature is gauge-centered under G0.
- Normalization uses correlated tensor Gauss–Hermite quadrature. Fit the summed NLL
  with double-precision full-batch L-BFGS for at most 250 iterations.
- Draw independent local pairs with contexts uniform on `{0,.25,.5,.75,1}` and use
  nested sizes `64,128,256,512`.

The decision panel uses 128 frozen Sobol mean/scale states, each duplicated at `r=0`
and `r=1`. Uniform tie-aware regret prevents arbitrary indexing from crediting G0/U
for their exact context ties. The panel is invalid unless at least 75% of endpoint
pairs have 1% oracle contrast and G0 tie-aware global regret is at least 1%.

## Execution gate

Smoke is one CPU seed at `n=128`, 32-by-32 normalization, and fixed QMC truth. It
authorizes the full study only when every identity and panel check passes, P converges,
P improves r=1 joint KL over U by 20%, P reduces mean r>0 qEI error by 25% versus
both baselines, and maximum P q=1 EI error is below 2%.

Full evidence, if authorized, uses eight seeds, every nested size, 48-by-48
normalization, and A100-batched evaluation. The frozen A–D gates are encoded by the
runner and cover marginal safety, dependence learning, batch decisions, and compute.

## Prohibited additions

No q>2, learned geometry, sequential BO, SAAS/NUTS, molecular data, neural EBM,
acquisition sampler, Task 04B implementation, or retrospective threshold change.
