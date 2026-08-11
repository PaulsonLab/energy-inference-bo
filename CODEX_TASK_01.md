# CODEX TASK 01 — Mathematical and oracle validation

Read `AGENTS.md` and `MATH_AND_SCOPE.md` completely before coding.

## Goal

Build the smallest transparent implementation that can test four things:

1. the oracle non-Gaussian predictive shape can matter even when mean and variance match a Gaussian exactly;
2. a tiny regularized residual energy can learn useful higher-order shape from modest data;
3. the augmented-probability identity reproduces expected utility and its replicated sharpening;
4. for an actual small exact GP with q=1, the augmented construction reproduces ordinary EI / LogEI decisions.

This task is a mathematical validation task, not a performance benchmark.

Do not implement Vecchia, SAAS, DKL, q>1 batch BO, molecular optimization, acquisition-aware training, or a neural-network EBM.

---

## Part A — oracle non-Gaussian shape

### A1. Distributions

Implement

\[
Z_\star
\sim
0.8N(-0.3,0.8^2)
+
0.2N(1.2,0.8^2)
\]

and

\[
Z_G\sim N(0,1).
\]

Verify analytically/numerically that the mixture mean is 0 and variance is 1.

### A2. Conditional objective family

Use \(x\in[0,1]\).

Implement a small **predetermined** set (for example 3) of smooth scenarios for \(\mu(x)\), \(\sigma(x)>0\), and \(f_{\rm best}\). Do not search a large space of scenarios to manufacture a positive result.

For each scenario,

\[
Y_\star(x)=\mu(x)+\sigma(x)Z_\star
\]

and

\[
Y_G(x)=\mu(x)+\sigma(x)Z_G
\]

must have identical mean and variance at every x.

At least one scenario may be deliberately designed to make upper-tail differences decision-relevant, but label it clearly.

### A3. EI truth

Compute:
- exact/high-accuracy EI under the mixture;
- analytic Gaussian EI;
- independent quadrature checks.

On a dense x-grid report:
- max absolute EI error;
- relative error near high-EI regions;
- EI argmax;
- Spearman rank correlation;
- overlap of top 5% acquisition locations.

Plot true and Gaussian EI.

---

## Part B — tiny residual energy

Implement

\[
q_\phi(z)
=
\frac{
\varphi(z)e^{-R_\phi(z)}
}{
Z_\phi
}
\]

with

\[
R_\phi(z)=
\sum_{k=1}^{K}
\phi_k\tilde b_k(z).
\]

Use:
- K = 9 by default;
- bounded RBF basis functions;
- centers roughly spanning [-3,3];
- reference-centered basis functions as described in `MATH_AND_SCOPE.md`;
- phi initialized at 0;
- strong configurable L2 penalty;
- 64-point Gauss-Hermite quadrature by default.

Fit by regularized maximum likelihood to iid samples from \(Z_\star\).

Development sample sizes:
- 20,
- 50,
- 100,
- 200,
- 500.

Development seeds:
- 3 locally;
- make a configuration for 10+ seeds that can later be run in Colab.

### Fair baselines

Compare:
1. fixed N(0,1);
2. Gaussian location/scale MLE fitted on the same samples;
3. residual energy;
4. Student-t location/scale if it is simple and does not delay completion.

Do not call the residual-energy model better merely because it adjusts finite-sample mean or variance.

### Metrics

Report:
- held-out log score;
- KL(true || model) by high-accuracy numerical integration;
- fitted mean/variance;
- selected upper-tail probability errors;
- EI curve error when embedded in the oracle Y(x) family;
- EI rank correlation;
- top-5% acquisition overlap;
- EI argmax.

CRPS is optional if implementation is straightforward.

---

## Part C — augmented expected-utility identity

Use the known oracle conditional distribution first.

For raw improvement utility

\[
u(x,y)=(y-f_{\rm best})_+,
\]

verify numerically

\[
\pi_1(x)\propto a(x)=E[u(x,Y)].
\]

Use integration and a simple importance sampler:
- x ~ Uniform(0,1);
- y ~ q(y|x);
- weight by u.

Then for M in {1,2,4}, verify

\[
\pi_M(x)\propto a(x)^M
\]

under uniform p0.

Report:
- normalized marginal error;
- inferred mode;
- effective sample size;
- dependence on particle count and M.

Do not hide the expected increase in weight degeneracy as M grows.

If a strictly positive utility is needed for log-energy code, run a separate epsilon-smoothed version and explicitly compare against `EI + epsilon`, not EI.

---

## Part D — exact GP q=1 sanity check

This part is intentionally small.

Use BoTorch/GPyTorch to fit an exact GP to a simple 1D toy objective using roughly 8–12 observations.

Use a standard Matérn kernel and ordinary marginal-likelihood fitting.

On a dense candidate grid:

1. compute the GP posterior mean/variance;
2. compute analytic EI (and LogEI where available);
3. independently compute EI by quadrature over the Gaussian posterior;
4. construct the augmented target using the same posterior and raw improvement utility;
5. verify that its x-marginal is proportional to EI;
6. verify that the maximizer agrees with EI / LogEI.

Also test M = 2 sharpening and verify the marginal is proportional to EI^2 under a uniform design prior.

This is a correctness check only. Do not claim the sampler is superior to LogEI for q=1 Gaussian BO.

---

## Part E — package and tests

Suggested minimal structure:

```text
src/energy_bo/
    oracle/
        distributions.py
        residual_energy.py
        acquisition.py
        augmentation.py
    gp/
        exact_gp.py
    experiments/
        oracle_shape.py
        augmented_inference.py
        gp_q1_sanity.py
    metrics.py
```

Tests must include:
- mixture mean/variance;
- residual-energy normalization;
- R_phi = 0 recovers N(0,1);
- energy gauge-centering behavior;
- mixture EI analytic vs quadrature;
- augmented marginal identity for M=1;
- replicated identity for M=2 on a grid;
- exact-GP analytic EI vs quadrature;
- exact-GP augmented marginal vs EI.

Use:
- Python 3.11+;
- torch.double;
- PyTorch;
- BoTorch / GPyTorch for the exact GP;
- scipy for independent checks;
- pytest;
- matplotlib.

---

## Part F — compute and Colab behavior

The local machine is a MacBook Air with 16 GB RAM.

Run only:
- unit tests;
- 3-seed development versions;
- small particle counts sufficient for a smoke check.

Do not launch large seed sweeps or expensive particle studies.

Create a thin Colab notebook or `COLAB.md` showing exactly how to:
1. clone the public GitHub repository;
2. install the package;
3. run a larger seed sweep;
4. save/download results.

If the smoke tests indicate a materially expensive next experiment, stop and report the recommended Colab configuration instead of running it locally.

---

## Completion report

Report clearly:

1. Does oracle higher-order shape change EI ranking or maximization in any predetermined scenario despite identical mean/variance?
2. Does the residual energy beat a fitted Gaussian location/scale model, and at what sample sizes?
3. Are any EBM gains actually explained by shifts in fitted mean/variance?
4. Does the M=1 augmented marginal match expected utility?
5. Do M=2 and M=4 match the predicted power-sharpened marginals?
6. How quickly does particle ESS deteriorate with M?
7. Does the exact GP q=1 augmented construction reproduce EI / LogEI's maximizer?
8. Based on the evidence, should Task 02 prioritize:
   - exact-GP corrected prequential training;
   - annealed SMC for augmented inference;
   - both;
   - or stop/rethink?

Do not implement Task 02 automatically.
