# Task 04A-E summary

**Decision: `LEARNING_NO_GO`. The full A100 study and Task 04B are not authorized.**

The single frozen CPU smoke used seed 0, 128 independent local pairs, all five
contexts, 32-by-32 Gauss–Hermite normalization, and 65,536-point endpoint QMC. It is
a mechanism gate, not a multi-seed batch-BO benchmark.

## Eight completion answers

1. **Calibration passed.** The latent t-copula correlation was `0.5132882`; four
   independent normal-score correlations were `0.499944–0.500041`. Endpoint mean
   errors were below `1.3e-5`, variance errors below `4.2e-5`, and covariance mismatch
   was `5.45e-5`.
2. **The oracle was decision-relevant.** Gaussian/t qEI was
   `0.052644/0.048023`, an `8.78%` contrast. Every paired panel state exceeded the 1%
   contrast threshold, and G0's tie-aware global regret was `3.41%`.
3. **The mathematical implementation passed.** Copula density, normalization,
   symmetry, exact G0 recovery, gradients, strongly convex Hessians, marginal
   integration, and tie handling pass the test suite.
4. **P did not meet the density-learning gate.** At `r=1`, U/P joint KL was
   `0.053909/0.048590`: a `0.005319`-nat or `9.87%` gain, below the required 20%.
   Both fits converged in 15 closure calls, so this is not an optimizer failure.
5. **Marginal safety failed.** P's marginal KL remained numerically small
   (`0.00162–0.00201`), but its location shift drove q=1 EI relative error from
   `10.1%` at `r=0` to `18.2%` at `r=1`, versus the 2% cap. Pearson error reached
   `0.01585`.
6. **P worsened q=2 decisions.** Mean r>0 qEI relative error was `5.89%` for G0,
   `18.27%` for U, and `19.92%` for P. On the valid paired panel, P chose the wrong
   endpoint for every significant pair and doubled tie-aware regret from `3.41%` to
   `6.82%`.
7. **Compute was bounded.** U/P took `0.028/0.0088 s`, peak process RSS was about
   `617 MiB`, and both were finite and converged. CUDA parity remains an unexecuted
   test skip because the full study was not authorized.
8. **The prespecified outcome is `LEARNING_NO_GO`.** The oracle and decision panel
   were valid, but P missed density, marginal-safety, and qEI gates. Do not tune the
   penalty, basis, seed, or thresholds retrospectively; a new contract is required
   for any further work.

## Why this outcome is coherent

The model showed a small amount of the intended dependence learning. At `r=1`, its
predicted joint upper-tail co-exceedance moved from G0's `0.01396` toward the true
`0.02563`, reaching `0.01886`. Stronger co-movement makes a two-point batch more
redundant and should lower qEI, matching the direction of the oracle's Gaussian-to-t
change.

That useful signal was overwhelmed by marginal drift. P shifted the standardized
mean by as much as `0.0565`; at the tail incumbent `1.5`, this small location error
inflated q=1 EI by `18.2%`. Its predicted qEI therefore increased from `0.05769` at
`r=0` to `0.06084` at `r=1`, while the oracle decreased from `0.05264` to `0.04802`.
The paired decision consequently reversed in every case.

This is plausible at `n=128`: P has 35 fitted coordinates, the t-copula distinction
is concentrated in comparatively rare joint-tail observations, and precision-10
shrinkage is intentionally strong. The true mixture log-density also varies
nonlinearly with `r`, whereas P uses a single affine-in-`r` energy direction. Finally,
P is tested for marginal preservation but does not enforce it structurally, so finite
data can trade a modest joint-likelihood gain for a decision-damaging marginal shift.

The result therefore rejects the specified low-data affine correction as a safe q=2
mechanism. It does not reject the copula construction, the convex energy mathematics,
or all possible marginal-preserving dependence models.

Compact reviewed evidence is in [results/task04ae/smoke](../../results/task04ae/smoke/README.md).
