# Task 04A-E mathematics

## Matched-marginal copula mixture

For `u=Phi(z1)`, `v=Phi(z2)`, and `xj=t_nu^{-1}(Phi(zj))`,

\[
p_T(z_1,z_2)=\phi(z_1)\phi(z_2)
\frac{t_{\nu,R}(x_1,x_2)}{t_\nu(x_1)t_\nu(x_2)}.
\]

Both coordinates are standard normal. The Gaussian endpoint has the same marginals.
Consequently `p_r=(1-r)p_G+r p_T` is normalized and retains those marginals. Pearson
covariance is an expectation and is therefore constant in `r` once the endpoint
normal-score correlations are matched.

## Convex bivariate correction

Let `b` be Task 04A's seven centered RBFs and `f_sym` the 28 nonredundant symmetric
pair products. The correction is

\[
\Delta E_\theta=a^T[b(z_1)+b(z_2)]+r\,\beta^Tf_{sym}(z_1,z_2).
\]

For fixed data and quadrature, each NLL term is affine in `theta` plus a log-sum-exp
of affine functions. Its Hessian is a tilted feature covariance. Adding precision
`10` gives minimum eigenvalue at least `10`, so the fitted problem is strongly convex.
Subtracting the G0 expectation from pair features is a context-dependent energy
constant; it cancels from the normalized density and fixes the numerical gauge.

## Marginals and qEI

The learned marginal is evaluated explicitly through the G0 conditional:

\[
p_\theta(z_1\mid r)=\phi(z_1)
\frac{E_{G0(Z_2\mid z_1)}[e^{-\Delta E_\theta(z_1,Z_2;r)}]}
{E_{G0}[e^{-\Delta E_\theta(Z_1,Z_2;r)}]}.
\]

This exposes marginal KL, CDF, moments, and q=1 EI rather than assuming that a joint
fit preserved them. For affine batch outputs, qEI is the normalized free-energy ratio

\[
\frac{E_{G0}[(\max(Y_1,Y_2)-y^\star)_+e^{-\Delta E_\theta}]}
{E_{G0}[e^{-\Delta E_\theta}]}.
\]

The r-grid assesses curve recovery. Paired Sobol batches provide the actual decision
test, with uniform averaging over numerical maximizer sets for exact G0/U ties.
Training normalizers use tensor Gauss–Hermite quadrature. qEI uses an independent
deterministic two-dimensional Gauss–Legendre rule split over the regions `Y1>Y2` and
`Y2>Y1`, avoiding slow quadrature convergence at the max-utility kink.
