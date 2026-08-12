# Task 03A mathematics

## Reference-PIT correction

For continuous mixture reference `p0`, let `u=F0(y|x,D)` and
`z=Phi^{-1}(u)`. Under the reference, z is standard normal. A normalized residual
law `fW(z|c)` therefore induces

\[
q_W(y\mid x,D)=p_0(y\mid x,D)\frac{f_W(z\mid c)}{\phi(z)}.
\]

For the affine energy `R_W(z,c)=b(z)^T W h(c)`, this ratio is
`exp(-R_W)/Z_W(c)`. Consequently

\[
Q_W(y\mid x,D)=G_W(\Phi^{-1}(F_0(y\mid x,D))\mid c).
\]

The implementation retains `Z_W(c)` everywhere. “Exact normalization” means exact
reduction to deterministic one-dimensional quadrature; it is not a closed form.

## Gauge and convexity

Each RBF is centered by its analytic expectation under N(0,1), removing the redundant
constant-energy direction. With feature `a(z,c)=b(z) tensor h(c)`, the penalized
objective has Hessian

\[
\sum_i \operatorname{Cov}_{q_i}[a(Z,c_i)] + \lambda I \succeq \lambda I.
\]

Positive-weight Gauss–Hermite quadrature preserves this log-sum-exp convexity. The
default precision is lambda=10 and zero initialization exactly recovers the reference.

## Warped-GP truth

The latent exact posterior is Gaussian. For `h(g)=expm1(alpha g)/alpha`, inverse
`log1p(alpha y)/alpha`, and maximization incumbent y-star, let g-star be its inverse.
Then

\[
EI=\frac{e^{\alpha\mu+\alpha^2s^2/2}
\Phi((\mu+\alpha s^2-g_\star)/s)
-e^{\alpha g_\star}\Phi((\mu-g_\star)/s)}{\alpha}.
\]

The alpha-to-zero regime uses ordinary Gaussian EI. Tests compare these identities,
the transformed density/CDF, and PIT-space quadrature against independent numerical
integration or Monte Carlo.
