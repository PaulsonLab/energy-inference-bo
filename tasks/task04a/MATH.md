# Task 04A mathematics

## Directed normalization and Gaussian nesting

For a fixed order and predecessor sets (C_i\subset\{1,\ldots,i-1\}),

\[
p_\Theta(y_{1:n}\mid X)=\prod_i p_\Theta(y_i\mid y_{C_i},X)
\]

is normalized because each scalar factor integrates to one in its response while its
parents remain fixed. It is an exact joint density for the specified directed graph,
not an undirected pseudo-likelihood.

For a zero-mean GP, conditioning gives

\[
\mu_i=K_{iC}K_{CC}^{-1}y_C,\qquad
\sigma_i^2=K_{ii}-K_{iC}K_{CC}^{-1}K_{Ci}+10^{-8}.
\]

Using every predecessor is the Gaussian chain rule and reproduces the GP joint
density. Restricting to the fixed nearest predecessors defines an ordering-dependent
Vecchia Gaussian approximation. G0 uses these truncated Gaussian factors exactly.

## Affine correction and convexity

Let the fixed local Gaussian reference coordinate be

\[
z_i(y)=\frac{y-\mu_i}{\sigma_i}.
\]

Let (b(z)\in\mathbb R^7) be fixed bounded standard-normal-centered RBFs and

\[
s_i=\sum_{j\in C_i}\omega_{ij}b(y_j).
\]

The correction is

\[
\Delta E_i(y)=a^Tb(z_i(y))+b(z_i(y))^TBs_i=\theta^T\phi_i(y).
\]

For each conditional,

\[
\nabla^2\left[E_i(y_i)+\log\int e^{-E_i(t)}dt\right]
=\operatorname{Cov}_{p_\theta(t\mid c_i)}[\phi_i(t)]\succeq0.
\]

Adding (lambda\|\theta\|^2/2) with (lambda=10) makes the discrete-quadrature
training objective strongly convex with Hessian at least (10I). This claim depends
on keeping geometry, Gaussian reference, scaling, bases, and neighbor responses fixed.
Standardizing the child by its fixed local reference does not change convexity and
prevents the correction from becoming constant merely because the local Gaussian
variance contracts.

Bounded RBF corrections cannot overcome the Gaussian quadratic tails, so every
conditional normalizer is finite. Standard-normal centering fixes the inherited
constant-energy gauge convention; the normalizer is retained in all evaluations.

## Oracle truths

G samples G0 directly. W samples latent local Gaussian (g_i) and applies
(h(g)=\operatorname{expm1}(0.6g)/0.6); density follows by the inverse-map Jacobian.

I uses

\[
\Delta E_i^\star(y)=2[u^Tb(z_i(y))][v^Ts_i],
\]

where

\[
u=(0,-1,0,2,0,-1,0)/\sqrt6,\quad
v=(0,0,-1,0,1,0,0)/\sqrt2.
\]

Changing (v^Ts_i) changes the coefficient of the nonconstant function
(u^Tb(z_i(y))).
The log-density ratio between two such contexts is therefore nonconstant in (y), so
no context-free unary correction can represent the entire conditional family.

## Free-energy EI

For (p(y\mid x)=e^{-E(y;x)}/Z_B(x)) and raw improvement
(u(y)=(y-y^\star)_+),

\[
EI(x)=\frac{\int u(y)e^{-E(y;x)}dy}{\int e^{-E(y;x)}dy}
=\frac{Z_U(x)}{Z_B(x)},\qquad
\log EI=F_B-F_U.
\]

The numerator is integrated directly on (y>y^\star); no log utility is evaluated
where improvement is zero. When the correction is zero, this reduces to analytic
Gaussian EI. W also has an independent transformed-Gaussian analytic EI; I is checked
by independent scalar integration.
