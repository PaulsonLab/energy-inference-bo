# T2-B Nonlinear-PDE Audit

## Scope and verdict

This audit closes only the concrete structural-influence construction for the
nonlinear-PDE factor family used by E2 and its family-level T4 mapping. It does
not address finite-sample inference certification, continuous action
optimization, samplers, additional PDE models, or whether the adaptive
experiment must choose exactly 40 factors.

Audit and implementation baseline:
`fe8d994119a47b0709651f26a418c946032e90f5`.

Archived prototype:
`notebooks/prototypes/DEC_Nonlinear_PDE_BO_Demo.ipynb`, SHA-256
`73459edc0545ea2470a0cba5ab3cf60d18508b78b0acae08068790905b48e6fd`.

**Verdict: PASS.** The Menz assumptions hold uniformly along every
active-to-full interpolation, the clean sparse comparison matrix agrees with a
literal reproduction of the archived construction to floating-point roundoff,
and the theorem-backed structural replay remains
`0.03874403301354687`.

## Exact model

The latent vector is the complete scalar field on a nonperiodic \(24\times24\)
grid. Its Gaussian-reference precision is

\[
Q=q_0I+q_LL,\qquad q_0=3.5,\qquad q_L=0.6,
\]

where \(L\) is the grid-graph Laplacian. There is no covariance normalization
or jitter in the archived construction.

For site \(j\), with nearest-neighbor set \(\mathcal N(j)\),

\[
r_j(y)=y_j-c\sum_{k\in\mathcal N(j)}y_k+\eta\sin y_j-b_j,
\]

\[
e_j(y)=\gamma\log\cosh(r_j(y)/\tau),
\]

with

\[
c=0.12,\qquad \eta=0.25,\qquad
\gamma=0.08,\qquad\tau=0.30.
\]

The useful constants are

\[
a=1+\eta=1.25,\quad
\alpha=\gamma/\tau^2=0.8888888889,\quad
\beta=\gamma/\tau=0.2666666667,\quad
\delta_{\rm nl}=\gamma\eta/\tau=0.0666666667.
\]

## Derivative audit

With \(u_i\) the coordinate vector at site \(i\),

\[
\nabla r_j=(1+\eta\cos y_j)u_j-c\sum_{k\in\mathcal N(j)}u_k,
\]

\[
\nabla^2r_j=-\eta\sin y_j\,u_ju_j^\top.
\]

For \(z_j=r_j/\tau\), the factor derivatives are

\[
\nabla e_j=\frac{\gamma}{\tau}\tanh(z_j)\nabla r_j,
\]

\[
\nabla^2e_j=
\frac{\gamma}{\tau^2}\operatorname{sech}^2(z_j)
\nabla r_j\nabla r_j^\top
+\frac{\gamma}{\tau}\tanh(z_j)\nabla^2r_j.
\]

Both chain-rule terms are necessary. The first is positive semidefinite; the
second is center-diagonal and can be negative. Globally,

\[
\nabla^2e_j\succeq-\frac{\gamma\eta}{\tau}u_ju_j^\top.
\]

Thus the negative-curvature scale is \(\gamma\eta/\tau\), not
\(\gamma\eta/\tau^2\). Finite-difference tests cover the residual gradient,
residual Hessian, factor gradient, and full two-term factor Hessian.

## Menz comparison construction

The covariance comparison uses Georg Menz, “A Brascamp–Lieb type covariance
estimate,” *Electronic Journal of Probability* 19 (2014), no. 78, Theorem
2.3.

Along the active-to-full path, let each factor coefficient
\(\theta_j\in[0,1]\). For block \(i\), every factor whose support overlaps \(i\)
appears in the exact conditional Hessian. Neighbor-centered factors contribute
nonnegative outer-product curvature. Only the factor centered at \(i\) has a
possibly negative residual-Hessian term. Therefore

\[
\rho_i=Q_{ii}-\gamma\eta/\tau.
\]

For distinct blocks, let

\[
d_{ji}=(1+\eta)\mathbf 1\{i=j\}
+c\mathbf 1\{i\in\mathcal N(j)\}.
\]

Every factor containing both blocks must be counted:

\[
\kappa_{ik}=|Q_{ik}|+\frac{\gamma}{\tau^2}
\sum_{j:\{i,k\}\subseteq D_j}d_{ji}d_{jk}.
\]

The comparison matrix is

\[
A_{ii}=\rho_i,\qquad A_{ik}=-\kappa_{ik}\quad(i\ne k).
\]

The potential is globally strongly convex because

\[
\nabla^2V_{S,s}\succeq Q-(\gamma\eta/\tau)I,
\]

so Menz’s convex-at-infinity assumption holds. The one-block constants,
mixed-Hessian bounds, and \(A\) are independent of the active set, interpolation
parameter, and latent state.

## Positivity and overlap accounting

A domain-independent sufficient condition is

\[
q_0>\frac{\gamma\eta}{\tau}
+\frac{8\gamma(1+\eta)c}{\tau^2}
+\frac{12\gamma c^2}{\tau^2}.
\]

The right side is `1.2869333333`, below \(q_0=3.5\). The coefficients count:

- four nearest-neighbor pairs with two center–neighbor products each;
- four diagonal-neighbor pairs with two neighbor–neighbor products each;
- four axial-distance-two pairs with one neighbor–neighbor product each.

Boundary rows omit some couplings. The minimum row margin is consequently the
interior value `2.2130666667`, independent of the expanding domain size.

## Numerical diagnostics

The clean \(N=576\) sparse construction gives:

| Quantity | Value |
|---|---:|
| minimum \(\rho_i\) | `4.6333333333` |
| maximum \(\kappa_{ik}\) | `0.8666666667` |
| nearest-neighbor coupling | `0.8666666667` |
| diagonal-neighbor coupling | `0.0256` |
| axial-distance-two coupling | `0.0128` |
| minimum row margin | `2.2130666667` |
| \(\lambda_{\min}(A)\) | `2.2366796511` |
| \(\lambda_{\max}(A)\) | `9.1204570358` |
| \(\operatorname{cond}_2(A)\) | `4.0776769402` |
| maximum nonzeros per row | `13` |

The sparse solve residual in the structural replay is
\(7.8\times10^{-15}\). The implementation applies \(A^{-1}\) only through a
sparse solve.

## Notebook-versus-proof comparison

The archived notebook constructs a dense precision, initializes
\(\kappa=|Q|\) off diagonal, then loops through every residual support and adds
all ordered cross-coordinate outer-product bounds. The clean implementation
uses sparse graph Laplacians and the equivalent sparse product of the
factor-derivative-bound matrix.

For the frozen parameters:

- maximum absolute precision difference: `0.0`;
- maximum absolute comparison-matrix difference:
  `1.1102230246251565e-16`;
- rigorous correction factor relative to the notebook construction: `1.0`.

The last-bit difference comes only from floating-point accumulation order. It
does not alter the comparison matrix or structural result mathematically.

## EI structural bound and non-vacuity

Here the finite latent vector is the complete discretized field, so
Rao–Blackwellization is the identity. For distinct action coordinates \(x\)
and \(\widehat x\),

\[
L_i(\bar F_{x,\widehat x})
=\mathbf 1\{i=x\}+\mathbf 1\{i=\widehat x\}.
\]

The factor sensitivity and omitted load are

\[
L_i(e_j)=\frac{\gamma}{\tau}d_{ji},
\]

\[
(h_U)_i=\frac{\gamma}{\tau}\left[
(1+\eta)\mathbf 1\{i\in U\}
+c|\{j\in U:i\in\mathcal N(j)\}|\right].
\]

Consequently,

\[
|G_C(x,\widehat x)-G_S(x,\widehat x)|
\le L(\bar F_{x,\widehat x})^\top A^{-1}h_U.
\]

This calculation uses only the reference precision, supports, parameter bounds,
active mask, and action pair. It does not evaluate omitted residual values,
omitted factor energies, full-target samples, or full-target acquisitions.

For the archived active set with \(M=40\), leader `(14, 12)`, and challenger
`(9, 12)`, the clean replay gives `0.03874403301354686`; the locked notebook
value is `0.03874403301354687`. Thus the theorem-backed structural term is
non-vacuous and unchanged.

The associated sparse gap `-0.0127500536`, empirical inference allowance
`0.0235285003`, and empirical total envelope `0.0495224797` remain historical
replay values. Only the structural component is theorem-backed. The inference
allowance is asymptotic/empirical, so the total is not an end-to-end rigorous
finite-sample certificate.

## T4 classification

**PROVED FOR THIS FAMILY.** The matrices \(A_n\) are a uniformly SPD,
fixed-range sparse family with bounded graph degree and a domain-independent
spectral interval. Benzi and Razouk, “Decay bounds and \(O(n)\) algorithms for
approximating functions of sparse matrices,” *ETNA* 28 (2007), Theorem 3.4,
applied to \(f(A)=A^{-1}\), gives domain-independent exponential inverse decay
in the sparsity-graph distance. Combined with bounded residual support/load and
two-dimensional polynomial graph growth, this verifies the concrete
assumptions used by T4A.

This theorem does not prove that the adaptive experiment must select exactly
\(M=40\); that count remains empirical evidence.

## Implementation record

- reusable code: `src/conditioned_bo/nonlinear_pde_influence.py`;
- focused tests: `tests/test_nonlinear_pde_influence.py`;
- validation runner:
  `experiments/nonlinear_pde/run_structural_validation.py`;
- outputs:
  `experiments/nonlinear_pde/outputs/t2b_structural_validation/`.

## Limitations

- Finite-sample, adaptivity-safe inference-error certification remains open.
- The archived total stopping envelope is empirical despite the now-proved
  structural component.
- The result is for this globally bounded sine nonlinearity and the stated
  stable parameter regime; it does not cover arbitrary nonlinear PDE factors.
- The action comparison is on the archived finite grid.
- The T4 result is a family-level inverse-decay mapping, not a prediction of the
  adaptive factor count on every instance.

**The structural-influence blocker is closed for the main factor families.**
