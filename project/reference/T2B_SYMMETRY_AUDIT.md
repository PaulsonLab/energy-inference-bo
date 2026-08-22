# T2-B Reflection-Symmetry Audit

## Scope and verdict

This audit resolves only the reflection-symmetry portion of T2-B. It does not
address the nonlinear-PDE factor family, the finite-sample inference allowance,
or continuous-action optimization error.

Repository baseline: `ba4c1a75fc799a5d05095d35dbee4f492f4033f6`.

> **Post-audit status note (2026-08-21):** the nonlinear-PDE T2-B
> construction was subsequently proved and regression-tested; see
> `T2B_NONLINEAR_PDE_AUDIT.md`. References below to that construction as
> unresolved are preserved as dated audit provenance. The finite-sample
> inference theory was subsequently completed and its locked prospective
> finite-grid pilot passed. Continuous-action optimization remains outside the
> finite-grid claim, and no guarantee transfers automatically to other
> inference backends.
> See `INFERENCE_CERTIFICATION_IMPLEMENTATION_HANDOFF.md` and `THEORY.tex`.

**Verdict: PASS.** The analytic construction is valid under its stated
assumptions, the implementation reproduces the archived structural constants,
and the prospective EI validation shows a meaningfully non-vacuous bound in the
intended OU regime.

The archived prototype remains unchanged at
`notebooks/prototypes/DEC_Symmetry_Continuous_BO_Demo.ipynb` (SHA-256
`74631161db99d37b386a49bcdffcb9a1e364a8330c3eb7eb1ae76a7569705636`).

## Published covariance result

The construction uses Georg Menz, “A Brascamp–Lieb type covariance estimate,”
*Electronic Journal of Probability* 19 (2014), no. 78, 1–15,
[Theorem 2.3](https://doi.org/10.1214/EJP.v19-2997), labeled “Covariance
estimate, Otto & Menz.”

For a Gibbs measure

\[
\mu(dy)=Z^{-1}e^{-V(y)}dy
\]

on a product of Euclidean blocks, the result requires:

1. the preceding convex-at-infinity assumption on \(V\), which supplies a
   global Poincaré inequality;
2. a one-block Poincaré constant \(\rho_i>0\), uniform in the other blocks;
3. uniform mixed-Hessian bounds
   \(\lVert\nabla_i\nabla_jV\rVert_{\mathrm{op}}\le\kappa_{ij}\);
4. positive definiteness of

   \[
   A_{ii}=\rho_i,\qquad A_{ij}=-\kappa_{ij}\quad(i\ne j).
   \]

It then gives

\[
|\operatorname{Cov}_{\mu}(g,h)|
\le
\sum_{i,j}(A^{-1})_{ij}
\left(\mathbb E_\mu\lVert\nabla_i g\rVert_2^2\right)^{1/2}
\left(\mathbb E_\mu\lVert\nabla_j h\rVert_2^2\right)^{1/2}.
\]

The proof also establishes that \(A^{-1}\) is entrywise nonnegative. The
Bakry–Émery criterion supplies a block Poincaré constant when a conditional
potential has a uniform Hessian lower bound.

## Exact symmetry target

At one BO iteration, let

\[
Y=(Y_1,\ldots,Y_N),\qquad
Y_j=\begin{bmatrix}f(-r_j)\\f(r_j)\end{bmatrix},\qquad
Y\sim\mathcal N(m,K),\qquad Q=K^{-1}.
\]

With \(b=(-1,1)^\top\), the factors are

\[
e_j(Y_j)=\gamma_j\log\cosh\!\left(\frac{b^\top Y_j}{\tau_j}\right).
\]

For active set \(S\) and interpolation \(s\in[0,1]\), the exact marginal
potential is

\[
V_{S,s}(y)
=
\frac12(y-m)^\top Q(y-m)
+\sum_{j=1}^N
\left(\mathbf 1\{j\in S\}+s\mathbf 1\{j\notin S\}\right)e_j(y_j).
\]

The block derivatives are

\[
\nabla_j e_j
=\frac{\gamma_j}{\tau_j}
\tanh\!\left(\frac{b^\top y_j}{\tau_j}\right)b,
\]

\[
\nabla_j^2e_j
=\frac{\gamma_j}{\tau_j^2}
\operatorname{sech}^2\!\left(\frac{b^\top y_j}{\tau_j}\right)bb^\top
\succeq0.
\]

Thus the exact factor sensitivity is

\[
L_i(e_j)=\mathbf 1\{i=j\}\frac{\sqrt2\,\gamma_j}{\tau_j}.
\]

Conditioning on all blocks except \(i\), the conditional Hessian is bounded
below by \(Q_{ii}\). Bakry–Émery therefore gives the uniform choice

\[
\rho_i=\lambda_{\min}(Q_{ii}).
\]

The factors have no cross-block Hessian, so for \(i\ne j\),

\[
\nabla_i\nabla_jV_{S,s}=Q_{ij},\qquad
\kappa_{ij}=\lVert Q_{ij}\rVert_{\mathrm{op}}.
\]

Also, \(\nabla^2V_{S,s}\succeq Q\succ0\), which is stronger than the
convex-at-infinity requirement. None of these constants depends on \(S\),
\(s\), or \(y\). Consequently, whenever \(A\succ0\),

\[
C_S\equiv C=A^{-1}
\]

is one valid conservative operator for every active set and every point on the
active-to-full path.

## Analytic OU positivity

For the equally spaced OU/Matérn-\(1/2\) kernel, let

\[
q=e^{-\Delta x/\ell}.
\]

The spatial precision is exactly tridiagonal, with endpoint diagonal
\(1/(1-q^2)\), interior diagonal \((1+q^2)/(1-q^2)\), and adjacent
off-diagonal \(-q/(1-q^2)\). Pairing \((-r_j,r_j)\) in inner-to-outer order
makes the comparison matrix \(A\) tridiagonal. For \(N\ge2\), its minimum row
diagonal-dominance margin is

\[
\frac{1-q}{1+q}>0.
\]

Strict row diagonal dominance is a sufficient positivity proof; it is not
equivalent to positive definiteness in general. A numerical eigenvalue check is
therefore retained only as a regression diagnostic.

For the archived parameters \(\Delta x=0.05\), \(\ell=0.125\), and \(N=40\),
the clean implementation independently obtains:

| Quantity | Value |
|---|---:|
| \(q\) | 0.6703200460356393 |
| \(\rho_{\rm inner}\) | 1.4146538810285458 |
| \(\rho_{\rm middle}\) | 2.631932441832188 |
| \(\rho_{\rm outer}\) | 1.815966220916094 |
| \(\kappa_{\rm adjacent}\) | 1.2172785608036423 |
| minimum row margin | 0.19737532022490356 |
| \(\lambda_{\min}(A)\) | 0.1990359303880976 |
| \(\lambda_{\max}(A)\) | 5.059021531108995 |
| \(\kappa_2(A)\) | 25.4176294765 |

The sparse solve residual in the clean EI run was
\(5.05\times10^{-15}\). The implementation constructs the analytic AR(1)
precision and applies \(A^{-1}\) only through sparse linear solves.

## Factor-sufficient representation

The finite vector \(Y\) need only be sufficient for the factors; it need not
determine the whole function. For any full-function observable \(g(f)\), define

\[
\bar g(Y)=\mathbb E_{P_{0,t}}[g(f)\mid Y].
\]

Because every likelihood ratio along the conditioning path is measurable with
respect to \(Y\), the conditional law of \(f\mid Y\) is unchanged from the
reference model. Therefore

\[
\operatorname{Cov}_{\pi_{S,s}}(g,e_j)
=
\operatorname{Cov}_{\mu_{S,s}}(\bar g,\varepsilon_j).
\]

This identity is the only reduction needed for Menz’s finite-dimensional
theorem. A deterministic representation \(f=\mathcal T_tY\) remains a special
case.

## EI decision sensitivity

Under the reference GP,

\[
f(x)\mid Y=y
\sim
\mathcal N\!\left(m(x)+a_x^\top(y-m),\sigma_x^2\right),
\qquad a_x=K^{-1}k_{Yx}.
\]

For \(u_x(f)=(f(x)-y_t^\star)_+\), the Rao–Blackwellized EI observable has,
when \(\sigma_x>0\),

\[
\nabla_i\bar u_x(y)=\Phi(z_x)a_{x,i}.
\]

The gradient of an EI gap has the form
\(p a_{x,i}-q a_{\widehat x,i}\) for \(p,q\in[0,1]\). This vector is a convex
combination of \(0\), \(a_{x,i}\), \(-a_{\widehat x,i}\), and
\(a_{x,i}-a_{\widehat x,i}\). Hence

\[
d_i^{\rm EI}(x,\widehat x)
=
\max\!\left\{
\lVert a_{x,i}\rVert_2,
\lVert a_{\widehat x,i}\rVert_2,
\lVert a_{x,i}-a_{\widehat x,i}\rVert_2
\right\}
\]

is valid. At \(\sigma_x=0\), conditional EI is a positive-part function. Smooth
1-Lipschitz approximations with derivatives in \([0,1]\), or equivalently weak
derivatives, preserve the bound and justify passage to the nonsmooth limit.

With

\[
(h_U)_i
=\mathbf 1\{i\notin S\}\frac{\sqrt2\,\gamma_i}{\tau_i},
\]

the resulting reflection-symmetry bound is

\[
|G_C(x,\widehat x)-G_S(x,\widehat x)|
\le
\bigl(d^{\rm EI}(x,\widehat x)\bigr)^\top A^{-1}h_U.
\]

## Archived prototype alignment

The archived notebook uses exponential utility and differences of **log
acquisitions**, not EI. Its footprint

\[
d_i=\beta\lVert[a_x-a_{\widehat x}]_{Y_i}\rVert_2
\]

is valid after an additional interpolation between the two linear action tilts:

\[
V_{S,s,r}^{x,\widehat x}(y)
=V_{S,s}(y)
-\beta\{(1-r)a_{\widehat x}+r a_x\}^\top(y-m).
\]

The added term is linear, so it changes no Hessian, Poincaré, coupling, or
comparison-matrix constant. Integrating Menz’s covariance bound in \(r\) and
\(s\) produces the notebook’s \(h_U^\top A^{-1}d\) term without an extra
factor. This justifies the historical structural calculation but does not turn
it into an EI experiment.

The clean implementation reproduces the notebook’s five recorded structural
bounds:

\[
0.995451334989,
\quad 0.666572995738,
\quad 0.472007142052,
\quad 0.447917578966,
\quad 0.082866650278.
\]

The notebook audit also confirmed:

- its \(Q_{ii}\), \(\lVert Q_{ij}\rVert_{\mathrm{op}}\), signs, factor bound,
  and linear-solve orientations are correct;
- omitted factors are not evaluated during screening, and all factors are first
  evaluated in the final validation cell;
- sampling uses a Cholesky factor of \(K+10^{-12}I\) while the precision and OU
  conditional representation use \(K\), a formally inconsistent but numerically
  negligible mismatch here;
- `np.linalg.inv(K)` obscures the exact sparse AR(1) precision;
- the notebook’s final validation withholds factor evaluation but reuses the
  screening Gaussian draws, so it is not statistically independent held-out
  Monte Carlo;
- its Monte Carlo and action-grid errors remain separate from the now-proved
  structural bound.

## Clean EI non-vacuity validation

The frozen validation is recorded under
`experiments/symmetry/outputs/t2b_ei_validation/`. Before any full-target
calculation, it fixed:

- the committed clean-EI validation's \(N=40\), OU, factor, mean, and
  action-domain parameters; its reference mean is not the archived notebook's
  separate log-acquisition mean;
- incumbent \(y^\star=0.50\);
- EI tolerance \(0.01\), explicitly not the archived \(0.03\) log-acquisition
  tolerance;
- 80,000 screening draws with seed 123, 401 actions, and batches of three
  factors;
- a prospective non-vacuity criterion requiring at least 20% of factors to
  remain omitted;
- eight independent full-target validation seeds with 20,000 draws each.

The screen stopped with 15 of 40 factors active, leaving 25 factors (62.5%)
unevaluated before validation. The final screened action was \(-0.2082\), the
sparse gap was \(-0.00025484\), the structural term was \(0.00533922\), and the
optimistic envelope was \(0.00508439<0.01\). Refining from 401 to 1601 actions
moved the leader by \(0.00065\).

Fresh full-target validation placed the empirical leaders between \(-0.2147\)
and \(-0.2043\), with the same coarse-grid action in four of eight replicates
and a maximum action displacement of \(0.0065\). The largest observed
full-target EI regret at the screened action was \(3.78\times10^{-4}\), and all
eight replicates were below the fixed EI tolerance. Full-target
importance-sampling ESS ranged from 66.1% to 66.8%.

Only the structural term is rigorous here. The importance-sampling and grid
checks are high-accuracy empirical diagnostics, not a finite-sample inference
bound or continuous-action certificate.

## Intrinsic and instantiation-specific assumptions

Intrinsic to the screening method are a path-uniform covariance bound, cheap
factor and decision sensitivities, an influence application cheaper than full
factor processing, and componentwise nonnegative attribution for the current
factor-ranking formula.

Sufficient only for this instantiation are the finite Gaussian factor vector,
convex block-local `logcosh` factors, Bakry–Émery block constants, Menz’s
mixed-Hessian comparison, global essential-supremum gradient bounds, a common
\(A^{-1}\), and the OU/AR(1) kernel. Other valid constructions need not share
these choices. The matrix may also change across BO iterations as \(K_t\)
changes.

## Remaining limitations

- Non-vacuity is demonstrated for one frozen mechanism regime, not across the
  full E1 repeated-trial and baseline program.
- This dated validation itself supplies no finite-sample, adaptivity-safe
  inference allowance; the later exact-rejection construction and separate
  prospective finite-grid pilot passed.
- The action-grid refinement is not a rigorous continuous optimizer bound.
- At this audit's baseline, the nonlinear-PDE T2-B construction remained
  unresolved; it was subsequently proved as noted above.
- This result does not establish T4 influence decay for the experimental graph.
