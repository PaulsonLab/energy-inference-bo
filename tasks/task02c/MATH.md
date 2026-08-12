# Task 02C mathematics

## Exact decision tilt

Let (p(\eta\mid D)) be the normalized SAAS structural posterior in NumPyro's
unconstrained coordinates and let (a_\eta(x)=EI_\eta(x)>0). Define

\[
A(x)=E_p[a_\eta(x)].
\]

For every density (q\ll p),

\[
E_q[\log a_\eta(x)]-KL(q\|p)
=\log A(x)-KL(q\|q_x^\star),
\]

where

\[
q_x^\star(\eta)=\frac{p(\eta\mid D)a_\eta(x)}{A(x)}.
\]

The result follows by substituting
(\log q_x^\star=\log p+\log a-\log A). Therefore the supremum is
(\log A(x)), attained exactly by (q_x^\star). This requires finite, nonzero
(A(x)) and positive EI almost surely. Gaussian EI is strictly positive when latent
predictive variance is positive; the implementation uses stable LogEI without adding
an epsilon.

## Envelope identity and annealed path

If differentiation and posterior expectation may be interchanged,

\[
\nabla_x\log A(x)
=\frac{E_p[\nabla_x a_\eta(x)]}{A(x)}
=E_{q_x^\star}[\nabla_x\log a_\eta(x)].
\]

For

\[
q_{x,\beta}(\eta)\propto p(\eta\mid D)a_\eta(x)^\beta,
\]

the corresponding homotopy objective is

\[
F_\beta(x)=\frac{1}{\beta}\log E_p[a_\eta(x)^\beta],
\qquad
\nabla_xF_\beta=E_{q_{x,\beta}}[\nabla_x\log a_\eta(x)].
\]

At beta zero the continuous limit is (E_p[\log a_\eta(x)]); at beta one it is
(\log A(x)). Intermediate beta targets are a continuation device, not the final
fully Bayesian acquisition.

Particle locations must be retargeted when x changes. Task 02C therefore controls both
beta increments and design movement with conditional ESS, and finishes every design
move with a structural correction block. Without that restriction the envelope
average would be computed under a stale structural tilt.

## Exact SAAS energy

The trusted reference target is NumPyro's `initialize_model` potential in its own
unconstrained latent tree. It includes all support transformations and log-Jacobian
terms for outputscale, global SAAS scale, and local inverse lengthscale squares.

The runtime fused potential is

\[
U_{\rm prior}(\eta)+NLL_{\rm GP}(\eta)-\beta\,LogEI_\eta(x).
\]

The exact GP likelihood and LogEI share one Cholesky factorization. The fused posterior
is used only after its value differences and unconstrained gradients agree with the
trusted NumPyro posterior. Fixed observation noise enters the training covariance;
EI uses latent posterior variance with the BoTorch-compatible (10^{-12}) floor.

## Fairness and scope

P-SVGD and DT-SVGD share teacher-free initialization, whitening, particles, optimizer,
kernel, structural-step count, cache-build count, design-attempt count, and initial x.
Fresh NUTS is evaluation only. The study is q=1 on the existing D=10 embedded Branin
cases and does not define a BO loop or any later-stage model.
