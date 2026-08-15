# Task 05A mathematics

For encoded sequences (x,x'), S0 uses

\[
k_0(x,x')=\sigma_f^2\exp[-\rho\,d_H(x,x')].
\]

S1 embeds amino acids with the positive spectral part of BLOSUM62, sums residue
inner products over aligned positions, and applies the Tanimoto normalization
(k=s/(n_x+n_{x'}-s)). S2 follows the LOCK-GP equation: BLOSUM50 is converted to
a median-normalized correlation matrix, a locally exponentiated product term is
multiplied by one transformed-correlation sum, and a second sum is added. Its
positive parameters carry the published Gamma and LogNormal priors.

Each model is an ordinary exact GP. For training-only standardization
\(\tilde y=(y-m)/s\), raw predictive moments are
\(\mu_y=m+s\mu_{\tilde y}\) and
\(v_y=s^2v_{\tilde y}\). Gaussian NLL/calibration uses likelihood variance;
LogEI uses latent variance. With incumbent (f^\star),

\[
EI(x)=(\mu-f^\star)\Phi(z)+\sigma\phi(z),\qquad
z=(\mu-f^\star)/\sigma.
\]

Sequential finite-pool regret is

\[
R_t=\frac{f_{\max}-f_t^\star}{f_{\max}-f_0^\star}.
\]

The experiment measures belief quality, not a novel energy formulation. No
generator, semantic design prior, neural EBM, rollout, or downstream composition
is introduced here.
