# Task 02C — local implementation and preflight summary

## Status

The bounded Task 02C implementation is complete and the prescribed local checks pass.
The full six-case, three-repeat Colab comparison has **not** been run. Therefore the
current research recommendation is **NO-GO pending full evidence**, not a negative
judgment on the method.

## Eight completion questions

1. **Does the Gibbs/envelope mathematics pass?** Yes. Across 90 saved-teacher checks,
   the maximum autodiff-versus-tilted envelope error was `2.178e-12`; the maximum
   second-order finite-difference error was `6.297e-07`.
2. **Does the implementation agree across frameworks?** Yes. Maximum JAX/PyTorch
   objective and gradient discrepancies were `1.688e-13` and `6.366e-12`.
3. **Does the fused target retain the exact SAAS potential?** Yes. Centered potential
   and unconstrained-gradient errors were `3.979e-13` and `8.527e-13`, including
   NumPyro's transformation Jacobians.
4. **Is the teacher decision tilt immediately degenerate?** No in this preflight. At
   beta one, ESS/P ranged `0.6484`–`0.9869` across MAP, teacher, and midpoint designs
   for all six archived early/late cases. This does not establish that K=8 is enough.
5. **Is stable LogEI validated?** Yes. Tests cover extreme standardized improvements,
   BoTorch agreement, latent-variance flooring, and design gradients without adding an
   EI epsilon.
6. **Is the comparison computationally matched?** Yes. The smoke charged 112 total
   factorization-equivalents to each method, including four shared initialization
   steps; each branch had 80 post-initialization equivalents and four design attempts.
7. **What happened in the reduced smoke?** P-SVGD and DT-SVGD normalized regrets were
   `0.4619` and `0.4609`. The four-step beta bridge forced all four increments and is
   deliberately too small for scientific interpretation. Particle median distances
   remained above `5.13` in whitened coordinates, so the wiring failure mode was not
   immediate particle duplication.
8. **Is Task 02C scientifically supported yet?** **No conclusion pending the full
   Colab study.** Only that study can decide whether DT-SVGD consistently improves
   teacher regret at matched compute. Do not implement a subsequent task.

Local machine-readable evidence is indexed under
[`results/task02c/smoke/`](../../results/task02c/smoke/). Raw traces and plots remain
ignored under `artifacts/task02c/`.
