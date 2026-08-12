# Task 02C — full decision-tilted SVGD result

## Scope and conclusion

Task 02C is complete. The full Colab falsification study used the six D=10
embedded-negative-Branin cases (seeds 0–2 at n=16 and n=40), K=8/16/32,
post-initialization budgets 32/64, and three paired repeats. Fresh 256-particle SAAS
NUTS teachers provided continuous q=1 EI references.

The result is **NO-GO for the tested decision-tilted SVGD configuration**. This does
not invalidate the Gibbs or envelope identities, which passed strict tests, and it is
not evidence against every possible decision-energy transport method.

## Eight completion questions

1. **Did the mathematical and teacher preflight pass?** Yes. Across 90 fresh-teacher
   checks, maximum envelope autodiff and finite-difference errors were `2.046e-12` and
   `6.337e-07`; JAX/PyTorch value and gradient discrepancies were at most `5.205e-13`
   and `1.125e-11`.
2. **Was the exact SAAS energy reproduced?** Yes. Maximum centered-potential and
   unconstrained-gradient errors were `4.403e-11` and `6.314e-11`, including
   NumPyro's support transformations and Jacobians.
3. **Was the decision tilt intrinsically degenerate?** No. At beta one, teacher
   ESS/P ranged `0.612–0.986` with median `0.866`; none of the 18 checks fell below
   `0.1`. The target itself was therefore not rejected by teacher-weight collapse.
4. **Was the comparison complete and computationally matched?** Yes. All 108 paired
   runs were present. Every pair had identical K, initial particles, structural steps,
   cache builds, design attempts, and factorization-equivalent counts.
5. **Did DT-SVGD improve q=1 decisions?** No. It reduced normalized regret in only
   `20/108` pairs (`18.5%`). Median regret was `0.4139` for P-SVGD and `0.5330` for
   DT-SVGD; the paired DT-minus-P median was `0.06984`. No run of either method reached
   10% regret. For context, MAP-SAAS median regret was lower at `0.3543`.
6. **Did more particles or structural work rescue the result?** No systematic rescue
   appeared. DT median regret remained `0.486–0.622` across K/budget groups, and its
   best observed regret was `0.1135`. K=8 was indeed a stress test, but K=32 and B=64
   did not establish acceptable decision accuracy.
7. **What do cost and geometry indicate?** Median charged times were `22.9 s` for
   P-SVGD, `33.1 s` for DT-SVGD, and `23.4 s` for fresh NUTS. No final run met the
   prespecified collapse condition and no bandwidth-floor event occurred. However,
   initialization forced its likelihood-tempering progress in `625/864` steps,
   102/108 runs per method experienced structural velocity clipping, and median
   repulsion/attraction was only about `0.008`. This points to weak SVGD geometry and
   initialization difficulty rather than an incorrect target or literal particle
   duplication.
8. **What should happen next?** Stop direct joint-energy SVGD development under this
   configuration. A future transport revisit would require a separately approved
   diagnostic contract addressing tempering, clipping, and weak repulsion without
   tuning on these six cases. The preferred new planning direction is the still
   underexplored modeling hypothesis: exact-GP prequential residual-energy correction
   against strong Gaussian calibration baselines. No subsequent task is active or
   implemented.

## Evidence and reproducibility

The reviewed full package and independent audit are in
[`results/task02c/full/`](../../results/task02c/full/). Raw traces and particle states
remain ignored under `artifacts/task02c/full/` and are identified by committed
SHA-256 checksums.

The run used source commit `9afa2a1`; the two later commits changed only the Colab
checkout/download handoff, not the scientific implementation. This remains a bounded
q=1 structural-transport diagnostic, not an end-to-end BO benchmark.
