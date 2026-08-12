# Task 02C — full decision-tilted SVGD result

This is a q=1, D=10 embedded-Branin falsification study. It is not an end-to-end BO
benchmark and contains no Vecchia, residual-output EBM, q>1, molecular optimization,
or post-Task-02C method.

## Eight completion questions

1. **Did the teacher preflight pass?** Yes. The largest envelope autodiff and finite-difference errors were `2.046e-12` and `6.337e-07`.
2. **Was the exact SAAS energy reproduced?** Yes. The largest centered-value and unconstrained-gradient errors were `4.403e-11` and `6.314e-11`.
3. **Was the decision tilt intrinsically degenerate?** `0` beta-one teacher checks had ESS/P below 0.1; the observed range was `0.612`–`0.986`.
4. **Was expensive compute matched?** Yes. Every posterior/decision pair used identical initial particles, K, initialization steps, structural steps, cache builds, and design attempts; recorded factorization-equivalent counts match pairwise.
5. **Did DT-SVGD improve decision regret?** It had lower regret in `18.5%` of paired runs. Median normalized regret was `0.4139` for P-SVGD and `0.533` for DT-SVGD.
6. **Were K=8–16 particles sufficient?** `0.0%` of DT-SVGD runs with K<=16 achieved below 5% normalized regret. K=8 remains a stress test.
7. **What is the compute and geometry interpretation?** Median charged DT-SVGD time was `33.1` s versus `23.4` s for fresh NUTS. Final-particle collapse occurred in `0.0%` of runs, forced beta progress in `0.0%` of DT structural steps, and the bandwidth floor in `0.0%` of all structural steps. Factorization-equivalents are recorded in `task02c_methods.csv`; shared initialization is charged equally.
8. **Should the structural decision-energy program continue?** **NO-GO.** decision tilting did not consistently outperform posterior-focused SVGD at matched compute.

Task 02B Frank–Wolfe K=8/K=16 results remain an unattainable oracle compression
ceiling and were never used for initialization.
