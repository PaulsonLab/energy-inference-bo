# Task 02C contract — decision-tilted structural transport

## Scientific question

Can teacher-free decision-tilted SVGD preserve the fresh fully Bayesian q=1 EI
decision better than posterior-focused SVGD with the same small particle and exact-GP
factorization budget?

This is a falsification experiment. Task 02B oracle coresets are unattainable ceilings,
not initialization data or deployable baselines.

## Required preflight

- Reuse the six saved Task 02B early/late teacher particle sets without rerunning NUTS.
- At MAP, continuous-teacher, and midpoint designs, measure tilt ESS, entropy-effective
  fraction, maximum weight, 50/90/99% mass counts, conditional ESS, and structural
  shifts for beta 0, 0.25, 0.5, 0.75, and 1.
- Check the beta-envelope gradient by JAX autodiff, weighted particle gradients,
  finite differences, and an independent PyTorch calculation.
- Expose the exact NumPyro unconstrained potential and validate the fused exact-GP
  target. Transport is forbidden unless all preflight thresholds pass.

## Transport

- Coordinates: NumPyro's D+3 unconstrained SAAS variables.
- Whitening: fixed median and IQR/1.349 from 4,096 teacher-free prior draws, with a
  0.25 scale floor.
- Initialization: deterministic maximin K of 4K valid `init_to_uniform` states, then
  16 shared likelihood-tempered SVGD steps. Reach likelihood power one by step 12 and
  use four stabilization steps.
- Kernel: whitened RBF with median-squared-distance divided by `log(K+1)` and a
  `1e-3` bandwidth floor.
- Optimizer: Adam, learning rate 0.01, standard 0.9/0.999 moments, `1e-8` epsilon,
  particle-direction norm cap 10, and at most three nonfinite backtracks.
- P-SVGD targets the posterior and optimizes the log of mean EI.
- DT-SVGD reaches beta one during the first half of its structural budget using a
  CESS/P target of 0.8 and spends the second half at beta one.
- Structural blocks contain four SVGD steps. Before each block both methods receive
  four projected x attempts, coordinate cap 0.02. DT moves also require retilting
  CESS/P at least 0.9.

## Full comparison

- Cases: D=10, seeds 0–2, n=16 and n=40.
- K: 8, 16, and 32; K=8 is a stress test.
- Post-initialization structural budgets: 32 and 64 steps per particle.
- Repeats: three paired repeats.
- Baselines: MAP-SAAS, matched P-SVGD, fresh 256-particle NUTS teacher, and previously
  published Task 02B oracle K=8/K=16 ceilings where comparable.
- Continuous teacher: 2,048 common Sobol candidates, 16 bounded multistarts plus MAP,
  and an independent 4,096-point Sobol check. Strengthen the reference if any method
  exceeds it beyond numerical tolerance.

Record normalized/absolute teacher regret, selected x, teacher EI, distance, wall time,
structural and design counts, factorization-equivalents, beta/CESS events, particle
distances, kernel bandwidth, score/attraction/repulsion norms, clipping, backtracking,
and fresh-vs-archived teacher discrepancies.

## Execution boundary

Local execution is limited to the test suite, the saved-teacher preflight, and one
D=10 seed-0 n=16 K=8 wiring smoke with four initialization and eight structural steps.
The full study runs sequentially in guarded GPU Colab cells. No local or notebook run
may advance to another task automatically.

Out of scope: MALA, Langevin, SMC, Vecchia, residual-output EBMs, q>1, molecular
optimization, and an end-to-end BO loop.
