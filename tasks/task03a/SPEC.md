# Task 03A contract — fast sparse reference + residual predictive energy

## Scientific question

Can BoTorch's four-component Ensemble MAP-SAAS model provide sparse structural
predictions cheaply enough that a small, reference-anchored scalar residual energy
corrects decision-relevant non-Gaussian misspecification while remaining safe under
Gaussian truth and substantially cheaper than fresh NUTS-SAAS?

## Required hierarchy

- B0: fresh fully Bayesian SAAS-NUTS.
- B1: raw four-component Ensemble MAP-SAAS.
- B2-G: global Gaussian PIT location/scale correction.
- B2-C: conditional Gaussian PIT location/scale correction using the same context as E1.
- B3: global skew-normal PIT correction.
- B4: strongly regularized two-Gaussian PIT mixture.
- E0: global nine-basis residual energy.
- E1: conditional nine-basis residual energy with intercept, log total reference scale,
  and ensemble-disagreement context.

Every reported correction is trained on genuine four-fold held-out reference
predictions. Analytic fixed-hyperparameter LOO is diagnostic only. The fully charged
cost includes four cross-fit ensembles, the final ensemble, and calibration fitting.

## Oracle and profiles

Use a known sparse Matérn-5/2 latent GP on coordinates 0–1 and paired identity/warped
outcomes, `expm1(0.6 g)/0.6`. Local smoke is one D=6 seed at n=16/32. Full Colab is
five D=20 seeds at n=16/32/64, 1,024 test points, 2,048 candidates, and the locked
512/512/thinning-2/tree-depth-6 NUTS reference.

The exact quantitative gates A–D are recorded in [SUMMARY.md](SUMMARY.md) and may not
be revised after seeing the full data. Task 03B is forbidden unless all gates pass.

The full runner also writes two non-gating sensitivity tables: eight MAP-SAAS
components for warped n=32, all seeds; and shrinkage precisions 3 and 30 for seed 0,
warped n=64, compared with the primary precision-10 rows. Neither can change the gate.

Out of scope: sequential BO, q>1, Vecchia, neural EBMs, molecular examples, new
particle samplers, and structural transport.
