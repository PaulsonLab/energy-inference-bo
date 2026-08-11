# Task 02A — SAAS structural-posterior reuse diagnostic

## Scope

This is only the Task 02A falsification experiment: trusted BoTorch SAAS NUTS particles are held fixed while exact one-step predictive likelihoods update their weights and rank-one formulas update their GP caches. It does **not** implement rejuvenation, particle movement, selective-coordinate updates, Vecchia, an output EBM, q>1 acquisition, or a BO loop.

Profile: `full`; seeds `[0, 1, 2]`; D=10; n0=16; nfinal=40; retained particles=256; NUTS warmup/samples/thinning/tree depth=512/512/2/6. These values are from the planned three-seed full Colab configuration.

## Eight completion questions

1. **Reuse horizon.** ESS/P < 0.75: `1, 2, 2` new observations by seed; ESS/P < 0.50: `2, 2, 3` new observations by seed; ESS/P < 0.25: `4, 3, 4` new observations by seed; ESS/P < 0.10: `4, 5, 7` new observations by seed.
2. **Does -log(ESS/P) track fresh-posterior discrepancy?** The Pearson correlation with pooled-standardized RBF MMD is `0.8354983838250684`. Final mean `-log(ESS/P)=5.268` and MMD=`0.7878`.
3. **Structural marginals while ESS is high.** There was no noninitial fresh checkpoint with ESS/P >= 0.5; using the final checkpoint only as low-ESS context: mean log-lengthscale W1=`0.9044`, maximum dimensionwise W1=`1.362`, and pooled MMD=`0.3778`. Final active-top-2 probability is `1.000` reused versus `1.000` fresh.
4. **Predictive mixtures while ESS is high.** On the same checkpoint basis, mean predictive-mean RMSE=`0.08151`, total-variance RMSE=`0.05457`, and reused-minus-fresh held-out mixture log score=`-0.5611`. Total variance includes between-particle variance; log score also includes fixed observation noise.
5. **q=1 EI decision.** At the final checkpoint, mean Spearman=`0.9589`, mean top-5% overlap=`0.489`, mean maximum-EI relative error=`0.5593`, selected indices reused/fresh by seed=`[1805, 1695, 1725]/[1805, 1823, 317]`, decision agreement=`0.333`, and mean candidate distance=`0.8444`.
6. **Measured cost.** Median sequential likelihood/weight/cache update=`0.0031` s versus median fresh NUTS fit=`10.76` s, a `3.47e+03x` wall-time ratio. Exact increment error was at most `8.751e-11` and rank-one/full Cholesky error at most `1.638e-13`; validation-only full recomputations are separately counted in JSON.
7. **Coordinate stability.** Mean fresh-posterior log-lengthscale W1 drift was `0.2064` for active dimensions 0–1 and `0.8083` for irrelevant dimensions. On this evidence, irrelevant dimensions were `not more stable`.
8. **Stage recommendation.** **02B-B** provisionally, because fixed-support importance weights collapsed severely. For the smoke profile this recommendation is diagnostic only: do not implement Task 02B until the full Colab evidence is reviewed.

## Reproduction and files

- `artifacts/task02a_full/task02a_config.json` records the frozen affine transform, seeds, numerical settings, package versions, JAX backend/devices, and counters.
- The same directory contains round, checkpoint, lengthscale, coordinate-drift, and EI CSVs plus ESS, lengthscale, drift, EI, and timing PNGs.
- Unit tests include exact marginal-increment, rank-one cache, GPyTorch-kernel, stable weighting, weighted-EI, and frozen-preprocessing identities.
- `COLAB.md` gives the exact CPU full-run command and optional NVIDIA JAX setup.

No Task 02B code is included.
