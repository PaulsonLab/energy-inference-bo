# Task 02A — SAAS structural-posterior reuse diagnostic

## Scope

This is only the Task 02A falsification experiment: trusted BoTorch SAAS NUTS particles are held fixed while exact one-step predictive likelihoods update their weights and rank-one formulas update their GP caches. It does **not** implement rejuvenation, particle movement, selective-coordinate updates, Vecchia, an output EBM, q>1 acquisition, or a BO loop.

Profile: `smoke`; seeds `[0]`; D=4; n0=8; nfinal=11; retained particles=32; NUTS warmup/samples/thinning/tree depth=32/32/1/4. This is one deliberately reduced NUTS smoke run; its posterior is not scientifically interpretable. The D=10, three-seed Colab study is required before advancing a stage.

## Eight completion questions

1. **Reuse horizon.** ESS/P < 0.75: `2` new observations by seed; ESS/P < 0.50: `3` new observations by seed; ESS/P < 0.25: `>3` new observations by seed; ESS/P < 0.10: `>3` new observations by seed.
2. **Does -log(ESS/P) track fresh-posterior discrepancy?** The Pearson correlation with pooled-standardized RBF MMD is `not estimable with fewer than three informative fresh checkpoints`. Final `-log(ESS/P)=0.8559` and MMD=`0.2138`.
3. **Structural marginals while ESS is high.** There was no noninitial fresh checkpoint with ESS/P >= 0.5; using the final checkpoint only as low-ESS context: mean log-lengthscale W1=`0.2869`, maximum dimensionwise W1=`0.4332`, and pooled MMD=`0.2138`. Final active-top-2 probability is `0.275` reused versus `0.406` fresh.
4. **Predictive mixtures while ESS is high.** On the same low-ESS checkpoint basis, predictive-mean RMSE=`0.1104`, total-variance RMSE=`0.09337`, and reused-minus-fresh held-out mixture log score=`-0.09973`. Total variance includes between-particle variance; log score also includes fixed observation noise.
5. **q=1 EI decision.** At the final checkpoint, Spearman=`0.9702`, top-5% overlap=`0.846`, maximum-EI relative error=`0.3205`, selected indices reused/fresh=`[91]/[91]`, decision agreement=`1.000`, and candidate distance=`0`.
6. **Measured cost.** Median sequential likelihood/weight/cache update=`0.000271` s versus median fresh NUTS fit=`2.652` s, a `9.78e+03x` wall-time ratio. Exact increment error was at most `4.008e-11` and rank-one/full Cholesky error at most `3.264e-14`; validation-only full recomputations are separately counted in JSON.
7. **Coordinate stability.** Mean fresh-posterior log-lengthscale W1 drift was `0.4896` for active dimensions 0–1 and `0.1832` for irrelevant dimensions. On this evidence, irrelevant dimensions were `more stable`.
8. **Stage recommendation.** **02B-A** provisionally, because reuse is finite but needs adaptive annealed resample-move updates. For the smoke profile this recommendation is diagnostic only: do not implement Task 02B until the full Colab evidence is reviewed.

## Reproduction and files

- `artifacts/task02a/smoke/task02a_config.json` records the frozen affine transform, seeds, numerical settings, package versions, JAX backend/devices, and counters.
- The same directory contains round, checkpoint, lengthscale, coordinate-drift, and EI CSVs plus ESS, lengthscale, drift, EI, and timing PNGs.
- Unit tests include exact marginal-increment, rank-one cache, GPyTorch-kernel, stable weighting, weighted-EI, and frozen-preprocessing identities.
- [COLAB.md](COLAB.md) gives the exact CPU full-run command and optional NVIDIA JAX setup.

No Task 02B code is included.
