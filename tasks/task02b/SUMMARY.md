# Task 02B — full decision-space compression result

## Scope and evidence

Task 02B is complete. The full Colab study used source commit `772f59f`, three seeds,
D=10, 18 fresh-NUTS checkpoints, 256 retained particles per checkpoint, and 2,048
fixed q=1 EI candidates. The runtime used Python 3.12.13 and GPU-backed JAX 0.9.2.
This is a structural-decision diagnostic, not an end-to-end BO benchmark or a Task 02C
implementation.

The imported package and independent integrity audit are in
[`results/task02b/full/`](../../results/task02b/full/). The complete 71 MB extraction is
retained locally under ignored `artifacts/task02b/full/`; raw signature matrices are
identified by committed SHA-256 checksums rather than added to Git.

## Eight completion questions

1. **How much more robust was decision regret than posterior fidelity?** Across the 15
   noninitial Task 02A checkpoints, median/mean/max normalized reuse regret was `0` /
   `0.1028` / `0.9827`; 11/15 were below 5%, while 4/15 exceeded 10%. Among the 10
   checkpoints with ESS/P below 0.1, 6 retained regret below 5%. Decision robustness is
   therefore real but not uniform.
2. **Did ESS predict regret?** Weakly at best. ESS/P versus regret had Pearson/Spearman
   correlations `-0.2480` / `-0.1898`; `-log(ESS/P)` had `0.2767` / `0.1898`. These
   are descriptive correlations over only 15 noninitial checkpoints.
3. **How compressible were acquisition signatures?** Acquisition entropy effective
   rank ranged `3.168`–`15.388` (median `7.363`, mean `8.206`), compared with
   standardized structural-coordinate rank `6.158`–`8.991` (median `7.649`, mean
   `7.607`). Acquisition ranks needed for 90%/95%/99% squared spectral energy had
   min/median/max `4/9.5/20`, `7/17/34`, and `15/41.5/83`, respectively. This is modest
   decision-space rank, but the 99% tail is not tiny.
4. **How many particles preserved the teacher decision?** Acquisition-space
   Frank–Wolfe achieved below 5% and 10% regret at every checkpoint with K=8, and below
   1% everywhere with K=16. At K=4 its mean/max regret was `0.02073` / `0.2059` with
   77.8% exact-index agreement; at K=8 it was `0.001494` / `0.02690` with 94.4%
   agreement; K=16 selected the teacher candidate at all 18 checkpoints.
5. **Did acquisition-space selection outperform baselines?** Yes for this oracle
   compression diagnostic. Mean normalized regret across tested budgets/checkpoints was
   `0.004445` for acquisition Frank–Wolfe, `0.01028` for posterior-space medoids, and
   `0.02586` for 32-repeat random equal thinning. All 3,060 finite-candidate regret
   bounds passed. Frank–Wolfe uses the teacher acquisition vector and is not yet an
   implementable posterior-inference method.
6. **Did M=1 recover fully Bayesian EI?** Yes at the representative seed-0, n=40
   checkpoint: maximum/L1 normalized-marginal errors were `2.776e-17` / `3.539e-16`,
   and both modes were candidate `1805`. Independently, all 18 archived teacher curves
   matched their particle EI means within `5.55e-16`.
7. **Did independent-replica M=2 recover squared EI?** Yes at the same checkpoint:
   maximum/L1 errors were exactly `0`, and both modes were candidate `1805`. The
   common-particle negative control differed from independent M=2 by L1 `0.03819`,
   confirming that `E[EI_theta(x)^2]` is not the independent-replica target
   `E[EI_theta(x)]^2`.
8. **Is Task 02C joint-energy transport justified?** **GO for a bounded Task 02C
   falsification experiment.** All 5 prespecified gates passed: 6/10 low-ESS
   checkpoints retained <5% regret; mean acquisition rank was below 32; K=8 preserved
   <5% regret everywhere; both joint identities passed below `1e-12`; and acquisition
   compression beat both baselines. This GO authorizes designing and testing transport;
   it is not evidence that transport will work or improve BO.

## Reproducibility caveats

- Fresh-NUTS teacher curves from this rerun differed from the saved Task 02A curves by
  median/max absolute error `0.01371` / `0.04298` and median/max RMSE `0.002345` /
  `0.01165`. This Monte Carlo/backend variation is retained, not hidden.
- The 18 NUTS fits totaled `233.22 s` on the recorded Colab GPU (`11.19 s` median,
  `10.20`–`21.76 s`). These timings are environment-specific.
- Local ARM reconstruction of the fixed Sobol candidates differed from the archived
  x86 Colab values by at most `2.98e-08`; archived candidates are identical across
  checkpoints within each seed and are the values used for every reported metric.
- No SVGD, MALA, annealed Langevin, SMC rejuvenation, Vecchia, residual-output EBM,
  q>1 BO, molecular optimization, or end-to-end BO loop was implemented.

Do not begin Task 02C until it has its own explicit specification and mathematical
contract.
