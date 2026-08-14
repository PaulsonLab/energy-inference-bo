# Task 04A contract — local conditional energy process with oracle geometry

## Question and scope

Test whether a small pairwise conditional energy adds useful nonlinear local
dependence beyond a Gaussian/Vecchia conditional and a unary correction when the
correct active geometry, kernel, ordering, and neighborhoods are supplied.

This is an oracle capability and computational-feasibility study restricted to q=1
one-step decisions. It does not learn metrics or kernels, run sequential BO, compare
SAAS/NUTS, add neural energies, use structured inputs, or implement q>1.

## Frozen construction

- Scrambled Sobol points are maximin ordered in active coordinates 0–1, starting at
  the point closest to `(0.5,0.5)`. Ties use original Sobol index. Prefixes are nested.
- Each point uses its eight nearest predecessors; candidates use their eight nearest
  observations. The known Matérn-5/2 kernel has lengthscales `(0.2,0.2)`, outputscale
  `1`, and nugget `1e-8`.
- Output coordinates are fixed before observing data: center `0`, scale `1`.
- Seven centered RBFs have centers `linspace(-3,3,7)` and bandwidth `0.8`. Child
  features are evaluated at the fixed local Gaussian coordinate
  `z=(y-mu_i)/sigma_i`; neighbor summaries remain fixed functions of earlier raw
  observations. This revision prevents the positive control from vanishing solely
  as local conditional variance contracts.
- G0 has no correction; U has seven unary parameters; P adds a full directed `7×7`
  interaction matrix. U/P use summed exact conditional NLL, L2 precision `10`, zero
  initialization, 64-point Gauss–Hermite normalization, and full-batch double L-BFGS.
- Truths are G (the exact local Gaussian), W (`expm1(0.6g)/0.6` applied to a latent
  local Gaussian), and I (the frozen rank-one interaction in [MATH.md](MATH.md)).

## Execution profiles and outputs

- Smoke: seed 0, D=6, n=64, all regimes, 128 tests, 256 candidates, CPU, 75 L-BFGS
  iterations. It is validation only.
- Preflight: seeds 0–2, D=6, G/I, n=64/128/256, 256 tests, 512 candidates, CPU. It
  must pass the recorded interaction-signal, learnability, safety, convergence, and
  decision-relevance checks before the full profile is authorized.
- Full: seeds 0–4, D=10, n=64/128/256/512, all regimes, 1,024 tests, 2,048
  candidates, CUDA/A100, 250 iterations. It contains 60 cases and 120 learned fits.
- The runner saves configuration/runtime JSON, resumable case JSON, metric/timing/
  scaling/parameter CSVs, gate JSON, summary Markdown, and reviewed diagnostic plots.

The preflight authorizes the full profile only when all of these frozen checks pass:

- G safety over n=64/128: U/P median correction KL at most `0.02`, median regret
  increase at most `0.02`, and at most one degradation above `0.10`.
- I density signal and learning at both n=128/256: mean G0 KL at least `0.01`, P
  improves mean KL over U by at least `0.005` and 20%, and P wins all three seeds.
- I decision relevance over the six n=128/256 cells: U regret is at least `0.01` in
  at least two cells, P wins at least two cells, and P median regret does not exceed U.
- Every fit is finite and converged, with no global n-by-n factorization.

These checks are an authorization screen, not a replacement for the five-seed full
gates. They must not be weakened after observing preflight output.

## Frozen decision

- Gate A: strict nesting/EI identities pass; U/P Gaussian n=64/128 median correction
  KL ≤0.02, median regret increase ≤0.02, and at most 2/10 degradations above 0.10.
- Gate B: at I n=128/256, P beats U in mean KL by ≥0.01 and ≥20%; at n=256 it
  improves median regret by ≥0.02 or ≥20% with ≥4/5 paired wins. At W n=256, use the
  same KL/regret thresholds with ≥3/5 paired regret wins.
- Gate C: by I n=256, P mean KL ≤0.05 and median regret ≤0.10, with its prescribed
  advantage already present at n=128.
- Gate D: finite converged cases, CPU/CUDA parity, peak task allocation <4 GB, no
  global n×n allocation, and ≤2.5× synchronized fit-time growth per doubling after n=128.

All gates must pass to authorize a separately specified Task 04B. Failure of the I
pairwise criterion is a strong NO-GO. Mathematical identity failure invalidates the
experiment rather than producing a scientific conclusion.
