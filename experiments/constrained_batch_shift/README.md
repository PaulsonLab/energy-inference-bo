# Constrained Batch Shift

## Question

Do large posterior-to-decision shifts occur naturally near high-acquisition batch decisions in a recognizable constrained BO model?

## Scientific role

The rare-mode experiment established the mathematical mechanism in a deliberately extreme mixture, but scrambled Sobol QMC became reliable by 1,024 samples. This experiment is the prospective falsification gate for whether the same statistical problem occurs near competitive decisions in a recognizable constrained, batch BO problem. It remains diagnostic: it compares posterior sampling against a high-budget reference and does not implement a decision-adapted proposal or sampler.

The machine-readable contract is frozen in [`config.json`](config.json). Any implementation must refuse a configuration with a different protocol hash when resuming results.

## Frozen problem and states

Use BoTorch 0.18.1's standard smooth constrained Hartmann6 construction on `[0,1]^6`:

- maximize negated Hartmann6, with no observation noise;
- constraint violation
  \[
  g(x)=\sum_{d=1}^6x_d^2-1\le0;
  \]
- exact feasible fraction \(\pi^3/384=0.08074551218828077\), independently checked with a fixed `2^20`-point scrambled Sobol set; under four independent uniform points the chance that a batch contains at least one feasible point is `0.28592630458803336`;
- batch size `q=4`.

Generate exactly eight states using seeds `3101` through `3108`. Never select, replace, or discard a state based on decision-shift or acquisition results. Each state starts from the fixed feasible anchor `(0.25,...,0.25)` plus 23 scrambled Sobol points, then runs eight ordinary constrained batch-BO rounds of four points, giving 56 observations. State generation uses independent standard Gaussian objective and constraint GPs and practical constrained `qLogExpectedImprovement` with 512 scrambled Sobol posterior samples. The frozen optimizer settings are 20 restarts, 1,024 raw Sobol starts, joint `q=4` L-BFGS-B, and 200 maximum iterations.

The best objective among observed feasible points is the incumbent. The feasible anchor guarantees that it is always defined. All objective and constraint observations, state-generation seeds, and model fits must be saved before the diagnostic beliefs are constructed.

## Matched beliefs

Fit the same two independent output processes—objective and constraint—to every frozen state. Train-only population centering/scaling is fixed separately for each output.

For each output, fit a constant mean and ARD Matérn-5/2 correlation kernel under one conjugate scale model

\[
y\mid\sigma^2\sim N(m,\sigma^2(K_\theta+10^{-6}I)),
\qquad
\sigma^2\sim\operatorname{InvGamma}(a_0=2,b_0=1).
\]

Use three deterministic lengthscale starts `(0.15, 0.30, 0.60)` in every dimension, log-lengthscale bounds `[log(0.05),log(2.0)]`, full-batch double-precision L-BFGS-B, and the largest finite marginal likelihood. No tail parameter is tuned.

The non-Gaussian belief is the exact conjugate Student-t-process predictive distribution. At `n=56`, its predictive degrees of freedom are fixed at `2(a0+n/2)=60`. The Gaussian control uses the identical predictive location and the exact Student-t predictive covariance. Thus the two beliefs have matching first and second moments, fitted geometry, data, and incumbent; only their higher-order joint tails differ. Objective and constraint scale variables are independent, while each scale variable is shared across the four batch outcomes for its output.

Before analysis, verify Gaussian/Student-t means and covariances agree to `1e-8` and record marginal tail diagnostics. A mismatch invalidates the comparison.

## Practical acquisition

The acquisition is BoTorch's positive smoothed constrained `qLogExpectedImprovement` utility, not a custom softplus proxy:

- `q=4`;
- objective `samples[..., 0]`;
- feasibility convention `samples[..., 1] < 0`;
- `fat=True`, `eta=1e-3`, `tau_max=1e-2`, and `tau_relu=1e-6`;
- latent, noiseless posterior outcomes;
- the best observed feasible raw objective as `best_f`.

The per-world positive utility is exactly the one whose log is aggregated by this pinned qLogEI implementation. Reference utility moments use the same smoothed max, improvement, and feasibility operations. For every candidate batch compute

\[
\alpha=E_P[U],\quad
\chi^2=E_P[U^2]/E_P[U]^2-1,\quad
D_2=\log(1+\chi^2),\quad
ESS/N=e^{-D_2}.
\]

Sampling uses transformed scrambled Sobol base uniforms. The Gaussian and Student-t beliefs share all normal base coordinates; the Student-t belief uses one additional scale coordinate per output. Common random numbers are used across candidate batches within every state, belief, sample count, and scramble. Stable log-domain evaluation is mandatory, but it does not replace the raw positive utility required for the moment identities.

## Candidate batches

Construct one common candidate panel per state from a prospective union, then evaluate every batch under both beliefs:

1. Run 32 separate high-budget optimizer restarts for each belief using 8,192 scrambled Sobol posterior samples; retain all 64 terminal batches, not just the winners.
2. For each belief, take its eight best terminal batches according to its high-budget optimizer estimate and add eight fixed-seed Gaussian perturbations per batch (`sd=0.025`, clipped to the domain), for 128 local batches.
3. Add 256 batches from one scrambled Sobol design in 24 dimensions.

This yields 448 batches before exact canonical deduplication. Sort the four points in every batch lexicographically, deduplicate only at tolerance `1e-10`, preserve source labels, and report the resulting counts. Candidate generation never uses decision-shift, ESS, or reference-error results.

For each belief and state, define reference acquisition quality as `alpha / max(alpha)` over this frozen panel. The primary high-acquisition set is the top decile by reference acquisition. It is valid only if it contains at least 32 distinct batches and has median quality at least `0.8`; otherwise the optimizer/candidate construction is invalid rather than negative scientific evidence. Also report the subset with quality at least `0.8` directly.

## High-budget reference and practical-QMC study

The A100 reference uses four independent scrambled Sobol replicates of `2^16=65,536` worlds each. Pool their first and second utility moments, and differentiate the log of the pooled acquisition estimate for the reference gradient. Process candidate batches in chunks of 16.

Numerical reference convergence is checked by comparing the first and second replicate pairs. At least 95% of high-acquisition batches must agree within 1% in value, have gradient cosine at least `0.99`, and have gradient-norm relative error at most 5%. A prospectively allowed numerical escalation doubles each replicate to `2^17`, then at most `2^18`; failure at the cap makes the state invalid. This rule changes only numerical precision, never candidate membership or scientific thresholds.

Compare practical sample counts `64, 128, 256, 512, 1024, 2048` using 32 independent scrambles. Use common random numbers across the full candidate panel. For outer-optimization reliability, run eight independent optimizations per sample count and belief from the same 20 restarts and 1,024 raw starts, and evaluate every terminal solution with the reference.

Record:

- acquisition value and log-value error;
- top-decile pairwise ranking disagreement, Kendall rank correlation, top-10% overlap, and selected-batch reference regret;
- reference and estimated gradient norms, relative errors, and cosine similarities;
- final optimizer batch, reference acquisition regret, reference rank, and distance to the reference-optimal set;
- `chi-square`, `D2`, population ESS fraction, and their relationship to acquisition quality;
- synchronized wall time, posterior worlds evaluated, peak CPU/GPU memory, and reference escalation.

IID MC is an optional theorem diagnostic only. It is not a primary baseline or gate input.

## Frozen decision rule

All mathematical, provenance, candidate-panel, and reference-convergence checks must pass first. Otherwise the result is `INVALID`.

A belief is **shift-positive** in a state when at least 20% of its valid high-acquisition set has population ESS fraction at most `0.05`. It has a **material 512-QMC failure** when at least two of these hold over the frozen repetitions:

1. median high-acquisition relative value error is at least `0.10`;
2. mean high-acquisition pairwise ranking disagreement is at least `0.10`;
3. median high-acquisition gradient cosine is at most `0.90`, or median relative gradient error is at least `0.25`;
4. at least 25% of optimizer runs have reference acquisition regret at least `0.05`.

The experiment is positive only if one matched belief has both properties in at least four of eight predetermined states, the pooled low-ESS fraction within its high-acquisition sets is at least 20%, and no single state contributes more than 40% of all pooled low-ESS high-acquisition batches.

- `GO_CONSTRAINT_BATCH` means the Gaussian control satisfies the rule: constrained/batch utility alone produces a practical integration problem.
- `GO_HEAVY_TAIL_AMPLIFIED` means the Student-t belief satisfies it and has at least twice the Gaussian pooled low-ESS fraction or at least twice its median 512-sample optimizer regret. Because the beliefs have matched moments and fixed `df=60`, this is evidence about higher-order tails rather than a tuned extreme posterior.
- `NO_GO_HIGH_VALUE_HEALTHY` means neither belief has the required low-ESS prevalence in high-acquisition regions.
- `NO_GO_QMC_RELIABLE` means shift is present but 512-sample scrambled Sobol QMC does not meet the material-error rule.
- `NO_GO_LOW_VALUE_ONLY` means substantial shift occurs mainly outside the high-acquisition sets.
- `NO_GO_NON_GAUSSIAN_ISOLATED` means only the Student-t belief shows isolated effects but fails the cross-state prevalence or twofold-amplification requirement.

The complete curves at every sample count remain primary evidence; the categories do not replace them. No positive result automatically authorizes a sampler. A negative result triggers reconsideration of the paper direction rather than stronger tails, tighter constraints, or post-hoc state selection.

## Compute and expected evidence

Implementation must first provide a CPU smoke using one truncated state-generation seed and tiny sample counts; smoke evidence cannot satisfy this gate. The full study is sharded one frozen state per resumable A100 job, with a target below three hours per shard. No implementation or run is part of this protocol-freeze commit.

Future full outputs must include the frozen config and hash, all eight state datasets, belief diagnostics, candidate provenance, reference-convergence records, per-scramble metrics, automatic gate result, environment/Git metadata, and plots of shift versus acquisition quality, practical-QMC error versus sample count, gradient reliability, optimizer regret, and Gaussian-versus-Student-t comparisons.

## Implementation and run handoff

Reusable conjugate-process, QMC, qLogEI-utility, gradient, and decision-shift code lives in `src/decision_tilt/`. The experiment CLI is [`run.py`](run.py). The implementation never changes the frozen JSON contract.

Local validation command:

```bash
uv run --no-sync pytest -q
uv run --no-sync python experiments/constrained_batch_shift/run.py smoke \
  --output-dir artifacts/constrained_batch_shift_smoke
```

The smoke uses seed 3101 with explicitly reduced wiring settings. It cannot produce a scientific gate. The full study must be run from the human-readable [A100 Colab notebook](../../notebooks/constrained_batch_shift_colab.ipynb), which checks out the stable tag `constrained-batch-shift-gpu-v2`, requires a matching GPU preflight, checkpoints every state, and packages exactly one `constrained_batch_shift_gpu_results.zip` plus `SHA256SUM.txt`. The superseded `v1` tag is retained only for provenance after its first external attempt exposed a summary-only aggregation bug before any state completed.

After download, put the ZIP in the repository root and request a separate frozen-results audit. Do not commit the raw ZIP or interpret incomplete states.

## Attempted A100 result

The `constrained-batch-shift-gpu-v2` campaign completed States 3101 and 3102,
then stopped prospectively at `INVALID_REFERENCE` during the Gaussian reference
for State 3103 after reaching the maximum $2^{18}$ samples per replicate.
The result is neither GO nor NO-GO. Do not rerun or relax the frozen reference
rule. See the [audited compact evidence](outputs/invalid_reference_v2/README.md).

Status: INVALID_REFERENCE — HUMAN REVIEW REQUIRED
