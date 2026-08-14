# Task 04A-D — frozen local decision-relevance diagnostic

## Question

Determine locally whether the standardized-child pairwise process is merely a better
conditional density or whether it learns context-dependent q=1 EI well enough to
improve controlled and natural decisions. This addendum does not revise the completed
Task 04A preflight, authorize its A100 run, or open Task 04B.

## Frozen experiment

- Withheld seeds `100–107`, D=6, neighborhood size 8, precision 10, and the existing
  coefficient-2 interaction truth.
- I at n=128/256; Gaussian safety G at n=256; 256 predictive points.
- Natural candidates are 4,096 active-coordinate Sobol points plus a 65-by-65 active
  grid, with inactive coordinates fixed at 0.5. An independent 4,096-point Sobol set
  is appended for all methods only when its oracle maximum exceeds the construction
  maximum by more than 0.5%.
- Real pairs are selected without oracle EI: candidates must have G0 EI at least 20%
  of its maximum, relative G0-EI mismatch at most 0.5%, and low/high context separation
  at least the interdecile range. Greedy selection is deterministic and disjoint, with
  at most 64 and at least 32 pairs per I case.
- A 32-state counterfactual panel holds each selected local mean/scale fixed while
  swapping actual summaries nearest the 10th/90th context-score quantiles.
- No coefficient, basis, shrinkage, neighborhood, warp, or threshold sweep is allowed.

## Frozen decision

Validity requires the mathematical suite, finite converged cases, teacher-free pair
construction, at least 32 real pairs per I case, and verified candidate maxima.

- Safety: G n=256 U/P median correction KL at most 0.02, median regret increase at
  most 0.02, and at most one degradation above 0.10.
- Density: I n=256 P improves mean KL over U by at least 0.01 nat and 20%, with at
  least 6/8 paired wins.
- Counterfactual mechanism: at least 75% of oracle pairs have normalized contrast at
  least 0.01; P has at least 80% sign accuracy and at most 30% median relative contrast
  error in at least 6/8 seeds.
- Real pairs: at I n=256, P has at least 70% accuracy, exceeds U by at least ten
  percentage points, and reduces margin-weighted pair regret by at least 25% in at
  least 6/8 seeds.
- Natural decisions: at least 3/8 n=256 cases give U regret at least 0.01 or place U
  outside the oracle 1%-optimal set. P wins at least two-thirds of eligible cases;
  its all-seed median regret gain is at least 0.01 absolute or 20%, with at most one
  degradation above 0.10.

Classify without reinterpretation as `LOCAL_GO`, `MECHANISM_ONLY`,
`LEARNING_NO_GO`, `ORACLE_NO_GO`, or `INVALID`. Even `LOCAL_GO` authorizes only a
separately planned confirmation study. The full Task 04A notebook remains disabled.

## Result

The complete 24-case run is classified **`INVALID`** because every I case had zero
qualifying real near-tie pairs, below the required 32. Only 6.64% of n=256
counterfactual oracle pairs reached the required 1% EI contrast, and U had no natural
decision opportunity in 0/8 medium-data cases. Density and safety passed: P reduced
mean I n=256 KL from 0.045686 to 0.022366 (51.0%) with 8/8 wins. This result must not
be reclassified by relaxing the frozen panel thresholds.
