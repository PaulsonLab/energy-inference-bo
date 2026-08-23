# Normalized PBE replacement-model gate

Terminal verdict: `PASS_NORMALIZED_PBE_MODEL`.

This target-blind E3 gate ran no Bayesian optimization, posterior inference,
or GW-target analysis.

## Frozen model

- Starting SHA: `810d3c29ac3579a1159ad7eba301df4069db8d66`
- Support / actions / added PBE-only nodes: `500` / `191` / `309`
- Strict factors / omitted exact-tie pairs: `124718` / `32`
- Exact-tie groups and size histogram: `23` / `{'2': 20, '3': 2, '4': 1}`
- Initial / final descriptor covering radius: `34.92084178302644` / `5.073016093657024`
- Maximum weighted row sum / `||W_pbe||_2`: `0.9999999999999997` / `0.9997442203602465`

The bank is a normalized composite/generalized-Bayes ranking energy, not an
independent all-pairs likelihood. Support selection uses descriptors and action
membership only; exact PBE ties create no factor.

## Existing Menz compatibility

- Definition: `A0 = Q0 - 0.25 W_pbe`
- Analytic / numerical `lambda_min(A0)`: `0.75` / `0.9003424446737381`
- Symmetric / nonpositive off-diagonals / SPD: `True` / `True` / `True`
- Factorization / 191-RHS solve time: `0.0326173750218004` / `0.0386099589522928` seconds
- Solve relative residual: `1.6157178193330642e-15`

The weighted logistic Hessian is PSD and has mixed-curvature magnitude at most
`omega/4`. Later nonnegative diagonal observation precision retains the
committed M-matrix/resolvent monotonicity, so `C=A0^-1` is conservative. No
dense inverse was formed and no theory-ledger change is proposed.

## PBE-only MAP signal

Normalized dense support/action Spearman: `0.9897442696242639` / `0.9881802838247451`.
Adjacent baseline support/action Spearman: `0.06724929032854206` / `0.09758197850647561`.

The complete MAP standard deviations, ranges, strict-pair accuracies, and
top-minus-bottom-decile contrasts are recorded in `pbe_signal_summary.json`.

## Target-blind influence diagnostic

The deterministic diagnostic has 256 action pairs, 64 from each
descriptor-graph-distance rank quartile. Fractions of strict factors needed to
reach structural influence totals are:

- 50 percent: 25th / median / 75th / 90th percentile `0.005877259096521753` / `0.006161901249218236` / `0.006578841867252522` / `0.0069556920412450485`.
- 75 percent: 25th / median / 75th / 90th percentile `0.02059646562645328` / `0.029831299411472282` / `0.03914030051796854` / `0.04561490723071249`.
- 90 percent: 25th / median / 75th / 90th percentile `0.3925656280568964` / `0.39678715181449353` / `0.40096056703924055` / `0.4039513141647557`.
- 95 percent: 25th / median / 75th / 90th percentile `0.6731426097275454` / `0.674662037556728` / `0.6763919402171299` / `0.6773801696627593`.

Total structural influence is summarized in `influence_summary.json`. These
quantities have no sparsity pass/fail threshold.

## Isolation

GW oracle read: `False`. GW target statistics computed: `False`. The input
interface is an exact allowlist of hashes and contains no target table.
