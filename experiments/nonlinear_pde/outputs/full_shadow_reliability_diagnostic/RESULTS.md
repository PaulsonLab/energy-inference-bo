# E2 FULL-Shadow Reliability Diagnostic

Status: **DEVELOPMENT ONLY — NO PROSPECTIVE SEED EXECUTED**

This diagnostic used only source seed `2026082401` from unexecuted replacement
preregistration `0dcb76f5f2d053e098b472ac9984182b837295b5`, with frozen-config
SHA-256 `2717ece2e5581a7224e1a7a5cb5f69c8291c14ad0ea5183aff54154072b4748b`.
It did not modify or execute the scientific design. The frozen FULL-shadow rule
remained action agreement, both ESS fractions at least `0.20`, and uniform
maximum acquisition-vector difference at most `0.01`.

The n=40 8,192 and 16,384 seeds exactly reproduce the prior development
profile. The n=24 states use the same development-only trajectory identifier
`-2`; 32,768 uses the next independent seed pair. The optional 65,536 level
was not used because the cause was unambiguous after 32,768.

## Mechanical classification

A failure is labeled `OTHER_NUMERICAL_FAILURE` for nonconvergence/nonfinite
output, then `LOW_ESS` if either ESS-fraction rule fails. With adequate ESS,
action disagreement is `NEAR_TIE_ACTION_INSTABILITY` when both the maximum
reciprocal cross-batch regret and pooled top-two gap are at most the existing
`0.01` decision-quality scale; otherwise it is
`ACTION_INSTABILITY_WITH_MATERIAL_GAP`. If actions and ESS pass but only the
uniform vector check fails, it is `GLOBAL_VECTOR_DISAGREEMENT_ONLY`.
Simultaneous secondary failures are retained in the component columns and raw
record rather than hidden by this precedence.

## Complete decision and inference diagnostics

Top-five entries are `action:acquisition`. `max diff/top-5 diff` reports the
uniform maximum followed by the maximum restricted to the union of both
top-five sets. `cross regret A/B` means A's action evaluated under B followed
by B's action evaluated under A. Pooled results concatenate both batches'
samples and SNIS weights and are diagnostic only. Full precision and the
pooled top five are in `diagnostic.json`.

| n | checkpoint | samples/batch | actions A/B | top five A | top five B | rank gaps A/B | ESS frac. A/B (absolute) | max diff/top-5 diff | cross regret A/B | pooled action (gap) | classification |
|---:|---|---:|---|---|---|---|---|---|---|---|---|
| 24 | early | 8192 | 347/227 | 347:0.423705 227:0.423605 348:0.423500 228:0.413542 323:0.256475 | 227:0.421675 348:0.420563 228:0.417118 347:0.416906 252:0.249100 | 0.000100/0.001111 | 0.534/0.528 (4372/4324) | 0.009292/0.009292 | 0.004769/0.000100 | 227 (0.000607) | NEAR_TIE_ACTION_INSTABILITY |
| 24 | early | 16384 | 347/347 | 347:0.424629 228:0.421475 227:0.419703 348:0.415833 371:0.254743 | 347:0.425503 348:0.423150 227:0.420908 228:0.420511 371:0.260302 | 0.003154/0.002353 | 0.559/0.551 (9161/9023) | 0.007317/0.007317 | 0/0 | 347 (0.004072) | PASS |
| 24 | early | 32768 | 227/227 | 227:0.428293 347:0.420458 348:0.419188 228:0.415796 371:0.256281 | 227:0.423404 348:0.421965 347:0.421664 228:0.418472 371:0.254555 | 0.007835/0.001439 | 0.543/0.554 (17804/18157) | 0.004889/0.004889 | 0/0 | 227 (0.004791) | PASS |
| 24 | middle | 8192 | 203/324 | 203:0.054018 371:0.053870 252:0.053256 324:0.053148 323:0.052545 | 324:0.057381 203:0.057300 371:0.056261 323:0.053814 204:0.053574 | 0.000148/0.000081 | 0.533/0.551 (4364/4512) | 0.004323/0.004233 | 0.000081/0.000870 | 203 (0.000394) | NEAR_TIE_ACTION_INSTABILITY |
| 24 | middle | 16384 | 203/203 | 203:0.056466 371:0.054800 323:0.054373 252:0.053308 204:0.053085 | 203:0.057685 371:0.056570 323:0.055180 252:0.053911 204:0.052821 | 0.001666/0.001115 | 0.546/0.528 (8938/8648) | 0.002338/0.001770 | 0/0 | 203 (0.001389) | PASS |
| 24 | middle | 32768 | 371/371 | 371:0.057606 204:0.056929 203:0.056090 323:0.055111 324:0.053078 | 371:0.056026 203:0.055260 204:0.054654 323:0.054455 251:0.052959 | 0.000677/0.000766 | 0.510/0.526 (16718/17240) | 0.002353/0.002276 | 0/0 | 371 (0.001022) | PASS |
| 24 | late | 8192 | 372/371 | 372:0.050402 251:0.050402 324:0.050357 371:0.050091 226:0.025209 | 371:0.056650 324:0.056515 251:0.050520 372:0.050419 229:0.024411 | 0.000001/0.000135 | 0.527/0.531 (4319/4346) | 0.006559/0.006559 | 0.006231/0.000311 | 324 (0.000064) | NEAR_TIE_ACTION_INSTABILITY |
| 24 | late | 16384 | 371/371 | 371:0.055103 324:0.052114 251:0.051029 372:0.048903 229:0.025280 | 371:0.052978 324:0.051875 251:0.051282 372:0.046770 226:0.025873 | 0.002989/0.001103 | 0.521/0.529 (8539/8663) | 0.002322/0.002133 | 0/0 | 371 (0.002047) | PASS |
| 24 | late | 32768 | 371/371 | 371:0.053816 324:0.053195 372:0.052224 251:0.050520 226:0.025041 | 371:0.055275 324:0.051698 251:0.051114 372:0.048840 229:0.024893 | 0.000621/0.003577 | 0.532/0.518 (17421/16971) | 0.003385/0.003385 | 0/0 | 371 (0.002102) | PASS |
| 40 | early | 8192 | 900/900 | 900:0.429753 899:0.427713 699:0.422183 700:0.402363 939:0.259663 | 900:0.435504 699:0.433350 899:0.433321 700:0.406587 740:0.262692 | 0.002040/0.002153 | 0.216/0.170 (1771/1393) | 0.024832/0.011167 | 0/0 | 900 (0.002112) | LOW_ESS |
| 40 | early | 16384 | 699/900 | 699:0.423413 900:0.422842 899:0.421627 700:0.418375 939:0.253716 | 900:0.438896 899:0.428829 700:0.420271 699:0.414454 860:0.258154 | 0.000571/0.010066 | 0.199/0.198 (3265/3252) | 0.016054/0.016054 | 0.024442/0.000571 | 900 (0.005689) | LOW_ESS |
| 40 | early | 32768 | 900/900 | 900:0.430037 899:0.425917 699:0.418437 700:0.416549 860:0.259438 | 900:0.429696 899:0.428101 699:0.420518 700:0.407901 659:0.254979 | 0.004120/0.001594 | 0.200/0.211 (6559/6926) | 0.011589/0.008649 | 0/0 | 900 (0.002872) | GLOBAL_VECTOR_DISAGREEMENT_ONLY |
| 40 | middle | 8192 | 740/860 | 740:0.069081 939:0.066398 860:0.063562 859:0.060245 659:0.058365 | 860:0.058967 939:0.058725 940:0.058243 859:0.058084 659:0.056381 | 0.002683/0.000243 | 0.163/0.232 (1339/1904) | 0.013909/0.013909 | 0.003795/0.005520 | 939 (0.000463) | LOW_ESS |
| 40 | middle | 16384 | 659/939 | 659:0.064250 860:0.063469 939:0.062341 859:0.059763 660:0.059637 | 939:0.063216 659:0.061154 860:0.061014 740:0.059046 739:0.058585 | 0.000781/0.002062 | 0.232/0.195 (3807/3192) | 0.004585/0.003096 | 0.002062/0.001909 | 939 (0.000052) | LOW_ESS |
| 40 | middle | 32768 | 939/860 | 939:0.062576 659:0.061638 860:0.060534 740:0.058027 859:0.057968 | 860:0.063395 939:0.063260 859:0.061955 659:0.061365 740:0.060772 | 0.000938/0.000135 | 0.197/0.192 (6459/6280) | 0.003987/0.003987 | 0.000135/0.002041 | 939 (0.000948) | LOW_ESS |
| 40 | late | 8192 | 739/660 | 739:0.062401 740:0.058250 660:0.057747 939:0.054046 698:0.033857 | 660:0.063824 939:0.061856 739:0.059229 740:0.056059 901:0.033321 | 0.004151/0.001968 | 0.146/0.173 (1199/1421) | 0.007810/0.007810 | 0.004595/0.004654 | 739 (0.000028) | LOW_ESS |
| 40 | late | 16384 | 740/740 | 740:0.060860 739:0.058294 660:0.057970 939:0.055571 698:0.026646 | 740:0.060827 939:0.057185 660:0.056573 739:0.051499 698:0.028774 | 0.002566/0.003642 | 0.217/0.183 (3557/3006) | 0.006795/0.006795 | 0/0 | 740 (0.003572) | LOW_ESS |
| 40 | late | 32768 | 740/739 | 740:0.059764 660:0.059601 739:0.059075 939:0.056910 698:0.026422 | 739:0.059804 660:0.058016 740:0.058015 939:0.055744 698:0.026644 | 0.000163/0.001788 | 0.191/0.160 (6257/5230) | 0.003294/0.001749 | 0.001789/0.000689 | 739 (0.000551) | LOW_ESS |

## Laplace convergence and resources

Every mode converged in three accepted Newton steps. No nonfinite value or
optimization failure occurred. Gradient values are infinity norms. Batch wall
times are A/B; peak RSS is isolated per state/sample-count worker.

| n | checkpoint | samples/batch | converged A/B | iterations A/B | gradient A/B | batch wall A/B s | pair wall s | peak RSS GB |
|---:|---|---:|---|---|---|---|---:|---:|
| 24 | early | 8192 | yes/yes | 3/3 | 4.751e-14/4.751e-14 | 0.277/0.248 | 0.528 | 0.396 |
| 24 | early | 16384 | yes/yes | 3/3 | 4.751e-14/4.751e-14 | 0.527/0.500 | 1.034 | 0.637 |
| 24 | early | 32768 | yes/yes | 3/3 | 4.751e-14/4.751e-14 | 1.358/1.147 | 2.525 | 0.892 |
| 24 | middle | 8192 | yes/yes | 3/3 | 4.755e-14/4.755e-14 | 0.241/0.242 | 0.487 | 0.402 |
| 24 | middle | 16384 | yes/yes | 3/3 | 4.755e-14/4.755e-14 | 0.517/0.495 | 1.019 | 0.647 |
| 24 | middle | 32768 | yes/yes | 3/3 | 4.755e-14/4.755e-14 | 1.117/1.077 | 2.210 | 1.040 |
| 24 | late | 8192 | yes/yes | 3/3 | 4.756e-14/4.756e-14 | 0.243/0.242 | 0.488 | 0.400 |
| 24 | late | 16384 | yes/yes | 3/3 | 4.756e-14/4.756e-14 | 0.518/0.497 | 1.020 | 0.641 |
| 24 | late | 32768 | yes/yes | 3/3 | 4.756e-14/4.756e-14 | 1.086/1.079 | 2.182 | 1.016 |
| 40 | early | 8192 | yes/yes | 3/3 | 3.652e-14/3.652e-14 | 0.938/0.923 | 1.869 | 0.920 |
| 40 | early | 16384 | yes/yes | 3/3 | 3.652e-14/3.652e-14 | 1.946/1.786 | 3.754 | 1.401 |
| 40 | early | 32768 | yes/yes | 3/3 | 3.652e-14/3.652e-14 | 4.102/4.253 | 8.414 | 2.200 |
| 40 | middle | 8192 | yes/yes | 3/3 | 3.652e-14/3.652e-14 | 1.000/0.986 | 1.994 | 0.921 |
| 40 | middle | 16384 | yes/yes | 3/3 | 3.652e-14/3.652e-14 | 2.063/2.050 | 4.150 | 1.347 |
| 40 | middle | 32768 | yes/yes | 3/3 | 3.652e-14/3.652e-14 | 4.919/4.569 | 9.540 | 2.493 |
| 40 | late | 8192 | yes/yes | 3/3 | 3.652e-14/3.652e-14 | 0.970/1.007 | 1.983 | 0.926 |
| 40 | late | 16384 | yes/yes | 3/3 | 3.652e-14/3.652e-14 | 2.460/2.549 | 5.041 | 1.531 |
| 40 | late | 32768 | yes/yes | 3/3 | 3.652e-14/3.652e-14 | 6.254/5.669 | 11.983 | 2.181 |

## Diagnosis

- At n=24, ESS is healthy (`0.510`--`0.559`; absolute ESS already
  `4,319`--`4,512` at 8,192). All three 8,192 pairs fail only because nearly
  tied actions flip; reciprocal regret is at most `0.006231` and pooled gaps
  are at most `0.000607`. The frozen 16,384 escalation passes all three states,
  as does 32,768.
- At n=40, every 8,192 and every 16,384 state fails the ESS-fraction rule.
  Across all six batches per state, mean ESS fraction is `0.19925` early,
  `0.20196` middle, and `0.17850` late. Absolute ESS nevertheless grows from
  `1,199`--`1,904` at 8,192 to `5,230`--`6,926` at 32,768. Relative ESS is an
  asymptotic proposal/target overlap quantity; more samples increase absolute
  ESS but do not systematically raise its fraction. This identifies
  proposal-target mismatch at n=40, especially late, rather than insufficient
  sample count or a failed Laplace optimizer.
- Most n=40 middle/late action flips are decision-negligible near ties:
  reciprocal regrets are at most `0.005520`, and pooled gaps are below
  `0.001`. The exception is n=40 early at 16,384, where action 699 has regret
  `0.024442` under batch B. Both 8,192 batches and both 32,768 batches select
  900, and every pooled estimate selects 900, but this material independent-
  batch discrepancy prevents treating all failures as harmless near ties.
- The uniform vector rule has one clean false-rejection pattern: at n=40 early
  and 32,768, both actions are 900, both ESS fractions pass, cross-regret is
  zero, and top-five-union difference is `0.008649`, yet the global maximum
  difference is `0.011589`. The rule rejects due to lower-ranked actions. At
  the frozen 8,192/16,384 budgets, however, uniform-vector failures coexist
  with low ESS and, at 16,384, the material action discrepancy. Replacing only
  the uniform rule would therefore not repair the frozen reference check.

## Recommendation

**Recommendation: improve the FULL-reference numerical backend before any
prospective run.** Do not simply run the current freeze: this development
sample would label all three n=40 states unreliable after the frozen 16,384
escalation, making an `INCONCLUSIVE_FULL_REFERENCE` outcome structurally
likely. Do not replace the criterion yet: the data do show that exact action
agreement and a uniform all-action norm can reject decision-negligible near
ties, but low relative ESS is the dominant n=40 failure and one frozen-budget
pair has material `0.024442` cross-regret.

The numerical improvement should target proposal/target overlap, not merely
increase sample count: for example, a prospectively specified and independently
validated defensive/heavy-tailed mixture, multi-mode Laplace mixture, or
standard sequential backend. Any later decision-aligned reliability rule
should control an upper confidence bound on reciprocal action regret or the
top-action gap at the scientific decision-quality scale, with its uncertainty
calibrated on independent development simulations. Conceptually that matches
the shadow's role—validating the FULL decision rather than every irrelevant
action—but it must be frozen only after a backend with adequate overlap is
demonstrated. No replacement criterion is proposed from these six cases.
