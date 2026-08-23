# Independent verification — Sun oxide GW BO value pilot

Terminal scientific verdict: `PASS_PBE_VALUE`.

Terminal Colab state: `PASS_PBE_VALUE_COLAB`.

This file records the post-run audit of the returned Colab archive. It was not
part of the prospectively frozen run artifact manifest.

## Provenance and archive integrity

- Returned ZIP SHA-256:
  `35608acb689a7425f2615c419d52e84f805f5fb79e1872df2926db3ff890723d`.
- Preregistration/run SHA:
  `44f58f100f41247afe0937e42eebe58055104225`.
- Frozen config SHA-256:
  `6cc47d41dfbdbf88187d535d405ca6afd971e4b07f91932d55dbbbf5c101ef0f`.
- The returned `frozen_config.json` is byte-identical to the committed config.
- All 27 manifest-declared files have the exact recorded size and SHA-256.
  The archive contains exactly those files plus `artifact_manifest.json`.
- Every frozen benchmark/model input hash matches both the config and the
  repository artifact at the run SHA.
- The isolated runtime was Python 3.12.13 with NumPy 2.3.5, Pandas 2.3.3,
  SciPy 1.18.0, and Matplotlib 3.11.1. The smoke test passed before oracle
  access and included the frozen dimensions, sparse solve, Schur-complement
  fixture, MAP/Hessian/Cholesky fixture, and Laplace samples.

## Independent metric recomputation

The audit recomputed the initialization, frozen scaling, observations, simple
regrets, per-seed AURCs, final regrets, top-10 times, optimum discoveries,
paired differences, medians, win fraction, and post-run Spearman diagnostic
from `trajectories.csv`, the frozen action mapping, and the oracle. All values
match `run_summary.json` and `seed_summaries.csv`.

| Metric | `NO_PBE` | `FULL_PBE` |
|---|---:|---:|
| Median AURC over 12 queries | 19.4445 eV | 1.0170 eV |
| Median final simple regret | 0.8310 eV | 0.0000 eV |
| Global-optimum discoveries | 4/12 | 10/12 |
| Paired AURC outcome | 0 wins, 2 ties | 10 wins, 2 ties |

The two non-wins for `FULL_PBE` were exact ties: in seeds 3 and 4, the shared
eight-action initialization already contained the global optimum. `FULL_PBE`
never had worse AURC. Its paired `FULL_PBE - NO_PBE` AURC differences were

```text
-13.975, -11.180, -52.275, 0.000, 0.000, -26.450,
-12.438, -50.580, -21.240, -52.520, -20.079, -1.463
```

The median simple-regret trajectories after queries 1 through 12 were:

```text
NO_PBE:   2.795, 2.795, 2.795, 2.795, 2.5065, 1.385,
          1.2835, 1.1915, 1.1915, 0.831, 0.831, 0.831
FULL_PBE: 0.596, 0.301, 0.1045, 0.1045, 0.1045, 0,
          0, 0, 0, 0, 0, 0
```

The preregistered AURC threshold was
`0.90 * 19.4445 = 17.50005` eV; the observed `FULL_PBE` value was 1.0170 eV.
The final-regret condition also passed (`0.0 <= 0.831`).

## Inference and oracle-isolation checks

All three frozen 4,096-sample Laplace-proposal SNIS validations passed:

| State | ESS / 4096 | Laplace action | IS-EI action | IS regret | Gap MC SE |
|---|---:|---:|---:|---:|---:|
| seed 0, initial | 0.905102 | 13 | 5 | 0.00289358 | 0.00137015 |
| seed 0, after 6 | 0.905340 | 134 | 179 | 0.00003532 | 0.00003532 |
| seed 0, after 12 | 0.906766 | 133 | 4 | 0.00008743 | 0.00008742 |

Each ESS fraction is far above 0.10, and every action-regret estimate is below
the frozen `max(0.02, 2 * MC_SE)` limit. An independent replay from the frozen
trajectory states, code, config, and RNG seeds selected the same Laplace and
IS actions and reproduced the reported numeric fields to maximum absolute
difference `3.6e-9`. These are empirical Monte Carlo diagnostics, not rigorous
posterior certificates; routine `FULL_PBE` uses a Laplace approximation, not
the exact conditioned posterior.

The oracle-access log has exactly 384 permitted per-action reads (96 shared
initial observations and 144 sequential observations per method), followed by
one post-run unlock and one full-oracle evaluation. The audit matched every
per-action access to the frozen trajectory order. Every routine inference
record selected the next recorded query before its oracle read. There was no
full-oracle evaluation before all 12 seeds and both methods completed, and no
unobserved GW value entered either acquisition.

All 288 routine decisions passed their frozen numerical requirements. Five
`FULL_PBE` decisions used the single permitted deterministic optimizer retry.
The maximum accepted gradient infinity norm was `9.984095848458718e-06`
(threshold `1e-5`), and the maximum Laplace solve residual was
`1.2997863077060485e-15` (threshold `1e-9`). Every reduced marginal used 500
RHS solves; no decision used the incorrect principal precision submatrix.

## Timing and scientific interpretation

Median routine decision time on the standard Colab CPU was about 0.240 seconds
for `NO_PBE` and 1.630 seconds for `FULL_PBE`. For `FULL_PBE`, the median major
components were 0.365 seconds for exact 500-RHS marginalization and 0.998
seconds for MAP optimization. The three IS validations took 32.265, 30.878,
and 32.207 seconds and were excluded from routine decision times. Timings are
diagnostics only; this run makes no adaptive-speedup claim.

The result is logically coherent with the allowed post-run diagnostic:
PBE-versus-GW Spearman rank correlation over the 191 actions is 0.8333, and
the GW-optimal action is also ranked first by PBE. Thus the frozen PBE ranking
contains unusually strong target-relevant ordering information, and full
conditioning exploits it quickly. This diagnostic explains the magnitude of
the gain but was not used to tune the model or protocol.

The supported conclusion is narrow but strong: on this frozen realistic E3
benchmark, conditioning on the complete normalized PBE ranking bank greatly
improves ordinary scalar-observation GW BO relative to the same Gaussian
reference with no PBE factors. This single benchmark does not establish
cross-dataset generalization, a rigorous Laplace certificate, or any adaptive
conditioning quality or speedup claim.
