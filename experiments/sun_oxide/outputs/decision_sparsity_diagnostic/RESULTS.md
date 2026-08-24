# Decision-sparsity diagnostic — DEVELOPMENT ONLY

Terminal development classification: `MIXED_DECISION_SPARSITY`.

This final diagnostic used only the frozen 500-support E3 model and
the three already-consumed seed-0 FULL_PBE states. It is not a fresh
scientific preregistration and does not treat the known value of PBE
preferences for PBE-to-GW prediction as a novel paper contribution.

- Starting GitHub main SHA: `5f49f140acc3532b0231b1d1c446d22cd0e168d8`
- Implementation SHA: `2ac5dd3576c548f0c9999cbea3bdf7d6f626656d`
- Config SHA-256: `281e9b173a234029563f8ed876b5d252befb6706849bf100cd61f281e65662e6`
- Peak process RSS: `0.222232576` GB
- Fresh seeds 12--31 accessed: `False`

`PASS_PBE_VALUE`, `ADAPTIVE_ENGINEERING_PATHOLOGICAL`, and
`FULL_ARCHIVE_NOT_HELPFUL` remain valid. The superseded Colab
preregistration remains unauthorized and unrun.

## FULL shadow reference

| State | Observations | Committed leader | Recomputed leader | Agreement | FULL fit s |
|---|---:|---:|---:|---:|---:|
| seed_0_initial | 8 | 13 | 13 | True | 0.264078 |
| seed_0_after_6_queries | 14 | 134 | 134 | True | 0.240358 |
| seed_0_after_12_queries | 20 | 133 | 133 | True | 0.234434 |

## Decision stabilization

| State | Path | First agreement | First stable | First <=0.01 regret | Certificate | Gap |
|---|---|---:|---:|---:|---:|---:|
| seed_0_initial | STATIC_INFLUENCE_PREFIX | 0.10 | 0.10 | 0.00 | 1.00 | 0.90 |
| seed_0_initial | RERANKED_FINE_PATH | 0.70 | 0.70 | 0.00 | 1.00 | 0.30 |
| seed_0_after_6_queries | STATIC_INFLUENCE_PREFIX | 0.20 | 0.20 | 0.00 | 1.00 | 0.80 |
| seed_0_after_6_queries | RERANKED_FINE_PATH | 0.20 | 0.20 | 0.00 | 1.00 | 0.80 |
| seed_0_after_12_queries | STATIC_INFLUENCE_PREFIX | 0.00 | 0.40 | 0.00 | 1.00 | 0.60 |
| seed_0_after_12_queries | RERANKED_FINE_PATH | 0.00 | 0.10 | 0.00 | 1.00 | 0.90 |

## Matched random subsets

| State | Fraction | Reranked agreement/regret | Random agreement | Random regret q25 / median / q75 |
|---|---:|---:|---:|---:|
| seed_0_initial | 0.10 | False / 0.00018875691 | 0.00 | 0.0015692659 / 0.0015692659 / 0.0015692659 |
| seed_0_initial | 0.20 | False / 0.00018875691 | 0.00 | 0.0015692659 / 0.0015692659 / 0.0015692659 |
| seed_0_initial | 0.40 | False / 0.00018875691 | 0.00 | 0.0011425637 / 0.0015692659 / 0.0015692659 |
| seed_0_after_6_queries | 0.10 | False / 1.2392895e-09 | 0.00 | 1.0686312e-06 / 1.0686312e-06 / 1.0686312e-06 |
| seed_0_after_6_queries | 0.20 | True / 0 | 0.00 | 7.7229601e-07 / 1.0686312e-06 / 1.0686312e-06 |
| seed_0_after_6_queries | 0.40 | True / 0 | 0.00 | 7.7229601e-07 / 7.7229601e-07 / 8.4637981e-07 |
| seed_0_after_12_queries | 0.10 | True / 0 | 1.00 | 0 / 0 / 0 |
| seed_0_after_12_queries | 0.20 | True / 0 | 1.00 | 0 / 0 / 0 |
| seed_0_after_12_queries | 0.40 | True / 0 | 1.00 | 0 / 0 / 0 |

## Frozen terminal interpretation

`RERANKED_FINE_PATH` is the primary diagnostic. STATIC and
RERANKED differences, random-subset comparisons, and the empirical
stabilization-versus-certificate gap are reported above without
relaxing `epsilon_struct=0.02` or changing any E3 model choice.

The result is mixed decision sparsity. STATIC and RERANKED differ
materially at the initial state (stable agreement 0.10 versus 0.70)
and after 12 queries (0.40 versus 0.10), while both stabilize at
0.20 after 6 queries. Influence ranking is useful but not uniformly:
at the middle state RERANKED agrees with FULL from 0.20 onward while
none of 20 matched random subsets agree through 0.40; initially it
reduces regret relative to random without exact agreement through
0.40; after 12 queries both ranked and random subsets agree.

The certificate is conservative in every state (certificate fraction
1.00), especially after observations, but empirical exact-action
sparsity is not uniformly strong because the initial RERANKED path
needs 0.70. The first <=0.01-regret fraction is 0.00 in every case
because FULL EI is already sufficiently flat at the empty model; this
does not imply exact action agreement. Under the frozen prospective
classification, these mixed results do not justify a major new
mathematical effort. Stop at these development numbers without
creating another experiment plan or spending fresh seeds 12--31.
