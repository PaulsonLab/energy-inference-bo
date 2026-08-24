# Decision-reset adaptive E3 engineering smoke

## Status

`ADAPTIVE_ENGINEERING_PATHOLOGICAL`

This engineering run used only the already-consumed seeds 0--2. It is not
scientific evidence, did not run SNIS, and did not access fresh seeds 12--31.

- Implementation SHA: `1fab6fb6afd626381a87df55d2d6b348e7475584`
- Config SHA-256: `01cabcda9c4d50ae6b6b498467fa570ea5167b7c89da4feb6a8ec890f6c33c4c`
- Superseded, never-executed preregistration commit:
  `70a9686b143c09f9f970306cc4489a2ce2b6e173`
- Optimized FULL decisions matching the committed FULL_PBE pilot: `36/36`
- Adaptive decisions certified or explicitly full-fallbacked: `36/36`
- Explicit full-bank fallbacks: `36/36`
- Shadow-FULL action agreement: `36/36`
- Maximum shadow-FULL Laplace-EI regret: `0.0`
- Oracle log: 96 queried-action reads, seeds exactly `{0,1,2}`; no shadow or
  unobserved-action oracle reads.

The only algorithmic lifecycle change from the superseded smoke was to start
each BO decision from an empty factor mask. Activation remained cumulative
within each decision. The preceding BO iteration's adaptive MAP was used to
warm-start the first active-target fit in all 33 decisions after decision 1;
no factor was forced active from a preceding BO iteration.

## Per-decision smoke results

Each row reports medians across seeds 0--2. Fallback and agreement are counts
out of three. Work columns are factor-element counts, shown as
`FULL_PBE_OPT / ADAPTIVE_PBE`. Shadow EI regret is standardized.

| BO decision | Final active factors (fraction) | Stages | Full fallback | Shadow agreement; max regret | Conditioning seconds FULL / ADAPTIVE | Energy-gradient work FULL / ADAPTIVE | Hessian work FULL / ADAPTIVE |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 124,718 (1.000) | 8 | 3/3 | 3/3; 0 | 0.220657 / 0.762077 | 11,723,492 / 36,888,001 | 124,718 / 1,051,118 |
| 2 | 124,718 (1.000) | 8 | 3/3 | 3/3; 0 | 0.232090 / 0.849942 | 11,099,902 / 36,891,040 | 124,718 / 1,054,460 |
| 3 | 124,718 (1.000) | 8 | 3/3 | 3/3; 0 | 0.173244 / 0.654374 | 9,478,568 / 31,840,960 | 124,718 / 1,053,464 |
| 4 | 124,718 (1.000) | 8 | 3/3 | 3/3; 0 | 0.177901 / 0.797105 | 9,603,286 / 40,953,521 | 124,718 / 1,051,655 |
| 5 | 124,718 (1.000) | 8 | 3/3 | 3/3; 0 | 0.119139 / 0.641687 | 6,485,336 / 31,547,981 | 124,718 / 1,047,876 |
| 6 | 124,718 (1.000) | 8 | 3/3 | 3/3; 0 | 0.128922 / 0.629392 | 6,734,772 / 31,084,996 | 124,718 / 1,048,957 |
| 7 | 124,718 (1.000) | 8 | 3/3 | 3/3; 0 | 0.187236 / 0.673188 | 9,229,132 / 32,679,130 | 124,718 / 1,050,368 |
| 8 | 124,718 (1.000) | 8 | 3/3 | 3/3; 0 | 0.190134 / 0.715796 | 10,351,594 / 36,689,364 | 124,718 / 1,049,390 |
| 9 | 124,718 (1.000) | 8 | 3/3 | 3/3; 0 | 0.173949 / 0.703715 | 9,353,850 / 35,461,972 | 124,718 / 1,046,772 |
| 10 | 124,718 (1.000) | 8 | 3/3 | 3/3; 0 | 0.192548 / 0.724017 | 9,977,440 / 31,838,427 | 124,718 / 1,045,083 |
| 11 | 124,718 (1.000) | 8 | 3/3 | 3/3; 0 | 0.150255 / 0.729083 | 7,857,234 / 36,537,040 | 124,718 / 1,042,578 |
| 12 | 124,718 (1.000) | 8 | 3/3 | 3/3; 0 | 0.175402 / 0.707145 | 8,979,696 / 35,488,170 | 124,718 / 1,039,047 |

## Decision 1 versus decisions 2--12

Decision 1 already required a median first batch of 115,373 factors
(`0.925071` of the bank). After eight stages it retained a median 116,195
factors (`0.931662`) before full fallback; the median remaining structural
envelope was `0.118337`, well above `epsilon_struct=0.02`.

Across the 33 decisions 2--12, re-screening did respond slightly to the
state: the median first batch was 115,079 factors (`0.922714`), and the median
eight-stage pre-fallback set was 115,765 factors (`0.928214`). This is not a
sparse regime. The median remaining stage-8 structural envelope was
`0.092544` (range `0.069804`--`0.111578`), so every decision exhausted the
eight stages and used the exact full-bank fallback.

For decisions 2--12:

- final active factors: median 124,718 (`1.0`), range 124,718--124,718;
- certification stages: median 8, with 33/33 full-bank fallbacks;
- shadow-FULL agreement: 33/33, maximum EI regret `0.0`;
- median conditioning seconds: FULL `0.175477`, ADAPTIVE `0.707145`;
- median paired decision-time ratio: `3.99777`;
- median energy-gradient element work: FULL `9,229,132`, ADAPTIVE
  `33,582,182`;
- median Hessian element work: FULL `124,718`, ADAPTIVE `1,047,989`.

Across complete trajectories, the per-seed ADAPTIVE/FULL cumulative
conditioning-time ratios were `4.23255`, `3.57476`, and `4.13067`. For
decisions 2--12 alone they were `4.24034`, `3.60823`, and `4.22847`.
ADAPTIVE energy-gradient work was `3.35`--`4.03` times FULL over complete
trajectories, and Hessian work was about `8.40` times FULL.

## Stop decision

The reset lifecycle preserves the FULL decision while making the computational
path strictly worse in this smoke. It provides no credible factor-work or
conditioning-time reduction, including after observations arrive. Therefore
the fresh 20-seed validation remains unspent, no replacement preregistration
is created, and the adaptive E3 efficiency claim is blocked under the frozen
algorithm and scientific parameters.

One-time reference costs were `0.0817368` seconds for the shared exact 500-D
Gaussian reference and `0.155669` seconds for `C_H0`. They do not alter the
routine-decision blocker above. Absolute seconds are machine-specific; the
factor-work counts independently show the same failure.
