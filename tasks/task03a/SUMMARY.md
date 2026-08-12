# Task 03A summary

**Status:** implementation complete only after local validation; full scientific
evidence is pending the guarded Colab run. Smoke results must not be used to pass the
research gate.

## Local smoke evidence

The single approved CPU smoke completed all four seed-0 cases (G/W at n=16/32),
producing 32 method rows and four timing rows. The complete repository suite passed
58 tests at final validation; the single CUDA-only parity test was skipped because the development host
has no NVIDIA device. Five PIT inversions were clamped at the documented numerical
bounds. Reduced NUTS fits took 0.60–3.36 seconds; fully charged E0 fits took
0.20–0.45 seconds. Across only these four non-scientific cases, median normalized
regret was 0.799 for B1/E0, 0.751 for E1, and 0.891 for reduced NUTS. These unstable
one-seed values validate execution only and do not answer any gate below.

## Prespecified completion questions

1. Does four-fold held-out PIT construction remain leak-free and computationally
   cheaper than NUTS after charging all five ensemble fits?
2. Does E0/E1 stay near the reference under Gaussian truth at n=16/32?
3. Under warped truth, does one energy variant improve true NLL beyond B1, B2-G,
   B2-C, B3, and B4 at n=32 and n=64?
4. Does that same variant improve one-step true decision regret at n=64 in at least
   four of five paired seeds?
5. Does it stay within 0.02 median regret of NUTS at n=32/64 while fully charged time
   remains at most 50% of NUTS?
6. Is energy optimization at most 10% of charged MAP-reference construction time?
7. Does correction stay small under Gaussian truth and unlock with repeated warped
   misspecification evidence?
8. Do all normalization, convexity, oracle, EI, CPU/CUDA, timing, and provenance
   checks pass?

## Frozen gates

- **A safety:** Gaussian n=16/32 excess true NLL <=0.01 nat, median regret increase
  <=0.02, at most 2/10 cells degrade by >0.10, median correction KL <=0.02.
- **B flexibility:** one variant improves warped mean true NLL by >=0.01 nat over the
  strongest calibration baseline at both n=32/64; at n=64 it improves median regret
  by >=20% or 0.02 and wins at least 4/5 seeds.
- **C quality/cost:** regret is within 0.02 of NUTS, charged time <=50% of NUTS, and
  energy optimization <=10% of reference construction.
- **D unlocking:** Gaussian correction KL <=0.02; warped correction KL and NLL gain
  at n=64 exceed n=16 in at least 3/5 seeds.

GO to a separately contracted Task 03B only if A–D pass. If flexibility fails but raw
Ensemble MAP-SAAS is within 0.02 median regret of NUTS and its final fit is <=25% of
NUTS time, recommend a structural-only pivot. Otherwise recommend NO-GO.
