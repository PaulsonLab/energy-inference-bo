# Sun oxide adaptive E3 engineering smoke

Terminal state: `ENGINEERING_SMOKE_PASS`.

This was an engineering run on the already-consumed seeds 0--2. It is not
scientific evidence and did not run SNIS validation.

- Implementation SHA: `7fbfb202268dd0fd92d35defbea2cc4990f089e2`
- Config SHA-256: `aa327b3a0462c103a2dfbfed721bc30b7946acdb7b3c02032078001dc186b1a9`
- FULL/ADAPTIVE decisions: `36` / `36`
- Optimized FULL decisions matching the committed FULL_PBE pilot: `36/36`
- Adaptive decisions structurally certified or explicitly full-fallback: `36/36`
- Explicit full-bank fallbacks: `3` (the first decision of each seed)
- Shadow-FULL action agreement at adaptive data states: `36/36`
- Maximum shadow-FULL Laplace-EI regret: `0.0`
- Maximum omitted-contribution sum discrepancy: `4.440892098500626e-16`

## Factor use and timing

- Median final adaptive active factor fraction across decisions: `1.0`.
- Adaptive active factor count range: `124718`--`124718`.
- Median PBE-conditioning time, FULL_PBE_OPT / ADAPTIVE_PBE:
  `0.17748841631691903` / `0.1772242084844038` seconds.
- Median ADAPTIVE/FULL conditioning-time ratio: `0.9985114080231384`.
- Median total decision time, FULL_PBE_OPT / ADAPTIVE_PBE:
  `0.18139845901168883` / `0.18039799958933145` seconds.
- Median factor energy-gradient element work, FULL_PBE_OPT / ADAPTIVE_PBE:
  `9416209` / `9291491`.
- Median factor Hessian element work, FULL_PBE_OPT / ADAPTIVE_PBE:
  `124718` / `124718`.

The first decision of each seed activated roughly 115,000 factors in the first
batch, remained uncertified after eight activation stages, and explicitly
fell back to the full bank. Because factors are cumulative within a seed, all
later decisions used the full bank and certified with zero structural load.
This smoke therefore supplies no adaptive-speedup evidence.

The frozen engineering-pathology stop requires both a median final active
fraction strictly above `0.95` and a median ADAPTIVE conditioning time strictly
above `1.25 * FULL_PBE_OPT`. Only the first condition held, so the stop did not
trigger and preregistration of the fresh validation is permitted.

## One-time precomputes

- Shared exact 500-D reference: `0.08688191720284522` seconds; amortized over
  the 72 routine FULL/ADAPTIVE smoke decisions, `0.0012066932944839614`
  seconds per decision.
- Adaptive `C_H0` precompute: `0.155905291903764` seconds; amortized over the
  36 adaptive smoke decisions, `0.004330702552882334` seconds per decision.

Absolute seconds are machine-specific. Complete per-decision timings, factor
work, trajectories, adaptive stages, shadow FULL records, checkpoints, and the
oracle-access log are stored beside this file.
