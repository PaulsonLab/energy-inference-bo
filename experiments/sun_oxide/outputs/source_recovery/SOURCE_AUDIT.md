# Sun oxide source audit

Terminal verdict: `JOIN_AMBIGUOUS`

This is a source-recovery verdict only. It is not evidence for the paper method.

## Reproduced source facts

- Current authoritative NREL/NLR finite-FERE query: 5604 raw rows, 2142 unique compositions.
- Current authoritative NREL/NLR finite-GW query: 244 raw rows, 194 unique compositions.
- Finite PBE/GW raw fields: 5604/244.
- Formula-level GW-to-PBE coverage: 193 of 194; missing composition count: 1.

## Blocking evidence

- The pinned PrefInt history uses `ElementProperty.from_preset(preset_name="magpie")` after formula-to-composition conversion and formula-only `drop_duplicates`.
- It records 244 raw GW rows and 194 rows after composition de-duplication.
- Its displayed raw head has counts {'O2 Sn': 2, 'O2 Ti': 3}; the current authoritative query has {'O2 Sn': 3, 'O2 Ti': 4} for the same formulas. The raw source has therefore drifted despite unchanged aggregate counts.
- The current GW source has 28 duplicate-formula groups and 50 extra polymorph rows. 27 groups span multiple final space groups and 28 span multiple stable MatDB family IDs.
- Stable identifiers do not rescue the historical selection because the committed author notebook discarded `id` before `drop_duplicates`, and the exact 2019/2020 CSV is absent from the repository and publisher supplement.

## Source-route findings

- The requested PrefInt tree at `b96a1eb0fc8667127d6fffffd92f8ee37b9641f5` contains molecular benchmark data, not the oxide CSV. The exact preprocessing evidence survives only in the recorded history commit and blob.
- The publisher supplement is a three-page molecular table (94 molecules), not an oxide data file.
- The NIMS/SAMURAI record was inspected, but exposes no downloadable MDR item; current MDR DOI/title searches did not recover the oxide assets.
- The cited Lany and Pilania DOI/source routes did not expose the exact historical Sun-study PBE/GW CSVs.

Implementation SHA: `46a08e09bce33d2189f3f3666035a72e08e608cb`.

No target ranking, correlation, histogram, model, descriptor regeneration, or BO operation was performed.
