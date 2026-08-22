# Sun et al. oxide source-recovery gate

This directory contains only the first source-recovery and benchmark-reconstruction gate for the realistic E3 oxide case. It contains no factor design, inference, influence analysis, modeling, performance baseline, or Bayesian optimization.

The audit queries the authoritative NREL/NLR Materials Database for finite PBE/FERE oxide rows and finite GW oxide rows, inspects the pinned PrefInt history, and checks source counts, stable identifiers, duplicates, polymorphs, and GW-to-PBE coverage. Raw downloads remain under the gitignored `data/external/sun_oxide/source_recovery/` directory.

Run from the repository root:

```bash
python experiments/sun_oxide/source_audit.py fetch \
  --config experiments/sun_oxide/configs/source_recovery.json \
  --cache-dir data/external/sun_oxide/source_recovery

python experiments/sun_oxide/source_audit.py audit \
  --config experiments/sun_oxide/configs/source_recovery.json \
  --cache-dir data/external/sun_oxide/source_recovery \
  --output-dir experiments/sun_oxide/outputs/source_recovery \
  --implementation-sha "$(git rev-parse HEAD)"
```

`fetch` verifies every cached file against `cache_manifest.json` on later runs. `audit` refuses to overwrite a nonempty terminal output directory. Use a new output path for an independent repeat.

The committed terminal result is `JOIN_AMBIGUOUS`: exact aggregate composition counts reproduce, but the historical raw GW selection has drifted, formula-only de-duplication spans multiple stable material families/polymorphs, and the exact author CSV is unavailable. No normalized benchmark or descriptor matrix is emitted.
