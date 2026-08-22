# Sun oxide source records and current-NLR benchmark

This directory contains two distinct records for the realistic E3 oxide case.
It contains no descriptors, graph, preference factors, theory calculation,
inference, performance baseline, or Bayesian optimization.

## Historical Sun reproduction

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

## `CURRENT_NLR_PBE_GW_V1`

`CURRENT_NLR_PBE_GW_V1` is “A current, reproducible NLR PBE→GW oxide
benchmark inspired by the legacy-data setting of Sun et al. (2020), not an
exact reproduction of their historical polymorph selection.” It freezes 2,142
canonical PBE/FERE legacy compositions and 191 strictly mapped GW actions.
Multi-polymorph GW compositions are resolved only by the finite total energy
per atom of each row's authoritative `wave` parent, with stable MatDB ID as the
exact-tie breaker. GW target magnitudes are isolated in a two-column oracle and
never enter selection or mapping.

The committed artifacts and complete NLR data-use notice are in
[`benchmark/`](benchmark/). Reproduce them from the repository root with:

```bash
python3 experiments/sun_oxide/current_nlr_benchmark.py fetch
python3 experiments/sun_oxide/current_nlr_benchmark.py build
python3 experiments/sun_oxide/gw_oracle.py write
```

The terminal benchmark verdict is `PASS_CURRENT_NLR_BENCHMARK`. This does not
alter the historical `JOIN_AMBIGUOUS` verdict and is not evidence for the paper
method.
