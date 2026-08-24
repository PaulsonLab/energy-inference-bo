# Sun oxide source, benchmark, reference, and frozen legacy-factor records

This directory contains the source/benchmark, descriptor/reference graph,
frozen legacy-factor records, and first preregistered GW BO value result for
the realistic E3 oxide case. Target-blind construction records remain separate
from the oracle-evaluated BO outputs.

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

## Descriptor/reference-graph compatibility gate

[`colab_descriptor_graph.ipynb`](colab_descriptor_graph.ipynb) is the
installation-safe Colab handoff for the next narrow gate. It provisions a
pinned standalone Python 3.12 interpreter when needed, creates an isolated
environment from the fully pinned
[`requirements-colab-graph.txt`](requirements-colab-graph.txt), constructs the
frozen 132-dimensional Magpie composition descriptors, builds the deterministic
10-NN-plus-MST graph, checks the sparse graph-Gaussian reference precision, and
maps the 191 actions to latent nodes.

The returned Colab artifacts are committed under
[`outputs/descriptor_graph/`](outputs/descriptor_graph/) and passed independent
ZIP, descriptor, deterministic-edge, sparse-Q0, solve-residual, and 191-action
mapping checks. The terminal verdict is `PASS_DESCRIPTOR_GRAPH`; see the
independent [`VERIFICATION.md`](outputs/descriptor_graph/VERIFICATION.md).
The prospective config's `PENDING_COLAB` field is preserved as run provenance.
The committed result directory carries the full NLR data-use notice.
No GW value was read, no benchmark PBE value entered descriptor or graph
construction, and no preference-factor, influence, inference, or BO code ran.

## Frozen adjacent PBE-order baseline and existing-theory gate

[`pbe_factor_theory.py`](pbe_factor_theory.py) constructs
`ADJACENT_STRICT_PBE_ORDER_V1` from the committed PBE gaps in
`current_nlr_legacy.csv`. It sorts all 2,142 nodes by increasing exact-decimal
PBE gap and then stable composition key, adds a temperature-one logistic factor
only across a strict consecutive increase, and omits every exact adjacent tie.

This is not Sun et al.'s all-pairs likelihood. It is a sparse, transparent
ordinal legacy-information model: every included relation is a true strict
ordering in the frozen PBE data, and no PBE numerical value is compared with a
GW numerical value. The model-specific existing-Menz-theory check passed as
`PASS_PBE_FACTOR_THEORY`; see the immutable
[`RESULTS.md`](outputs/pbe_factor_theory/RESULTS.md). The 90/95/99%-influence
factor fractions are target-blind diagnostics, not sparsity gates. This run
performed no posterior sampling, Laplace inference, or BO.

## Normalized PBE full-conditioning model

The target-blind replacement-model gate passed as
`PASS_NORMALIZED_PBE_MODEL`. [`normalized_pbe_model.py`](normalized_pbe_model.py)
freezes `PBE_SUPPORT_500_V1` with all 191 actions plus 309 deterministic
descriptor-space farthest-point nodes, and freezes all 124,718 strict PBE pairs
as `NORMALIZED_ALL_PAIRS_PBE_500_V1` with global weight `1/499`; 32 exact-tie
pairs are omitted. This is a normalized composite/generalized-Bayes ranking
energy, not an independent all-pairs likelihood.

The weighted graph and existing-Menz checks pass, with analytic `A0` eigenvalue
floor `0.75` and numerical minimum `0.9003424446737381`. The PBE-only MAP and
256 stratified action-pair influence summaries are diagnostics without a
signal or sparsity threshold. No GW value, posterior inference, or BO entered
the gate. The adjacent model above remains an immutable valid sparse baseline;
the normalized model is the proposed E3 full-conditioning model. See the
model [specification](NORMALIZED_PBE_MODEL.md) and immutable
[`RESULTS.md`](outputs/normalized_pbe_model/RESULTS.md).

## GW BO value pilot

The first preregistered target-value experiment passed as `PASS_PBE_VALUE`
(`PASS_PBE_VALUE_COLAB` terminal state) from run SHA
`44f58f100f41247afe0937e42eebe58055104225` and frozen config SHA-256
`6cc47d41dfbdbf88187d535d405ca6afd971e4b07f91932d55dbbbf5c101ef0f`.
It compared only `NO_PBE` and `FULL_PBE` under identical fixed initial sets,
per-seed target scaling, Gaussian-reference hyperparameters, and 12-query
budgets. No adaptive conditioning was used.

Median AURC was 19.4445 eV for `NO_PBE` and 1.0170 eV for `FULL_PBE`; median
final regret was 0.8310 and 0.0000 eV. `FULL_PBE` won 10/12 paired seeds, tied
two seeds whose shared initialization already contained the global optimum,
and never lost. All three frozen Laplace-proposal importance validations passed
with ESS fractions near 0.905--0.907, and the independent access-log audit
confirmed that no unobserved GW target entered either acquisition.

The post-run PBE-vs-GW Spearman diagnostic is 0.8333, and the GW-optimal action
is also the highest-PBE action, so the large gain is scientifically coherent.
The result establishes value for full normalized PBE conditioning on this
frozen benchmark; it does not yet establish adaptive conditioning quality,
cost savings, or cross-dataset generalization. See the immutable
[`RESULTS.md`](outputs/bo_value_pilot/RESULTS.md), complete archived outputs,
and independent [`VERIFICATION.md`](outputs/bo_value_pilot/VERIFICATION.md).

## Adaptive E3 engineering smoke and fresh preregistration

The adaptive implementation uses one exact 500-dimensional Gaussian support
reference for all methods, the exact state-specific support block of the Menz
operator, cumulative active factors, L-BFGS-B MAP warm starts, exact active/full
Hessians, dense 500-dimensional Laplace Cholesky factors, and the frozen EI.

The engineering smoke ran only the already-consumed seeds 0--2 from
implementation SHA `7fbfb202268dd0fd92d35defbea2cc4990f089e2`. It passed all
mechanical gates and exactly reproduced the 36 prior FULL decisions. All 36
adaptive decisions certified or explicitly full-fallbacked, and all 36 shadow
FULL actions agreed. Each seed's first decision reached the explicit full-bank
fallback, so the active fraction was 1.0 thereafter. The median
ADAPTIVE/FULL conditioning-time ratio was about 0.999; the frozen pathological
stop did not trigger because its ratio condition requires a value above 1.25.
This smoke is not scientific evidence. See its immutable
[`RESULTS.md`](outputs/adaptive_e3_smoke/RESULTS.md).

The fresh 20-seed validation is preregistered in
[`configs/adaptive_e3_validation.json`](configs/adaptive_e3_validation.json),
with SHA-256
`aa327b3a0462c103a2dfbfed721bc30b7946acdb7b3c02032078001dc186b1a9`, and is
launched by [`colab_adaptive_e3_validation.ipynb`](colab_adaptive_e3_validation.ipynb)
on a standard CPU Colab runtime. It freezes seeds 12--31 and the methods
`NO_PBE`, `FULL_PBE_OPT`, and `ADAPTIVE_PBE`. No fresh-seed oracle execution
has occurred at preregistration time.
