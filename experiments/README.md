# Experiment index

Experiment directory paths are stable provenance and are not reorganized when
their status changes. The current claims, prospective criteria, and execution
order live in [`project/EXPERIMENTS.md`](../project/EXPERIMENTS.md).

## Active main experiments

| ID | Directory | Status | Canonical committed result |
|---|---|---|---|
| E1 | [`symmetry/`](symmetry/) | T2-B validation and finite-grid pilot passed; repeated coverage/baselines are next | [finite-grid `RESULTS.md`](symmetry/outputs/inference_certification_pilot/RESULTS.md) |
| E2 | [`nonlinear_pde/`](nonlinear_pde/) | T2-B and family T4 passed; `FULL_REFERENCE_BACKEND_UNRESOLVED`, so prospective scaling remains blocked | [structural result](nonlinear_pde/outputs/t2b_structural_validation/RESULTS.md), [backend rescue](nonlinear_pde/outputs/full_shadow_backend_rescue/RESULTS.md) |

## Realistic E3 candidate

The exact historical Sun et al. PBE→GW oxide reconstruction remains closed as
`JOIN_AMBIGUOUS`. The separately specified current-data benchmark
`CURRENT_NLR_PBE_GW_V1` is frozen under [`sun_oxide/`](sun_oxide/) with terminal
verdict `PASS_CURRENT_NLR_BENCHMARK`; its 132-dimensional descriptor and sparse
reference-graph compatibility gate passed as `PASS_DESCRIPTOR_GRAPH`, and its
frozen PBE-order existing-theory gate passed as `PASS_PBE_FACTOR_THEORY`.
The first frozen GW BO value pilot passed as `PASS_PBE_VALUE`: full normalized
PBE conditioning reduced median AURC from 19.4445 to 1.0170 eV and median final
regret from 0.8310 to 0.0000 eV, with all three prospectively frozen Laplace/IS
checks passing. Adaptive conditioning against `FULL_PBE` is next. See the
[E3 registry entry](../project/EXPERIMENTS.md#e3--realistic-non-pde-bo-candidate).

## Supplementary

| ID | Directory | Status |
|---|---|---|
| E4 | [`linear_pde/`](linear_pde/) | Historical prototype control; clean main-paper expansion is not planned |

## Closed negative experiments

| Directory | Exact terminal status | Canonical committed record |
|---|---|---|
| [`preference_bo/`](preference_bo/) | Both synthetic pilots `FAIL-P2`; value/performance gates passed, sparsity gates failed | [minimal](preference_bo/outputs/minimal_pilot/RESULTS.md), [redundant bank](preference_bo/outputs/redundant_bank_pilot/RESULTS.md) |
| [`gp2_preference_bo/`](gp2_preference_bo/) | `PREPROCESSING_INVALID`; no scientific P1 comparison ran | [`PREPROCESSING_INVALID.json`](gp2_preference_bo/outputs/preflight/PREPROCESSING_INVALID.json) |
| [`gp2_proxy_bo/`](gp2_proxy_bo/) | `PREPROCESSING_INVALID / GP2_ABANDONED`; no theory matrix, inference, or BO ran | [`RESULTS.md`](gp2_proxy_bo/outputs/structural_preflight/RESULTS.md) |

All committed configs, outputs, failed/intermediate attempts, and scientific
code are preserved in their original directories. Status text embedded inside
an immutable old output records the state at run time; use the current registry
for present status.
