# Experiment index

Experiment directory paths are stable provenance and are not reorganized when
their status changes. The current claims, prospective criteria, and execution
order live in [`project/EXPERIMENTS.md`](../project/EXPERIMENTS.md).

## Active main experiments

| ID | Directory | Status | Canonical committed result |
|---|---|---|---|
| E1 | [`symmetry/`](symmetry/) | T2-B validation and finite-grid pilot passed; repeated coverage/baselines are next | [finite-grid `RESULTS.md`](symmetry/outputs/inference_certification_pilot/RESULTS.md) |
| E2 | [`nonlinear_pde/`](nonlinear_pde/) | T2-B and family T4 passed; repeated scaling/baselines remain | [structural `RESULTS.md`](nonlinear_pde/outputs/t2b_structural_validation/RESULTS.md) |

## Planned main candidate

E3's next realistic candidate is the Sun et al. PBE→GW oxide legacy-data
problem. It has not been designed or implemented, so there is intentionally no
experiment directory yet. See the [E3 registry entry](../project/EXPERIMENTS.md#e3--realistic-non-pde-bo-candidate).

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
