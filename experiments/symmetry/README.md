# E1 — Nonlocal Reflection Symmetry

Status: finite-grid prospective inference certification passed; broader clean
reproduction and baselines remain. See the
[E1 specification](../../project/EXPERIMENTS.md#e1--nonlocal-reflection-symmetry).

## T2-B EI validation

The focused runner [`run_ei_validation.py`](run_ei_validation.py) validates the
newly proved EI-specific structural bound in the archived OU parameter regime.
It is deliberately narrower than the full E1 experiment. The archived notebook
uses exponential utility and a log-acquisition certificate; this clean runner
uses ordinary EI and a raw acquisition gap.

Run from the repository root:

```bash
uv run python experiments/symmetry/run_ei_validation.py
```

The frozen configuration, per-round screen, fresh held-out replicates, summary,
and interpretation are in
[`outputs/t2b_ei_validation/`](outputs/t2b_ei_validation/). Full-factor
evaluation occurs only after the screened action is frozen. Its Monte Carlo and
action-grid checks are empirical diagnostics, not a rigorous end-to-end
certificate.

T2-B reflection-symmetry verdict: **PASS**. The nonlinear-PDE construction was
subsequently proved; see
[`project/T2B_NONLINEAR_PDE_AUDIT.md`](../../project/T2B_NONLINEAR_PDE_AUDIT.md).
The exact-rejection/strong-log-concavity finite-sample inference construction
and its locked prospective finite-grid pilot passed. The frozen configuration is
[`configs/inference_certification_pilot.json`](configs/inference_certification_pilot.json)
and complete results are in
[`outputs/inference_certification_pilot/`](outputs/inference_certification_pilot/).
The exhaustive 401-action grid has zero optimization remainder; no
continuous-action certificate is claimed, and the guarantee does not transfer
automatically to HMC, SMC, FlowGP, or another inference backend. Status fields
inside the frozen 2026-08-20 output are historical provenance, not the current
project ledger.
