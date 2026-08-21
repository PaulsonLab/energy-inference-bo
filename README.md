# Decision-Relevant Conditioning for Bayesian Optimization

This repository studies how much structured information—such as symmetry, preferences, shape constraints, or physics residuals—must be incorporated before the next Bayesian optimization action is determined to a prescribed tolerance relative to fully conditioned BO.

## Status

The project has a BO-first formulation, proved and regression-tested structural-influence constructions for the main symmetry and nonlinear-PDE factor families, and historical prototype evidence. The remaining technical blocker is a rigorous finite-sample inference-error construction for an end-to-end certificate; the broader experiment suite has not yet been cleanly reproduced.

## Project documents

- [Paper story](project/PAPER_STORY.md)
- [Problem formulation](project/PROBLEM_FORMULATION.tex)
- [Theory ledger](project/THEORY.tex)
- [Experiment registry](project/EXPERIMENTS.md)
- [Related work and novelty boundaries](project/RELATED_WORK.md)
- [Project handoff](project/PROJECT_HANDOFF.md)

## Repository layout

- `project/`: scientific sources of truth
- `notebooks/prototypes/`: immutable prototype evidence
- `src/conditioned_bo/`: reusable clean implementations
- `experiments/`: experiment-specific drivers and protocols
- `results/`: structured outputs from clean experiment code
- `tests/`: lightweight and mathematical validation

## Evidence provenance

The initial symmetry and PDE results are historical prototype evidence until they are reproduced from clean code. The original notebooks are preserved under `notebooks/prototypes/` and should not be refactored in place.

## Compute policy

Use a 16 GB MacBook Air and CPU for smoke tests and inexpensive falsification. Use one Colab A100 only for larger experiments; do not launch large sweeps locally.
