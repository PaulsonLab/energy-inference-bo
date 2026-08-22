# Decision-Relevant Conditioning for Bayesian Optimization

This repository studies how much structured information—such as symmetry,
preferences, shape constraints, or physics residuals—must be incorporated
before the next Bayesian optimization action is determined to a prescribed
tolerance relative to fully conditioned BO.

## Start here

1. Read the [project handoff](project/PROJECT_HANDOFF.md) for the locked thesis,
   current evidence, closed failures, and next task.
2. Read the [paper story](project/PAPER_STORY.md) for the narrative and claim
   boundaries.
3. Then use the source of truth for the work at hand:
   [problem formulation](project/PROBLEM_FORMULATION.tex),
   [theory ledger](project/THEORY.tex),
   [experiment registry](project/EXPERIMENTS.md), or
   [related work](project/RELATED_WORK.md).

The supporting audits in [`project/reference/`](project/reference/) and closed
handoffs in [`project/archive/`](project/archive/) are provenance and technical
reference material, not default required reading.

## Current status

The BO-first formulation and the main abstract theory are proved. Concrete
structural-influence constructions are proved and regression-tested for the
reflection-symmetry and nonlinear-PDE families. A locked prospective
reflection-symmetry pilot passed for its exhaustive 401-action grid and exact
rejection backend; it does not imply a continuous-action or other-backend
guarantee. The synthetic preference pilots failed their sparsity gates, and
Gp2 is closed and abandoned as E3 after invalid preprocessing gates. See the
[experiment index](experiments/README.md) for the concise active/supplementary/
closed map.

## Repository layout

- `project/`: the six active scientific sources of truth
- `project/reference/`: authoritative completed theory/prototype audits
- `project/archive/`: closed implementation and gate handoffs
- `notebooks/prototypes/`: immutable historical prototype evidence
- `src/conditioned_bo/`: reusable implementations
- `experiments/`: stable experiment paths, protocols, and committed outputs
- `results/`: structured outputs from other clean experiment code
- `tests/`: lightweight and mathematical validation

## Evidence and compute policy

Historical notebooks remain provenance until clean code reproduces their
claims. Do not refactor them in place. Use a 16 GB MacBook Air and CPU for smoke
tests and inexpensive falsification; reserve a single Colab A100 for larger
runs, and do not launch large sweeps locally.
