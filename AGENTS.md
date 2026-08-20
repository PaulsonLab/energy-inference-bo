# AGENTS.md

## Required reading

Before scientific or code changes, read:

- `project/PROJECT_HANDOFF.md`;
- the source-of-truth file relevant to the task.

## Scientific discipline

- Keep the framing Bayesian-optimization-first.
- Do not reopen unrelated research directions unless an explicit blocker invalidates the current thesis.
- Distinguish `PROVED`, `PROOF TASK`, `BLOCKER`, `EXISTING EVIDENCE`, and `PLANNED` claims.
- Do not claim sampler novelty for standard IS, SMC, HMC, ESS, or diffusion methods.
- Do not call empirical Monte Carlo diagnostics rigorous certificates.
- Every new experiment must map to a claim in `project/EXPERIMENTS.md`.
- Every experiment must have a prospective failure criterion.
- Update `project/EXPERIMENTS.md` when empirical status changes.
- Do not silently change mathematical behavior that conflicts with `project/PROBLEM_FORMULATION.tex` or `project/THEORY.tex`.

## Coding discipline

- Use explicit random seeds.
- Put reusable code in `src/`.
- Keep notebooks as thin drivers or immutable prototypes.
- Test mathematical identities before performance runs.
- Report discrepancies rather than loosening tests.
- Keep dependencies minimal.
- Never commit secrets.

## Compute discipline

- Use the MacBook Air and CPU for smoke tests.
- Use a single Colab A100 for larger experiments.
- Do not launch large sweeps locally.
- Full runs must print progress and save structured results.

## Prototype rule

Files under `notebooks/prototypes/` are archival evidence. Do not silently edit them once added. Clean implementations belong under `src/` and `experiments/`.
