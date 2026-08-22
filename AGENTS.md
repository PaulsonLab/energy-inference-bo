# AGENTS.md

## Required reading

Before scientific or code changes:

1. read `project/PROJECT_HANDOFF.md`;
2. read the active source-of-truth file relevant to the task:
   `project/PAPER_STORY.md`, `project/PROBLEM_FORMULATION.tex`,
   `project/THEORY.tex`, `project/EXPERIMENTS.md`, or
   `project/RELATED_WORK.md`;
3. for experiment work, also read `experiments/README.md` and the relevant
   experiment directory's `README.md`/committed result record.

Files under `project/reference/` and `project/archive/` are supporting audits
and closed handoffs. They are not default required reading; consult them only
when an active source links to them or the task concerns their provenance.

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

## Prototype and provenance rule

Files under `notebooks/prototypes/` and committed experiment output directories
are archival evidence. Do not silently edit them once added. Clean
implementations belong under `src/` and `experiments/`.
