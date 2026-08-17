# AGENTS.md

## Scientific contract

- Read `PAPER_PLAN.md` and `STATUS.md` first.
- Do not change the paper thesis during an implementation call.
- Every code change must support a named paper claim, figure, table, or mathematical test. Otherwise, stop and report that it is out of scope.

## Research discipline

- Never weaken a GO/NO-GO criterion after seeing results. Negative results are valid.
- Do not invent a new EBM sampler unless explicitly authorized.
- Do not replace outer L-BFGS/Adam acquisition optimization with sampling over `X`.
- Treat qLogEI with scrambled Sobol-QMC as a strong baseline, not a straw man.
- Preserve arbitrary non-Gaussian posterior support in mathematics and interfaces.
- Prefer the simplest correct implementation.

## Compute

- Local target: a 16 GB MacBook Air. Tests and `rare_mode_mechanism` must be CPU-compatible.
- GPU work uses one Colab A100. Design each shard for at most roughly 2–3 hours and make it checkpointed or cleanly restartable.

## Organization

- Reusable math/model code: `src/decision_tilt/`.
- Prospective paper experiments: `experiments/<descriptive_name>/`.
- Human-readable analysis and Colab workflows: `notebooks/`.
- Mathematical identities: unit tests.
- Do not create numbered research-task directories.

## Notebooks

Every substantive notebook must contain: question, paper relevance, mathematics, protocol, GO/NO-GO, results, interpretation, and next action.

## End of call

Update `STATUS.md` on every Codex call. Do not automatically begin the next experiment unless the human explicitly requested it.
