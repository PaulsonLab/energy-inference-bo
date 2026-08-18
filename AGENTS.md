# AGENTS.md

## Scientific workflow

- Read `PAPER_DESIGN.md` and `STATUS.md` first.
- Implement only the experiment explicitly authorized in `STATUS.md`.
- Never change the paper thesis during an implementation call.
- Never introduce a rollout policy.
- Never introduce a critic or value network unless a future human-approved plan changes the hypothesis.
- Preserve causality: the policy sees only observed history, never the latent sampled world.
- Use direct pathwise gradients where the model permits them.
- Compare plain direct policy optimization against causal path-KL transport.
- Treat negative results as valid.
- Keep notebooks human-readable.
- Do not create numbered task folders.
- Update `STATUS.md` after every scientific call.

## Compute

- Local target: MacBook Air with 16 GB RAM.
- The first `policy_kill` experiment must be CPU practical.
- Use a single Colab A100 only when later experiments genuinely need it.
