# Paper Experiments

Each directory corresponds to a prospective paper figure or result. Reusable implementation belongs in `src/decision_tilt/`; experiment directories contain frozen protocols and experiment-specific entry points only after authorization.

- [`rare_mode_mechanism/`](rare_mode_mechanism/): completed CPU mechanism experiment; candidate Figure 1 and frozen evidence await human review.
- [`constrained_batch_shift/`](constrained_batch_shift/): permanently invalid attempted A100 diagnostic; retained for auditability.
- [`welded_beam_shift/`](welded_beam_shift/): completed valid q=1 Gaussian diagnostic; decision shift was observed, but practical QMC decision failure was not.
