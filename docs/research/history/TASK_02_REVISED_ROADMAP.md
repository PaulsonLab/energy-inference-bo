# Revised roadmap after Task 02A

This roadmap motivated Task 02B and is retained as history. The active specification
is [`../../../tasks/task02b/SPEC.md`](../../../tasks/task02b/SPEC.md).

## Task 02B — decision-relevant structural compression

- quantify decision regret from Task 02A;
- measure acquisition-signature effective rank;
- test small weighted structural coresets;
- verify the exact joint structural-decision energy marginal with NUTS as teacher.

Only if this diagnostic passes should Task 02C test direct joint structural-decision
energy transport. Candidate future methods include SVGD, annealed MALA/Langevin, and
resample-move SMC, but none is selected or implemented by this historical roadmap.

Later work may consider general expected utility, residual predictive energy,
Vecchia/local likelihoods, structured domains, and coherent batch BO. Those remain
out of scope for Task 02B.
