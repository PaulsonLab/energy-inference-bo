# T2-B Reflection-Symmetry EI Validation

Status: **PASS**

This run tests only whether the proved EI structural bound is numerically
non-vacuous in the archived OU/symmetry regime. It is not a full E1 reproduction
and not a rigorous end-to-end action certificate.

## Prospective protocol

- incumbent: `0.50`;
- EI tolerance: `0.010` (fixed before full-target evaluation and distinct
  from the archived `0.03` log-acquisition tolerance);
- screen: `80000` reference samples, seed
  `123`, `401` actions, factor batches of
  `3`;
- non-vacuity criterion: certify while omitting at least
  `20%` of factors;
- validation: `8` fresh replicates with
  `20000` samples each, evaluated only after
  the active set and action were frozen.

## Result

- active factors: `15/40`; omitted before validation:
  `25/40` (`62.5%`);
- screened EI action: `-0.208200`;
- final sparse gap: `-0.00025484`;
- structural bound: `0.00533922`;
- optimistic envelope: `0.00508439`
  versus tolerance `0.01000`;
- refined-grid leader drift: `0.00065`;
- held-out full leaders: `-0.214700` to
  `-0.204300`;
- exact coarse-grid action matches:
  `50.0%`;
- maximum held-out action distance:
  `0.0065`;
- maximum held-out EI regret at the screened action:
  `0.00037839759`;
- held-out runs within EI tolerance:
  `100.0%`;
- full-target importance-sampling ESS:
  `66.1%` to
  `66.8%`.

No omitted factor was evaluated during screening. Full-factor evaluation used
fresh samples and occurred only after the screened action was frozen.

## Error separation

- The structural term is the proved analytic bound.
- Importance-sampling variability is an empirical diagnostic, not a rigorous
  finite-sample inference allowance.
- The 401-to-1601 grid comparison is a numerical diagnostic, not a continuous
  global-optimization certificate.

## Decision

The reflection-symmetry T2-B construction is **PASS**. The
nonlinear-PDE construction and rigorous inference/optimization allowances remain
unresolved.
