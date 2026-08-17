# Constrained Batch Shift

## Question

Do large posterior-to-decision shifts occur naturally near high-acquisition batch decisions in a recognizable constrained BO model?

## Prospective protocol

The blocked experiment uses constrained Hartmann6 with `d=6`, batch size `q=4`, at least one constraint, and frozen late-stage BO states. Its primary belief is a non-Gaussian Student-t-process-style posterior obtained by integrating out an uncertain GP scale. A large scrambled Sobol-QMC reference measures constrained qLogEI values, gradients, utility moments, chi-square decision shift, and ESS fraction across optimizer batches, restarts, random batches, and local perturbations.

The analysis maps acquisition value against decision shift and compares ordinary QMC sample counts on ranking, gradients, optimizer outcomes, and wall-clock. An A100 is used for the large vectorized reference and candidate study. This stage remains diagnostic: it does not implement a decision-adapted sampler.

Status: BLOCKED UNTIL RARE_MODE_MECHANISM HUMAN REVIEW
