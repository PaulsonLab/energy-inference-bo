# Rare-Mode Mechanism

## Question

Can a rare but high-utility predictive mode make ordinary posterior Monte Carlo misrank two BO decisions even when acquisition arithmetic is numerically stable?

## Frozen scientific outline

The experiment uses an exactly evaluable, non-Gaussian two-component Gaussian mixture on `x in [-2, 2]`. A broad common mode produces a modest improvement peak near one decision, while a low-probability mode produces a larger peak near another. Exact componentwise truncated-normal first and second moments provide EI, utility second moments, chi-square posterior-to-decision shift, asymptotic ESS, and predicted Monte Carlo relative variance. Utility-tilted component weights expose how a rare posterior mode can dominate decision-relevant worlds.

The next implementation call must freeze the exact numerical parameters before final results. It must produce the intended Figure 1 panels: posterior versus decision worlds, exact versus finite-sample acquisition landscapes, theorem verification across sample counts, and decision reliability. A strictly positive smoothed utility must confirm that the phenomenon is not solely caused by zero improvement utility.

## Compute and gate

This experiment must run on CPU. It is a GO only if the identities match simulation, the rare-mode decision is better by a nontrivial margin, low/moderate posterior Monte Carlo frequently misranks the decisions in line with the predicted shift/sample-size relationship, and the effect survives smoothing. A failure of those conditions is a valid NO-GO and does not authorize redesigning the synthetic case after inspection.

Status: NEXT AUTHORIZED EXPERIMENT
