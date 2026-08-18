# Policy Kill Experiment

## Model system

Use a one-dimensional **two-regime Bayesian optimization problem** with latent regime

\[
M \in \{0,1\}
\]

and a non-degenerate prior. Each regime defines a smooth objective response \(f_M(x)\).

The functions should have:

- two competitive high-value exploit regions;
- modest differences between regimes at the exploit locations;
- a separate low-immediate-value diagnostic region where the two regimes predict very different observations.

Thus myopic BO should prefer exploitation, while a sufficiently nonmyopic policy can prefer a diagnostic measurement and then adapt its next action.

Observations are Gaussian:

\[
Y_t = f_M(X_t) + \epsilon_t.
\]

The posterior over \(M\) is updated exactly after every observation. This gives a genuinely non-Gaussian predictive mixture while remaining simple enough for accurate dynamic programming.

## Reference

For horizons \(H=1,2,3\) initially, construct an accurate numerical dynamic-programming reference using:

- a dense one-dimensional action grid;
- Gaussian quadrature for observation noise;
- interpolation over the low-dimensional sufficient belief state.

Derive the sufficient policy state carefully—expected to include time, the exact regime belief, and the reward-relevant incumbent—rather than storing an arbitrary full history.

## Candidate methods

Only:

1. accurate DP reference;
2. plain direct pathwise causal-policy optimization;
3. causal path-KL transport.

No rollout heuristic.

No critic.

No PPO.

## Main kill questions

- Does the exact optimal first action change with horizon?
- Does the learned causal policy recover the reference first action/value?
- Does path-KL transport materially improve optimization stability or sample efficiency over plain pathwise Adam?

Status: NEXT AUTHORIZED EXPERIMENT
