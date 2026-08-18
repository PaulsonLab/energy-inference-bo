# Causal Policy Transport for Nonmyopic Bayesian Optimization
## Mathematical Design Note and JES-Like Paper Blueprint

**Status:** paper-design hypothesis, not yet an approved algorithm  
**Central slogan:** **Optimize the policy, not the tree. Transport actions, not outcomes.**

---

# 0. Executive conclusion after adversarial audit

There is a mathematically coherent energy/path-space formulation of full nonmyopic Bayesian optimization (BO), but several initially attractive claims are already covered by prior work:

- **planning as inference over policies** is already explicit in *MDP Planning as Policy Inference* (2026);
- **pathwise posterior sampling + neural policies + long horizons** is already central to LookaHES (2026);
- **differentiating nonmyopic BO trajectories** is already developed in *Differentiating Policies for Non-Myopic Bayesian Optimization* (2024);
- **trust-region path-space measure transport** is already a general stochastic-control methodology (Blessing et al., 2025).

Therefore the paper cannot be “BO is an EBM over trajectories” or “use a neural policy instead of a tree.”

The remaining plausible paper is narrower:

> **Full finite-horizon BO can be posed as optimization over causal path measures. Naively reward-tilting Bayesian fantasy trajectories is invalid because it also tilts future observations. Projecting the reward-tilted path law back onto the family with fixed Bayesian observation dynamics yields an exact KL-proximal policy-improvement problem. When the Bayesian world model admits coherent differentiable root samples, this proximal objective can be optimized directly by backpropagating through whole adaptive campaigns—without an explicit scenario tree, fixed rollout policy, or learned critic.**

The intended practical advantage is **not an asymptotic theorem that planning becomes easy**. It is a regime-dependent computational claim:

1. scenario-tree representation grows exponentially with horizon;
2. a shared policy has horizon-independent parameter count and requires \(O(MH)\) world evaluations per gradient step;
3. unlike EARL-BO, the proposed continuous-domain solver does not require actor–critic value learning;
4. unlike GP-specific scenario-tree fantasization, it can consume a coherent non-Gaussian world sampler without repeatedly solving posterior-conditioning problems at fantasy nodes.

The idea should be killed if these distinctions do not produce a large empirical advantage.

---

# 1. Full causal BO problem

Let the current real dataset be

\[
D_0=\{(x_i,y_i)\}_{i=1}^{n}.
\]

Let \(F\) denote the uncertain latent objective (and, if necessary, other latent quantities such as constraint functions or model parameters). Assume a calibrated posterior

\[
P_0(dF)=P(dF\mid D_0).
\]

The observation model is

\[
L(dy\mid F,x).
\]

For \(t=0,\ldots,H-1\), define the history

\[
H_t=(D_0,x_0,y_0,\ldots,x_{t-1},y_{t-1}),
\]

with associated information filtration \(\mathcal F_t=\sigma(H_t)\).

A **causal policy** is a sequence of stochastic kernels

\[
\pi=\{\pi_t\}_{t=0}^{H-1},
\qquad
X_t\sim\pi_t(dx\mid H_t).
\]

The nonanticipativity requirement is simply

\[
X_t\perp (F,Y_{t:H-1})\mid H_t
\]

under the operational policy: the policy may use observed history, but may not see the latent world or future observations.

## 1.1 Reward

For the cleanest initial theory, consider noise-free maximization and let

\[
b_0=\max_{(x_i,y_i)\in D_0} y_i.
\]

Define terminal improvement

\[
R_H(F,\tau)
=
\left[
\max_{0\le t<H}F(X_t)-b_0
\right]_+.
\]

Equivalently, under noise-free observations, incremental improvement rewards telescope:

\[
r_t=(Y_t-b_t)_+,
\qquad
b_{t+1}=\max(b_t,Y_t),
\]

so

\[
\sum_{t=0}^{H-1}r_t=b_H-b_0=R_H.
\]

The full finite-horizon objective is

\[
\boxed{
J_H(\pi;D_0)
=
\mathbb E^\pi[R_H].
}
\]

The Bayes-optimal policy is

\[
\boxed{
\pi_H^\star
=
\arg\max_{\pi\in\Pi_{\rm causal}}J_H(\pi;D_0).
}
\]

For noisy observations, replace \(R_H\) by the desired latent terminal decision utility; the causal/path-measure derivations below are unchanged.

---

# 2. Standard BO is the one-step special case

For \(H=1\),

\[
J_1(\pi)
=
\int \pi_0(dx\mid D_0)
\mathbb E_{P_0}
[(F(x)-b_0)_+].
\]

Define

\[
EI(x)
=
\mathbb E_{P_0}
[(F(x)-b_0)_+].
\]

Then

\[
J_1(\pi)
=
\int EI(x)\pi_0(dx\mid D_0).
\]

Hence every optimal one-step policy is supported on

\[
\arg\max_x EI(x).
\]

Therefore

\[
\boxed{
H=1
\quad\Longrightarrow\quad
\text{ordinary EI decision}.
}
\]

This is the precise special-case statement. A finite proximal optimization step introduced later does not itself equal EI; the **underlying unregularized target** does.

---

# 3. Sequential Bayesian path measure

The ordinary posterior-predictive transition is

\[
K_t(dy\mid H_t,x)
=
\int L(dy\mid F,x)P(dF\mid H_t).
\]

A causal policy induces the history-space path law

\[
\boxed{
P_K^\pi(d\tau)
=
\prod_{t=0}^{H-1}
\pi_t(dx_t\mid H_t)
K_t(dy_t\mid H_t,x_t),
}
\]

where

\[
\tau=(x_0,y_0,\ldots,x_{H-1},y_{H-1}).
\]

The dynamic-programming formulation is

\[
V_t(H_t)
=
\sup_x
\mathbb E_{K_t}
[
r_t+V_{t+1}(H_{t+1})
].
\]

Its difficulty is that the decision rule must be defined over all possible future histories.

---

# 4. Proposition 1 — Root-world equivalence

Define instead the joint law

\[
\boxed{
\bar P^\pi(dF,d\tau)
=
P_0(dF)
\prod_{t=0}^{H-1}
\pi_t(dx_t\mid H_t)
L(dy_t\mid F,x_t).
}
\]

## Proposition

Marginalizing \(F\) from \(\bar P^\pi\) produces exactly the sequential Bayesian path law:

\[
\boxed{
\int \bar P^\pi(dF,d\tau)
=
P_K^\pi(d\tau).
}
\]

Consequently,

\[
\boxed{
J_H(\pi)
=
\mathbb E_{
F\sim P_0,\,
\tau\sim\pi,L(\cdot\mid F,\cdot)
}
[R_H(F,\tau)].
}
\]

## Proof sketch

After any realized history \(H_t\), Bayes' rule gives

\[
P(dF\mid H_t)
\propto
P_0(dF)
\prod_{s<t}L(dy_s\mid F,x_s).
\]

Therefore the conditional law of the next observation under the root-world factorization is

\[
\int L(dy_t\mid F,x_t)P(dF\mid H_t)
=
K_t(dy_t\mid H_t,x_t).
\]

Applying this sequentially yields the stated marginal path law.

## Computational interpretation

To generate a correct adaptive fantasy trajectory, it is sufficient to:

1. sample one coherent world \(F\sim P_0\);
2. let the causal policy choose \(X_t\) from the observed fantasy history;
3. generate \(Y_t\sim L(\cdot\mid F,X_t)\).

No explicit posterior refit is mathematically required merely to generate the trajectory.

### Important novelty correction

This root/pathwise idea is **not new by itself**. LookaHES explicitly uses pathwise posterior sampling to avoid exponentially growing rollout trees. The useful BO-specific consequence may instead arise when a complex non-Gaussian model supplies coherent world samples but repeated hypothetical conditioning would be expensive.

---

# 5. Why naive reward-tilted fantasy paths are invalid

Fix a current causal policy \(\pi^k\). A tempting EBM target is

\[
\boxed{
\widetilde Q_k(d\tau)
=
\frac{
e^{\eta R_H(\tau)}
P_K^{\pi^k}(d\tau)
}{
Z_k
}.
}
\]

This favors high-return futures.

However, it does not in general correspond to a realizable BO policy.

Define the continuation desirability

\[
\psi_{t+1}(H_{t+1})
=
\mathbb E_{P_K^{\pi^k}}
[
e^{\eta R_H}
\mid H_{t+1}
].
\]

Then the conditional observation distribution under the tilted path law satisfies

\[
\widetilde Q_k
(dy_t\mid H_t,x_t)
\propto
K_t(dy_t\mid H_t,x_t)
\psi_{t+1}(H_t,x_t,y_t).
\]

Unless \(\psi_{t+1}\) is constant in \(y_t\),

\[
\boxed{
\widetilde Q_k
(dy_t\mid H_t,x_t)
\neq
K_t(dy_t\mid H_t,x_t).
}
\]

Thus reward tilting changes the apparent probability of future experimental outcomes.

Interpretation:

> **Naively conditioning on successful Bayesian futures makes lucky outcomes look controllable.**

This is the central causality problem.

It is the stochastic-dynamics analogue of a known limitation in control-as-inference: unconstrained inference over high-reward trajectories does not generally preserve the true environment transition law.

---

# 6. Causal family of path measures

Define

\[
\mathcal M_K
=
\left\{
P_K^\pi:
\pi\in\Pi_{\rm causal}
\right\}.
\]

Every member of \(\mathcal M_K\) has:

- arbitrary allowed causal action kernels;
- the **same fixed Bayesian observation kernels** \(K_t\).

The only thing the decision maker may change is the action law.

This is the appropriate feasible set for energy/path inference in BO.

---

# 7. Proposition 2 — Causal KL projection

Project the noncausal reward-tilted target back onto the realizable causal family:

\[
\boxed{
P_K^{\pi^{k+1}}
=
\arg\min_{P_K^\pi\in\mathcal M_K}
KL
\left(
P_K^\pi
\Vert
\widetilde Q_k
\right).
}
\]

Since

\[
\log
\frac{d\widetilde Q_k}{dP_K^{\pi^k}}
=
\eta R_H-\log Z_k,
\]

we obtain

\[
KL(P_K^\pi\Vert\widetilde Q_k)
=
KL(P_K^\pi\Vert P_K^{\pi^k})
-
\eta J_H(\pi)
+
\log Z_k.
\]

Therefore the projection is exactly

\[
\boxed{
\pi^{k+1}
=
\arg\max_{\pi\in\Pi_{\rm causal}}
\left\{
J_H(\pi)
-
\frac{1}{\eta}
KL(P_K^\pi\Vert P_K^{\pi^k})
\right\}.
}
\]

Equivalently, use a trust-region form

\[
\boxed{
\max_\pi J_H(\pi)
\quad
\text{s.t.}
\quad
KL(P_K^\pi\Vert P_K^{\pi^k})
\le \delta_k.
}
\]

The constrained form is preferable algorithmically because the KL radius has a direct trust-region interpretation and the corresponding dual temperature can be chosen automatically.

---

# 8. Proposition 3 — Dynamics cancel from the path KL

Both \(P_K^\pi\) and \(P_K^{\pi^k}\) use exactly the same observation kernels. Hence

\[
\frac{dP_K^\pi}{dP_K^{\pi^k}}(\tau)
=
\prod_{t=0}^{H-1}
\frac{
\pi_t(x_t\mid H_t)
}{
\pi_t^k(x_t\mid H_t)
}.
\]

Therefore

\[
\boxed{
KL(P_K^\pi\Vert P_K^{\pi^k})
=
\sum_{t=0}^{H-1}
\mathbb E_{P_K^\pi}
\left[
KL
\left(
\pi_t(\cdot\mid H_t)
\Vert
\pi_t^k(\cdot\mid H_t)
\right)
\right].
}
\]

This is a useful BO interpretation:

> **The trust region constrains only decisions. The Bayesian observation model is never distorted or regularized.**

---

# 9. Proposition 4 — Exact proximal improvement is monotone

Assume the proximal subproblem is solved exactly.

Because the old policy \(\pi^k\) is feasible,

\[
J_H(\pi^{k+1})
-
\frac1\eta
KL(P^{\pi^{k+1}}\Vert P^{\pi^k})
\ge
J_H(\pi^k).
\]

Hence

\[
\boxed{
J_H(\pi^{k+1})
\ge
J_H(\pi^k)
+
\frac1\eta
KL(P^{\pi^{k+1}}\Vert P^{\pi^k})
\ge
J_H(\pi^k).
}
\]

Thus the KL term is an **algorithmic proximal device**, not a permanent entropy regularizer that changes the target BO objective.

### Important limitation

This monotonicity statement applies to the exact unrestricted proximal solve. A neural-policy approximation optimized only approximately does not inherit the guarantee automatically.

---

# 10. Exact proximal update and the hidden Bellman problem

For additive rewards and unrestricted stochastic policies, the proximal problem admits the recursion

\[
Q_t^k(H_t,x)
=
\mathbb E_{K_t}
[
r_t+V_{t+1}^k(H_{t+1})
],
\]

\[
\boxed{
V_t^k(H_t)
=
\tau_k
\log
\int
\pi_t^k(dx\mid H_t)
\exp
\left(
\frac{Q_t^k(H_t,x)}{\tau_k}
\right),
}
\]

with \(\tau_k=1/\eta_k\), and

\[
\boxed{
\pi_t^{k+1}(dx\mid H_t)
=
\pi_t^k(dx\mid H_t)
\exp
\left(
\frac{
Q_t^k(H_t,x)-V_t^k(H_t)
}{
\tau_k
}
\right).
}
\]

At \(H=1\),

\[
Q_0(D_0,x)=EI(x),
\]

so

\[
\pi_0^{k+1}(x)
\propto
\pi_0^k(x)
\exp(EI(x)/\tau_k).
\]

Repeated concentration or a zero-temperature limit recovers the ordinary EI maximizer.

## Crucial audit result

The energy formulation has **not** solved nonmyopic BO at this point.

Computing \(Q_t^k\) exactly is still the original dynamic program.

Therefore the paper requires a computational approximation that avoids explicitly learning or tabulating \(Q_t^k\).

---

# 11. Parametric causal policy

Use a shared policy

\[
\pi_\theta(dx_t\mid H_t,t).
\]

For continuous BO, a practical first form is a reparameterizable stochastic policy

\[
X_t
=
g_\theta(H_t,t,\zeta_t),
\qquad
\zeta_t\sim p_\zeta,
\]

with a tractable density \(\pi_\theta(x\mid H_t,t)\).

The history encoder must be permutation/size aware. A DeepSets or attention-set encoder is a natural baseline, but this architecture is not a contribution; EARL-BO already uses Attention-DeepSets.

A stochastic policy is useful during optimization because path-space KL is well defined. The final first action can be taken from the mode/mean or from a concentrated policy.

---

# 12. Differentiable root-world representation

Assume a coherent posterior world can be reparameterized as

\[
F=\mathcal G_\psi(\xi),
\qquad
\xi\sim p_\xi,
\]

and observations as

\[
Y_t
=
h(F,X_t,\epsilon_t),
\qquad
\epsilon_t\sim p_\epsilon.
\]

The complete trajectory is a deterministic function of base randomness

\[
\omega
=
(\xi,\zeta_{0:H-1},\epsilon_{0:H-1})
\]

and policy parameters:

\[
\tau_\theta=\mathcal T_\theta(\omega).
\]

Then

\[
J_H(\theta)
=
\mathbb E_\omega
[
R_H(\mathcal T_\theta(\omega))
].
\]

This representation does **not** reveal \(F\) to the policy. The policy receives only \(H_t\). Root-world information is used by the simulator, not by the decision rule.

---

# 13. Proposition 5 — Direct pathwise proximal gradient

Define

\[
\mathcal L_k(\theta)
=
J_H(\theta)
-
\tau_k
KL(P_K^{\pi_\theta}\Vert P_K^{\pi_{\theta_k}}).
\]

Using the KL factorization,

\[
\mathcal L_k(\theta)
=
\mathbb E_\omega
\left[
R_H(\tau_\theta)
-
\tau_k
\sum_{t=0}^{H-1}
\log
\frac{
\pi_\theta(X_t\mid H_t,t)
}{
\pi_{\theta_k}(X_t\mid H_t,t)
}
\right].
\]

Under standard reparameterization, differentiability, and dominated-convergence conditions,

\[
\boxed{
\nabla_\theta\mathcal L_k(\theta)
=
\mathbb E_\omega
\left[
\nabla_\theta
\left\{
R_H(\tau_\theta)
-
\tau_k
\sum_t
\log
\frac{\pi_\theta}{\pi_{\theta_k}}
\right\}
\right].
}
\]

Thus a Monte Carlo gradient is obtained simply by backpropagating through sampled adaptive campaigns.

No explicit \(Q\)-function or learned critic is required.

### Nonsmooth terminal improvement

The max in terminal improvement is differentiable almost everywhere under continuous outcomes; ties have probability zero under standard continuous models. In practice, automatic differentiation can use the active maximizer. A smooth-max relaxation is optional but should not be introduced unless necessary.

---

# 14. What is actually new versus existing policy-gradient methods?

The pathwise gradient alone is **not new**.

- *Differentiating Policies* differentiates BO rollout trajectories, but the future actions are generated by a chosen base acquisition policy.
- LookaHES already uses neural policies, pathwise posterior sampling, and chain-rule gradients in continuous problems.
- EARL-BO already optimizes a full adaptive BO policy, but with PPO actor–critic training.

The plausible intersection still not obviously covered is:

\[
\boxed{
\text{full classical finite-horizon improvement objective}
+
\text{shared causal policy}
+
\text{root-world pathwise gradients}
+
\text{causal path-KL trust-region transport}
}
\]

with no fixed rollout policy and no learned critic.

That is the actual candidate contribution.

---

# 15. Working algorithm: Causal Pathwise Policy Transport

**Working name only.**

At each real BO iteration:

### Input

- current posterior world model \(P_0(dF)\);
- horizon \(H\);
- causal policy \(\pi_{\theta_0}\);
- path-KL trust-region radius \(\delta\);
- \(M\) root worlds per gradient batch.

### Transport stage \(k=0,\ldots,S-1\)

1. Freeze the reference policy \(\pi_{\theta_k}\).
2. Draw or refresh \(M\) coherent root worlds.
3. Simulate complete \(H\)-step causal campaigns under candidate \(\pi_\theta\).
4. Estimate
   \[
   J_H(\theta)
   \]
   and
   \[
   KL(P^{\pi_\theta}\Vert P^{\pi_{\theta_k}}).
   \]
5. Solve approximately
   \[
   \max_\theta J_H(\theta)
   \quad
   \text{s.t.}
   \quad
   KL_{\rm path}(\theta,\theta_k)\le\delta
   \]
   using pathwise gradients and a dual variable for the KL constraint.
6. Set the optimized policy as the next reference.

### Execution

Execute only the first real action from the final policy, observe the true experiment, update the posterior, and replan.

### Initialization

The method does **not** assume a rollout policy. A broad policy can be used initially; after the first real BO iteration, the previous optimized policy provides a natural warm start.

---

# 16. Why call this “energy transport”?

The formal reward tilt

\[
\widetilde Q_k
\propto
P^{\pi_k}e^{\eta R}
\]

is an energy-based high-utility path target.

It is noncausal.

The reverse-KL projection

\[
\operatorname*{argmin}_{P^\pi\in\mathcal M_K}
KL(P^\pi\Vert\widetilde Q_k)
\]

moves the realizable policy-induced path law toward that target while preserving the observation dynamics.

Repeated KL-limited steps are analogous to trust-region / annealed path-space measure transport.

However, after projection the computational problem is also recognizable as relative-entropy / mirror-descent policy optimization.

Therefore:

> **The energy view motivates the target and causal projection, but the paper must not claim that KL-proximal policy optimization is itself new.**

The energy framing earns its place only if this transport interpretation yields a practically superior BO solver.

---

# 17. Scaling analysis

## 17.1 Explicit scenario tree

Suppose every fantasy action node branches into \(K\) possible observations.

For horizon \(H\), the number of action nodes is

\[
\boxed{
N_{\rm tree}(K,H)
=
1+K+\cdots+K^{H-1}
=
\frac{K^H-1}{K-1}.
}
\]

With input dimension \(d\), a one-shot tree contains approximately

\[
d\,N_{\rm tree}
\]

continuous action variables.

Examples:

| \(K\) | \(H\) | action nodes |
|---:|---:|---:|
| 4 | 4 | 85 |
| 4 | 6 | 1,365 |
| 4 | 8 | 21,845 |
| 4 | 10 | 349,525 |
| 8 | 4 | 585 |
| 8 | 6 | 37,449 |
| 8 | 8 | 2,396,745 |
| 8 | 10 | 153,391,689 |

One-shot optimization removes nested optimizers, but it does not remove this representational growth.

## 17.2 Shared pathwise policy

Let:

- \(p\): policy parameter count;
- \(M\): root worlds per gradient step;
- \(G\): gradient steps per transport stage;
- \(S\): transport stages.

The parameter count \(p\) does not grow with \(H\) for a shared time-conditioned policy.

Ignoring encoder details, world evaluations scale as

\[
\boxed{
N_{\rm eval}
=
O(SGMH).
}
\]

If root worlds are reused during a local optimization window,

\[
C_{\rm policy}
\approx
SM\,C_{\rm world}
+
SGMH\,C_{\rm eval},
\]

where \(C_{\rm world}\) is the cost of creating a coherent posterior world and \(C_{\rm eval}\) is the cost of evaluating that world at a new action.

Memory for direct reverse-mode differentiation is \(O(MH)\) activations, reducible by checkpointing.

### Important caveat

There is **no theorem that \(M,G,S\) stay constant with horizon**. Worst-case sample/optimization complexity can still be severe.

The valid claim is only:

> the **representation and per-trajectory simulation** do not explicitly branch exponentially.

## 17.3 Generic non-Gaussian model

Suppose a model-specific conditional posterior operation at a fantasy node costs \(C_{\rm cond}\).

A tree-like planner may incur roughly

\[
C_{\rm tree}
\sim
N_{\rm tree}(K,H)C_{\rm cond}
\]

in addition to optimization overhead.

If the same model permits coherent root-world samples, policy simulation costs approximately

\[
C_{\rm path}
\sim
SGM
\left[
C_{\rm world}
+
H C_{\rm eval}
\right].
\]

The most attractive non-Gaussian regime is therefore

\[
\boxed{
N_{\rm tree}(K,H)C_{\rm cond}
\gg
SGM(C_{\rm world}+HC_{\rm eval}).
}
\]

This is especially plausible when **conditioning a complex generative posterior is expensive but evaluating one already-sampled world is cheap**.

---

# 18. Comparison with EARL-BO

EARL-BO also avoids explicit scenario-tree parameter growth by learning a shared policy. Its implementation uses:

- an Attention-DeepSets state encoder;
- PPO;
- an actor;
- a critic/value network;
- model-based virtual BO trajectories;
- off-policy initialization and on-policy fine-tuning.

Its per-episode simulation cost is already \(O(H)\), so the proposed method has **no asymptotic horizon advantage over EARL-BO**.

The candidate advantage is instead:

\[
\boxed{
\text{direct differentiable model gradient}
+
\text{no critic}
+
\text{causal path-KL trust region}.
}
\]

The paper must show that this leads to substantially fewer virtual model evaluations, lower wall-clock, or greater training stability.

If not, EARL-BO already solves the main representational problem.

---

# 19. Comparison with LookaHES — the strongest overlap

LookaHES already has:

- a shared neural policy;
- pathwise posterior sampling;
- chain-rule differentiation in continuous problems;
- constant policy parameter count with horizon;
- long-horizon demonstrations beyond \(H=20\);
- structured discrete policies for protein design.

Therefore the proposed method **cannot claim** those capabilities as novel.

The meaningful distinctions are:

1. **Objective.**  
   The proposed method directly targets the classical finite-horizon expected terminal improvement \(J_H\). LookaHES optimizes a multi-step \(H\)-Entropy Search / decision-entropy objective with dynamic costs.

2. **Causal energy derivation.**  
   We explicitly show why unrestricted reward tilting of stochastic Bayesian futures is noncausal and derive the fixed-dynamics KL projection.

3. **Trust-region path geometry.**  
   Policy changes are constrained by exact induced path-measure KL rather than introduced as a generic optimizer heuristic.

4. **Model interface.**  
   The target API is a coherent differentiable world sampler rather than GP-specific analytic fantasization.

These are mathematically meaningful. Whether they are *enough for a paper* is an empirical question.

---

# 20. Comparison with “MDP Planning as Policy Inference”

The 2026 policy-inference paper already assigns an unnormalized density to policies that is monotone in expected return and uses VSMC for policy inference under stochastic dynamics.

Therefore we cannot claim:

> “Expected-return planning is Bayesian inference over policies.”

The distinction would be:

- continuous BO information states and continuous actions;
- Bayesian **world uncertainty** rather than a generic MDP simulator;
- root-world reparameterization;
- direct pathwise gradients;
- causal KL policy transport rather than VSMC over discrete deterministic policies.

Again, the algorithmic consequences must carry the novelty.

---

# 21. Comparison table

| Property | One-shot tree | EARL-BO | LookaHES | Proposed |
|---|---|---|---|---|
| Full adaptive policy target | Yes, tree approximation | Yes | Neural variational HES policy | **Yes, expected terminal improvement** |
| Fixed rollout heuristic | No | No | No | **No** |
| Policy representation | Separate action/tree node | Shared actor | Shared neural/LLM policy | **Shared causal policy** |
| Scenario branching | Explicit | No | No | **No** |
| Actor–critic | No | **Yes** | Not for all continuous cases; PPO in complex discrete case | **No in differentiable regime** |
| Pathwise sampling | GP fantasies | model-based trajectories | **Yes** | **Yes** |
| Pathwise policy gradient | Tree autodiff | PPO/score-style RL | **Yes in continuous case** | **Yes** |
| Standard expected-return objective | Multi-step EI | **Yes** | HES-style objective | **Yes** |
| Explicit causal path-KL derivation | No | No | No | **Candidate contribution** |
| Arbitrary coherent non-Gaussian world sampler | model-specific | not demonstrated | not central | **Target capability** |
| \(H=1\) target | EI | one-step MDP reward | depends HES setup | **EI** |

---

# 22. Model-quality scaling: planning has a calibration window

Long-horizon planning is more model-sensitive.

Assume the true predictive transition kernel is \(K\), the model uses \(\widetilde K\), and uniformly over relevant histories/actions,

\[
TV(K_t(\cdot\mid H,x),\widetilde K_t(\cdot\mid H,x))
\le\epsilon.
\]

For any fixed policy, a sequential coupling gives

\[
TV(P_K^\pi,P_{\widetilde K}^\pi)
\le
1-(1-\epsilon)^H
\le
H\epsilon.
\]

If

\[
0\le R_H\le R_{\max},
\]

then

\[
\boxed{
|J_K(\pi)-J_{\widetilde K}(\pi)|
\le
R_{\max}H\epsilon.
}
\]

Let \(\pi_K^\star\) be optimal under the true model and \(\pi_{\widetilde K}^\star\) optimal under the surrogate. Then

\[
\boxed{
J_K(\pi_K^\star)-J_K(\pi_{\widetilde K}^\star)
\le
2R_{\max}H\epsilon.
}
\]

This is a simple worst-case bound, not a tight BO theorem, but it captures an important regime condition:

> **The horizon must be long enough that trees are difficult, but not so long that model error dominates the possible nonmyopic gain.**

This supports treating calibrated world modeling as an assumption rather than another contribution.

---

# 23. Regime map

## Most promising

The method should be tested where all of the following approximately hold:

### A. Moderate-to-long horizon

\[
H\gtrsim 5
\]

so explicit scenario branching is already uncomfortable.

### B. Genuine adaptivity

The best action after step \(t\) depends substantially on observations collected earlier. Purely open-loop planning is insufficient.

### C. Compressible policy

A shared mapping

\[
H_t\mapsto X_t
\]

can represent good behavior with manageable capacity.

### D. Coherent differentiable world samples

The surrogate can produce

\[
F\sim P(F\mid D_0)
\]

and support differentiable evaluation \(F(x)\).

### E. Expensive hypothetical conditioning

Repeatedly constructing posteriors for many fantasy datasets is significantly more expensive than evaluating a sampled world.

### F. Model quality is credible

The expected nonmyopic benefit is large compared with the approximate \(O(H\epsilon)\) model-error amplification.

### G. Warm starts are possible

Successive real BO rounds have nearby beliefs, so the previous policy can seed the next policy-transport solve.

## Poor regime

Expect the approach to lose when:

- \(H\le2\) or \(3\);
- GP fantasization is extremely cheap;
- the policy is highly discontinuous/noncompressible;
- the surrogate lacks differentiable coherent world samples;
- action spaces are discrete and require high-variance score-function gradients;
- terminal reward is so sparse that direct pathwise optimization has no useful signal;
- model calibration is poor.

---

# 24. Primary algorithmic hypothesis

The paper should test one central algorithmic hypothesis:

\[
\boxed{
\text{Causal KL-proximal pathwise policy optimization
reaches high-quality full nonmyopic policies
with fewer virtual-world evaluations than actor–critic RL.}
}
\]

This is stronger and more falsifiable than “energy models can plan.”

A useful decomposition is:

### Direct pathwise policy optimization

\[
\max_\theta J_H(\theta)
\]

with Adam.

### Causal transport

\[
\max_\theta J_H(\theta)
\quad
\text{s.t.}
\quad
KL_{\rm path}(\theta,\theta_k)\le\delta.
\]

### EARL/PPO

actor–critic policy optimization.

If causal transport does not materially outperform plain pathwise Adam, then the energy/transport machinery is probably decorative.

If both direct pathwise methods perform similarly to EARL at similar cost, the paper is also weak.

---

# 25. JES-like paper thesis

## Working title

**Causal Policy Transport for Nonmyopic Bayesian Optimization**

Alternative:

**Nonmyopic Bayesian Optimization Without Trees or Critics**

The first is safer.

## One-sentence thesis

> **Rather than enumerate a branching fantasy tree, optimize the causal policy that generates it: a reward-tilted Bayesian future is generally noncausal, but its KL projection onto policies with fixed Bayesian dynamics yields a trust-region objective that can be optimized directly through coherent posterior worlds.**

## Memorable consequence

> **Transport actions, not outcomes.**

---

# 26. JES-like introduction structure

### Paragraph 1 — recognizable problem

Myopic BO ignores the effect of an experiment on future decisions. Full nonmyopic BO solves this correctly as a finite-horizon adaptive policy problem.

### Paragraph 2 — limitation of existing representations

Scenario-tree methods represent adaptivity explicitly, creating rapidly growing numbers of fantasy branches and decision variables. Neural-policy methods avoid this representation cost, but EARL-BO uses actor–critic RL, while recent pathwise methods optimize alternative long-horizon objectives.

### Paragraph 3 — structural observation

High-reward path conditioning is an attractive modern inference view, but it is wrong under stochastic Bayesian observations because it changes the law of future outcomes.

### Paragraph 4 — our insight

Restrict path inference to measures with the original Bayesian observation dynamics. Reverse-KL projection of a reward tilt onto this causal family is exactly a path-space trust-region policy-improvement step.

### Paragraph 5 — computational consequence

BO provides a coherent Bayesian world model. Root-sample a world once and differentiate the complete adaptive campaign with respect to a shared causal policy. This directly optimizes full terminal improvement without a tree or critic.

### Contributions

1. causal path-measure formulation and noncausal-tilt result;
2. KL-projection/proximal equivalence and path-KL factorization;
3. critic-free root-world pathwise algorithm for the full expected-return BO objective;
4. empirical scaling versus scenario trees and RL, including a non-Gaussian world model.

---

# 27. Minimal theory section for the paper

The main paper should contain only four central propositions:

### Proposition 1 — Root-world equivalence

\[
P_K^\pi(\tau)
=
\int P_0(dF)\,
P^\pi(\tau\mid F).
\]

### Proposition 2 — Reward-tilted Bayesian futures are generally noncausal

\[
\widetilde Q(dy_t\mid H_t,x_t)
\neq
K_t(dy_t\mid H_t,x_t).
\]

### Proposition 3 — Causal projection equals KL-proximal policy improvement

\[
\operatorname*{argmin}_{P^\pi\in\mathcal M_K}
KL(P^\pi\Vert \widetilde Q_k)
=
\operatorname*{argmax}_\pi
\left[
J_H(\pi)
-\tau_k KL(P^\pi\Vert P^{\pi_k})
\right].
\]

Together with monotonicity for exact proximal solves.

### Proposition 4 — Path KL decomposes into causal action KLs

\[
KL(P^\pi\Vert P^{\pi_k})
=
\sum_t
E_{P^\pi}
KL(\pi_t\Vert\pi_t^k).
\]

The direct pathwise-gradient equation follows as the computational corollary rather than another headline theorem.

---

# 28. Figure plan

## Figure 1 — Conceptual result

Three panels:

### A. Scenario tree

Show exponentially branching fantasies.

### B. Naive energy tilt

Show high-reward fantasy conditioning tilting both:

- actions;
- lucky observations.

Mark this **noncausal**.

### C. Causal transport

Show one shared policy across histories; transport changes action distributions while the observation arrows/dynamics remain fixed.

Bottom equation:

\[
\widetilde Q
\xrightarrow{\text{causal KL projection}}
P^{\pi^{k+1}}.
\]

This should make the main idea understandable without reading the derivation.

## Figure 2 — Exact short-horizon validation

Use a tiny problem where an accurate scenario-tree solution is available for \(H=2,3,4\).

Show:

- first action/value of tree;
- direct pathwise policy;
- causal transport policy.

The candidate method must reproduce the tree decision closely.

## Figure 3 — Horizon scaling

Plot versus \(H\):

- planning wall-clock;
- virtual world/model evaluations;
- terminal policy value.

Methods:

- one-shot tree;
- direct pathwise Adam;
- causal transport;
- EARL-BO;
- LookaHES where its objective/setup permits fair comparison.

The desired signature is that tree cost explodes while causal transport maintains strong value and uses materially fewer model interactions than RL.

## Figure 4 — Complex posterior capability

Replace the GP with a genuinely non-Gaussian coherent differentiable world model.

Use the same planner implementation.

Show that:

- planner mechanics are unchanged;
- repeated fantasy conditioning is expensive/awkward;
- root-world policy planning remains practical.

This is where the modern-generative-model connection should become concrete.

---

# 29. First model systems

Do not begin with a large protein task.

## Mechanism/validation system

A small adaptive problem with an exact or very accurate short-horizon reference.

Requirements:

- optimal first action changes with horizon;
- observation-dependent second actions matter;
- calibrated model known exactly;
- \(H=2\)–4 tree reference cheap.

This is an algorithm validation, not the flagship application.

## Long-horizon system

Use a recognized movement/edit-constrained BO benchmark where intermediate evaluations can be instrumentally useful.

This gives a structural reason for \(H\gg1\).

Because LookaHES already targets this regime, it must be treated as a serious baseline rather than merely inspiration.

## Non-Gaussian system

Only after the algorithm is validated.

Prefer a model where:

- coherent posterior worlds are available;
- world evaluation is differentiable;
- conditional inference after arbitrary fantasy datasets would be expensive.

A nonconjugately conditioned function process is more meaningful than a mildly heavy-tailed Student-t perturbation.

---

# 30. Empirical success criteria

The paper should not survive on small improvements.

## Short horizon

For \(H\le4\), candidate first-action value should be essentially indistinguishable from the scenario-tree reference.

If not, stop.

## Long horizon

At \(H\approx10\)–20, aim for:

- clear improvement over myopic BO;
- comparable or better terminal utility than EARL-BO/LookaHES in matched settings;
- a **large** reduction in virtual world evaluations or wall-clock versus actor–critic training.

A useful target would be at least a several-fold efficiency improvement, not 10–20%.

## Transport ablation

Causal KL transport must materially outperform plain direct pathwise Adam in at least the hard long-horizon/multimodal regimes.

Otherwise the energy component is not earning its complexity.

## Non-Gaussian result

The same planner should operate with minimal model-specific changes and exhibit a real computational advantage over methods that require repeated hypothetical conditioning.

---

# 31. Fatal tests

Stop this paper direction if any occurs.

### F1. Short-horizon policy mismatch

The method cannot reproduce scenario-tree first actions/value at \(H=2\)–4.

### F2. No transport advantage

Direct pathwise Adam performs as well as causal KL transport.

Then the EBM/transport contribution is largely unnecessary.

### F3. EARL/LookaHES parity at equal cost

Actor–critic/pathwise neural baselines achieve similar performance with similar or lower virtual-model cost.

### F4. Hidden Bellman/critic returns

The implementation requires learning a value function or explicitly computing long-term \(Q_t\) to work.

Then the claimed simplification collapses.

### F5. Non-Gaussian capability is artificial

The complex model does not provide coherent differentiable worlds, or the method requires bespoke conditioning machinery anyway.

### F6. Model error dominates

Planning gains disappear once realistic model error is introduced.

---

# 32. Adversarial reviewer audit

## Reviewer attack 1

> “This is just TRPO / mirror descent applied to BO.”

**Valid concern.**

The causal KL projection is closely related to established relative-entropy policy optimization.

Required response:

- the BO-specific result is the fixed Bayesian-dynamics path formulation;
- root-world simulation yields direct differentiation of information-gathering behavior;
- the full expected-terminal-improvement objective is optimized without a critic/tree;
- empirical advantage must be substantial.

If experiments do not show this, the reviewer wins.

## Reviewer attack 2

> “LookaHES already has pathwise neural policy optimization and long horizons.”

**Very serious.**

Required response:

- LookaHES already removes many of the same scaling problems;
- our distinction must be the **classical full finite-horizon expected-return objective**, causal path-KL transport, and complex world-model interface;
- direct comparison is necessary.

We cannot claim pathwise sampling or neural policy compression as novel.

## Reviewer attack 3

> “EARL-BO already optimizes the full policy.”

**Correct.**

Required response:

- ours uses direct model derivatives rather than PPO actor–critic estimation;
- no critic/off-policy warm-start policy is required;
- path-KL gives a model-derived trust region;
- demonstrate large sample/wall-clock benefit.

## Reviewer attack 4

> “Planning-as-policy-inference already exists.”

**Correct.**

Do not claim policy posterior inference as novelty.

Our method is a continuous, differentiable BO-specific causal transport solver, not the first policy-inference formulation.

## Reviewer attack 5

> “You still approximate the full policy with a neural network.”

**Correct.**

Scenario trees have discretization/tree error; neural policies have function-approximation error.

The method should be framed as replacing exponential explicit representation by a compressible policy class, not as exact dynamic programming.

## Reviewer attack 6

> “Your pathwise gradients require differentiability, so the arbitrary-surrogate claim is false.”

**Correct.**

Revise the claim:

> the critic-free version applies to **coherent differentiable world samplers**, which may be non-Gaussian.

For nondifferentiable discrete models, score-function or alternative inference is required and the advantage is less clear.

## Reviewer attack 7

> “The KL temperature changes the BO objective.”

The final target is unregularized \(J_H\).

Use a constrained trust-region formulation

\[
KL_{\rm path}\le\delta
\]

and interpret the dual temperature only as an optimization parameter. Exact proximal stages are monotone in the original return.

## Reviewer attack 8

> “Long-horizon planning on a bad model is dangerous.”

Agree explicitly.

The method assumes a credible posterior world model. The simple \(O(H\epsilon)\) path-model error bound should be included to motivate this limitation.

---

# 33. Corrected novelty claim

After the audit, the strongest defensible claim is:

> **We show that direct energy conditioning of high-reward Bayesian futures is generally noncausal, and derive the corresponding causal path-measure projection. This yields a KL-proximal optimization of the standard finite-horizon BO policy. For coherent differentiable posterior world models, the proximal objective admits direct pathwise gradients through full adaptive campaigns, enabling test-time optimization of the classical nonmyopic objective without explicit scenario trees, fixed rollout heuristics, or actor–critic value learning.**

This is narrower than the initial idea, but much more defensible.

---

# 34. Recommendation

This direction is mathematically coherent enough to justify **one implementation kill test**, but not yet a full paper-scale project.

The next code should test only:

1. exact/accurate scenario-tree reference on a small adaptive BO problem;
2. a shared causal policy;
3. direct pathwise optimization;
4. the path-KL trust-region variant.

Questions:

- Can direct policy optimization match the tree at \(H=2\)–4?
- Does the KL transport materially stabilize/improve it?
- Does policy simulation scale to \(H=10\)–20 without requiring a critic?

Do **not** add the complex non-Gaussian model until those questions are answered.

If the answer to either of the first two is no, this direction should stop before another major repository rebuild.

---

# References / nearest literature

- Jiang, Jiang, Balandat, Karrer, Gardner, Garnett. **Efficient Nonmyopic Bayesian Optimization via One-Shot Multi-Step Trees.** NeurIPS 2020. arXiv:2006.15779.
- Cheon, Lee, Koh, Tsay. **EARL-BO: Reinforcement Learning for Multi-Step Lookahead, High-Dimensional Bayesian Optimization.** arXiv:2411.00171, revised 2026.
- Truong et al. **Neural Nonmyopic Bayesian Optimization in Dynamic Cost Settings.** arXiv:2601.06505, 2026.
- Nwankwo, Bindel. **Differentiating Policies for Non-Myopic Bayesian Optimization.** arXiv:2408.07812, 2024.
- Tolpin. **MDP Planning as Policy Inference.** arXiv:2602.17375, 2026.
- Blessing et al. **Trust Region Constrained Measure Transport in Path Space for Stochastic Optimal Control and Inference.** arXiv:2508.12511, 2025.
- Levine. **Reinforcement Learning and Control as Probabilistic Inference: Tutorial and Review.** arXiv:1805.00909, 2018.
