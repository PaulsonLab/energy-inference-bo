# Task 02 Roadmap

## Task 02A — now
**Posterior continuity / reuse diagnostic**

Seed particles with trusted SAAS NUTS once. Add observations sequentially and update only weights. Measure ESS, posterior drift, coordinate stability, predictive mixture accuracy, q=1 EI agreement, and cached-update cost.

No new sampler.

### Gate
Proceed only if there is a useful reuse horizon and fresh-NUTS agreement remains good while ESS is healthy.

---

## Task 02B — only if 02A passes
**Adaptive annealed resample-move over full latent SAAS energy**

Implement exact SAAS latent energy in unconstrained variables, including Jacobians.

Use:
- adaptive likelihood tempering;
- ESS-controlled resampling;
- batched/parallel rejuvenation;
- cache reuse for unmoved particles;
- explicit full-factorization counters.

Compare against fresh NUTS.

This is still not automatically novel because SMC for BO hyperparameters is prior art.

---

## Task 02C — only if coordinate-stability evidence supports it
**Sparse selective structural transport**

Exploit SAAS structure:
- inactive/stable coordinates move rarely;
- active/uncertain coordinates receive more rejuvenation;
- explicit unlock/birth mechanism prevents permanently freezing a dimension.

Hypothesis:
computational cost should depend more on unresolved effective dimension than ambient dimension.

This is a more distinctive direction than generic SMC.

---

## Later
**Scale in n:** replace exact GP data energy with Vecchia/local factorization.

**Residual predictive energy:** only after a strong fully Bayesian structural reference is established, test whether persistent predictive miscalibration remains.

**Acquisition-side energy inference:** defer until a genuinely non-Gaussian/structured surrogate makes explicit acquisition evaluation a bottleneck.
