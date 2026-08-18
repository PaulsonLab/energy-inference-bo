# Welded Beam q=1 Decision-Shift Result

## Outcome

**WELDED_BEAM_SHIFT_NEGATIVE_REVIEW_REQUIRED**

The experiment used all three prospectively frozen states. The status is
computed from the frozen gate; it does not authorize q=4 or a new sampler.

All 21 independent GP fits converged. The stable analytic improvement moments
agreed with adaptive quadrature to maximum log error `4.44e-16`. The fixed
candidate set's true feasible fraction was `0.0010376` (17/16,384), consistent
with the narrow standard Welded Beam feasible region.

## State evidence

| State | Best ESS | Top-32 median ESS | 256 disagreement | 512 disagreement | 512 regret | Classification |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 4101 | 0.4089 | 0.1133 | 0.0214 | 0.0115 | 0.0000 | conclusively negative |
| 4102 | 0.4292 | 0.0745 | 0.0302 | 0.0194 | 0.0000 | conclusively negative |
| 4103 | 0.1360 | 0.0874 | 0.0634 | 0.0414 | 0.0000 | intermediate |

All 64 independent scrambles selected the exact candidate-set maximizer in
every state at every tested sample count from 64 through 1,024. Thus the
experiment verifies that ordinary constraints can reduce population ESS, but
it falsifies the required conjunction: the shift did not produce a material
practical-QMC decision error. State 4103 retained modest top-32 ordering error
at 256 samples, but it never changed the selected design or incurred regret.

The result should not be generalized to q=4 or every constrained BO state. It
does show that low ESS by itself is insufficient motivation for a new inner
inference method when common-random-number scrambled Sobol sampling preserves
the actual decision.

## Evidence

- [Gate record](gate_result.json)
- [State summary](state_summary.csv)
- [QMC summary](qmc_summary.csv)
- [Figure A](figure_a_shift_vs_quality.png)
- [Figure B](figure_b_qmc_reliability.png)
- [Figure C](figure_c_mechanism.png)

## Next action

Human review is required. Do not automatically authorize q=4, a non-Gaussian belief, or decision-adapted inference.
