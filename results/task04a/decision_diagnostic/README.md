# Task 04A-D local decision diagnostic

The frozen CPU diagnostic used withheld seeds 100–107, D=6, I at n=128/256, G at
n=256, 256 test points, and the prescribed natural and counterfactual decision panels.
All 24 cases completed in `14.89 s` of recorded case time; peak process RSS was
`783.3 MiB`.

**Official classification: `INVALID`.** All 16 I cases contained zero realizable
near-tie pairs under the frozen 20%-of-maximum, 0.5%-G0-EI, and interdecile-context
requirements, rather than the required 32. Only `6.64%` of n=256 counterfactual
oracle pairs had at least 1% normalized EI contrast. U had no natural decision
opportunity in any of the eight n=256 cases, and both U/P had zero median regret.

The invalid decision experiment still supplies a consistent secondary result: P
reduced mean n=256 conditional KL from `0.045686` to `0.022366` (`51.0%`) and won all
8 seeds. Gaussian safety passed. The process therefore learned pairwise density, but
the frozen oracle/candidate construction did not provide the decision variation
needed to test a q=1 BO advantage.

- [Frozen gate record](gate_status.json)
- [Per-seed n=256 audit](n256_seed_metrics.csv)
- [Task-level interpretation](../../../tasks/task04a/SUMMARY.md)

Raw case JSON, partial tables, and generated figures remain ignored under
`artifacts/task04a/decision_diagnostic/local_final/`.
