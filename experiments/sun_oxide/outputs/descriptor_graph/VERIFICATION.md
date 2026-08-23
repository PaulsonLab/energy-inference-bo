# Independent descriptor/graph result verification

Terminal verdict: `PASS_DESCRIPTOR_GRAPH`.

The returned Colab ZIP was verified independently on 2026-08-22 before these
outputs were committed. The upload SHA-256 was
`a1f2b2d18894d078ee5696ef76dfcc7ff0931f415625ee3dd8668fe7d1d41fb3`;
its ten paths were exactly the nine scientific outputs declared by
`artifact_manifest.json` plus that manifest, with no nested or unexpected
entries. All declared file sizes and SHA-256 hashes matched. The artifact
manifest SHA-256 is
`c908c7b6450016807346b9b892bbe83590ed1f10f48a75f7fe323c1b1dda09b8`.
The private transfer ZIP is not committed. An exact copy of the benchmark's
`NLR_DATA_USE_NOTICE.txt` accompanies the committed derived artifacts to retain
the source's redistribution notice.

The recorded Colab `RUN_SHA` is
`843b2173454b70cf12a6199b4d1a32740e60315e`, the intended implementation
commit. Independent verification used the committed benchmark keys and the
saved raw descriptor matrix; it did not regenerate descriptors or open the GW
oracle. It confirmed:

- the frozen benchmark remains 2,142 legacy compositions and 191 actions;
- the raw matrix is `(2142, 132)`, `float64`, with zero nonfinite values and 15
  zero-variance features;
- graph-only standardization and deterministic reconstruction reproduce the
  saved 14,063 10-NN edges, 2,141 MST edges, and 14,072-edge union exactly;
- the graph has one connected component, no isolated nodes, and degree
  min/median/mean/max `10 / 13.0 / 13.139122315592903 / 26`;
- the saved sparse `Q0` equals the reconstruction from the saved edge list,
  is symmetric with nonpositive off-diagonals, and has independently computed
  extremal eigenvalues approximately `1.0000000000000024` and
  `2.424183846275552`;
- independent sparse-solve relative residuals were
  `9.151770264032934e-16`, `7.395735936125527e-16`, and
  `5.834647719940069e-16`, each below `1e-10`; and
- all 191 action rows reproduce the committed benchmark mapping exactly, with
  unique action keys, composition keys, and node indices.

The Colab wall times remain diagnostics only. The tiny cross-machine
differences in recomputed eigensolver values and residuals are at floating-point
roundoff scale and do not affect any frozen criterion. No GW target value was
read, and no PBE value was used for descriptor or graph construction. This
result contains no preference factors, influence calculation, inference, or
Bayesian optimization.

The standard Magpie preset itself names six elemental aggregate features with
`GSbandgap`; these are required preset metadata and are not PBE or GW targets
from the frozen compound benchmark. The 20 reported exact-distance ties occur
in the complete-graph MST candidate ordering, are resolved by the frozen stable
composition-key rule, and leave no unresolved graph ambiguity.
