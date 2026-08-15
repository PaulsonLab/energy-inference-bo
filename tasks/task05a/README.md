# Task 05A — Structured Belief Gate

Task 05A asks whether a credible probabilistic belief exists for low-data measured
protein design and whether biological sequence structure materially improves
finite-pool BO decisions.

- [Frozen experiment contract](SPEC.md)
- [Mathematical definitions](MATH.md)
- [Current evidence and gate](SUMMARY.md)
- [A100 execution guide](COLAB.md)
- [Frozen configurations](configs/)

The only implementation in scope is Task 05A. A smoke run checks wiring but can
never authorize Task 05B. The A100 notebook first profiles one shard, then manages all
twenty frozen shards as a Drive-backed resumable campaign with one final ZIP.
