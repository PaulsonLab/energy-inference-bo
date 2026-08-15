# Results index

Only compact, reviewed evidence belongs here. Regenerated and large machine outputs
belong under ignored `artifacts/` until selected for publication.

| Stage | Evidence |
| --- | --- |
| Task 01 | [three-seed smoke evidence](task01/README.md) |
| Task 02A | [full three-seed Colab evidence](task02a/README.md) |
| Task 02B | [retrospective, local smoke, and reviewed full Colab evidence](task02b/README.md) |
| Task 02C | [local checks and reviewed full Colab NO-GO evidence](task02c/README.md) |
| Task 03A | [reviewed full Colab NO-GO evidence](task03a/README.md) |
| Task 04A | [density-positive but decision-diagnostic INVALID; full study disabled](task04a/README.md) |
| Task 04A-E | [valid q=2 oracle but CPU smoke LEARNING_NO_GO](task04ae/README.md) |
| Task 05A | [implementation complete; full measured-data evidence pending](../tasks/task05a/SUMMARY.md) |

Each result package records its own runtime/configuration metadata. Do not treat a
smoke run as a multi-seed performance benchmark. Interpret stage results through the
[current research roadmap](../docs/research/ROADMAP.md).

Automatic runners never write here by default. They write to ignored
`artifacts/<task>/<profile>/`; evidence enters `results/` only after a human reviews
the downloaded/local output and deliberately copies a compact subset.
