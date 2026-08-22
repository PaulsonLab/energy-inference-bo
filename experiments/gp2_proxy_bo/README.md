# Gp2 E3 structural preflight

This directory implements only the frozen, target-blind structural preflight
specified in
[`../../project/E3_GP2_STRUCTURAL_PREFLIGHT_HANDOFF.md`](../../project/E3_GP2_STRUCTURAL_PREFLIGHT_HANDOFF.md).
It calibrates the fixed Sort1/Sort8 proxy model from the historical source,
constructs test-only candidate-local proxy factors and the frozen Hamming
8-NN graph, verifies the model-specific Menz specialization, and evaluates the
all-unordered-action-pair structural-sparsity gate.

It does not implement Bayesian optimization, Laplace or importance-sampling
inference, adaptive conditioning, or the later scalar-only versus full-proxy
comparison. Held-out `SH_Average_bc` magnitudes are unavailable to graph,
factor, theory, and structural interfaces.

After the implementation and tests are committed as a clean preregistration,
run the real preflight exactly once:

```bash
python experiments/gp2_proxy_bo/run_structural_preflight.py
```

If the repository uses its project virtual environment directly, the
equivalent command is:

```bash
.venv/bin/python experiments/gp2_proxy_bo/run_structural_preflight.py
```

The runner verifies pinned source hashes and requires a clean working tree. It
creates the immutable output directory
`outputs/structural_preflight/` and refuses to overwrite it. Any verdict other
than `PASS_STRUCTURAL_PREFLIGHT` abandons Gp2 as E3; no tuning or rescue is
permitted in this task.
