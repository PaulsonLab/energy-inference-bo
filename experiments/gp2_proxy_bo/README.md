# Gp2 E3 structural preflight

Status: **CLOSED — `PREPROCESSING_INVALID / GP2_ABANDONED`**. The frozen run
stopped at 47 proxy factors, below the minimum of 75. No theory matrices,
structural distribution, inference, or BO were run.

This directory implements only the frozen, target-blind structural preflight
specified in
[`../../project/archive/e3/E3_GP2_STRUCTURAL_PREFLIGHT_HANDOFF.md`](../../project/archive/e3/E3_GP2_STRUCTURAL_PREFLIGHT_HANDOFF.md).
It calibrates the fixed Sort1/Sort8 proxy model from the historical source,
constructs test-only candidate-local proxy factors and the frozen Hamming
8-NN graph, verifies the model-specific Menz specialization, and evaluates the
all-unordered-action-pair structural-sparsity gate.

It does not implement Bayesian optimization, Laplace or importance-sampling
inference, adaptive conditioning, or the later scalar-only versus full-proxy
comparison. Held-out `SH_Average_bc` magnitudes are unavailable to graph,
factor, theory, and structural interfaces.

The real preflight was run exactly once from its clean preregistration with:

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
