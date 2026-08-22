# Gp2 E3 P1 gate

Status: **CLOSED — `PREPROCESSING_INVALID`**. Only 169 candidates survived,
below the frozen minimum of 250, so no smoke or scientific P1 comparison ran.
Gp2 is abandoned as E3.

This directory implements the prospective P1-only gate specified in
[`../../project/archive/e3/E3_GP2_P1_GATE_HANDOFF.md`](../../project/archive/e3/E3_GP2_P1_GATE_HANDOFF.md).
It compares only ordinary scalar graph-Gaussian EI with fully
preference-conditioned EI. It does not implement adaptive factor screening or
the final E3 baseline suite.

The runner downloads and verifies the two files from the pinned DevRep commit
into the git-ignored `data/external/gp2_devrep/` cache. Every output records the
external paths and SHA-256 hashes, exact config hash, and current Git SHA.

The historical target-blind preprocessing command was:

```bash
uv run python experiments/gp2_preference_bo/run_p1_gate.py \
  --config experiments/gp2_preference_bo/configs/p1_gate.json \
  --mode preflight
```

Only if that reports `PREPROCESSING_VALID`, run the mechanical smoke:

```bash
uv run python experiments/gp2_preference_bo/run_p1_gate.py \
  --config experiments/gp2_preference_bo/configs/p1_gate.json \
  --mode smoke
```

The scientific command must be run separately from a clean preregistration
commit:

```bash
uv run python experiments/gp2_preference_bo/run_p1_gate.py \
  --config experiments/gp2_preference_bo/configs/p1_gate.json \
  --mode scientific
```

All three canonical output directories are immutable. The runner refuses to
overwrite an existing preflight, smoke, or scientific result directory. ESS
and split-half discrepancies are empirical numerical reliability diagnostics,
not rigorous certificates.
