# E3 — Preference-Conditioned Sequential BO

Status: **CLOSED — both synthetic pilots `FAIL-P2`**. Their preference-value
and performance gates passed, but their preregistered sparsity gates failed.
Do not tune or rerun them; the next realistic-E3 candidate is tracked separately
in [`project/EXPERIMENTS.md`](../../project/EXPERIMENTS.md#e3--realistic-non-pde-bo-candidate).

The minimal phenomenon pilot is preregistered in
[`../../project/archive/e3/E3_PREFERENCE_BO_PILOT_HANDOFF.md`](../../project/archive/e3/E3_PREFERENCE_BO_PILOT_HANDOFF.md)
and frozen in [`configs/minimal_pilot.json`](configs/minimal_pilot.json). It runs
exactly scalar-only GP-EI, full preference-informed GP-EI, and adaptive
preference-informed GP-EI. This is a preserved closed protocol, not the active
E3 plan.

Run the reduced numerical smoke profile first:

```bash
.venv/bin/python experiments/preference_bo/run_minimal_pilot.py --smoke
```

After its mechanical checks pass, run the unchanged 12-seed pilot:

```bash
.venv/bin/python experiments/preference_bo/run_minimal_pilot.py
```

Output directories are immutable: the runner refuses to overwrite an existing
smoke or pilot directory. Empirical importance-sampling allowances and ESS are
diagnostics, not rigorous finite-sample certificates.

## Redundant-bank follow-up

The separate 24-edge follow-up is preregistered in
[`../../project/archive/e3/E3_PREFERENCE_BO_REDUNDANT_BANK_PILOT_HANDOFF.md`](../../project/archive/e3/E3_PREFERENCE_BO_REDUNDANT_BANK_PILOT_HANDOFF.md)
and frozen in
[`configs/redundant_bank_pilot.json`](configs/redundant_bank_pilot.json). It
retains the same BO problem and exactly the same three methods, but uses 24
unique overlapping preference edges and a 17-scalar-block influence matrix.
It does not alter the first pilot's `FAIL-P2` result or `0.65` sparsity gate.

Run its mechanical smoke profile and then its unchanged 12-seed pilot with:

```bash
.venv/bin/python experiments/preference_bo/run_redundant_bank_pilot.py --smoke
.venv/bin/python experiments/preference_bo/run_redundant_bank_pilot.py
```

Its separate prospective sparsity threshold is `0.80`. Final outputs are in
[`outputs/redundant_bank_pilot/`](outputs/redundant_bank_pilot/); incomplete
intermediate output attempts are retained alongside it and are not used for
the final gates.
