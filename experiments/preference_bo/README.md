# E3 — Preference-Conditioned Sequential BO

The minimal phenomenon pilot is preregistered in
[`../../project/E3_PREFERENCE_BO_PILOT_HANDOFF.md`](../../project/E3_PREFERENCE_BO_PILOT_HANDOFF.md)
and frozen in [`configs/minimal_pilot.json`](configs/minimal_pilot.json). It runs
exactly scalar-only GP-EI, full preference-informed GP-EI, and adaptive
preference-informed GP-EI. The broader final-E3 baseline suite remains deferred.

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
