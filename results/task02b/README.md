# Task 02B results

This directory separates evidence that required no new NUTS from the deliberately
small local teacher smoke test.

- [`retrospective/`](retrospective/) recomputes decision regret at all 18 published
  Task 02A checkpoints from saved fresh/reused EI curves and checkpoint diagnostics.
- [`smoke/`](smoke/) contains the D=4, two-checkpoint, 32-particle CPU smoke spectra,
  coreset metrics, exact joint-target checks, figures, and compact signatures.

The smoke NUTS chains are implementation checks and are not scientific evidence for
Task 02C. The full D=10, three-seed, 18-checkpoint extraction is intentionally deferred
to the [Task 02B Colab procedure](../../tasks/task02b/COLAB.md); its large signatures remain under
ignored `artifacts/` until reviewed.
