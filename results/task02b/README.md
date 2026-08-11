# Task 02B results

This directory separates evidence that required no new NUTS, the deliberately small
local teacher smoke test, and the reviewed full Colab study.

- [`retrospective/`](retrospective/) recomputes decision regret at all 18 published
  Task 02A checkpoints from saved fresh/reused EI curves and checkpoint diagnostics.
- [`smoke/`](smoke/) contains the D=4, two-checkpoint, 32-particle CPU smoke spectra,
  coreset metrics, exact joint-target checks, figures, and compact signatures.
- [`full/`](full/) contains the D=10, three-seed, 18-checkpoint quantitative summary,
  aggregate tables/configuration, figures, environment manifest, integrity audit, and
  checksums for the excluded raw signature matrices.

The smoke NUTS chains remain implementation checks. The full evidence passes the
prespecified gate for designing a bounded Task 02C experiment, but it is not evidence
that joint-energy transport works. Raw full signatures remain ignored; the
[Task 02B Colab procedure](../../tasks/task02b/COLAB.md) is retained for reproduction.
