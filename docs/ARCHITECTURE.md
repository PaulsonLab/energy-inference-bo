# Architecture and contribution map

The repository separates reusable mathematical components from bounded task drivers.
Task documents explain *why* a stage exists; source modules implement the reusable
pieces; results preserve reviewed evidence.

| Location | Responsibility | Add new work here when… |
| --- | --- | --- |
| `src/energy_bo/oracle/` | scalar oracle distributions, residual energy, and augmented-inference checks | validating a mathematical identity without a GP structural posterior |
| `src/energy_bo/gp/` | small exact-GP q=1 validation helpers | extending the Task 01 exact-GP sanity layer |
| `src/energy_bo/structural/` | fixed SAAS particles, exact GP cache, preprocessing, and NUTS references | working with structural posterior state or exact sequential GP calculations |
| `src/energy_bo/decision/` | EI signatures, decision metrics, coresets, and joint-target identities | adding a reusable decision-space diagnostic |
| `src/energy_bo/transport/` | exact unconstrained SAAS energy, stable JAX LogEI, teacher preflight, and SVGD primitives | working on the bounded Task 02C structural transport experiment |
| `src/energy_bo/experiments/` | reproducible task orchestration and `run_task*.py` CLIs | combining reusable modules into a bounded experiment |
| `tests/` | mathematical identity tests mirroring the source domains | adding a correctness test before an experiment |

## Evidence and execution

- `results/` contains compact, reviewed evidence that is safe to track.
- `artifacts/<task>/<profile>/` is ignored and holds every automatic run output,
  including reports, plots, raw signature matrices, and environment manifests.
- `notebooks/` contains one thin, guarded Colab driver per runnable task. Reusable
  logic belongs in `src/`; notebooks call the same experiment CLIs used locally.
- `tasks/<task>/` is the canonical specification and reporting hierarchy. Each folder
  begins with `README.md` and contains `SPEC.md`, optional `MATH.md`, `SUMMARY.md`, and
  `COLAB.md`. Use `tasks/ACTIVE_TASK.md` to begin work.
- `docs/research/ROADMAP.md` is the current human overview; it links to, but never
  replaces, task mathematics and specifications.

The promotion boundary is intentional: runners do not overwrite reviewed evidence.
After a local or Colab run, inspect `artifacts/<task>/<profile>/`, then copy only the
small summary, tables, and figures needed for audit into
`results/<task>/<profile>/`. Raw chains/signatures remain outside Git.

Do not reorganize a mathematical module solely to match a task number. The current
domain structure is intentional: later tasks may reuse `structural` or `decision`
components without inheriting a prior task's experiment driver.

Before handing off documentation changes, run
`python scripts/check_markdown_links.py` from the repository root to verify every
repository-relative Markdown link without accessing the network.
