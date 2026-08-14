# Task registry

Task documents are the canonical record of each research stage. Each task has one
self-contained folder; task-specific files are not duplicated at the repository root.

| Stage | Status | Canonical material |
| --- | --- | --- |
| Task 01 — oracle shape and q=1 identities | Complete | [task entry point](task01/README.md) |
| Task 02A — fixed-support SAAS reuse | Complete | [task entry point](task02a/README.md) |
| Task 02B — decision-space compression | Complete; Task 02C gate passed | [task entry point](task02b/README.md) |
| Task 02C — decision-tilted structural SVGD | Complete; tested configuration NO-GO | [task entry point](task02c/README.md) |
| Task 03A — fast sparse reference + residual predictive energy | Complete; Task 03B gate NO-GO | [task entry point](task03a/README.md) |
| Task 04A — local conditional energy process, oracle geometry | Decision diagnostic INVALID; full study disabled | [task entry point](task04a/README.md) |

## Document convention

| Document | Answers |
| --- | --- |
| Global invariants | What mathematics and scope rules every task must obey (`AGENTS.md`, `MATH_AND_SCOPE.md`) |
| `SPEC.md` | What this task may implement, measure, and test |
| Task math note | Why the task's identities and diagnostics are mathematically valid |
| `SUMMARY.md` and `results/` | What was actually observed and whether the next gate is justified |
| `COLAB.md` | How to reproduce the task's approved larger run |
| `README.md` | Where to start, what to run locally, and what to do with downloaded outputs |

## Working on a task

1. Read [AGENTS.md](../AGENTS.md) and [MATH_AND_SCOPE.md](../MATH_AND_SCOPE.md).
2. Read [ACTIVE_TASK.md](ACTIVE_TASK.md), then the canonical task material it names.
3. Keep reusable mathematics in `src/energy_bo/`, task orchestration in
   `src/energy_bo/experiments/`, automatic outputs in `artifacts/<task>/<profile>/`,
   and reviewed compact evidence in `results/<task>/<profile>/`.
4. Do not advance a stage automatically; each task summary states the evidence gate.

Task 04A's withheld-seed decision diagnostic replicated density learning but was
invalid for decision comparison because the frozen oracle supplied no qualifying
natural pairs. The full Colab study is disabled and Task 04B is not authorized.

The [current research roadmap](../docs/research/ROADMAP.md) is the human overview.
[Historical documents](../docs/research/history/README.md) never override the active
task specification.
