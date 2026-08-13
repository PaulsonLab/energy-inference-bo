# Notebooks

Notebooks are thin execution handoffs, not the implementation home. Put reusable
mathematics, data preparation, and diagnostics in `src/energy_bo/`; keep notebooks to
environment setup, explicit run configuration, and artifact download.

| Task | Notebook | Default hardware | Full run guard |
| --- | --- | --- | --- |
| Task 01 | [oracle/q=1 validation](task01_colab.ipynb) | CPU | `RUN_FULL = False` |
| Task 02A | [SAAS reuse diagnostic](task02a_colab.ipynb) | CPU; GPU optional for NUTS | `RUN_FULL = False` |
| Task 02B | [decision-space signatures](task02b_colab.ipynb) | NVIDIA GPU preferred | `RUN_FULL = False` |
| Task 02C | [decision-tilted SVGD](task02c_colab.ipynb) | NVIDIA GPU preferred; completed study, reproduction only | `RUN_PREFLIGHT = False`, `RUN_FULL = False` |
| Task 03A | [MAP-SAAS plus residual predictive energy](task03a_colab.ipynb) | NVIDIA GPU preferred; MAP fitting device profiled | `RUN_FULL = False` |

Each notebook follows the same cell order: configure → validate Python → clone an
explicit Git revision → install → validate backend and tests → opt into the full run
→ write a manifest → download one ZIP. The final Markdown cell says exactly which
files may be promoted to `results/`; downloading never modifies GitHub.

Task 03A's notebook deliberately installs its wheels into an isolated `/content`
package directory and disables global site packages for every scientific subprocess.
It does not rely on Colab `venv`/`ensurepip`. Do not restart between its setup and
preflight cells or install repair packages into Colab's global Python.

See the [task registry](../tasks/README.md) for the current task and terminal/Colab
procedures.
