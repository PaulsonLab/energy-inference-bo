import json
from pathlib import Path

import numpy as np
import pandas as pd

from decision_tilt.welded_beam import canonical_hash


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "welded_beam_shift"
OUTPUT = EXPERIMENT / "outputs"
NOTEBOOK = ROOT / "notebooks" / "welded_beam_shift.ipynb"


def test_completed_welded_beam_evidence_is_complete_and_finite() -> None:
    expected_rows = {
        "candidate_metrics.csv": 49152,
        "qmc_reliability.csv": 960,
        "qmc_summary.csv": 15,
        "gp_fits.csv": 21,
        "state_summary.csv": 3,
    }
    for filename, count in expected_rows.items():
        frame = pd.read_csv(OUTPUT / filename)
        assert len(frame) == count
        numeric = frame.select_dtypes(include=[np.number]).to_numpy()
        assert np.isfinite(numeric).all()
    fits = pd.read_csv(OUTPUT / "gp_fits.csv")
    assert fits.converged.all()
    states = pd.read_csv(OUTPUT / "state_summary.csv")
    assert float(states.moment_log_error.max()) < 1e-9

    protocol = json.loads((EXPERIMENT / "config.json").read_text())
    copied = json.loads((OUTPUT / "frozen_config.json").read_text())
    gate = json.loads((OUTPUT / "gate_result.json").read_text())
    assert protocol == copied
    assert gate["protocol_hash"] == canonical_hash(protocol)
    assert gate["valid"] is True
    assert gate["status"] == "WELDED_BEAM_SHIFT_NEGATIVE_REVIEW_REQUIRED"
    for stem in (
        "figure_a_shift_vs_quality",
        "figure_b_qmc_reliability",
        "figure_c_mechanism",
    ):
        for suffix in ("png", "pdf", "svg"):
            assert (OUTPUT / f"{stem}.{suffix}").stat().st_size > 1000


def test_welded_beam_notebook_is_executed_and_ordered() -> None:
    notebook = json.loads(NOTEBOOK.read_text())
    assert notebook["nbformat"] == 4
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert code_cells
    assert all(cell["execution_count"] is not None for cell in code_cells)
    assert not any(
        output.get("output_type") == "error"
        for cell in code_cells
        for output in cell.get("outputs", [])
    )
    markdown = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "markdown"
    )
    headings = [
        "## 1. Question",
        "## 2. Why Welded Beam",
        "## 3. Frozen protocol",
        "## 4. Exact mathematics",
        "## 5. Three frozen states",
        "## 6. Decision-shift results",
        "## 7. QMC ranking results",
        "## 8. Main figures",
        "## 9. Interpretation",
        "## 10. GO / NO-GO",
        "## 11. Next human decision",
    ]
    positions = [markdown.index(heading) for heading in headings]
    assert positions == sorted(positions)
