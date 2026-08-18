from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from causal_policy_bo.model import TwoRegimeConfig, TwoRegimeModel
from causal_policy_bo.reference import ReferenceLevel, solve_reference_level


def test_bayesian_regime_update_matches_normalized_bayes_rule() -> None:
    model = TwoRegimeModel(TwoRegimeConfig(delta=0.2, noise_std=0.15))
    prior = 0.37
    x = 0.48
    y = 0.11
    updated_log_odds = model.posterior_log_odds_numpy(
        np.log(prior / (1.0 - prior)), x, y
    )
    mu0, mu1 = model.means_numpy(x)
    likelihood0 = norm.pdf(y, loc=mu0, scale=model.config.noise_std)
    likelihood1 = norm.pdf(y, loc=mu1, scale=model.config.noise_std)
    expected = prior * likelihood1 / (
        (1.0 - prior) * likelihood0 + prior * likelihood1
    )
    assert np.isclose(model.probability_m1(updated_log_odds), expected, atol=1e-13)


def test_regime_update_is_finite_for_extreme_evidence() -> None:
    model = TwoRegimeModel(TwoRegimeConfig(delta=0.2, noise_std=0.1))
    updated = model.posterior_log_odds_numpy(0.0, np.array([0.5]), np.array([-5.0]))
    assert np.isfinite(updated).all()


def test_two_step_root_world_and_sequential_density_are_equal() -> None:
    model = TwoRegimeModel(TwoRegimeConfig(delta=0.25, noise_std=0.14))
    prior = model.config.prior_m1
    x0, y0 = 0.46, 0.08
    log_odds1 = model.posterior_log_odds_numpy(model.config.prior_log_odds, x0, y0)
    probability1 = float(model.probability_m1(log_odds1))
    x1 = 0.25 if probability1 < 0.5 else 0.75
    y1 = 0.91
    mu00, mu01 = model.means_numpy(x0)
    mu10, mu11 = model.means_numpy(x1)

    root_density = (1.0 - prior) * norm.pdf(
        y0, mu00, model.config.noise_std
    ) * norm.pdf(y1, mu10, model.config.noise_std) + prior * norm.pdf(
        y0, mu01, model.config.noise_std
    ) * norm.pdf(y1, mu11, model.config.noise_std)
    predictive0 = (1.0 - prior) * norm.pdf(
        y0, mu00, model.config.noise_std
    ) + prior * norm.pdf(y0, mu01, model.config.noise_std)
    predictive1 = (1.0 - probability1) * norm.pdf(
        y1, mu10, model.config.noise_std
    ) + probability1 * norm.pdf(y1, mu11, model.config.noise_std)
    assert np.isclose(root_density, predictive0 * predictive1, rtol=1e-12)


def test_tiny_reference_has_ordered_values_and_valid_actions() -> None:
    config = TwoRegimeConfig(delta=0.15, noise_std=0.1)
    solution = solve_reference_level(
        config, ReferenceLevel("test", 41, 7, 17, 21, state_batch=8)
    )
    values = [float(row["value"]) for row in solution.rows]
    actions = [float(row["first_action"]) for row in solution.rows]
    assert values[0] <= values[1] <= values[2]
    assert all(0.0 <= action <= 1.0 for action in actions)


def test_committed_negative_evidence_is_complete_and_finite() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    outputs = repository_root / "experiments" / "policy_kill" / "outputs"
    summary = json.loads((outputs / "summary.json").read_text())
    frozen = json.loads((outputs / "frozen_config.json").read_text())
    scan = pd.read_csv(outputs / "design_scan.csv")
    reference = pd.read_csv(outputs / "exact_reference.csv")

    assert summary["classification"] == "POLICY_KILL_NEGATIVE_REVIEW_REQUIRED"
    assert summary["candidate_count"] == 20
    assert summary["passing_candidate_count"] == 0
    assert summary["reference_convergence"]["passes"] is True
    assert frozen["selection_status"] == "FAILED_NO_CANDIDATE"
    assert frozen["selected_config"] is None
    assert len(scan) == 60
    assert len(reference) == 9
    assert scan.select_dtypes("number").notna().all().all()
    assert reference.select_dtypes("number").notna().all().all()

    for stem in (
        "figure1_planning_opportunity",
        "figure2_design_scan_failure",
        "figure3_reference_convergence",
    ):
        assert (outputs / f"{stem}.png").stat().st_size > 10_000
        assert (outputs / f"{stem}.pdf").stat().st_size > 5_000


def test_policy_kill_notebook_has_required_sections() -> None:
    notebook_path = Path(__file__).resolve().parents[1] / "notebooks" / "policy_kill.ipynb"
    notebook = json.loads(notebook_path.read_text())
    text = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    for section in (
        "# 1. Question",
        "# 2. Why this model system",
        "# 3. Frozen model",
        "# 4. Exact DP reference",
        "# 5. Causal policy",
        "# 6. Plain pathwise optimization",
        "# 7. Causal KL transport",
        "# 8. Results",
        "# 9. Figures",
        "# 10. Kill decision",
        "# 11. Next human decision",
    ):
        assert section in text
