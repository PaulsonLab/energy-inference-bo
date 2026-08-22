from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import conditioned_bo.gp2_proxy_bo as gp2


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    REPOSITORY_ROOT
    / "experiments/gp2_proxy_bo/configs/structural_preflight.json"
)


def _base4(value: int, width: int = 4) -> str:
    alphabet = "ACGT"
    digits = []
    for _ in range(width):
        digits.append(alphabet[value % 4])
        value //= 4
    return "".join(reversed(digits))


def _historical_frame(count: int = 120) -> pd.DataFrame:
    index = np.arange(count, dtype=float)
    sort1 = np.sin(index / 9.0) + index / 100.0
    sort8 = np.cos(index / 11.0) - index / 150.0
    target = 0.7 * sort1 - 0.25 * sort8 + 0.12 * np.sin(index / 3.0)
    return pd.DataFrame(
        {
            "Stop": False,
            "Paratope": ["H" + _base4(value) for value in range(count)],
            "SH_Average_bc": target,
            "Sort1_mean_score": sort1,
            "Sort8_mean_score": sort8,
        }
    )


def _prospective_frame(count: int = 12) -> pd.DataFrame:
    index = np.arange(count, dtype=float)
    return pd.DataFrame(
        {
            "Stop": False,
            "Paratope": ["T" + _base4(value) for value in range(count)],
            "SH_Average_bc": 10.0 + index,
            "Sort1_mean_score": index / 5.0,
            "Sort8_mean_score": -index / 7.0,
        }
    )


def test_frozen_config_and_source_roles_are_exact() -> None:
    config = gp2.load_structural_preflight_config(CONFIG_PATH)
    assert config["source_roles"] == {
        "historical_calibration": gp2.HISTORICAL_SOURCE_PATH,
        "prospective_actions": gp2.PROSPECTIVE_SOURCE_PATH,
        "union_sources_for_actions": False,
    }
    assert [item["sha256"] for item in config["external_data"]["files"]] == [
        "4b2c1697798764eb6c76c9297c77d5e36b548cb0a6bf44b376edfea065beb52f",
        "eab318cf049552706ad9feb28e2cbf144b457b618135652a5bea2af7aee0a9c2",
    ]
    assert config["columns"]["proxies"] == [
        "Sort1_mean_score",
        "Sort8_mean_score",
    ]


def test_source_roles_and_heldout_target_magnitudes_are_separated() -> None:
    historical = gp2.prepare_historical_rows(_historical_frame())
    prospective = _prospective_frame()
    altered = prospective.copy()
    altered[gp2.TARGET_COLUMN] = np.linspace(-1e6, 1e6, len(altered))
    first, first_summary = gp2.prepare_prospective_actions(
        prospective,
        historical_paratopes=historical.paratopes,
        historical_sequence_length=historical.sequence_length,
    )
    second, _ = gp2.prepare_prospective_actions(
        altered,
        historical_paratopes=historical.paratopes,
        historical_sequence_length=historical.sequence_length,
    )
    pd.testing.assert_frame_equal(first, second)
    assert gp2.TARGET_COLUMN not in first.columns
    assert first_summary["source_path"] == gp2.PROSPECTIVE_SOURCE_PATH
    assert historical.summary["source_path"] == gp2.HISTORICAL_SOURCE_PATH
    for function in (
        gp2.hamming_knn_graph,
        gp2.build_proxy_factor_bank,
        gp2.construct_and_verify_theory,
        gp2.compute_structural_sparsity,
    ):
        assert "target" not in inspect.signature(function).parameters


def test_missing_proxy_removes_factor_but_never_action() -> None:
    historical = gp2.prepare_historical_rows(_historical_frame())
    calibration = gp2.fit_proxy_calibration(historical)
    prospective = _prospective_frame(4)
    prospective.loc[1, "Sort1_mean_score"] = np.nan
    prospective.loc[2, "Sort8_mean_score"] = np.inf
    actions, _ = gp2.prepare_prospective_actions(
        prospective,
        historical_paratopes=historical.paratopes,
        historical_sequence_length=historical.sequence_length,
    )
    factors = gp2.build_proxy_factor_bank(actions, calibration.pipeline)
    assert len(actions) == 4
    assert factors["action_index"].tolist() == [0, 3]


def test_historical_scaling_and_calibration_are_deterministic() -> None:
    historical = gp2.prepare_historical_rows(_historical_frame())
    first = gp2.fit_proxy_calibration(historical)
    second = gp2.fit_proxy_calibration(historical)
    targets = historical.rows[gp2.TARGET_COLUMN].to_numpy(dtype=float)
    assert first.mu_hist == float(np.mean(targets))
    assert first.s_hist == float(np.std(targets, ddof=1))
    assert first.historical_target_scale_count == 120
    assert first.calibration_count == 120
    assert first.oof_rmse == second.oof_rmse
    assert first.s_proxy == second.s_proxy
    features = historical.rows.loc[:, gp2.PROXY_COLUMNS].to_numpy(dtype=float)
    np.testing.assert_array_equal(
        first.pipeline.predict(features), second.pipeline.predict(features)
    )
    assert first.pipeline.named_steps["ridge"].alpha == 1.0


def test_duplicate_handling_is_deterministic_and_ambiguous_values_stop() -> None:
    historical_frame = _historical_frame()
    duplicate = pd.concat(
        [historical_frame, historical_frame.iloc[[0]]], ignore_index=True
    )
    historical = gp2.prepare_historical_rows(duplicate)
    assert len(historical.rows) == len(historical_frame)
    assert historical.summary["duplicate_row_count"] == 1

    bad_historical = duplicate.copy()
    bad_historical.loc[len(bad_historical) - 1, "Sort1_mean_score"] += 1e-4
    with pytest.raises(gp2.PreprocessingAmbiguity, match="Sort1_mean_score"):
        gp2.prepare_historical_rows(bad_historical)

    prospective = _prospective_frame()
    prospective_duplicate = pd.concat(
        [prospective, prospective.iloc[[0]]], ignore_index=True
    )
    actions, summary = gp2.prepare_prospective_actions(
        prospective_duplicate,
        historical_paratopes=historical.paratopes,
        historical_sequence_length=historical.sequence_length,
    )
    assert len(actions) == len(prospective)
    assert summary["duplicate_row_count"] == 1
    bad_prospective = prospective_duplicate.copy()
    bad_prospective.loc[len(bad_prospective) - 1, gp2.TARGET_COLUMN] += 1e-4
    with pytest.raises(gp2.PreprocessingAmbiguity, match=gp2.TARGET_COLUMN):
        gp2.prepare_prospective_actions(
            bad_prospective,
            historical_paratopes=historical.paratopes,
            historical_sequence_length=historical.sequence_length,
        )


def test_historical_prospective_overlap_is_preprocessing_ambiguity() -> None:
    historical = gp2.prepare_historical_rows(_historical_frame())
    prospective = _prospective_frame()
    prospective.loc[0, "Paratope"] = next(iter(historical.paratopes))
    with pytest.raises(gp2.PreprocessingAmbiguity, match="overlap"):
        gp2.prepare_prospective_actions(
            prospective,
            historical_paratopes=historical.paratopes,
            historical_sequence_length=historical.sequence_length,
        )


def test_hamming_graph_is_deterministic_with_lexicographic_ties() -> None:
    sequences = ["AAA", "AAB", "AAC", "ABA", "BAA"]
    first = gp2.hamming_knn_graph(sequences, k=1)
    second = gp2.hamming_knn_graph(sequences, k=1)
    np.testing.assert_array_equal(first.edges, second.edges)
    assert (0, 1) in {tuple(edge) for edge in first.edges.tolist()}


def test_component_rule_retains_only_a_large_deterministic_component() -> None:
    actions = pd.DataFrame(
        {
            "action_index": np.arange(5),
            "Paratope": ["A", "B", "C", "D", "E"],
            "Sort1_mean_score": 0.0,
            "Sort8_mean_score": 0.0,
        }
    )
    graph = gp2.HammingGraph(
        edges=np.asarray([[0, 1], [1, 2], [3, 4]], dtype=int),
        component_sizes=(3, 2),
        largest_component_indices=np.asarray([0, 1, 2], dtype=int),
        largest_component_fraction=0.6,
    )
    retained, edges, restricted = gp2.apply_component_rule(
        actions, graph, minimum_largest_fraction=0.6
    )
    assert restricted
    assert retained["Paratope"].tolist() == ["A", "B", "C"]
    np.testing.assert_array_equal(edges, np.asarray([[0, 1], [1, 2]]))
    with pytest.raises(gp2.PreprocessingInvalid):
        gp2.apply_component_rule(actions, graph, minimum_largest_fraction=0.9)


def test_normalized_laplacian_precision_matches_direct_fixture() -> None:
    edges = np.asarray([[0, 1], [1, 2]], dtype=int)
    q0 = gp2.normalized_laplacian_precision(3, edges)
    adjacency = np.asarray(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]
    )
    inverse_sqrt = np.diag(1.0 / np.sqrt(adjacency.sum(axis=1)))
    expected = 2.0 * np.eye(3) - inverse_sqrt @ adjacency @ inverse_sqrt
    np.testing.assert_allclose(q0, expected, atol=1e-14)
    assert np.linalg.eigvalsh(q0).min() > 0.0


def test_local_logcosh_factor_gradient_and_hessian_finite_differences() -> None:
    z = 0.37
    mu_proxy = -0.21
    s_proxy = 0.43
    step = 1e-5
    finite_gradient = (
        gp2.local_proxy_factor_energy(z + step, mu_proxy, s_proxy)
        - gp2.local_proxy_factor_energy(z - step, mu_proxy, s_proxy)
    ) / (2.0 * step)
    analytic_gradient = gp2.local_proxy_factor_gradient(z, mu_proxy, s_proxy)
    np.testing.assert_allclose(analytic_gradient, finite_gradient, atol=1e-10)
    finite_hessian = (
        gp2.local_proxy_factor_gradient(z + step, mu_proxy, s_proxy)
        - gp2.local_proxy_factor_gradient(z - step, mu_proxy, s_proxy)
    ) / (2.0 * step)
    analytic_hessian = gp2.local_proxy_factor_hessian(z, mu_proxy, s_proxy)
    np.testing.assert_allclose(analytic_hessian, finite_hessian, atol=1e-9)
    assert analytic_hessian >= 0.0


def test_menz_matrix_equals_q0_and_k0_is_nonnegative() -> None:
    edges = np.asarray([[0, 1], [1, 2]], dtype=int)
    theory = gp2.construct_and_verify_theory(
        3, edges, proxy_means=np.zeros(3), s_proxy=0.7
    )
    np.testing.assert_allclose(theory.a, theory.q0, atol=1e-14)
    np.testing.assert_allclose(theory.c, theory.k0, atol=1e-14)
    np.testing.assert_allclose(theory.q0 @ theory.k0, np.eye(3), atol=1e-14)
    assert theory.k0.min() >= -1e-10


def test_graph_distance_covariance_bound_holds_for_fixture() -> None:
    edges = np.asarray([[0, 1], [1, 2], [2, 3]], dtype=int)
    theory = gp2.construct_and_verify_theory(
        4, edges, proxy_means=np.zeros(4), s_proxy=0.9
    )
    bound = np.exp2(-theory.graph_distances)
    assert np.max(theory.k0 - bound) <= 1e-10
    assert theory.graph_distances[0, 3] == 3


def test_hand_checkable_structural_tail_and_pairwise_fixture() -> None:
    contributions = np.asarray([0.04, 0.03, 0.02, 0.01])
    assert gp2.structural_tail_count(contributions, epsilon=0.05) == 2
    assert gp2.structural_tail_count(contributions, epsilon=0.10) == 0

    k0 = np.diag([0.04, 0.04, 0.04])
    distances = np.asarray(
        [[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]]
    )
    result = gp2.compute_structural_sparsity(
        k0,
        np.asarray([0, 1, 2]),
        1.0,
        distances,
        epsilon_struct=0.05,
    )
    assert result.pairwise["M_0_05"].tolist() == [1, 1, 1]
    np.testing.assert_allclose(result.pairwise["R_0_05"], 1.0 / 3.0)
    assert result.verdict == "PASS_STRUCTURAL_PREFLIGHT"


def test_immutable_output_directory_refuses_overwrite(tmp_path: Path) -> None:
    destination = tmp_path / "structural_preflight"
    gp2.create_immutable_output_directory(destination)
    with pytest.raises(FileExistsError):
        gp2.create_immutable_output_directory(destination)


def test_handoff_copy_is_byte_identical_to_attachment() -> None:
    repository_copy = (
        REPOSITORY_ROOT
        / "project/archive/e3/E3_GP2_STRUCTURAL_PREFLIGHT_HANDOFF.md"
    )
    attachment = Path("/Users/joelpaulson/Downloads/E3_GP2_STRUCTURAL_PREFLIGHT_HANDOFF.md")
    if attachment.exists():
        assert repository_copy.read_bytes() == attachment.read_bytes()
