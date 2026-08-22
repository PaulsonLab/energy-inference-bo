from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import conditioned_bo.gp2_preference_bo as gp2
from conditioned_bo.preference_influence import (
    logistic_preference_energy,
    logistic_preference_gradient,
    logistic_preference_hessian,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPOSITORY_ROOT / "experiments/gp2_preference_bo/configs/p1_gate.json"


def _row(paratope: str = "AAAA", **updates):
    row = {
        "Paratope": paratope,
        "Stop": False,
        "SH_Average_bc": 1.5,
        "Sort1_1_score": 1.0,
        "Sort1_2_score": 2.0,
        "Sort1_3_score": 3.0,
        "Sort8_1_score": 4.0,
        "Sort8_2_score": 5.0,
        "Sort8_3_score": 6.0,
    }
    row.update(updates)
    return row


def _tiny_config() -> dict:
    config = json.loads(CONFIG_PATH.read_text())
    config["inference"] = {
        **config["inference"],
        "minimum_ess_fraction": 0.0,
        "maximum_split_half_ei_discrepancy": 10.0,
        "draw_chunk_size": 16,
        "factor_chunk_size": 4,
    }
    return config


def _tiny_prepared() -> gp2.PreparedGp2Data:
    sequences = [f"{value:04b}" for value in range(12)]
    candidates = pd.DataFrame(
        {
            "candidate_index": np.arange(12),
            "Paratope": sequences,
            "SH_Average_bc": np.linspace(-1.0, 1.2, 12),
        }
    )
    edges = np.asarray(
        sorted({tuple(sorted((index, (index + 1) % 12))) for index in range(12)}),
        dtype=int,
    )
    factors = pd.DataFrame(
        [
            {
                "factor_index": 0,
                "left_action_index": 0,
                "right_action_index": 1,
                "left_paratope": sequences[0],
                "right_paratope": sequences[1],
                "assay": "Sort1",
                "preference_sign": -1,
                "provenance": "Sort1_strict_2_of_3_replicate_vote",
            }
        ]
    )
    return gp2.PreparedGp2Data(
        candidates=candidates,
        edges=edges,
        factors=factors,
        preprocessing_summary={"verdict": "PREPROCESSING_VALID"},
        source_provenance=(),
    )


def test_frozen_config_and_pinned_provenance_are_exact() -> None:
    config = gp2.load_gate_config(CONFIG_PATH)
    files = config["external_data"]["files"]
    assert config["external_data"]["commit"] == "e05023a8abe7be6c2e22f42d523b20bd76cd8da5"
    assert [item["path"] for item in files] == [
        "datasets/assay_to_yield_training_sequences.csv",
        "datasets/test_sequences.csv",
    ]
    assert [item["sha256"] for item in files] == [
        "4b2c1697798764eb6c76c9297c77d5e36b548cb0a6bf44b376edfea065beb52f",
        "eab318cf049552706ad9feb28e2cbf144b457b618135652a5bea2af7aee0a9c2",
    ]
    assert config["preference"]["excluded_assays"] == ["Sort10"]


def test_pinned_download_hash_is_verified_and_corruption_is_rejected(tmp_path) -> None:
    payloads = {
        "datasets/assay_to_yield_training_sequences.csv": b"a\n1\n",
        "datasets/test_sequences.csv": b"a\n2\n",
    }
    config = {
        "external_data": {
            "commit": "abc123",
            "raw_base_url": "https://example.invalid/repo",
            "cache_directory": "data/cache",
            "files": [
                {"path": path, "sha256": hashlib.sha256(payload).hexdigest()}
                for path, payload in payloads.items()
            ],
        }
    }

    def downloader(url: str, destination: Path) -> None:
        source_path = next(path for path in payloads if url.endswith(path))
        destination.write_bytes(payloads[source_path])

    sources = gp2.ensure_pinned_sources(config, tmp_path, downloader=downloader)
    assert [source.row_count for source in sources] == [1, 1]
    sources[0].local_path.write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="cached pinned-data hash mismatch"):
        gp2.ensure_pinned_sources(config, tmp_path, downloader=downloader)


def test_candidate_filtering_uses_every_frozen_criterion() -> None:
    rows = [
        _row("AAAA"),
        _row("AAAB", Stop=True),
        _row("AAAC", Paratope=""),
        _row("AAAD", SH_Average_bc=np.nan),
        _row("AAAE", Sort1_2_score=np.nan),
        _row("AAAF", Sort8_3_score=np.inf),
    ]
    candidates, summary = gp2.canonical_candidates(
        [pd.DataFrame(rows)], ["fixture.csv"]
    )
    assert candidates["Paratope"].tolist() == ["AAAA"]
    assert summary["filtering_counts"]["union_rows"] == 6
    assert summary["retained_rows_before_duplicates"] == 1


def test_consistent_duplicates_collapse_and_inconsistent_duplicates_stop() -> None:
    consistent = pd.DataFrame([_row("AAAA"), _row("AAAA")])
    candidates, summary = gp2.canonical_candidates([consistent], ["fixture.csv"])
    assert len(candidates) == 1
    assert summary["duplicate_row_count"] == 1

    inconsistent = pd.DataFrame(
        [_row("AAAA"), _row("AAAA", SH_Average_bc=1.5001)]
    )
    with pytest.raises(gp2.PreprocessingAmbiguity):
        gp2.canonical_candidates([inconsistent], ["fixture.csv"])


def test_hamming_knn_is_deterministic_and_uses_lexical_ties() -> None:
    sequences = ["AAA", "AAB", "AAC", "ABA", "BAA"]
    first = gp2.hamming_knn_graph(sequences, k=1)
    second = gp2.hamming_knn_graph(sequences, k=1)
    np.testing.assert_array_equal(first.edges, second.edges)
    assert (0, 1) in {tuple(edge) for edge in first.edges.tolist()}


def test_connected_component_rule_is_deterministic() -> None:
    candidates = pd.DataFrame(
        {"candidate_index": range(5), "Paratope": ["A", "B", "C", "D", "E"]}
    )
    graph = gp2.HammingGraph(
        edges=np.asarray([[0, 1], [1, 2], [3, 4]], dtype=int),
        component_sizes=(3, 2),
        retained_indices=np.asarray([0, 1, 2], dtype=int),
        largest_component_fraction=0.6,
    )
    retained, edges, restricted = gp2.apply_component_rule(
        candidates, graph, minimum_largest_fraction=0.6
    )
    assert restricted
    assert retained["Paratope"].tolist() == ["A", "B", "C"]
    np.testing.assert_array_equal(edges, np.asarray([[0, 1], [1, 2]]))
    with pytest.raises(gp2.PreprocessingInvalid):
        gp2.apply_component_rule(candidates, graph, minimum_largest_fraction=0.9)


def test_strict_replicate_vote_treats_zero_as_abstention() -> None:
    assert gp2.preference_vote([3, 0, 2], [1, 1, 1]) == 1
    assert gp2.preference_vote([0, 2, 0], [1, 1, 1]) == -1
    assert gp2.preference_vote([0, 1, 2], [1, 1, 1]) is None


def test_sort1_sort8_factor_generation_and_canonical_order() -> None:
    scores = {
        "Sort1": np.asarray([[3, 0, 2], [1, 1, 1], [4, 4, 4]], dtype=float),
        "Sort8": np.asarray([[0, 0, 2], [1, 1, 1], [2, 2, 2]], dtype=float),
    }
    factors = gp2.build_preference_bank(
        ["AAA", "AAB", "AAC"], scores, np.asarray([[1, 2], [1, 0]])
    )
    assert factors[["left_action_index", "right_action_index", "assay"]].values.tolist() == [
        [0, 1, "Sort1"],
        [0, 1, "Sort8"],
        [1, 2, "Sort1"],
        [1, 2, "Sort8"],
    ]
    assert factors["preference_sign"].tolist() == [1, -1, -1, -1]


def test_graph_and_factor_interfaces_cannot_receive_target_values() -> None:
    assert "target" not in inspect.signature(gp2.hamming_knn_graph).parameters
    assert "target" not in inspect.signature(gp2.build_preference_bank).parameters
    scores = {assay: np.asarray([[2, 2, 2], [1, 1, 1]], float) for assay in gp2.ASSAYS}
    first = gp2.build_preference_bank(["AA", "AB"], scores, [[0, 1]])
    second = gp2.build_preference_bank(["AA", "AB"], scores, [[0, 1]])
    pd.testing.assert_frame_equal(first, second)


def test_normalized_laplacian_precision_matches_direct_reference() -> None:
    edges = np.asarray([[0, 1], [1, 2]], dtype=int)
    precision = gp2.normalized_laplacian_precision(3, edges)
    adjacency = np.asarray([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
    degree = np.diag(adjacency.sum(axis=1))
    inverse_sqrt = np.diag(1.0 / np.sqrt(np.diag(degree)))
    expected = np.eye(3) + np.eye(3) - inverse_sqrt @ adjacency @ inverse_sqrt
    np.testing.assert_allclose(precision, expected, atol=1e-14)
    assert np.linalg.eigvalsh(precision).min() > 0.0


def test_gaussian_conditioning_matches_direct_covariance_formula() -> None:
    precision = np.asarray([[2.0, -0.4, 0.0], [-0.4, 2.0, -0.3], [0.0, -0.3, 2.0]])
    observed = np.asarray([0, 2])
    values = np.asarray([0.7, -0.2])
    noise = 0.05
    posterior = gp2.graph_gaussian_posterior(precision, observed, values, noise)
    prior_covariance = np.linalg.inv(precision)
    cross = prior_covariance[:, observed]
    observed_covariance = prior_covariance[np.ix_(observed, observed)] + noise**2 * np.eye(2)
    expected_mean = cross @ np.linalg.solve(observed_covariance, values)
    expected_covariance = prior_covariance - cross @ np.linalg.solve(
        observed_covariance, cross.T
    )
    np.testing.assert_allclose(posterior.mean, expected_mean, atol=1e-12)
    np.testing.assert_allclose(posterior.covariance, expected_covariance, atol=1e-12)


def test_preference_energy_gradient_and_hessian_finite_differences() -> None:
    latent = np.asarray([0.4, -0.2, 0.7])
    pair = (0, 2)
    sign = -1
    temperature = 1.0
    epsilon = 1e-5
    analytic_gradient = logistic_preference_gradient(latent, pair, sign, temperature)
    finite_gradient = np.asarray(
        [
            (
                logistic_preference_energy(latent + epsilon * np.eye(3)[index], pair, sign, temperature)
                - logistic_preference_energy(latent - epsilon * np.eye(3)[index], pair, sign, temperature)
            )
            / (2 * epsilon)
            for index in range(3)
        ]
    )
    np.testing.assert_allclose(analytic_gradient, finite_gradient, atol=2e-10)
    analytic_hessian = logistic_preference_hessian(latent, pair, sign, temperature)
    finite_hessian = np.column_stack(
        [
            (
                logistic_preference_gradient(latent + epsilon * np.eye(3)[index], pair, sign, temperature)
                - logistic_preference_gradient(latent - epsilon * np.eye(3)[index], pair, sign, temperature)
            )
            / (2 * epsilon)
            for index in range(3)
        ]
    )
    np.testing.assert_allclose(analytic_hessian, finite_hessian, atol=2e-10)


def test_target_scaling_uses_only_five_initial_observations() -> None:
    initial = np.asarray([1.0, 2.0, 3.0, 4.0, 8.0])
    scaling = gp2.fit_target_scaling(initial)
    assert scaling.mean == np.mean(initial)
    assert scaling.standard_deviation == np.std(initial, ddof=1)
    assert gp2.fit_target_scaling(np.ones(5)).standard_deviation == 1e-6


def test_initial_design_matches_exact_frozen_numpy_expression() -> None:
    for seed in (0, 7, 19):
        expected = np.random.default_rng(seed).choice(300, size=5, replace=False)
        np.testing.assert_array_equal(gp2.initial_design(300, seed), expected)


def test_scalar_and_full_share_initial_observations_and_never_repeat() -> None:
    prepared = _tiny_prepared()
    config = _tiny_config()
    scalar_rows, _, scalar_reliable = gp2._run_method(
        prepared=prepared,
        config=config,
        config_hash="abc",
        seed=0,
        method="scalar_only",
        horizon=2,
        draw_schedule=[64],
        profile="test",
    )
    full_rows, _, full_reliable = gp2._run_method(
        prepared=prepared,
        config=config,
        config_hash="abc",
        seed=0,
        method="full_preference",
        horizon=2,
        draw_schedule=[64],
        profile="test",
    )
    assert scalar_reliable and full_reliable
    assert scalar_rows[0]["initial_observation_sha256"] == full_rows[0]["initial_observation_sha256"]
    for rows in (scalar_rows, full_rows):
        selected = [row["selected_action_index"] for row in rows]
        initial = json.loads(rows[0]["initial_action_indices"])
        assert len(selected) == len(set(selected))
        assert not set(selected).intersection(initial)


def test_inference_streams_are_reproducible_and_independent() -> None:
    first = gp2.inference_rng(3, 2, 1, 0).standard_normal(20)
    repeated = gp2.inference_rng(3, 2, 1, 0).standard_normal(20)
    other_half = gp2.inference_rng(3, 2, 1, 1).standard_normal(20)
    other_stage = gp2.inference_rng(3, 2, 2, 0).standard_normal(20)
    np.testing.assert_array_equal(first, repeated)
    assert not np.array_equal(first, other_half)
    assert not np.array_equal(first, other_stage)


def test_output_schema_checks_sharing_and_overwrite_protection(tmp_path) -> None:
    prepared = _tiny_prepared()
    config = _tiny_config()
    scalar_rows, _, _ = gp2._run_method(
        prepared=prepared,
        config=config,
        config_hash="abc",
        seed=0,
        method="scalar_only",
        horizon=2,
        draw_schedule=[64],
        profile="test",
    )
    full_rows, inference_rows, _ = gp2._run_method(
        prepared=prepared,
        config=config,
        config_hash="abc",
        seed=0,
        method="full_preference",
        horizon=2,
        draw_schedule=[64],
        profile="test",
    )
    gp2.validate_output_schema(scalar_rows + full_rows, inference_rows)
    destination = tmp_path / "immutable"
    gp2.create_immutable_output_directory(destination)
    with pytest.raises(FileExistsError):
        gp2.create_immutable_output_directory(destination)


def test_gate_evaluator_known_outcomes() -> None:
    passing = [
        {"seed": seed, "method": method, "r_5": value}
        for seed in range(20)
        for method, value in (("scalar_only", 0.4), ("full_preference", 0.3))
    ]
    assert gp2.evaluate_p1_gate(passing)["verdict"] == "PASS"
    failing = [
        {"seed": seed, "method": method, "r_5": value}
        for seed in range(20)
        for method, value in (("scalar_only", 0.4), ("full_preference", 0.31))
    ]
    assert gp2.evaluate_p1_gate(failing)["verdict"] == "FAIL_P1"
    ceiling = [
        {"seed": seed, "method": method, "r_5": 0.0}
        for seed in range(20)
        for method in ("scalar_only", "full_preference")
    ]
    assert gp2.evaluate_p1_gate(ceiling)["verdict"] == "GATE_UNINFORMATIVE_SCALAR_CEILING"
