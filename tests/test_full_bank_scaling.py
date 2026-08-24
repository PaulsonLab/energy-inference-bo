from __future__ import annotations

from decimal import Decimal
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy import sparse

from conditioned_bo.adaptive_pbe import exact_activation_batch
from conditioned_bo.full_bank_scaling import (
    CompleteMinusTieAdjacency,
    ImplicitMenzSystem,
    ResourceGuard,
    build_compact_pair_bank,
    construct_support_precision_reference,
    regression_against_dense_500,
    stable_exact_activation_batch,
    update_fixed_reference_state,
)
from conditioned_bo.pbe_factor_theory import load_action_mapping, load_legacy_nodes


ROOT = Path(__file__).resolve().parents[1]
SUN = ROOT / "experiments/sun_oxide"


def _load_driver():
    path = SUN / "full_bank_scaling_probe.py"
    specification = importlib.util.spec_from_file_location("full_bank_scaling_probe", path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_compact_complete_minus_ties_matches_explicit_adjacency() -> None:
    bank = build_compact_pair_bank(
        np.arange(5),
        [Decimal("0"), Decimal("1"), Decimal("1"), Decimal("2"), Decimal("3")],
        model_name="fixture",
    )
    assert bank.endpoint_pairs.dtype == np.int32
    assert bank.signs.dtype == np.int8
    assert bank.factor_count == 9
    assert bank.omitted_tie_pair_count == 1
    assert bank.weight == 0.25
    assert bank.maximum_weighted_incident_degree == 1.0
    operator = CompleteMinusTieAdjacency(bank, 5)
    vector = np.asarray([0.3, -0.2, 0.5, 1.2, -0.7])
    np.testing.assert_allclose(
        operator.matvec(vector), operator.explicit_sparse() @ vector, atol=1e-15
    )


def test_implicit_menz_solve_matches_direct_fixture() -> None:
    q0 = sparse.csr_matrix(
        np.asarray(
            [
                [1.5, -0.2, 0.0, 0.0],
                [-0.2, 1.4, -0.1, 0.0],
                [0.0, -0.1, 1.3, -0.1],
                [0.0, 0.0, -0.1, 1.2],
            ]
        )
    )
    bank = build_compact_pair_bank(
        np.arange(4),
        [Decimal("0"), Decimal("1"), Decimal("2"), Decimal("2")],
        model_name="fixture",
    )
    system = ImplicitMenzSystem(q0, bank, [1], residual_tolerance=1e-11)
    rhs = np.asarray([1.0, -0.2, 0.4, 0.7])
    iterative, diagnostics = system.solve(rhs)
    explicit = (
        q0
        - 0.25 * CompleteMinusTieAdjacency(bank, 4).explicit_sparse()
        + sparse.diags([0.0, 400.0, 0.0, 0.0])
    ).toarray()
    np.testing.assert_allclose(iterative, np.linalg.solve(explicit, rhs), atol=1e-11)
    assert diagnostics["relative_residual"] <= 1e-11


def test_support_reference_and_observation_update_are_exact() -> None:
    q0 = sparse.csr_matrix(
        np.asarray(
            [
                [2.0, -0.2, 0.0, 0.0],
                [-0.2, 1.8, -0.1, 0.0],
                [0.0, -0.1, 1.6, -0.1],
                [0.0, 0.0, -0.1, 1.5],
            ]
        )
    )
    guard = ResourceGuard.create(1.0, {})
    support = np.asarray([0, 2, 3])
    reference = construct_support_precision_reference(
        q0, support, residual_tolerance=1e-12, guard=guard
    )
    expected_covariance = np.linalg.inv(q0.toarray())[np.ix_(support, support)]
    np.testing.assert_allclose(
        reference.precision, np.linalg.inv(expected_covariance), atol=1e-14
    )
    state = update_fixed_reference_state(
        reference,
        [1],
        [0.4],
        [0, 2],
        residual_tolerance=1e-12,
    )
    expected_precision = np.asarray(reference.precision).copy()
    expected_precision[1, 1] += 400.0
    expected_information = np.asarray([0.0, 160.0, 0.0])
    np.testing.assert_allclose(state.mean, np.linalg.solve(expected_precision, expected_information))
    expected_inverse = np.linalg.inv(expected_precision)
    np.testing.assert_allclose(state.action_variances, np.diag(expected_inverse)[[0, 2]])


def test_scaling_activation_rule_is_the_frozen_exact_rule() -> None:
    omitted = np.arange(7, dtype=np.int64)
    contributions = np.asarray([0.08, 0.08, 0.04, 0.03, 0.02, 0.01, 0.01])
    expected = exact_activation_batch(
        omitted,
        contributions,
        active_gap=-0.01,
        epsilon_struct=0.02,
        rho=0.8,
    )
    observed = stable_exact_activation_batch(
        omitted,
        contributions,
        active_gap=-0.01,
        epsilon=0.02,
        rho=0.8,
    )
    np.testing.assert_array_equal(observed, expected)


def test_real_500_bank_and_implicit_regression_match_committed_model() -> None:
    config = json.loads((SUN / "configs/full_bank_scaling_probe.json").read_text())
    nodes = load_legacy_nodes(ROOT / config["inputs"]["legacy_pbe"]["path"], 2142)
    actions = load_action_mapping(
        ROOT / config["inputs"]["action_node_mapping"]["path"], 191, 2142
    )
    driver = _load_driver()
    support = driver._frozen_support_nodes(
        ROOT / config["inputs"]["frozen_support_500"]["path"]
    )
    bank = build_compact_pair_bank(
        support,
        [node.pbe_band_gap for node in nodes],
        model_name="NORMALIZED_ALL_PAIRS_PBE_500_V1",
    )
    driver._validate_frozen_bank_500(
        ROOT / config["inputs"]["frozen_factor_bank_500"]["path"], bank
    )
    assert bank.factor_count == 124718
    q0 = sparse.load_npz(ROOT / config["inputs"]["q0"]["path"])
    action_nodes = np.asarray([int(action["node_index"]) for action in actions])
    regression = regression_against_dense_500(q0, bank, action_nodes)
    assert regression["w_max_abs_error"] <= 1e-8
    assert regression["v_max_abs_error"] <= 1e-8


def test_development_state_loader_reads_only_each_authorized_prefix() -> None:
    driver = _load_driver()
    config = json.loads((SUN / "configs/full_bank_scaling_probe.json").read_text())
    actions = driver._read_csv(
        ROOT / config["inputs"]["action_node_mapping"]["path"]
    )
    fixture = ROOT / config["inputs"]["development_seed0_states"]["path"]
    for count in (8, 14, 20):
        state = driver._load_authorized_state(
            fixture,
            observation_count=count,
            action_rows=actions,
            expected_mean=config["seed0_state_provenance"]["target_mean_ev"],
            expected_scale=config["seed0_state_provenance"]["target_scale_ev"],
        )
        assert state["rows_read"] == count
        assert state["later_rows_read"] == 0
        assert set(state["observed_action_positions"]).isdisjoint(range(191, 211))


def test_driver_has_no_fresh_oracle_interface() -> None:
    source = (SUN / "full_bank_scaling_probe.py").read_text()
    assert "gw_oracle.csv" not in source
    config = json.loads((SUN / "configs/full_bank_scaling_probe.json").read_text())
    assert config["fresh_seed_policy"]["forbidden_seeds"] == list(range(12, 32))
    assert not config["fresh_seed_policy"]["scientific_preregistration_created"]
