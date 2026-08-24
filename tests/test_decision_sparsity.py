from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from decimal import Decimal

import numpy as np
from scipy import sparse

from conditioned_bo.decision_sparsity import (
    classify_development,
    deterministic_random_subset,
    fit_active_acquisition,
    run_influence_path,
    stabilization_quantities,
    target_factor_counts,
)
from conditioned_bo.full_bank_scaling import (
    FixedReferenceState,
    ImplicitMenzSystem,
    ResourceGuard,
    build_compact_pair_bank,
)


ROOT = Path(__file__).resolve().parents[1]
SUN = ROOT / "experiments/sun_oxide"
CONFIG = SUN / "configs/decision_sparsity_diagnostic.json"


def _load_driver():
    specification = importlib.util.spec_from_file_location(
        "decision_sparsity_diagnostic",
        SUN / "decision_sparsity_diagnostic.py",
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_frozen_fraction_grid_has_exact_rounded_counts() -> None:
    fractions = [
        0,
        0.01,
        0.02,
        0.05,
        0.10,
        0.20,
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
        0.95,
        1.00,
    ]
    np.testing.assert_array_equal(
        target_factor_counts(fractions, 124718),
        [
            0,
            1247,
            2494,
            6236,
            12472,
            24944,
            37415,
            49887,
            62359,
            74831,
            87303,
            99774,
            112246,
            118482,
            124718,
        ],
    )


def test_deterministic_random_subsets_are_exact_and_order_independent() -> None:
    first = deterministic_random_subset(
        100, 20, base_seed=20260823, state_index=1, fraction_index=2, replicate=3
    )
    repeat = deterministic_random_subset(
        100, 20, base_seed=20260823, state_index=1, fraction_index=2, replicate=3
    )
    different = deterministic_random_subset(
        100, 20, base_seed=20260823, state_index=1, fraction_index=2, replicate=4
    )
    np.testing.assert_array_equal(first, repeat)
    assert first.size == 20
    assert np.all(np.diff(first) > 0)
    assert not np.array_equal(first, different)


def test_stabilization_quantities_require_suffix_stability() -> None:
    fractions = [0.0, 0.1, 0.2, 0.4, 1.0]
    agreement = [False, True, False, True, True]
    regret = [0.2, 0.0, 0.03, 0.005, 0.0]
    certificate = [False, False, False, False, True]
    records = [
        {
            "requested_fraction": fraction,
            "action_agreement": agree,
            "full_laplace_ei_regret": value,
            "theorem_certificate_passed": certified,
        }
        for fraction, agree, value, certified in zip(
            fractions, agreement, regret, certificate, strict=True
        )
    ]
    summary = stabilization_quantities(records)
    assert summary == {
        "first_agreement_fraction": 0.1,
        "first_stable_agreement_fraction": 0.4,
        "first_small_regret_fraction": 0.4,
        "certificate_fraction": 1.0,
        "stable_to_certificate_gap": 0.6,
    }


def test_terminal_classification_rules_are_literal() -> None:
    def row(stable: float, gap: float, suffix: bool = True):
        return {
            "first_stable_agreement_fraction": stable,
            "stable_to_certificate_gap": gap,
            "stable_suffix_regret_at_most_0_01": suffix,
        }

    assert classify_development([row(0.2, 0.8), row(0.4, 0.6), row(0.3, 0.7)]) == (
        "STRONG_BOUND_CONSERVATISM"
    )
    assert classify_development([row(0.4, 0.6), row(0.5, 0.2), row(0.7, 0.3)]) == (
        "MIXED_DECISION_SPARSITY"
    )
    assert classify_development([row(0.4, 0.6), row(0.6, 0.4), row(0.8, 0.2)]) == (
        "DECISION_DENSE"
    )


def test_both_influence_paths_reach_exact_checkpoints_and_full_target() -> None:
    q0 = sparse.eye(5, format="csr") * 2.0
    bank = build_compact_pair_bank(
        np.arange(4, dtype=np.int32),
        [Decimal("0"), Decimal("1"), Decimal("2"), Decimal("3")],
        model_name="toy",
    )
    reference = FixedReferenceState(
        mean=np.asarray([0.0, 0.1, -0.1, 0.2]),
        precision=np.eye(4) * 2.0,
        action_variances=np.full(3, 0.5),
        diagnostics={},
    )
    actions = np.asarray([0, 1, 2])
    keys = ["a", "b", "c"]
    menz = ImplicitMenzSystem(q0, bank, [], residual_tolerance=1e-9)
    guard = ResourceGuard.create(1.0, {"influence_path": 60.0})
    settings = {
        "gradient_tolerance": 1e-5,
        "optimizer_gradient_tolerance": 1e-8,
        "function_tolerance": 1e-15,
        "maximum_iterations": 2000,
        "residual_tolerance": 1e-9,
    }
    full_ei, full_leader, _, _ = fit_active_acquisition(
        reference,
        bank,
        np.arange(bank.factor_count),
        actions,
        [],
        keys,
        0.0,
        None,
        chunk_size=64,
        laplace_settings=settings,
        guard=guard,
        phase_name="influence_path",
    )
    for path_name in ("STATIC_INFLUENCE_PREFIX", "RERANKED_FINE_PATH"):
        rows = run_influence_path(
            path_name,
            [0.0, 0.5, 1.0],
            reference,
            menz,
            bank,
            actions,
            [],
            keys,
            0.0,
            full_ei,
            full_leader,
            epsilon_struct=0.02,
            chunk_size=64,
            laplace_settings=settings,
            guard=guard,
            phase_name="influence_path",
        )
        assert [row["active_factor_count"] for row in rows] == [0, 3, 6]
        assert rows[-1]["action_agreement"]
        assert rows[-1]["theorem_certificate_passed"]


def test_driver_freezes_model_rules_and_has_no_fresh_target_interface() -> None:
    driver = _load_driver()
    config = driver._load_config(CONFIG)
    assert config["starting_main_sha"] == (
        "5f49f140acc3532b0231b1d1c446d22cd0e168d8"
    )
    assert config["model"]["support_count"] == 500
    assert config["model"]["strict_factor_count"] == 124718
    assert config["epsilon_struct"] == 0.02
    assert config["fresh_seed_policy"]["forbidden_seeds"] == list(range(12, 32))
    assert not config["fresh_seed_policy"]["scientific_preregistration_created"]
    assert not config["resource_guard"]["colab_fallback_allowed"]
    source = (SUN / "decision_sparsity_diagnostic.py").read_text(encoding="utf-8")
    assert "gw_oracle.csv" not in source
    assert "1000" not in json.dumps(config["model"])
    assert "2142" not in json.dumps(config["model"])


def test_state_loader_reads_only_each_authorized_prefix() -> None:
    driver = _load_driver()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
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
