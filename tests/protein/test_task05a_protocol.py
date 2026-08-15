from __future__ import annotations

from energy_bo.protein.gate import evaluate_task05a_gate


def _evidence(*, structured_gain: float = 0.25, credible: bool = True):
    offline, sequential = [], []
    for dataset in ("trpb", "creilov"):
        for model in ("S0", "S1", "S2"):
            for n in (48, 96, 192):
                for seed in range(10):
                    offline.append({
                        "dataset": dataset, "model": model, "n": n, "seed": seed,
                        "interval_calibration_error": 0.05 if credible else 0.3,
                        "coverage_90": 0.9,
                        "high_utility_nll": 1.0,
                        "constant_high_utility_nll": 1.0,
                        "converged": credible, "finite": True,
                    })
            for seed in range(10):
                auc = 0.5
                if model == "S1" and dataset == "trpb":
                    auc *= 1 - structured_gain
                sequential.append({"dataset": dataset, "model": model, "seed": seed, "regret_auc": auc, "top1_hit": model == "S1" and dataset == "trpb", "top5_hit": True})
    return offline, sequential


def test_smoke_can_never_pass() -> None:
    gate = evaluate_task05a_gate([], [], profile="smoke", git_sha="x", config_hash="y")
    assert gate["status"] == "INCONCLUSIVE"
    assert gate["gate_checks"]["smoke_only"]


def test_gate_pass_a_is_mechanical() -> None:
    offline, sequential = _evidence(structured_gain=0.25)
    for row in offline:
        row["converged"], row["finite"] = str(row["converged"]), str(row["finite"])
    for row in sequential:
        row["top1_hit"], row["top5_hit"] = str(row["top1_hit"]), str(row["top5_hit"])
    gate = evaluate_task05a_gate(offline, sequential, profile="full", git_sha="x", config_hash="y")
    assert gate["status"] == "PASS"
    assert gate["gate_checks"]["path"] == "PASS-A"


def test_gate_pass_b_is_mechanical() -> None:
    offline, sequential = _evidence(structured_gain=0.0)
    for row in sequential:
        row["top1_hit"] = False
    gate = evaluate_task05a_gate(offline, sequential, profile="full", git_sha="x", config_hash="y")
    assert gate["status"] == "PASS"
    assert gate["gate_checks"]["path"] == "PASS-B"


def test_gate_fail_and_incomplete_are_not_authorization() -> None:
    offline, sequential = _evidence(credible=False)
    gate = evaluate_task05a_gate(offline, sequential, profile="full", git_sha="x", config_hash="y")
    assert gate["status"] == "FAIL"
    incomplete = evaluate_task05a_gate(offline[:-1], sequential, profile="full", git_sha="x", config_hash="y")
    assert incomplete["status"] == "INCONCLUSIVE"
