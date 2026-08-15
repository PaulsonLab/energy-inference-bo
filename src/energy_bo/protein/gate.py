"""Prospective automatic Task 05A gate evaluation."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        if value.lower() in {"true", "1"}:
            return True
        if value.lower() in {"false", "0", ""}:
            return False
        raise ValueError(f"invalid serialized boolean: {value}")
    return bool(value)


def _rows(rows: list[dict[str, Any]], *, dataset: str, model: str, n: int | None = None) -> list[dict[str, Any]]:
    return [row for row in rows if row["dataset"] == dataset and row["model"] == model and (n is None or int(row.get("n", -1)) == n)]


def _credibility(offline: list[dict[str, Any]], dataset: str, model: str) -> tuple[bool, dict[str, Any]]:
    checks = []
    details: dict[str, Any] = {}
    for n in (48, 96):
        rows = _rows(offline, dataset=dataset, model=model, n=n)
        if len(rows) != 10:
            details[str(n)] = {"complete": False, "rows": len(rows)}
            checks.append(False)
            continue
        ice = float(np.median([float(row["interval_calibration_error"]) for row in rows]))
        coverage_ok = sum(abs(float(row["coverage_90"]) - 0.9) <= 0.20 for row in rows)
        nll_gap = float(np.median([float(row["high_utility_nll"]) - float(row["constant_high_utility_nll"]) for row in rows]))
        converged = sum(_as_bool(row["converged"]) and _as_bool(row["finite"]) for row in rows)
        passed = ice <= 0.10 and coverage_ok >= 8 and nll_gap <= 0.10 and converged >= 9
        details[str(n)] = {"complete": True, "median_interval_error": ice, "coverage_ok": coverage_ok, "median_nll_gap": nll_gap, "finite_converged": converged, "passed": passed}
        checks.append(passed)
    return all(checks), details


def _catastrophic(offline: list[dict[str, Any]], dataset: str, model: str) -> bool:
    for n in (48, 96):
        rows = _rows(offline, dataset=dataset, model=model, n=n)
        if len(rows) != 10:
            return True
        ice = float(np.median([float(row["interval_calibration_error"]) for row in rows]))
        nll_gap = float(np.median([float(row["high_utility_nll"]) - float(row["constant_high_utility_nll"]) for row in rows]))
        failures = sum(not (_as_bool(row["converged"]) and _as_bool(row["finite"])) for row in rows)
        if ice > 0.20 or nll_gap > 0.25 or failures > 2:
            return True
    return False


def _sequential_summary(sequential: list[dict[str, Any]], dataset: str, model: str) -> dict[str, Any]:
    rows = _rows(sequential, dataset=dataset, model=model)
    return {
        "rows": len(rows),
        "median_auc": float(np.median([float(row["regret_auc"]) for row in rows])) if rows else float("nan"),
        "top1_rate": float(np.mean([_as_bool(row["top1_hit"]) for row in rows])) if rows else float("nan"),
        "top5_rate": float(np.mean([_as_bool(row["top5_hit"]) for row in rows])) if rows else float("nan"),
    }


def evaluate_task05a_gate(
    offline: list[dict[str, Any]],
    sequential: list[dict[str, Any]],
    *,
    profile: str,
    git_sha: str,
    config_hash: str,
) -> dict[str, Any]:
    if profile != "full":
        return {"task_id": "05A", "status": "INCONCLUSIVE", "protocol_version": "task05a-v1", "git_sha": git_sha, "config_hash": config_hash, "gate_checks": {"smoke_only": True}, "primary_metrics": {}, "notes": "Smoke profiles can never pass a research gate."}
    expected_offline = 2 * 3 * 3 * 10
    expected_sequential = 2 * 3 * 10
    if len(offline) != expected_offline or len(sequential) != expected_sequential:
        return {"task_id": "05A", "status": "INCONCLUSIVE", "protocol_version": "task05a-v1", "git_sha": git_sha, "config_hash": config_hash, "gate_checks": {"complete": False, "offline_rows": len(offline), "expected_offline_rows": expected_offline, "sequential_rows": len(sequential), "expected_sequential_rows": expected_sequential}, "primary_metrics": {}, "notes": "Full gate requires all frozen shards."}

    datasets = ("trpb", "creilov")
    models = ("S0", "S1", "S2")
    credible, credibility_details = {}, {}
    summaries = {}
    for dataset in datasets:
        for model in models:
            credible[(dataset, model)], credibility_details[f"{dataset}:{model}"] = _credibility(offline, dataset, model)
            summaries[(dataset, model)] = _sequential_summary(sequential, dataset, model)

    pass_a_models = []
    gains: dict[str, Any] = {}
    for model in ("S1", "S2"):
        for dataset in datasets:
            other = datasets[1 - datasets.index(dataset)]
            base_rows = {int(row["seed"]): row for row in _rows(sequential, dataset=dataset, model="S0")}
            model_rows = {int(row["seed"]): row for row in _rows(sequential, dataset=dataset, model=model)}
            relative = [(float(base_rows[s]["regret_auc"]) - float(model_rows[s]["regret_auc"])) / max(float(base_rows[s]["regret_auc"]), 1e-15) for s in base_rows]
            wins = sum(float(model_rows[s]["regret_auc"]) < float(base_rows[s]["regret_auc"]) for s in base_rows)
            top_gain = summaries[(dataset, model)]["top1_rate"] - summaries[(dataset, "S0")]["top1_rate"]
            median_gain = float(np.median(relative))
            auc_signal = median_gain >= 0.20 and wins >= 7
            tail_signal = top_gain >= 0.15
            other_base = summaries[(other, "S0")]
            other_model = summaries[(other, model)]
            auc_worsening_abs = other_model["median_auc"] - other_base["median_auc"]
            auc_worsening_rel = auc_worsening_abs / max(other_base["median_auc"], 1e-15)
            severe = (auc_worsening_rel > 0.20 and auc_worsening_abs > 0.05) or (other_base["top1_rate"] - other_model["top1_rate"] >= 0.20)
            passed = credible[(dataset, model)] and (auc_signal or tail_signal) and not _catastrophic(offline, other, model) and not severe
            gains[f"{dataset}:{model}"] = {"median_relative_auc_gain": median_gain, "paired_auc_wins": wins, "top1_rate_gain": top_gain, "other_dataset_severe_regression": severe, "passed": passed}
            if passed:
                pass_a_models.append((model, dataset, median_gain, top_gain))

    if pass_a_models:
        winner = max(pass_a_models, key=lambda value: (value[2], value[3], value[0]))
        status, path, selected = "PASS", "PASS-A", winner[0]
    else:
        baseline_credible = all(credible[(dataset, "S0")] for dataset in datasets)
        immaterial = True
        for model in ("S1", "S2"):
            for dataset in datasets:
                item = gains[f"{dataset}:{model}"]
                if item["median_relative_auc_gain"] >= 0.10 or item["top1_rate_gain"] >= 0.10:
                    immaterial = False
        if baseline_credible and immaterial:
            status, path, selected = "PASS", "PASS-B", "S0"
        elif not any(credible.values()):
            status, path, selected = "FAIL", "FAIL", None
        else:
            status, path, selected = "INCONCLUSIVE", "INCONCLUSIVE", None
    primary = {f"{dataset}:{model}": summaries[(dataset, model)] for dataset in datasets for model in models}
    return {"task_id": "05A", "status": status, "protocol_version": "task05a-v1", "git_sha": git_sha, "config_hash": config_hash, "gate_checks": {"path": path, "selected_model": selected, "credibility": credibility_details, "structured_gains": gains}, "primary_metrics": primary, "notes": "Computed mechanically from the frozen Task 05A protocol."}
