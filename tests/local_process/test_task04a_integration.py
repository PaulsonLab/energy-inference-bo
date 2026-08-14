from __future__ import annotations

import json

from energy_bo.experiments.task04a import Task04AConfig, analyze_full_gates, analyze_preflight, run_task04a


def test_smoke_runner_is_deterministic_and_complete(tmp_path):
    config=Task04AConfig("test",(0,),4,(16,),("G","W","I"),16,12,24,max_iterations=20,metric_points=32)
    first=run_task04a(config,tmp_path/"run",device="cpu")
    second=run_task04a(config,tmp_path/"run",device="cpu")
    assert first["rows"]==second["rows"]
    assert len(first["rows"])==12 and len(first["timing"])==3
    assert all(row["conditional_kl"]>=0 for row in first["rows"])
    assert all(float(row["objective_gradient_seconds"])<=float(row["fit_seconds"])+1e-9 for row in ({"objective_gradient_seconds":t[f"{method}_objective_gradient_seconds"],"fit_seconds":t[f"{method}_fit_seconds"]} for t in first["timing"] for method in ("U","P")))
    assert json.loads((tmp_path/"run"/"task04a_config.json").read_text())["output_scale"]==1.0
    assert first["gate"]["decision"]=="NOT_EVALUATED_FROM_SMOKE"


def test_full_gate_requires_complete_evidence():
    assert analyze_full_gates(Task04AConfig.full(),[],[])["decision"]=="PENDING_REVIEW"
    assert analyze_preflight(Task04AConfig.preflight(),[],[])["decision"]=="INCOMPLETE_PREFLIGHT"
    assert Task04AConfig.preflight().child_feature_coordinate=="local_standardized_residual"
