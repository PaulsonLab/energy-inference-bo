from __future__ import annotations

import json

from energy_bo.experiments.task04ae import Task04AEConfig, run_task04ae


def test_small_runner_writes_complete_resumable_evidence(tmp_path):
    config=Task04AEConfig("test",(9,),(32,),24,12,calibration_power=16,verification_power=14,max_iterations=75,panel_count=32)
    output=tmp_path/"task04ae"
    first=run_task04ae(config,output,device="cpu")
    case=output/"case_seed9_n32.json"; before=case.stat().st_mtime_ns
    second=run_task04ae(config,output,device="cpu")
    assert case.stat().st_mtime_ns==before
    assert first["preflight"]["status"]=="PASS" and second["preflight"]["status"]=="PASS"
    assert len(second["metrics"])==15 and len(second["marginal"])==15 and len(second["panels"])==3
    assert len(second["timing"])==2
    for name in ("metrics.csv","marginal_safety.csv","batch_decisions.csv","timing.csv","parameters.csv","gate_status.json","TASK_04AE_SUMMARY.md"):
        assert (output/name).is_file()
    saved=json.loads(case.read_text())
    assert saved["seed"]==9 and saved["n"]==32 and sum(saved["context_counts"].values())==32
    assert json.loads((output/"task04ae_config.json").read_text())["l2_precision"]==10.0
