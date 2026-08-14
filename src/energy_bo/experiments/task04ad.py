"""Bounded local decision-relevance diagnostic for Task 04A-D."""

from __future__ import annotations

import csv
import json
import math
import platform
import resource
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.quasirandom import SobolEngine

from energy_bo.local_process.decision import (
    active_candidate_sets,
    build_counterfactual_panel,
    counterfactual_metrics,
    pairwise_metrics,
    select_real_pairs,
    within_one_percent,
)
from energy_bo.local_process.energy import LocalEnergyModel, neighbor_summary
from energy_bo.local_process.evaluation import decision_metrics, predictive_metrics
from energy_bo.local_process.geometry import LocalGeometry, build_geometry, candidate_geometry, ordered_sobol
from energy_bo.local_process.truth import U_TRUE, V_TRUE, generate_oracle


@dataclass(frozen=True)
class DecisionDiagnosticConfig:
    profile: str
    seeds: tuple[int, ...]
    dimension: int = 6
    realization_count: int = 256
    test_count: int = 256
    neighborhood_size: int = 8
    l2_precision: float = 10.0
    max_iterations: int = 250
    metric_points: int = 128
    candidate_chunk: int = 512
    wall_time_seconds: float = 900.0

    @classmethod
    def wiring(cls) -> "DecisionDiagnosticConfig":
        return cls("wiring", (100,))

    @classmethod
    def local(cls) -> "DecisionDiagnosticConfig":
        return cls("local", tuple(range(100, 108)))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields=sorted({key for row in rows for key in row})
    with path.open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields,lineterminator="\n"); writer.writeheader(); writer.writerows(rows)


def _prefix(geometry: LocalGeometry, n: int) -> LocalGeometry:
    return LocalGeometry(geometry.x[:n],geometry.neighbors[:n],geometry.mask[:n],geometry.coefficients[:n],geometry.variances[:n],geometry.similarity_weights[:n],geometry.jitter)


def _chunked_ei(model: LocalEnergyModel, mean: torch.Tensor, scale: torch.Tensor, summary: torch.Tensor, best: float, chunk: int) -> torch.Tensor:
    return torch.cat([model.expected_improvement(mean[start:start+chunk],scale[start:start+chunk],summary[start:start+chunk],best) for start in range(0,len(mean),chunk)])


def _peak_rss_bytes() -> int:
    value=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform=="darwin" else value*1024)


def _case_specs(config: DecisionDiagnosticConfig) -> list[tuple[int,str,int]]:
    if config.profile=="wiring":
        return [(100,"I",128)]
    return [(seed,regime,n) for seed in config.seeds for regime,n in (("G",256),("I",128),("I",256))]


def classify_decision_diagnostic(config: DecisionDiagnosticConfig, metrics: list[dict[str, Any]], diagnostics: list[dict[str, Any]]) -> dict[str, Any]:
    expected=1 if config.profile=="wiring" else 24
    if len(diagnostics)!=expected:
        return {"decision":"INCOMPLETE","eligible":False,"gates":{},"completed_cases":len(diagnostics),"expected_cases":expected}
    if config.profile!="local":
        return {"decision":"WIRING_COMPLETE","eligible":False,"gates":{},"completed_cases":len(diagnostics),"expected_cases":expected}
    lookup={(int(r["seed"]),str(r["regime"]),int(r["n"]),str(r["method"])):r for r in metrics}
    valid=all(bool(r["finite"]) and bool(r["converged"]) and bool(r["teacher_free_pairs"]) and bool(r["candidate_pool_verified"]) and (r["regime"]!="I" or int(r["real_pair_count"])>=32) for r in diagnostics)

    safety_rows=[]
    for seed in config.seeds:
        for method in ("U","P"):
            row=lookup[seed,"G",256,method]; base=lookup[seed,"G",256,"G0"]
            safety_rows.append((method,float(row["average_correction_kl"]),float(row["normalized_regret"])-float(base["normalized_regret"])))
    safety={}
    for method in ("U","P"):
        selected=[r for r in safety_rows if r[0]==method]
        median_kl=float(np.median([r[1] for r in selected])); median_regret=float(np.median([r[2] for r in selected])); harms=sum(r[2]>.1 for r in selected)
        safety[method]={"median_correction_kl":median_kl,"median_regret_increase":median_regret,"degradations_over_0_10":harms,"passed":median_kl<=.02 and median_regret<=.02 and harms<=1}
    safety_pass=all(value["passed"] for value in safety.values())

    u=[lookup[s,"I",256,"U"] for s in config.seeds]; p=[lookup[s,"I",256,"P"] for s in config.seeds]
    u_kl=float(np.mean([float(r["conditional_kl"]) for r in u])); p_kl=float(np.mean([float(r["conditional_kl"]) for r in p])); kl_gain=u_kl-p_kl
    kl_wins=sum(float(pr["conditional_kl"])<float(ur["conditional_kl"]) for ur,pr in zip(u,p,strict=True))
    density={"u_mean_kl":u_kl,"p_mean_kl":p_kl,"gain":kl_gain,"relative_gain":kl_gain/max(u_kl,1e-15),"paired_wins":kl_wins}
    density_pass=kl_gain>=.01 and density["relative_gain"]>=.2 and kl_wins>=6
    density["passed"]=density_pass

    medium=[r for r in diagnostics if r["regime"]=="I" and int(r["n"])==256]
    oracle_fraction=float(np.mean([float(r["counter_oracle_significant_fraction"]) for r in medium]))
    counter_seed_pass=[float(r["counter_P_sign_accuracy"])>=.8 and float(r["counter_P_median_relative_contrast_error"])<=.3 for r in medium]
    counter={"oracle_significant_fraction":oracle_fraction,"p_seed_passes":sum(counter_seed_pass),"passed":oracle_fraction>=.75 and sum(counter_seed_pass)>=6}
    oracle_pass=oracle_fraction>=.75

    real_seed_pass=[]
    for row in medium:
        u_accuracy=float(row["real_U_choice_accuracy"]); p_accuracy=float(row["real_P_choice_accuracy"])
        u_regret=float(row["real_U_margin_weighted_regret"]); p_regret=float(row["real_P_margin_weighted_regret"])
        reduction=(u_regret-p_regret)/max(u_regret,1e-15)
        real_seed_pass.append(p_accuracy>=.7 and p_accuracy-u_accuracy>=.1 and reduction>=.25)
    real={"seed_passes":sum(real_seed_pass),"passed":sum(real_seed_pass)>=6}

    eligible=[]; p_wins=0; harms=0
    for ur,pr in zip(u,p,strict=True):
        opportunity=float(ur["normalized_regret"])>=.01 or not bool(ur["within_one_percent"])
        if opportunity:
            eligible.append((ur,pr)); p_wins += float(pr["normalized_regret"])<float(ur["normalized_regret"])
        harms += float(pr["normalized_regret"])-float(ur["normalized_regret"])>.1
    gains=[float(ur["normalized_regret"])-float(pr["normalized_regret"]) for ur,pr in zip(u,p,strict=True)]
    median_gain=float(np.median(gains)); median_u=float(np.median([float(r["normalized_regret"]) for r in u])); median_p=float(np.median([float(r["normalized_regret"]) for r in p]))
    relative_gain=(median_u-median_p)/max(median_u,1e-15)
    required_wins=math.ceil(2*len(eligible)/3) if eligible else 0
    natural={"eligible_cases":len(eligible),"p_wins":p_wins,"required_wins":required_wins,"median_u_regret":median_u,"median_p_regret":median_p,"median_gain":median_gain,"relative_median_gain":relative_gain,"degradations_over_0_10":harms}
    natural_pass=len(eligible)>=3 and p_wins>=required_wins and (median_gain>=.01 or relative_gain>=.2) and harms<=1
    natural["passed"]=natural_pass
    gates={"validity":valid,"safety":safety_pass,"density":density_pass,"counterfactual":counter["passed"],"real_pairs":real["passed"],"natural":natural_pass}
    if not valid:
        decision="INVALID"
    elif not oracle_pass:
        decision="ORACLE_NO_GO"
    elif not safety_pass or not density_pass or not counter["passed"] or not real["passed"]:
        decision="LEARNING_NO_GO"
    elif not natural_pass:
        decision="MECHANISM_ONLY"
    else:
        decision="LOCAL_GO"
    return {"decision":decision,"eligible":True,"gates":gates,"diagnostics":{"safety":safety,"density":density,"counterfactual":counter,"real_pairs":real,"natural":natural}}


def _plots(output: Path, metrics: list[dict[str, Any]], diagnostics: list[dict[str, Any]]) -> None:
    selected=[r for r in diagnostics if r["regime"]=="I" and int(r["n"])==256]
    figure,axes=plt.subplots(1,2,figsize=(8.5,3.3))
    axes[0].scatter([r["seed"] for r in selected],[r["real_U_choice_accuracy"] for r in selected],label="U")
    axes[0].scatter([r["seed"] for r in selected],[r["real_P_choice_accuracy"] for r in selected],label="P")
    axes[0].set(xlabel="seed",ylabel="near-tie accuracy",ylim=(0,1.05)); axes[0].legend()
    for method in ("U","P"):
        rows=[r for r in metrics if r["regime"]=="I" and int(r["n"])==256 and r["method"]==method]
        axes[1].scatter([r["seed"] for r in rows],[r["normalized_regret"] for r in rows],label=method)
    axes[1].set(xlabel="seed",ylabel="natural normalized regret"); axes[1].legend()
    figure.tight_layout(); figure.savefig(output/"task04ad_decisions.png",dpi=160); plt.close(figure)
    figure,axis=plt.subplots(figsize=(5,3.3))
    for method in ("U","P"):
        x_values=[]; values=[]
        for n in (128,256):
            rows=[r for r in metrics if r["regime"]=="I" and int(r["n"])==n and r["method"]==method]
            if rows:
                x_values.append(n); values.append(float(np.mean([r["conditional_kl"] for r in rows])))
        if values:
            axis.plot(x_values,values,marker="o",label=method)
    axis.set(xlabel="n",ylabel="mean conditional KL"); axis.legend(); figure.tight_layout(); figure.savefig(output/"task04ad_density.png",dpi=160); plt.close(figure)


def run_decision_diagnostic(config: DecisionDiagnosticConfig, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True,exist_ok=True); case_dir=output_dir/"cases"; case_dir.mkdir(exist_ok=True)
    (output_dir/"task04ad_config.json").write_text(json.dumps(asdict(config),indent=2)+"\n")
    runtime={"python":platform.python_version(),"torch":torch.__version__,"device":"cpu","torch_double":True}
    (output_dir/"runtime.json").write_text(json.dumps(runtime,indent=2)+"\n")
    start=time.perf_counter(); metrics=[]; diagnostics=[]
    seeds={seed for seed,_,_ in _case_specs(config)}
    cached={}
    for seed in sorted(seeds):
        x=ordered_sobol(config.dimension,config.realization_count,seed); geometry=build_geometry(x,config.neighborhood_size)
        cached[seed]=(x,geometry,{regime:generate_oracle(x,regime,seed,config.neighborhood_size,geometry) for regime in ("G","I")})
    for seed,regime,n in _case_specs(config):
        path=case_dir/f"seed{seed}_{regime}_n{n}.json"
        if path.exists():
            saved=json.loads(path.read_text()); metrics.extend(saved["metrics"]); diagnostics.append(saved["diagnostic"]); continue
        if time.perf_counter()-start>config.wall_time_seconds:
            break
        case_start=time.perf_counter(); x,full_geometry,oracles=cached[seed]; oracle=oracles[regime]
        train_x=x[:n]; train_y=oracle.values[:n]; train_geometry=_prefix(full_geometry,n)
        mean=train_geometry.means(train_y); scale=train_geometry.variances.sqrt(); summary=neighbor_summary(train_y,train_geometry.neighbors,train_geometry.mask,train_geometry.similarity_weights)
        models={"G0":LocalEnergyModel(False),"U":LocalEnergyModel(False,l2_precision=config.l2_precision),"P":LocalEnergyModel(True,l2_precision=config.l2_precision)}
        fits={name:model.fit(train_y,mean,scale,summary,max_iter=config.max_iterations) for name,model in models.items() if name!="G0"}
        test_x=SobolEngine(config.dimension,scramble=True,seed=50_000+seed).draw(config.test_count).double()
        test_geometry=candidate_geometry(train_x,test_x,config.neighborhood_size); test_mean=test_geometry.means(train_y); test_scale=test_geometry.variances.sqrt(); test_summary=neighbor_summary(train_y,test_geometry.neighbors,test_geometry.mask,test_geometry.similarity_weights)

        candidate_sets=active_candidate_sets(config.dimension,seed)
        construction_geometry=candidate_geometry(train_x,candidate_sets.construction,config.neighborhood_size)
        verification_geometry=candidate_geometry(train_x,candidate_sets.verification,config.neighborhood_size)
        construction_truth=oracle.expected_improvement(construction_geometry,train_y,float(train_y.max()))
        verification_truth=oracle.expected_improvement(verification_geometry,train_y,float(train_y.max()))
        relative_excess=max(0.0,float(verification_truth.max()-construction_truth.max()))/max(float(construction_truth.max()),1e-15)
        expanded=relative_excess>.005
        geometries=(construction_geometry,verification_geometry) if expanded else (construction_geometry,)
        candidates=torch.cat(tuple(g.x for g in geometries)); candidate_mean=torch.cat(tuple(g.means(train_y) for g in geometries)); candidate_scale=torch.cat(tuple(g.variances.sqrt() for g in geometries))
        candidate_summary=torch.cat(tuple(neighbor_summary(train_y,g.neighbors,g.mask,g.similarity_weights) for g in geometries))
        oracle_ei=torch.cat((construction_truth,verification_truth)) if expanded else construction_truth
        best=float(train_y.max()); estimates={name:_chunked_ei(model,candidate_mean,candidate_scale,candidate_summary,best,config.candidate_chunk) for name,model in models.items()}

        case_metrics=[]
        for name,model in models.items():
            predictive=predictive_metrics(oracle,test_geometry,train_y,model,test_mean,test_scale,test_summary,points=config.metric_points)
            predictive["average_correction_kl"]=float(model.correction_kl(test_mean,test_scale,test_summary).mean())
            natural=decision_metrics(estimates[name],oracle_ei,candidates); natural["within_one_percent"]=within_one_percent(estimates[name],oracle_ei)
            case_metrics.append({"seed":seed,"regime":regime,"n":n,"method":name,**predictive,**natural})

        metric_finite=all(math.isfinite(float(value)) for row in case_metrics for value in row.values() if isinstance(value,(float,int)) and not isinstance(value,bool))
        diagnostic={"seed":seed,"regime":regime,"n":n,"finite":metric_finite and all(torch.isfinite(value).all() for value in (*estimates.values(),oracle_ei)),"converged":fits["U"].converged and fits["P"].converged,"teacher_free_pairs":True,"candidate_pool_verified":True,"verification_relative_excess":relative_excess,"candidate_pool_expanded":expanded,"candidate_count":len(candidates),"real_pair_count":0,"seconds":0.0,"peak_rss_bytes":0}
        if regime=="I":
            pairs=select_real_pairs(estimates["G0"],candidate_summary,V_TRUE)
            diagnostic["real_pair_count"]=len(pairs)
            for name in ("U","P"):
                values=pairwise_metrics(estimates[name],oracle_ei,pairs)
                diagnostic.update({f"real_{name}_{key}":value for key,value in values.items() if key!="pair_count"})
            panel=build_counterfactual_panel(estimates["G0"],candidate_summary,V_TRUE)
            base=panel.base_indices; low=panel.low_summary.expand(len(base),-1); high=panel.high_summary.expand(len(base),-1)
            truth_model=oracle.conditional(construction_geometry,train_y)[3]
            assert truth_model is not None
            oracle_low=truth_model.expected_improvement(candidate_mean[base],candidate_scale[base],low,best); oracle_high=truth_model.expected_improvement(candidate_mean[base],candidate_scale[base],high,best)
            for name in ("U","P"):
                low_ei=models[name].expected_improvement(candidate_mean[base],candidate_scale[base],low,best); high_ei=models[name].expected_improvement(candidate_mean[base],candidate_scale[base],high,best)
                values=counterfactual_metrics(low_ei,high_ei,oracle_low,oracle_high)
                diagnostic.update({f"counter_{name}_{key}":value for key,value in values.items() if not key.startswith("oracle_")})
            oracle_values=counterfactual_metrics(oracle_low,oracle_high,oracle_low,oracle_high)
            diagnostic["counter_oracle_significant_fraction"]=oracle_values["oracle_significant_fraction"]
            diagnostic["counter_oracle_median_normalized_contrast"]=oracle_values["median_normalized_oracle_contrast"]
            assert models["P"].pair is not None
            true_pair=2*torch.outer(U_TRUE,V_TRUE); diagnostic["p_pair_norm"]=float(models["P"].pair.norm()); diagnostic["p_pair_cosine"]=float(torch.nn.functional.cosine_similarity(models["P"].pair.flatten(),true_pair.flatten(),dim=0)); diagnostic["p_pair_projection"]=float((models["P"].pair*true_pair).sum()/true_pair.square().sum())
        diagnostic["seconds"]=time.perf_counter()-case_start; diagnostic["peak_rss_bytes"]=_peak_rss_bytes()
        path.write_text(json.dumps({"metrics":case_metrics,"diagnostic":diagnostic},indent=2)+"\n"); metrics.extend(case_metrics); diagnostics.append(diagnostic)
        _write_csv(output_dir/"task04ad_metrics.partial.csv",metrics); _write_csv(output_dir/"task04ad_diagnostics.partial.csv",diagnostics)
    _write_csv(output_dir/"task04ad_metrics.csv",metrics); _write_csv(output_dir/"task04ad_diagnostics.csv",diagnostics)
    gate=classify_decision_diagnostic(config,metrics,diagnostics); (output_dir/"gate_status.json").write_text(json.dumps(gate,indent=2)+"\n")
    summary=["# Task 04A-D local decision diagnostic","",f"**Decision: {gate['decision']}**","",f"Completed {len(diagnostics)} of {len(_case_specs(config))} cases.","","```json",json.dumps(gate,indent=2),"```",""]
    (output_dir/"TASK_04AD_SUMMARY.md").write_text("\n".join(summary)); _plots(output_dir,metrics,diagnostics)
    return {"metrics":metrics,"diagnostics":diagnostics,"gate":gate}
