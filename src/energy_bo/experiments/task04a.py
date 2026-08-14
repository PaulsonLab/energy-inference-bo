"""Task 04A local conditional energy oracle study."""

from __future__ import annotations

import csv
import json
import math
import platform
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

from energy_bo.local_process.energy import LocalEnergyModel, neighbor_summary
from energy_bo.local_process.evaluation import decision_metrics, oracle_predictive_metrics, predictive_metrics
from energy_bo.local_process.geometry import build_geometry, candidate_geometry, ordered_sobol
from energy_bo.local_process.truth import V_TRUE, generate_oracle


@dataclass(frozen=True)
class Task04AConfig:
    profile: str
    seeds: tuple[int, ...]
    dimension: int
    counts: tuple[int, ...]
    regimes: tuple[str, ...]
    realization_count: int
    test_count: int
    candidate_count: int
    neighborhood_size: int = 8
    quadrature_points: int = 64
    metric_points: int = 160
    l2_precision: float = 10.0
    max_iterations: int = 250
    output_center: float = 0.0
    output_scale: float = 1.0
    child_feature_coordinate: str = "local_standardized_residual"

    @classmethod
    def smoke(cls) -> "Task04AConfig":
        return cls("smoke", (0,), 6, (64,), ("G", "W", "I"), 64, 128, 256, max_iterations=75, metric_points=96)

    @classmethod
    def full(cls) -> "Task04AConfig":
        return cls("full", (0,1,2,3,4), 10, (64,128,256,512), ("G","W","I"), 512, 1024, 2048)

    @classmethod
    def preflight(cls) -> "Task04AConfig":
        return cls("preflight", (0,1,2), 6, (64,128,256), ("G","I"), 256, 256, 512, max_iterations=100, metric_points=128)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows: return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields,lineterminator="\n"); writer.writeheader(); writer.writerows(rows)


def _sync(device: torch.device) -> None:
    if device.type == "cuda": torch.cuda.synchronize(device)


def _system(device: torch.device) -> dict[str, Any]:
    return {"python":platform.python_version(),"torch":torch.__version__,"device":str(device),"device_name":torch.cuda.get_device_name(device) if device.type=="cuda" else platform.processor(),"cuda":torch.version.cuda,"torch_double":True}


def _chunked_ei(model: LocalEnergyModel, mean: torch.Tensor, scale: torch.Tensor, summary: torch.Tensor, best: float, chunk: int = 512) -> torch.Tensor:
    return torch.cat([model.expected_improvement(mean[start:start+chunk],scale[start:start+chunk],summary[start:start+chunk],best) for start in range(0,len(mean),chunk)])


def _chunked_cdf(model: LocalEnergyModel, value: torch.Tensor, mean: torch.Tensor, scale: torch.Tensor, summary: torch.Tensor, chunk: int = 512) -> torch.Tensor:
    return torch.cat([model.cdf(value[start:start+chunk],mean[start:start+chunk],scale[start:start+chunk],summary[start:start+chunk]) for start in range(0,len(mean),chunk)])


def _gate_placeholder(profile: str) -> dict[str, Any]:
    return {"eligible":profile=="full","decision":"PENDING_REVIEW" if profile=="full" else "NOT_EVALUATED_FROM_SMOKE","gates":{"A":False,"B":False,"C":False,"D":False}}


def analyze_preflight(config: Task04AConfig, rows: list[dict[str, Any]], timing: list[dict[str, Any]]) -> dict[str, Any]:
    """Frozen CPU check that decides whether the expensive full profile is warranted."""
    expected=len(config.seeds)*len(config.regimes)*len(config.counts)*4
    if config.profile!="preflight" or len(rows)!=expected or len(timing)!=18:
        return {"eligible":False,"decision":"INCOMPLETE_PREFLIGHT","gates":{}}
    lookup={(int(r["seed"]),str(r["regime"]),int(r["n"]),str(r["method"])):r for r in rows}
    safety={}; safety_pass=True
    for method in ("U","P"):
        cells=[(seed,n) for seed in config.seeds for n in (64,128)]
        correction=[float(lookup[s,"G",n,method]["average_correction_kl"]) for s,n in cells]
        regret=[float(lookup[s,"G",n,method]["normalized_regret"])-float(lookup[s,"G",n,"G0"]["normalized_regret"]) for s,n in cells]
        result={"median_correction_kl":float(np.median(correction)),"median_regret_increase":float(np.median(regret)),"regret_degradation_over_0_10":sum(value>.1 for value in regret)}
        result["passed"]=result["median_correction_kl"]<=.02 and result["median_regret_increase"]<=.02 and result["regret_degradation_over_0_10"]<=1
        safety[method]=result; safety_pass &= result["passed"]
    density={}; density_pass=True
    decision_cells=[]
    for n in (128,256):
        g0=[lookup[s,"I",n,"G0"] for s in config.seeds]
        u=[lookup[s,"I",n,"U"] for s in config.seeds]
        p=[lookup[s,"I",n,"P"] for s in config.seeds]
        g0_kl=float(np.mean([float(r["conditional_kl"]) for r in g0]))
        u_kl=float(np.mean([float(r["conditional_kl"]) for r in u])); p_kl=float(np.mean([float(r["conditional_kl"]) for r in p]))
        gain=u_kl-p_kl; wins=sum(float(pr["conditional_kl"])<float(ur["conditional_kl"]) for ur,pr in zip(u,p,strict=True))
        passed=g0_kl>=.01 and gain>=.005 and gain/max(u_kl,1e-15)>=.2 and wins==len(config.seeds)
        density[f"I_n{n}"]={"available_g0_kl":g0_kl,"u_mean_kl":u_kl,"p_mean_kl":p_kl,"p_gain":gain,"relative_gain":gain/max(u_kl,1e-15),"paired_kl_wins":wins,"passed":passed}
        density_pass &= passed
        decision_cells.extend((float(ur["normalized_regret"]),float(pr["normalized_regret"])) for ur,pr in zip(u,p,strict=True))
    signal_cells=sum(u_regret>=.01 for u_regret,_ in decision_cells)
    p_wins=sum(p_regret<u_regret-1e-12 for u_regret,p_regret in decision_cells)
    u_median=float(np.median([u for u,_ in decision_cells])); p_median=float(np.median([p for _,p in decision_cells]))
    decision_pass=signal_cells>=2 and p_wins>=2 and p_median<=u_median
    compute_pass=all(bool(t["finite"]) and bool(t["converged"]) and int(t["global_n_by_n_factorizations"])==0 for t in timing)
    gates={"safety":safety_pass,"density_signal_and_learning":density_pass,"decision_relevance":decision_pass,"compute":compute_pass}
    return {"eligible":True,"decision":"READY_FOR_FULL" if all(gates.values()) else "PAUSE_FULL_STUDY","gates":gates,"diagnostics":{"safety":safety,"density":density,"decision":{"u_regret_signal_cells":signal_cells,"p_regret_wins":p_wins,"u_median_regret":u_median,"p_median_regret":p_median},"compute":{"all_finite_converged":compute_pass}}}


def analyze_full_gates(config: Task04AConfig, rows: list[dict[str, Any]], timing: list[dict[str, Any]]) -> dict[str, Any]:
    expected=len(config.seeds)*len(config.regimes)*len(config.counts)*4
    if config.profile!="full" or len(rows)!=expected or len(timing)!=60: return _gate_placeholder(config.profile)
    lookup={(int(r["seed"]),str(r["regime"]),int(r["n"]),str(r["method"])):r for r in rows}
    safety={}; gate_a=True
    for method in ("U","P"):
        keys=[(seed,"G",n) for seed in config.seeds for n in (64,128)]
        kls=[float(lookup[*key,method]["average_correction_kl"]) for key in keys]
        regrets=[float(lookup[*key,method]["normalized_regret"])-float(lookup[*key,"G0"]["normalized_regret"]) for key in keys]
        result={"median_correction_kl":float(np.median(kls)),"median_regret_increase":float(np.median(regrets)),"regret_degradation_over_0_10":sum(v>.1 for v in regrets)}
        result["passed"]=result["median_correction_kl"]<=.02 and result["median_regret_increase"]<=.02 and result["regret_degradation_over_0_10"]<=2
        safety[method]=result; gate_a &= result["passed"]
    pairwise={}; gate_b=True
    for regime,n,required_wins in (("I",128,None),("I",256,4),("W",256,3)):
        u=[lookup[s,regime,n,"U"] for s in config.seeds]; p=[lookup[s,regime,n,"P"] for s in config.seeds]
        u_kl=float(np.mean([float(r["conditional_kl"]) for r in u])); p_kl=float(np.mean([float(r["conditional_kl"]) for r in p])); kl_gain=u_kl-p_kl
        u_reg=float(np.median([float(r["normalized_regret"]) for r in u])); p_reg=float(np.median([float(r["normalized_regret"]) for r in p])); reg_gain=u_reg-p_reg
        wins=sum(float(pr["normalized_regret"])<float(ur["normalized_regret"]) for ur,pr in zip(u,p,strict=True))
        kl_pass=kl_gain>=.01 and (regime!="I" or kl_gain/max(u_kl,1e-15)>=.2)
        decision_pass=True if required_wins is None else ((reg_gain>=.02 or reg_gain/max(u_reg,1e-15)>=.2) and wins>=required_wins)
        passed=kl_pass and decision_pass
        pairwise[f"{regime}_n{n}"]={"u_mean_kl":u_kl,"p_mean_kl":p_kl,"kl_gain":kl_gain,"relative_kl_gain":kl_gain/max(u_kl,1e-15),"u_median_regret":u_reg,"p_median_regret":p_reg,"regret_gain":reg_gain,"paired_regret_wins":wins,"passed":passed}; gate_b &= passed
    p_i=[lookup[s,"I",256,"P"] for s in config.seeds]
    gate_c=float(np.mean([float(r["conditional_kl"]) for r in p_i]))<=.05 and float(np.median([float(r["normalized_regret"]) for r in p_i]))<=.1 and pairwise["I_n128"]["passed"]
    ratios=[]
    for method in ("U","P"):
        med={n:float(np.median([float(t[f"{method}_objective_gradient_seconds"]) for t in timing if int(t["n"])==n])) for n in (128,256,512)}
        ratios.extend((med[256]/max(med[128],1e-15),med[512]/max(med[256],1e-15)))
    gate_d=all(bool(t["finite"]) and bool(t["converged"]) and float(t["peak_cuda_bytes"])<4e9 and int(t["global_n_by_n_factorizations"])==0 for t in timing) and max(ratios)<=2.5
    decision="GO_TASK_04B" if all((gate_a,gate_b,gate_c,gate_d)) else ("STRONG_NO_GO" if not pairwise["I_n256"]["passed"] else "NO_GO")
    return {"eligible":True,"decision":decision,"gates":{"A":gate_a,"B":gate_b,"C":gate_c,"D":gate_d},"diagnostics":{"A_safety":safety,"B_pairwise":pairwise,"C":{"p_i_n256_mean_kl":float(np.mean([float(r["conditional_kl"]) for r in p_i])),"p_i_n256_median_regret":float(np.median([float(r["normalized_regret"]) for r in p_i]))},"D":{"doubling_ratios":ratios}}}


def _plots(output: Path, rows: list[dict[str, Any]], timing: list[dict[str, Any]]) -> None:
    specs=[("kl_learning_curve","conditional_kl","Conditional KL"),("regret","normalized_regret","Normalized regret")]
    for filename,key,label in specs:
        figure,axes=plt.subplots(1,3,figsize=(11,3.2),sharey=True)
        for axis,regime in zip(axes,("G","W","I"),strict=True):
            for method in ("G0","U","P"):
                selected=[r for r in rows if r["regime"]==regime and r["method"]==method]
                if selected: axis.scatter([r["n"] for r in selected],[r[key] for r in selected],s=14,label=method,alpha=.7)
            axis.set_title(regime); axis.set_xlabel("n")
        axes[0].set_ylabel(label); axes[-1].legend(fontsize=7); figure.tight_layout(); figure.savefig(output/f"task04a_{filename}.png",dpi=160); plt.close(figure)
    figure,axis=plt.subplots(figsize=(5,3.3))
    for method in ("U","P"):
        axis.scatter([t["n"] for t in timing],[t[f"{method}_fit_seconds"] for t in timing],label=method,s=16)
    axis.set(xlabel="n",ylabel="fit seconds"); axis.legend(); figure.tight_layout(); figure.savefig(output/"task04a_timing_scaling.png",dpi=160); plt.close(figure)
    extras=[("gaussian_safety","average_correction_kl","Average correction KL",lambda r:r["regime"]=="G"),("interaction_context","truth_context_median_abs","Median |interaction context|",lambda r:r["regime"]=="I"),("ei","ei_spearman","EI Spearman",lambda r:r["method"]!="Oracle")]
    for filename,key,label,condition in extras:
        figure,axis=plt.subplots(figsize=(5,3.3))
        for method in ("G0","U","P"):
            selected=[r for r in rows if r["method"]==method and condition(r)]
            axis.scatter([r["n"] for r in selected],[r[key] for r in selected],label=method,s=16,alpha=.7)
        axis.set(xlabel="n",ylabel=label); axis.legend(); figure.tight_layout(); figure.savefig(output/f"task04a_{filename}.png",dpi=160); plt.close(figure)


def run_task04a(config: Task04AConfig, output_dir: Path, *, device: str | None = None) -> dict[str, Any]:
    selected=torch.device(device or ("cuda" if config.profile=="full" and torch.cuda.is_available() else "cpu"))
    if config.profile=="full" and selected.type!="cuda": raise RuntimeError("full Task 04A requires explicit CUDA/A100 execution")
    if config.profile=="full" and "A100" not in torch.cuda.get_device_name(selected): raise RuntimeError(f"full Task 04A requires an A100, got {torch.cuda.get_device_name(selected)}")
    output_dir.mkdir(parents=True,exist_ok=True); cases=output_dir/"cases"; cases.mkdir(exist_ok=True)
    (output_dir/"task04a_config.json").write_text(json.dumps(asdict(config),indent=2)+"\n")
    (output_dir/"runtime.json").write_text(json.dumps(_system(selected),indent=2)+"\n")
    rows=[]; timing=[]; parameters=[]
    for seed in config.seeds:
        geometry_start=time.perf_counter(); x=ordered_sobol(config.dimension,config.realization_count,seed)
        test_x=SobolEngine(config.dimension,scramble=True,seed=seed+1000).draw(config.test_count).double()
        candidates=SobolEngine(config.dimension,scramble=True,seed=seed+2000).draw(config.candidate_count).double()
        shared_geometry=build_geometry(x,config.neighborhood_size); training_geometry_seconds=time.perf_counter()-geometry_start
        oracles={regime:generate_oracle(x,regime,seed,config.neighborhood_size,shared_geometry) for regime in config.regimes}
        for regime,oracle in oracles.items():
            for n in config.counts:
                path=cases/f"seed{seed}_{regime}_n{n}.json"
                if path.exists():
                    saved=json.loads(path.read_text()); rows.extend(saved["rows"]); timing.append(saved["timing"]); parameters.extend(saved["parameters"]); continue
                train_x=x[:n]; train_y=oracle.values[:n]
                train_geometry=oracle.geometry
                # Geometry is nested, so its prefix remains valid.
                train_geometry=type(train_geometry)(train_geometry.x[:n],train_geometry.neighbors[:n],train_geometry.mask[:n],train_geometry.coefficients[:n],train_geometry.variances[:n],train_geometry.similarity_weights[:n],train_geometry.jitter)
                start=time.perf_counter(); mean=train_geometry.means(train_y).to(selected); scale=train_geometry.variances.sqrt().to(selected); y=train_y.to(selected); _sync(selected); reference_seconds=time.perf_counter()-start
                start=time.perf_counter(); summary=neighbor_summary(train_y,train_geometry.neighbors,train_geometry.mask,train_geometry.similarity_weights).to(selected); _sync(selected); feature_seconds=time.perf_counter()-start
                if selected.type=="cuda": torch.cuda.reset_peak_memory_stats(selected)
                models={"G0":LocalEnergyModel(False).to(selected),"U":LocalEnergyModel(False,l2_precision=config.l2_precision).to(selected),"P":LocalEnergyModel(True,l2_precision=config.l2_precision).to(selected)}
                fits={name:model.fit(y,mean,scale,summary,max_iter=config.max_iterations) for name,model in models.items() if name!="G0"}
                start=time.perf_counter(); test_geometry=candidate_geometry(train_x,test_x,config.neighborhood_size).to(selected); candidate_geo=candidate_geometry(train_x,candidates,config.neighborhood_size).to(selected); _sync(selected); candidate_neighborhood_seconds=time.perf_counter()-start
                source=train_y.to(selected); test_mean=test_geometry.means(source); test_scale=test_geometry.variances.sqrt(); test_summary=neighbor_summary(source,test_geometry.neighbors,test_geometry.mask,test_geometry.similarity_weights)
                candidate_mean=candidate_geo.means(source); candidate_scale=candidate_geo.variances.sqrt(); candidate_summary=neighbor_summary(source,candidate_geo.neighbors,candidate_geo.mask,candidate_geo.similarity_weights)
                best=float(train_y.max()); true_ei=oracle.expected_improvement(candidate_geo,source,best)
                _,_,truth_test_summary,_=oracle.conditional(test_geometry,source)
                truth_context=truth_test_summary@V_TRUE.to(selected) if regime=="I" else torch.zeros(len(test_x),dtype=torch.double,device=selected)
                context_diagnostics={"truth_context_min":float(truth_context.min()),"truth_context_max":float(truth_context.max()),"truth_context_median_abs":float(truth_context.abs().median())}
                start=time.perf_counter()
                for model in models.values(): model.log_normalizer(mean,scale,summary)
                _sync(selected); normalization_seconds=time.perf_counter()-start
                case_rows=[]
                oracle_decision=decision_metrics(true_ei,true_ei,candidates)
                oracle_scores=oracle_predictive_metrics(oracle,test_geometry,source,points=config.metric_points)
                case_rows.append({"seed":seed,"regime":regime,"n":n,"method":"Oracle",**oracle_scores,**oracle_decision,"pi_max_abs_error":0.0,**context_diagnostics})
                prediction_seconds=0.0; ei_seconds=0.0
                for name,model in models.items():
                    prediction_start=time.perf_counter()
                    predictive=predictive_metrics(oracle,test_geometry,source,model,test_mean,test_scale,test_summary,points=config.metric_points)
                    predictive["average_correction_kl"]=float(model.correction_kl(test_mean,test_scale,test_summary).mean())
                    _sync(selected); prediction_seconds += time.perf_counter()-prediction_start
                    ei_start=time.perf_counter()
                    estimate=_chunked_ei(model,candidate_mean,candidate_scale,candidate_summary,best)
                    model_pi=1-_chunked_cdf(model,torch.full_like(candidate_mean,best),candidate_mean,candidate_scale,candidate_summary)
                    if regime=="W":
                        latent_best=torch.log1p(.6*torch.as_tensor(best,dtype=torch.double,device=selected))/.6
                        true_pi=.5*(1+torch.erf((candidate_mean-latent_best)/candidate_scale/math.sqrt(2)))
                    elif regime=="G":
                        true_pi=.5*(1+torch.erf((candidate_mean-best)/candidate_scale/math.sqrt(2)))
                    else:
                        truth_mean,truth_scale,truth_summary,truth_model=oracle.conditional(candidate_geo,source)
                        true_pi=1-truth_model.cdf(torch.full_like(truth_mean,best),truth_mean,truth_scale,truth_summary)
                    decision=decision_metrics(estimate,true_ei,candidates); _sync(selected); ei_seconds += time.perf_counter()-ei_start
                    case_rows.append({"seed":seed,"regime":regime,"n":n,"method":name,**predictive,**decision,"pi_max_abs_error":float((model_pi-true_pi).abs().max()),**context_diagnostics})
                    parameters.append({"seed":seed,"regime":regime,"n":n,"method":name,"unary_norm":float(model.unary.norm()),"pair_norm":0.0 if model.pair is None else float(model.pair.norm()),"parameter_count":0 if name=="G0" else model.parameter_count})
                peak=torch.cuda.max_memory_allocated(selected) if selected.type=="cuda" else 0
                case_timing={"seed":seed,"regime":regime,"n":n,"training_geometry_seconds":training_geometry_seconds,"candidate_neighborhood_seconds":candidate_neighborhood_seconds,"local_reference_seconds":reference_seconds,"feature_seconds":feature_seconds,"normalization_seconds":normalization_seconds,"U_fit_seconds":fits["U"].seconds,"P_fit_seconds":fits["P"].seconds,"U_objective_gradient_seconds":fits["U"].objective_gradient_seconds,"P_objective_gradient_seconds":fits["P"].objective_gradient_seconds,"prediction_metrics_seconds":prediction_seconds,"ei_seconds":ei_seconds,"U_iterations":fits["U"].iterations,"P_iterations":fits["P"].iterations,"converged":fits["U"].converged and fits["P"].converged,"finite":all(math.isfinite(float(v)) for r in case_rows for v in r.values() if isinstance(v,(float,int))),"peak_cuda_bytes":peak,"global_n_by_n_factorizations":0}
                saved={"rows":case_rows,"timing":case_timing,"parameters":parameters[-3:]}; path.write_text(json.dumps(saved,indent=2)+"\n"); rows.extend(case_rows); timing.append(case_timing)
                _write_csv(output_dir/"task04a_metrics.partial.csv",rows); _write_csv(output_dir/"task04a_timing.partial.csv",timing)
    _write_csv(output_dir/"task04a_metrics.csv",rows); _write_csv(output_dir/"task04a_timing.csv",timing); _write_csv(output_dir/"task04a_parameters.csv",parameters)
    scaling=[]
    for n in config.counts:
        for method in ("U","P"):
            values=[float(t[f"{method}_fit_seconds"]) for t in timing if int(t["n"])==n]
            objective=[float(t[f"{method}_objective_gradient_seconds"]) for t in timing if int(t["n"])==n]
            scaling.append({"n":n,"method":method,"median_fit_seconds":float(np.median(values)),"mean_fit_seconds":float(np.mean(values)),"median_objective_gradient_seconds":float(np.median(objective)),"mean_objective_gradient_seconds":float(np.mean(objective))})
    _write_csv(output_dir/"task04a_scaling.csv",scaling)
    gate=analyze_preflight(config,rows,timing) if config.profile=="preflight" else analyze_full_gates(config,rows,timing)
    (output_dir/"gate_status.json").write_text(json.dumps(gate,indent=2)+"\n")
    interpretation="The preflight is a frozen authorization check for the full profile." if config.profile=="preflight" else "Smoke output validates wiring only; only the complete five-seed full profile may evaluate the frozen gates."
    summary=["# Task 04A run summary","",f"**Decision: {gate['decision']}**","",f"Profile `{config.profile}` produced {len(rows)} metric rows across {len(timing)} cases.","",interpretation,"","```json",json.dumps(gate,indent=2),"```",""]
    (output_dir/"TASK_04A_COLAB_SUMMARY.md").write_text("\n".join(summary)); _plots(output_dir,rows,timing)
    return {"rows":rows,"timing":timing,"parameters":parameters,"gate":gate}
