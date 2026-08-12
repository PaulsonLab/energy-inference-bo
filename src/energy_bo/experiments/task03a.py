"""Task 03A oracle capability study and smoke/full configurations."""

from __future__ import annotations

import csv
import json
import math
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from torch.quasirandom import SobolEngine

from energy_bo.oracle.warped_gp import SparseWarpedGPOracle
from energy_bo.predictive.corrected import PITCorrectedPredictive
from energy_bo.predictive.crossfit import cross_fit_map_saas, mixture_raw_context
from energy_bo.predictive.residuals import (
    ConditionalGaussianResidual,
    ContextualResidualEnergy,
    GlobalGaussianMixtureResidual,
    GlobalSkewNormalResidual,
)
from energy_bo.structural.map_saas import fit_map_saas_reference, seeded_half_cauchy_taus, map_component_mll
from energy_bo.structural.preprocessing import FrozenOutputTransform
from energy_bo.structural.saas_reference import NutsConfig, fit_saas_reference
from energy_bo.structural.exact_gp import ExactGPBatchState


@dataclass(frozen=True)
class Task03AConfig:
    profile: str
    seeds: tuple[int, ...]
    dimension: int
    counts: tuple[int, ...]
    regimes: tuple[str, ...]
    test_count: int
    candidate_count: int
    map_components: int
    map_max_iterations: int
    nuts_warmup: int
    nuts_samples: int
    nuts_thinning: int
    nuts_tree_depth: int
    fold_count: int = 4
    noise_variance: float = 1e-6

    @classmethod
    def smoke(cls) -> "Task03AConfig":
        return cls("smoke", (0,), 6, (16, 32), ("G", "W"), 128, 256, 4, 75, 32, 32, 1, 4)

    @classmethod
    def full(cls) -> "Task03AConfig":
        return cls("full", (0,1,2,3,4), 20, (16,32,64), ("G","W"), 1024, 2048, 4, 250, 512, 512, 2, 6)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows: return
    fields=sorted({key for row in rows for key in row})
    with path.open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields,lineterminator="\n"); writer.writeheader(); writer.writerows(rows)


def charged_fit_seconds(crossfit_seconds: float, final_seconds: float, calibration_seconds: float) -> float:
    return float(crossfit_seconds + final_seconds + calibration_seconds)


def _write_plots(output_dir:Path,rows:list[dict[str,Any]],timing:list[dict[str,Any]])->None:
    if not rows:return
    methods=sorted({str(r["method"]) for r in rows}); figure,axes=plt.subplots(1,2,figsize=(9,3.8))
    for method in methods:
        selected=[r for r in rows if r["method"]==method and "normalized_regret" in r]
        axes[0].scatter([r["n"] for r in selected],[r["normalized_regret"] for r in selected],s=15,label=method,alpha=.7)
    axes[0].set(xlabel="training n",ylabel="normalized true decision regret",ylim=(-.02,1.02)); axes[0].legend(fontsize=6,ncol=2)
    if timing:
        axes[1].scatter([r["nuts_seconds"] for r in timing],[r["E1_charged_fit_seconds"] for r in timing],c=[0 if r["regime"]=="G" else 1 for r in timing]); limit=max(max(float(r["nuts_seconds"] or 0) for r in timing),max(float(r["E1_charged_fit_seconds"]) for r in timing)); axes[1].plot([0,limit],[0,limit],"k--",lw=.8); axes[1].set(xlabel="NUTS fit seconds",ylabel="fully charged E1 seconds")
    figure.tight_layout(); figure.savefig(output_dir/"task03a_overview.png",dpi=160); plt.close(figure)


def _gate_status(config:Task03AConfig)->dict[str,Any]:
    return {"eligible":config.profile=="full","decision":"PENDING_REVIEW" if config.profile=="full" else "NOT_EVALUATED_FROM_SMOKE","gates":{"A":"pending full five-seed evidence","B":"pending full five-seed evidence","C":"pending full five-seed evidence","D":"pending full five-seed evidence"}}


def _median(values:list[float])->float:
    return float(np.median(values))


def analyze_full_gates(
    config: Task03AConfig,
    rows: list[dict[str, Any]],
    timing: list[dict[str, Any]],
) -> dict[str, Any]:
    expected_rows=len(config.seeds)*len(config.regimes)*len(config.counts)*8
    if config.profile!="full" or len(rows)!=expected_rows or len(timing)!=30:
        return _gate_status(config)
    lookup={(int(r["seed"]),str(r["regime"]),int(r["n"]),str(r["method"])):r for r in rows}
    diagnostics:dict[str,Any]={}
    safety={}
    for method in ("E0","E1"):
        keys=[(seed,"G",n) for seed in config.seeds for n in (16,32)]
        excess=[float(lookup[*key,method]["expected_true_nll"])-float(lookup[*key,"B1"]["expected_true_nll"]) for key in keys]
        regret=[float(lookup[*key,method]["normalized_regret"])-float(lookup[*key,"B1"]["normalized_regret"]) for key in keys]
        correction=[float(lookup[*key,method]["average_correction_kl"]) for key in keys]
        safety[method]={"mean_excess_nll":float(np.mean(excess)),"median_regret_increase":_median(regret),"regret_degradation_over_0_10":sum(value>.10 for value in regret),"median_correction_kl":_median(correction)}
        safety[method]["passed"]=(safety[method]["mean_excess_nll"]<=.01 and safety[method]["median_regret_increase"]<=.02 and safety[method]["regret_degradation_over_0_10"]<=2 and safety[method]["median_correction_kl"]<=.02)
    gate_a=all(safety[m]["passed"] for m in ("E0","E1")); diagnostics["A_safety"]=safety
    baseline_methods=("B1","B2-G","B2-C","B3","B4"); flexibility={}
    qualifier=None
    for method in ("E1","E0"):
        nll_checks={}; strongest={}
        for n in (32,64):
            means={baseline:float(np.mean([lookup[seed,"W",n,baseline]["expected_true_nll"] for seed in config.seeds])) for baseline in baseline_methods}
            strongest[n]=min(means,key=means.get); energy_mean=float(np.mean([lookup[seed,"W",n,method]["expected_true_nll"] for seed in config.seeds])); nll_checks[n]={"strongest":strongest[n],"baseline_mean_nll":means[strongest[n]],"energy_mean_nll":energy_mean,"gain":means[strongest[n]]-energy_mean}
        baseline=strongest[64]; energy_regrets=[float(lookup[seed,"W",64,method]["normalized_regret"]) for seed in config.seeds]; baseline_regrets=[float(lookup[seed,"W",64,baseline]["normalized_regret"]) for seed in config.seeds]
        baseline_median=_median(baseline_regrets); energy_median=_median(energy_regrets); absolute=baseline_median-energy_median; relative=absolute/max(baseline_median,1e-15); wins=sum(e<b for e,b in zip(energy_regrets,baseline_regrets,strict=True))
        passed=all(nll_checks[n]["gain"]>=.01 for n in (32,64)) and (absolute>=.02 or relative>=.20) and wins>=4
        flexibility[method]={"nll":nll_checks,"n64_regret_absolute_gain":absolute,"n64_regret_relative_gain":relative,"paired_seed_wins":wins,"passed":passed}
        if passed and qualifier is None: qualifier=method
    gate_b=qualifier is not None; diagnostics["B_flexibility"]={"qualifier":qualifier,"variants":flexibility}
    gate_c=False; quality={"passed":False}
    gate_d=False; unlocking={"passed":False}
    if qualifier is not None:
        regret_excess={}
        for regime in ("G","W"):
            for n in (32,64):
                regret_excess[f"{regime}_n{n}"]=_median([float(lookup[seed,regime,n,qualifier]["normalized_regret"]) for seed in config.seeds])-_median([float(lookup[seed,regime,n,"B0-NUTS"]["normalized_regret"]) for seed in config.seeds])
        charged=[float(r[f"{qualifier}_charged_fit_seconds"])/float(r["nuts_seconds"]) for r in timing]
        energy_fraction=[float(r[f"{qualifier}_fit_seconds"])/(float(r["crossfit_seconds"])+float(r["final_reference_seconds"])) for r in timing]
        quality={"regret_excess":regret_excess,"median_charged_to_nuts_ratio":_median(charged),"median_energy_to_reference_ratio":_median(energy_fraction)}
        gate_c=max(regret_excess.values())<=.02 and quality["median_charged_to_nuts_ratio"]<=.5 and quality["median_energy_to_reference_ratio"]<=.1; quality["passed"]=gate_c
        gaussian_kl={n:_median([float(lookup[seed,"G",n,qualifier]["average_correction_kl"]) for seed in config.seeds]) for n in (16,32)}
        paired=0
        for seed in config.seeds:
            kl16=float(lookup[seed,"W",16,qualifier]["average_correction_kl"]); kl64=float(lookup[seed,"W",64,qualifier]["average_correction_kl"])
            gain16=float(lookup[seed,"W",16,"B1"]["expected_true_nll"])-float(lookup[seed,"W",16,qualifier]["expected_true_nll"]); gain64=float(lookup[seed,"W",64,"B1"]["expected_true_nll"])-float(lookup[seed,"W",64,qualifier]["expected_true_nll"])
            paired+=int(kl64>kl16 and gain64>gain16)
        gate_d=max(gaussian_kl.values())<=.02 and paired>=3; unlocking={"gaussian_median_correction_kl":gaussian_kl,"warped_unlocking_seed_count":paired,"passed":gate_d}
    diagnostics["C_quality_cost"]=quality; diagnostics["D_unlocking"]=unlocking
    structural_close=all(_median([float(lookup[seed,regime,n,"B1"]["normalized_regret"]) for seed in config.seeds])-_median([float(lookup[seed,regime,n,"B0-NUTS"]["normalized_regret"]) for seed in config.seeds])<=.02 for regime in ("G","W") for n in (32,64))
    final_ratio=_median([float(r["final_reference_seconds"])/float(r["nuts_seconds"]) for r in timing]); structural_pivot=(not gate_b and structural_close and final_ratio<=.25)
    decision="GO_TASK_03B" if all((gate_a,gate_b,gate_c,gate_d)) else ("STRUCTURAL_ONLY_PIVOT" if structural_pivot else "NO_GO")
    return {"eligible":True,"decision":decision,"qualifying_energy":qualifier,"gates":{"A":gate_a,"B":gate_b,"C":gate_c,"D":gate_d},"diagnostics":diagnostics,"structural_only":{"regret_condition":structural_close,"median_final_map_to_nuts_ratio":final_ratio,"passed":structural_pivot}}


def _write_full_summary(path:Path,gate:dict[str,Any],rows:list[dict[str,Any]],timing:list[dict[str,Any]])->None:
    lines=["# Task 03A Colab summary","",f"**Decision: {gate['decision']}**","",f"The full package contains {len(rows)} primary method rows and {len(timing)} configurations.","","## Eight completion questions","","1. Cross-fit provenance is enforced in code and all PIT clamp counts/times are recorded.","2. Gaussian safety is reported by Gate A below.","3. Warped predictive flexibility against all five reference/calibration baselines is Gate B.","4. Paired n=64 decision wins and absolute/relative regret gains are in Gate B diagnostics.","5. NUTS decision quality and fully charged cost are Gate C.","6. Residual optimization fractions are included in Gate C.","7. Evidence-dependent correction behavior is Gate D.","8. Numerical identities are gated by the notebook test suite; GPU/CPU profiling is in the manifest and device profile.","","## Frozen gate evaluation","","```json",json.dumps(gate,indent=2),"```","","This automatically generated report requires local import audit before any next task. It does not authorize Task 03B by itself."]
    path.write_text("\n".join(lines)+"\n")


def _top_overlap(a: torch.Tensor,b: torch.Tensor,fraction:float=.05)->float:
    k=max(1,math.ceil(len(a)*fraction)); return len(set(torch.topk(a,k).indices.tolist()) & set(torch.topk(b,k).indices.tolist()))/k


def _decision_metrics(model_ei:torch.Tensor,true_ei:torch.Tensor)->dict[str,float|int]:
    model_ei=model_ei.detach().cpu(); true_ei=true_ei.detach().cpu(); chosen=int(torch.argmax(model_ei)); optimum=float(true_ei.max()); selected=float(true_ei[chosen]); absolute=optimum-selected; regret=absolute/max(optimum,1e-15)
    return {"ei_spearman":float(spearmanr(model_ei.numpy(),true_ei.numpy()).statistic),"top5_overlap":_top_overlap(model_ei,true_ei),"chosen_index":chosen,"true_ei_optimum":optimum,"true_ei_at_selection":selected,"absolute_decision_regret":absolute,"normalized_regret":regret,"max_ei_abs_error":float((model_ei-true_ei).abs().max())}


def _expected_nll(model:PITCorrectedPredictive,true_posterior,nodes:int=32)->float:
    device=model.reference.means.device; true_mean=true_posterior.mean.to(device); true_variance=true_posterior.variance.to(device)
    x,w=np.polynomial.hermite.hermgauss(nodes); latent=true_mean[:,None]+true_variance.sqrt()[:,None]*torch.tensor(np.sqrt(2)*x,dtype=torch.double,device=device)
    values=true_posterior.transform(latent); weights=torch.tensor(w/np.sqrt(np.pi),dtype=torch.double,device=device)
    components,points=model.reference.means.shape
    repeated=type(model.reference)(model.reference.means[:,:,None].expand(components,points,nodes).reshape(components,-1),model.reference.variances[:,:,None].expand(components,points,nodes).reshape(components,-1),model.reference.weights)
    repeated_context=model.context[:,None,:].expand(points,nodes,-1).reshape(points*nodes,-1)
    repeated_corrected=PITCorrectedPredictive(repeated,model.residual,repeated_context)
    nll=-repeated_corrected.log_prob(values.reshape(-1)).reshape(points,nodes)
    return float(torch.mean(torch.sum(nll*weights,dim=1)))


def _predictive_metrics(model:PITCorrectedPredictive,true_posterior,best:float,nodes:int)->dict[str,float]:
    device=model.reference.means.device; true_mean=true_posterior.mean.to(device); true_variance=true_posterior.variance.to(device); true_predictive_mean=true_posterior.predictive_mean.to(device); true_predictive_variance=true_posterior.predictive_variance.to(device)
    nll=_expected_nll(model,true_posterior,nodes)
    true_entropy=0.5*torch.log(2*torch.pi*math.e*true_variance)
    if true_posterior.alpha!=0: true_entropy=true_entropy+true_posterior.alpha*true_mean
    model_mean,model_variance=model.moments(quadrature_points=48 if nodes<=16 else 96)
    pi=model.probability_improvement(best); true_pi=true_posterior.probability_improvement(best).to(device)
    gh_x,gh_w=np.polynomial.hermite.hermgauss(nodes); latent=true_mean[:,None]+true_variance.sqrt()[:,None]*torch.tensor(np.sqrt(2)*gh_x,dtype=torch.double,device=device); values=true_posterior.transform(latent); weights=torch.tensor(gh_w/np.sqrt(np.pi),dtype=torch.double,device=device)
    components,points=model.reference.means.shape
    repeated=type(model.reference)(model.reference.means[:,:,None].expand(components,points,nodes).reshape(components,-1),model.reference.variances[:,:,None].expand(components,points,nodes).reshape(components,-1),model.reference.weights)
    repeated_context=model.context[:,None,:].expand(points,nodes,-1).reshape(points*nodes,-1)
    pit=PITCorrectedPredictive(repeated,model.residual,repeated_context).cdf(values.reshape(-1)).reshape(points,nodes)
    pit_mean=float(torch.mean(torch.sum(pit*weights,dim=1))); pit_second=float(torch.mean(torch.sum(pit.square()*weights,dim=1))); pit_variance=pit_second-pit_mean**2
    calibration={f"quantile_{int(level*100):02d}_error":abs(float(torch.mean(torch.sum(weights*(pit<=level),dim=1)))-level) for level in (.1,.5,.9)}
    return {"expected_true_nll":nll,"true_to_model_kl":max(0.0,nll-float(true_entropy.mean())),"mean_rmse":float(torch.mean((model_mean-true_predictive_mean).square()).sqrt()),"variance_rmse":float(torch.mean((model_variance-true_predictive_variance).square()).sqrt()),"pi_max_abs_error":float((pi-true_pi).abs().max()),"pit_mean_error":abs(pit_mean-.5),"pit_variance_error":abs(pit_variance-1/12),**calibration}


def profile_map_devices(train_x:torch.Tensor,train_y:torch.Tensor,taus:torch.Tensor,max_iterations:int)->dict[str,Any]:
    devices=["cpu"]+(["cuda"] if torch.cuda.is_available() else [])
    fits={}; predictions={}; mll={}; failures={}
    probe=train_x[:min(32,len(train_x))]
    for device in devices:
        try:
            if device=="cuda": torch.cuda.synchronize()
            start=time.perf_counter(); fit=fit_map_saas_reference(train_x,train_y,taus=taus,max_iterations=min(max_iterations,75),device=device)
            if device=="cuda": torch.cuda.synchronize()
            fits[device]=time.perf_counter()-start; predictions[device]=fit.posterior(probe).to("cpu"); mll[device]=map_component_mll(fit)
        except Exception as error:
            if device=="cpu": raise
            failures[device]=f"{type(error).__name__}: {error}"
    selected="cpu"; agreement=None
    if "cuda" in fits:
        agreement=max(float((predictions["cpu"].mean-predictions["cuda"].mean).abs().max()),float((predictions["cpu"].variance-predictions["cuda"].variance).abs().max()),float((mll["cpu"]-mll["cuda"]).abs().max()))
        if fits["cpu"]/fits["cuda"]>=1.2 and agreement<=1e-6: selected="cuda"
    return {"selected_device":selected,"seconds":fits,"failures":failures,"cpu_cuda_max_abs":agreement,"speedup":None if "cuda" not in fits else fits["cpu"]/fits["cuda"]}


def _fit_models(z:torch.Tensor,context:torch.Tensor,max_iter:int)->dict[str,Any]:
    one=context[:,:1]
    models={"B2-G":ConditionalGaussianResidual(1),"B2-C":ConditionalGaussianResidual(context.shape[1]),"B3":GlobalSkewNormalResidual(),"B4":GlobalGaussianMixtureResidual(),"E0":ContextualResidualEnergy(1),"E1":ContextualResidualEnergy(context.shape[1])}
    fits={}
    for name,model in models.items():
        start=time.perf_counter(); fit=model.fit(z,one if name in ("B2-G","B3","B4","E0") else context,max_iter=max_iter); fits[name]={"fit":fit,"seconds":time.perf_counter()-start}
    return {"models":models,"fits":fits}


def run_task03a(config:Task03AConfig,output_dir:Path,*,run_nuts:bool=True)->dict[str,Any]:
    output_dir.mkdir(parents=True,exist_ok=True); case_dir=output_dir/"cases"; case_dir.mkdir(exist_ok=True); rows=[]; timing=[]; structural=[]; precision_sensitivity=[]
    (output_dir/"config.json").write_text(json.dumps(asdict(config),indent=2)+"\n")
    map_device="cpu"; device_profile={"selected_device":"cpu","reason":"smoke fixes MAP fitting to CPU"}
    for seed in config.seeds:
        taus=seeded_half_cauchy_taus(seed+30_000,config.map_components)
        base=SparseWarpedGPOracle.generate(config.dimension,seed,64,alpha=0)
        test_x=SobolEngine(config.dimension,scramble=True,seed=seed+1000).draw(config.test_count).double()
        candidates=SobolEngine(config.dimension,scramble=True,seed=seed+2000).draw(config.candidate_count).double()
        if config.profile=="full" and seed==config.seeds[0]:
            device_profile=profile_map_devices(base.train_x_all[:16],base.outcomes(16),taus,config.map_max_iterations); map_device=str(device_profile["selected_device"]); (output_dir/"map_device_profile.json").write_text(json.dumps(device_profile,indent=2)+"\n")
        for regime in config.regimes:
            alpha=0 if regime=="G" else .6; oracle=SparseWarpedGPOracle(base.train_x_all,base.latent_all,alpha)
            for count in config.counts:
                case_path=case_dir/f"seed{seed}_{regime}_n{count}.json"
                if config.profile=="full" and case_path.exists():
                    saved=json.loads(case_path.read_text()); rows.extend(saved["rows"]); timing.append(saved["timing"]); structural.extend(saved.get("structural",[])); precision_sensitivity.extend(saved.get("precision_sensitivity",[])); continue
                row_start=len(rows); structural_start=len(structural); precision_start=len(precision_sensitivity)
                train_x=oracle.train_x_all[:count]; train_y=oracle.outcomes(count); best=float(train_y.max())
                cross=cross_fit_map_saas(train_x,train_y,taus=taus,seed=seed,max_iterations=config.map_max_iterations,noise_variance=config.noise_variance,device=map_device)
                context=cross.context(); fitted=_fit_models(cross.z,context,config.map_max_iterations)
                start=time.perf_counter(); final=fit_map_saas_reference(train_x,train_y,taus=taus,noise_variance=config.noise_variance,max_iterations=config.map_max_iterations,device=map_device); final_seconds=time.perf_counter()-start
                loo_start=time.perf_counter(); loo=final.fixed_hyperparameter_loo(); loo_seconds=time.perf_counter()-loo_start; loo_probability=loo.cdf(train_y).clamp(1e-12,1-1e-12); loo_z=torch.special.ndtri(loo_probability)
                prediction_device="cuda" if config.profile=="full" and torch.cuda.is_available() else "cpu"
                if prediction_device=="cuda": torch.cuda.reset_peak_memory_stats()
                final.to(prediction_device)
                test_reference=final.posterior(test_x); candidate_reference=final.posterior(candidates)
                test_context=cross.context(mixture_raw_context(test_reference)); candidate_context=cross.context(mixture_raw_context(candidate_reference))
                true_test=oracle.posterior(test_x,count); true_candidates=oracle.posterior(candidates,count); true_ei=true_candidates.expected_improvement(best)
                variants={"B1":None,**fitted["models"]}
                for name,residual in variants.items():
                    if residual is None:
                        residual=ContextualResidualEnergy(1); test_c=test_context[:,:1]; cand_c=candidate_context[:,:1]
                    elif name in ("B2-G","B3","B4","E0"):
                        test_c=test_context[:,:1]; cand_c=candidate_context[:,:1]
                    else: test_c=test_context; cand_c=candidate_context
                    if hasattr(residual,"to"): residual.to(prediction_device)
                    prediction_start=time.perf_counter(); corrected_test=PITCorrectedPredictive(test_reference,residual,test_c); corrected_candidates=PITCorrectedPredictive(candidate_reference,residual,cand_c); predictive_metrics=_predictive_metrics(corrected_test,true_test,best,16 if config.profile=="smoke" else 32); prediction_seconds=time.perf_counter()-prediction_start
                    normalization_start=time.perf_counter()
                    if isinstance(residual,ContextualResidualEnergy): residual.log_normalizer(cand_c)
                    normalization_seconds=time.perf_counter()-normalization_start
                    ei_start=time.perf_counter(); ei=corrected_candidates.expected_improvement(best,quadrature_points=48 if config.profile=="smoke" else 96); ei_seconds=time.perf_counter()-ei_start
                    row={"seed":seed,"regime":regime,"n":count,"method":name,**predictive_metrics,**_decision_metrics(ei.cpu(),true_ei)}
                    row.update({"prediction_seconds":prediction_seconds,"normalization_seconds":normalization_seconds,"ei_seconds":ei_seconds})
                    if name=="B1":
                        relevance=final.model.covar_module.base_kernel.lengthscale.detach().reshape(config.map_components,-1).reciprocal().median(dim=0).values.cpu()
                        row["active_top2_recall"]=len(set(torch.topk(relevance,2).indices.tolist())&{0,1})/2
                    if name.startswith("E"): row["coefficient_norm"]=float(residual.coefficients.norm()); row["average_correction_kl"]=float(residual.correction_kl(test_c).mean())
                    rows.append(row)
                map_lengthscales=final.model.covar_module.base_kernel.lengthscale.detach().reshape(config.map_components,-1).cpu()
                for dimension in range(config.dimension): structural.append({"seed":seed,"regime":regime,"n":count,"method":"B1","dimension":dimension,"median_inverse_lengthscale":float(map_lengthscales[:,dimension].reciprocal().median())})
                if config.profile=="full" and seed==0 and regime=="W" and count==64:
                    for precision in (3.0,30.0):
                        for energy_name,energy_context,test_c,cand_c in (("E0",context[:,:1],test_context[:,:1],candidate_context[:,:1]),("E1",context,test_context,candidate_context)):
                            energy=ContextualResidualEnergy(energy_context.shape[1],l2_precision=precision); start=time.perf_counter(); energy.fit(cross.z,energy_context,max_iter=config.map_max_iterations); fit_seconds=time.perf_counter()-start; energy.to(prediction_device); corrected_test=PITCorrectedPredictive(test_reference,energy,test_c); corrected_candidates=PITCorrectedPredictive(candidate_reference,energy,cand_c); ei=corrected_candidates.expected_improvement(best)
                            precision_sensitivity.append({"seed":seed,"regime":regime,"n":count,"method":energy_name,"precision":precision,"fit_seconds":fit_seconds,"coefficient_norm":float(energy.coefficients.norm()),"average_correction_kl":float(energy.correction_kl(test_c).mean()),**_predictive_metrics(corrected_test,true_test,best,32),**_decision_metrics(ei.cpu(),true_ei)})
                nuts_seconds=None
                if run_nuts:
                    transform=FrozenOutputTransform.fit(train_y); nuts=fit_saas_reference(train_x,transform.transform(train_y),config.noise_variance,NutsConfig(config.nuts_warmup,config.nuts_samples,config.nuts_thinning,config.nuts_tree_depth,seed+count)); nuts_seconds=nuts.elapsed_seconds
                    nuts_prediction_start=time.perf_counter(); state=ExactGPBatchState.build(nuts.particles,train_x,transform.transform(train_y),config.noise_variance); mean,var=state.predict(candidates,chunk_size=256); mean=transform.untransform(mean); var=var*transform.scale**2
                    from energy_bo.predictive.mixture import GaussianMixtureMarginals
                    nuts_candidate=GaussianMixtureMarginals(mean,var); nuts_ei_start=time.perf_counter(); nuts_ei=nuts_candidate.expected_improvement(best); nuts_ei_seconds=time.perf_counter()-nuts_ei_start
                    test_mean,test_var=state.predict(test_x,chunk_size=256); test_mean=transform.untransform(test_mean); test_var=test_var*transform.scale**2; nuts_test=GaussianMixtureMarginals(test_mean,test_var); identity=ContextualResidualEnergy(1); nuts_corrected=PITCorrectedPredictive(nuts_test,identity,torch.ones((config.test_count,1))); nuts_prediction_seconds=time.perf_counter()-nuts_prediction_start
                    rows.append({"seed":seed,"regime":regime,"n":count,"method":"B0-NUTS",**_predictive_metrics(nuts_corrected,true_test,best,16 if config.profile=="smoke" else 32),**_decision_metrics(nuts_ei,true_ei),"prediction_seconds":nuts_prediction_seconds,"ei_seconds":nuts_ei_seconds,"active_top2_recall":len(set(torch.topk(nuts.particles.lengthscales.reciprocal().median(0).values,2).indices.tolist())&{0,1})/2})
                    for dimension in range(config.dimension): structural.append({"seed":seed,"regime":regime,"n":count,"method":"B0-NUTS","dimension":dimension,"median_inverse_lengthscale":float(nuts.particles.lengthscales[:,dimension].reciprocal().median())})
                residual_times={name:float(item["seconds"]) for name,item in fitted["fits"].items()}
                peak_cuda_memory=0 if prediction_device=="cpu" else int(torch.cuda.max_memory_allocated())
                crossfit_forward=sum(int(record.fit_info["forward_evaluations"]) for record in cross.folds if record.fit_info is not None)
                crossfit_iterations=sum(int(record.fit_info["optimizer_iterations"]) for record in cross.folds if record.fit_info is not None)
                timing_row={"seed":seed,"regime":regime,"n":count,"map_device":map_device,"prediction_device":prediction_device,"peak_torch_cuda_bytes":peak_cuda_memory,"crossfit_forward_evaluations":crossfit_forward,"crossfit_optimizer_iterations":crossfit_iterations,"crossfit_seconds":cross.elapsed_seconds,"final_reference_seconds":final_seconds,"analytic_loo_seconds":loo_seconds,"crossfit_pit_mean":float(cross.z.mean()),"crossfit_pit_std":float(cross.z.std(unbiased=False)),"loo_pit_mean":float(loo_z.mean()),"loo_pit_std":float(loo_z.std(unbiased=False)),"nuts_seconds":nuts_seconds,"pit_clamps":cross.clamp_count,"map_optimizer_iterations":final.fit_info.optimizer_iterations,"map_optimizer_callbacks":final.fit_info.optimizer_callbacks,"map_retries":final.fit_info.retries,"map_forward_evaluations":final.fit_info.forward_evaluations,**{f"{name}_fit_seconds":value for name,value in residual_times.items()},**{f"{name}_charged_fit_seconds":charged_fit_seconds(cross.elapsed_seconds,final_seconds,value) for name,value in residual_times.items()}}
                timing.append(timing_row)
                if config.profile=="full": case_path.write_text(json.dumps({"rows":rows[row_start:],"timing":timing_row,"structural":structural[structural_start:],"precision_sensitivity":precision_sensitivity[precision_start:]},indent=2)+"\n")
                _write_csv(output_dir/"metrics.partial.csv",rows); _write_csv(output_dir/"timing.partial.csv",timing)
    if config.profile=="full":
        sensitivity=[]
        for seed in config.seeds:
            base=SparseWarpedGPOracle.generate(config.dimension,seed,64,alpha=.6); train_x=base.train_x_all[:32]; train_y=base.outcomes(32); taus=seeded_half_cauchy_taus(seed+80_000,8); start=time.perf_counter(); reference=fit_map_saas_reference(train_x,train_y,taus=taus,noise_variance=config.noise_variance,max_iterations=config.map_max_iterations,device=map_device); fit_seconds=time.perf_counter()-start
            test_x=SobolEngine(config.dimension,scramble=True,seed=seed+1000).draw(config.test_count).double(); candidates=SobolEngine(config.dimension,scramble=True,seed=seed+2000).draw(config.candidate_count).double(); best=float(train_y.max()); test_reference=reference.posterior(test_x); candidate_reference=reference.posterior(candidates); identity=ContextualResidualEnergy(1); corrected=PITCorrectedPredictive(test_reference,identity,torch.ones((config.test_count,1))); true_test=base.posterior(test_x,32); true_candidates=base.posterior(candidates,32)
            sensitivity.append({"seed":seed,"regime":"W","n":32,"components":8,"fit_seconds":fit_seconds,"optimizer_iterations":reference.fit_info.optimizer_iterations,**_predictive_metrics(corrected,true_test,best,32),**_decision_metrics(candidate_reference.expected_improvement(best).cpu(),true_candidates.expected_improvement(best))})
        _write_csv(output_dir/"map_components_sensitivity.csv",sensitivity)
    _write_csv(output_dir/"metrics.csv",rows); _write_csv(output_dir/"timing.csv",timing); _write_csv(output_dir/"structural_relevance.csv",structural); _write_csv(output_dir/"precision_sensitivity.csv",precision_sensitivity)
    _write_plots(output_dir,rows,timing); gates=analyze_full_gates(config,rows,timing); (output_dir/"gate_status.json").write_text(json.dumps(gates,indent=2)+"\n")
    if config.profile=="full": _write_full_summary(output_dir/"TASK_03A_COLAB_SUMMARY.md",gates,rows,timing)
    summary={"profile":config.profile,"scientific_evidence":config.profile=="full","rows":len(rows),"timing_rows":len(timing),"map_device_profile":device_profile,"gate_status":gates,"status":"smoke wiring only" if config.profile=="smoke" else "full run requires reviewed gate analysis","environment":{"python":platform.python_version(),"torch":torch.__version__}}
    (output_dir/"SUMMARY.json").write_text(json.dumps(summary,indent=2)+"\n"); return summary
