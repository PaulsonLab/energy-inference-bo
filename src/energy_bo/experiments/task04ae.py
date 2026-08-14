"""Task 04A-E matched-marginal q=2 dependence experiment."""

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

from energy_bo.local_process.copulas import MatchedCopulaOracle, calibrate_t_copula_correlation
from energy_bo.local_process.joint_decision import paired_batch_panel, paired_endpoint_metrics, tie_aware_decision_metrics
from energy_bo.local_process.joint_energy import BivariateEnergyModel


@dataclass(frozen=True)
class Task04AEConfig:
    profile: str
    seeds: tuple[int, ...]
    sizes: tuple[int, ...]
    quadrature_points: int
    evaluation_power: int
    calibration_power: int = 20
    verification_power: int = 18
    max_iterations: int = 250
    l2_precision: float = 10.0
    incumbent: float = 1.5
    panel_count: int = 128

    @classmethod
    def smoke(cls) -> "Task04AEConfig":
        return cls("smoke", (0,), (128,), 32, 16)

    @classmethod
    def full(cls) -> "Task04AEConfig":
        return cls("full", tuple(range(8)), (64, 128, 256, 512), 48, 18)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def _peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def _truth_expectation(
    oracle: MatchedCopulaOracle,
    gaussian: torch.Tensor,
    student: torch.Tensor,
    context: float,
    function,
) -> torch.Tensor:
    left = function(gaussian)
    right = function(student)
    return (1 - context) * left.mean() + context * right.mean()


def _density_rows(
    oracle: MatchedCopulaOracle,
    endpoint,
    models: dict[str, BivariateEnergyModel],
    seed: int,
    n: int,
    best: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metrics: list[dict[str, Any]] = []
    marginal: list[dict[str, Any]] = []
    contexts = torch.tensor([0, .25, .5, .75, 1], dtype=torch.double)
    standard_mean = torch.zeros(5, 2, dtype=torch.double)
    standard_scale = torch.ones(5, 2, dtype=torch.double)
    truth_g, truth_t = oracle.endpoint_qei(endpoint, standard_mean[:1], standard_scale[:1], best)
    truth_qei = (1 - contexts) * truth_g[0] + contexts * truth_t[0]
    true_q1 = oracle.q1_ei(best)
    for name, model in models.items():
        device = model.parameter.device
        device_contexts = contexts.to(device)
        predicted_qei = model.qei(standard_mean.to(device), standard_scale.to(device), device_contexts, best).cpu()
        marginal_result = model.marginal_metrics(device_contexts, best)
        means, variances, correlations = model.moments(device_contexts)
        cdf_values = torch.tensor([-1.5, 0.0, 1.5], dtype=torch.double, device=device)
        repeated_context = device_contexts[:, None].expand(-1, 3)
        cdf = model.marginal_cdf(cdf_values[None].expand(5, -1), repeated_context)
        true_cdf = 0.5 * (1 + torch.erf(cdf_values / math.sqrt(2)))
        for index, context in enumerate(contexts.tolist()):
            r = torch.full((len(endpoint.gaussian),), context, dtype=torch.double)
            true_log_g = oracle.log_prob(endpoint.gaussian, context)
            true_log_t = oracle.log_prob(endpoint.student, context)
            model_log_g = model.log_prob(endpoint.gaussian.to(device), r.to(device)).cpu()
            model_log_t = model.log_prob(endpoint.student.to(device), r.to(device)).cpu()
            cross_entropy = float(-((1-context)*model_log_g.mean()+context*model_log_t.mean()))
            entropy = float(-((1-context)*true_log_g.mean()+context*true_log_t.mean()))
            weights = model.normalized_node_weights(torch.tensor([context],dtype=torch.double,device=device))[0]
            nodes = model.base_nodes.to(device)
            tail = float((weights * ((nodes[:,0] > best) & (nodes[:,1] > best))).sum())
            true_tail = float(_truth_expectation(oracle, endpoint.gaussian, endpoint.student, context, lambda value: ((value[:,0]>best)&(value[:,1]>best)).double()))
            metrics.append({
                "seed":seed,"n":n,"context":context,"method":name,
                "joint_nll":cross_entropy,"joint_kl":max(0.0,cross_entropy-entropy),
                "qei":float(predicted_qei[index]),"true_qei":float(truth_qei[index]),
                "qei_abs_error":abs(float(predicted_qei[index]-truth_qei[index])),
                "qei_relative_error":abs(float(predicted_qei[index]-truth_qei[index]))/max(float(truth_qei[index]),1e-15),
                "tail_coexceedance":tail,"true_tail_coexceedance":true_tail,
                "tail_abs_error":abs(tail-true_tail),
            })
            marginal.append({
                "seed":seed,"n":n,"context":context,"method":name,
                "marginal_kl":float(marginal_result["kl"][index]),
                "marginal_mean_abs_error":abs(float(marginal_result["mean"][index])),
                "marginal_variance_abs_error":abs(float(marginal_result["variance"][index]-1)),
                "pearson_abs_error":abs(float(correlations[index]-.5)),
                "q1_ei":float(marginal_result["q1_ei"][index]),
                "q1_ei_relative_error":abs(float(marginal_result["q1_ei"][index])-true_q1)/true_q1,
                "marginal_cdf_max_error":float((cdf[index]-true_cdf).abs().max()),
                "joint_mean_max_abs":float(means[index].abs().max()),
                "joint_variance_max_abs_error":float((variances[index]-1).abs().max()),
            })
    return metrics, marginal


def _panel_rows(
    oracle: MatchedCopulaOracle,
    endpoint,
    models: dict[str, BivariateEnergyModel],
    seed: int,
    n: int,
    best: float,
    count: int,
) -> list[dict[str, Any]]:
    panel = paired_batch_panel(count)
    gaussian, student = oracle.endpoint_qei(endpoint, panel.mean[::2], panel.scale[::2], best)
    truth = torch.stack((gaussian, student), -1).reshape(-1)
    rows: list[dict[str, Any]] = []
    for name, model in models.items():
        device = model.parameter.device
        estimate = model.qei(panel.mean.to(device), panel.scale.to(device), panel.context.to(device), best).cpu()
        row = {"seed":seed,"n":n,"method":name,**tie_aware_decision_metrics(estimate,truth),**paired_endpoint_metrics(estimate,truth)}
        rows.append(row)
    return rows


def oracle_preflight(config: Task04AEConfig) -> tuple[MatchedCopulaOracle, Any, dict[str, Any]]:
    calibration = calibrate_t_copula_correlation(calibration_power=config.calibration_power,verification_power=config.verification_power)
    oracle = MatchedCopulaOracle(latent_rho=calibration.latent_rho)
    endpoint = oracle.endpoint_qmc(config.evaluation_power, 40_406)
    means = {name:[float(value) for value in sample.mean(0)] for name,sample in (("gaussian",endpoint.gaussian),("student",endpoint.student))}
    variances = {name:[float(value) for value in sample.var(0,unbiased=False)] for name,sample in (("gaussian",endpoint.gaussian),("student",endpoint.student))}
    correlations = {name:float((sample[:,0]*sample[:,1]).mean()) for name,sample in (("gaussian",endpoint.gaussian),("student",endpoint.student))}
    mean = torch.zeros(1,2,dtype=torch.double); scale = torch.ones(1,2,dtype=torch.double)
    gaussian_qei, student_qei = oracle.endpoint_qei(endpoint,mean,scale,config.incumbent)
    contrast = abs(float(student_qei-gaussian_qei))/float(gaussian_qei)
    panel = paired_batch_panel(config.panel_count)
    panel_g,panel_t=oracle.endpoint_qei(endpoint,panel.mean[::2],panel.scale[::2],config.incumbent)
    panel_truth=torch.stack((panel_g,panel_t),-1).reshape(-1); panel_base=torch.stack((panel_g,panel_g),-1).reshape(-1)
    pair=paired_endpoint_metrics(panel_base,panel_truth); decision=tie_aware_decision_metrics(panel_base,panel_truth)
    calibration_ok=max(abs(value-.5) for value in calibration.verification_correlations)<.005
    moment_ok=max(abs(value) for values in means.values() for value in values)<.001 and max(abs(value-1) for values in variances.values() for value in values)<.002
    covariance_ok=abs(correlations["gaussian"]-correlations["student"])<.005
    panel_ok=pair["oracle_significant_fraction"]>=.75 and decision["tie_aware_regret"]>=.01
    status="PASS" if calibration_ok and moment_ok and covariance_ok and contrast>=.05 and panel_ok else ("INVALID_ORACLE_PANEL" if not panel_ok else "INVALID_ORACLE")
    report={"status":status,"calibration":calibration.to_dict(),"means":means,"variances":variances,"correlations":correlations,"gaussian_qei":float(gaussian_qei),"student_qei":float(student_qei),"relative_qei_contrast":contrast,"analytic_q1_ei":oracle.q1_ei(config.incumbent),"panel":{**pair,**decision},"endpoint_clamp_count":endpoint.clamp_count}
    return oracle,endpoint,report


def classify(config: Task04AEConfig, preflight: dict[str,Any], metrics: list[dict[str,Any]], marginal: list[dict[str,Any]], panels: list[dict[str,Any]], timing: list[dict[str,Any]]) -> dict[str,Any]:
    if preflight["status"]!="PASS": return {"decision":preflight["status"],"authorize_full":False,"gates":{}}
    if config.profile=="smoke":
        lookup={(row["method"],row["context"]):row for row in metrics}
        p=lookup["P",1.0]; u=lookup["U",1.0]
        kl_gain=float(u["joint_kl"])-float(p["joint_kl"]); relative=kl_gain/max(float(u["joint_kl"]),1e-15)
        qei={method:float(np.mean([row["qei_relative_error"] for row in metrics if row["method"]==method and row["context"]>0])) for method in ("G0","U","P")}
        qei_pass=qei["P"]<=.75*qei["G0"] and qei["P"]<=.75*qei["U"]
        p_q1=max(row["q1_ei_relative_error"] for row in marginal if row["method"]=="P")
        convergence=all(row["converged"] for row in timing if row["method"] in ("U","P"))
        gates={"dependence":{"gain":kl_gain,"relative_gain":relative,"passed":kl_gain>0 and relative>=.2},"qei":{"mean_relative_errors":qei,"passed":qei_pass},"marginal":{"p_max_q1_relative_error":p_q1,"passed":p_q1<.02},"optimization":{"passed":convergence}}
        if not convergence: decision="LEARNING_NO_GO"
        elif not gates["dependence"]["passed"]: decision="LEARNING_NO_GO"
        elif not qei_pass: decision="DECISION_NO_GO"
        elif not gates["marginal"]["passed"]: decision="MARGINAL_NO_GO"
        else: decision="AUTHORIZE_FULL"
        return {"decision":decision,"authorize_full":decision=="AUTHORIZE_FULL","gates":gates}
    if config.profile!="full":
        return {"decision":"WIRING_COMPLETE","authorize_full":False,"gates":{},"completed_fits":len(timing)}
    # Full gates are intentionally evaluated only when every expected row is present.
    expected=len(config.seeds)*len(config.sizes)
    if len(timing)!=2*expected: return {"decision":"INCOMPLETE","authorize_full":False,"gates":{},"completed_fits":len(timing),"expected_fits":2*expected}
    sizes=(128,256)
    gate_a=[];gate_b=[];gate_c=[]
    for n in sizes:
        for seed in config.seeds:
            p_m=[r for r in marginal if r["seed"]==seed and r["n"]==n and r["method"]=="P"]
            gate_a.append((n,seed,max(r["marginal_kl"] for r in p_m)<=.01 and max(r["q1_ei_relative_error"] for r in p_m)<=.01 and max(r["pearson_abs_error"] for r in p_m)<=.01))
            p_kl=np.mean([r["joint_kl"] for r in metrics if r["seed"]==seed and r["n"]==n and r["method"]=="P" and r["context"]>=.5]);u_kl=np.mean([r["joint_kl"] for r in metrics if r["seed"]==seed and r["n"]==n and r["method"]=="U" and r["context"]>=.5])
            gate_b.append((n,seed,p_kl<u_kl,(u_kl-p_kl)/max(u_kl,1e-15)))
            errors={method:np.mean([r["qei_relative_error"] for r in metrics if r["seed"]==seed and r["n"]==n and r["method"]==method and r["context"]>0]) for method in ("G0","U","P")}
            gate_c.append((n,seed,errors["P"]<=.7*errors["G0"] and errors["P"]<=.7*errors["U"]))
    a_pass=all(sum(ok for nn,_,ok in gate_a if nn==n)>=7 for n in sizes)
    b_pass=all(sum(win for nn,_,win,_ in gate_b if nn==n)>=7 and np.median([gain for nn,_,_,gain in gate_b if nn==n])>=.25 for n in sizes)
    c_curve=all(sum(ok for nn,_,ok in gate_c if nn==n)>=7 for n in sizes)
    panel256=[r for r in panels if r["n"]==256]; p={r["seed"]:r for r in panel256 if r["method"]=="P"};u={r["seed"]:r for r in panel256 if r["method"]=="U"};g={r["seed"]:r for r in panel256 if r["method"]=="G0"}
    panel_wins=sum(p[s]["tie_aware_regret"]<u[s]["tie_aware_regret"] and p[s]["tie_aware_regret"]<g[s]["tie_aware_regret"] and p[s]["spearman"]>=max(u[s]["spearman"],g[s]["spearman"]) and p[s]["top10_overlap"]>=max(u[s]["top10_overlap"],g[s]["top10_overlap"]) for s in p)
    base_regret=np.median([min(u[s]["tie_aware_regret"],g[s]["tie_aware_regret"]) for s in p]);p_regret=np.median([p[s]["tie_aware_regret"] for s in p]);panel_reduction=(base_regret-p_regret)/max(base_regret,1e-15)
    c_pass=c_curve and panel_wins>=7 and panel_reduction>=.3
    d_pass=all(r["converged"] and math.isfinite(r["seconds"]) for r in timing) and max(r["peak_rss_bytes"] for r in timing)<4*1024**3
    gates={"A":{"passed":a_pass},"B":{"passed":b_pass},"C":{"passed":c_pass,"panel_wins":panel_wins,"panel_median_reduction":panel_reduction},"D":{"passed":d_pass}}
    decision="GO_TASK04B" if all(v["passed"] for v in gates.values()) else ("MARGINAL_NO_GO" if not a_pass else ("LEARNING_NO_GO" if not b_pass else ("STRONG_DECISION_NO_GO" if not c_pass else "COMPUTE_NO_GO")))
    return {"decision":decision,"authorize_full":False,"gates":gates}


def _plots(output:Path,metrics:list[dict[str,Any]],panels:list[dict[str,Any]])->None:
    if not metrics:return
    figure,axes=plt.subplots(1,2,figsize=(9,3.5))
    for method in ("G0","U","P"):
        rows=[r for r in metrics if r["method"]==method]
        axes[0].plot([r["context"] for r in rows],[r["joint_kl"] for r in rows],"o-",label=method)
        axes[1].plot([r["context"] for r in rows],[r["qei_relative_error"] for r in rows],"o-",label=method)
    axes[0].set(xlabel="r",ylabel="joint KL");axes[1].set(xlabel="r",ylabel="relative qEI error");axes[1].legend();figure.tight_layout();figure.savefig(output/"task04ae_density_qei.png",dpi=160);plt.close(figure)
    if panels:
        figure,axis=plt.subplots(figsize=(5,3.5));methods=[r["method"] for r in panels];axis.bar(methods,[r["tie_aware_regret"] for r in panels]);axis.set_ylabel("tie-aware normalized regret");figure.tight_layout();figure.savefig(output/"task04ae_panel_regret.png",dpi=160);plt.close(figure)


def profile_fit_devices(config:Task04AEConfig,oracle:MatchedCopulaOracle)->dict[str,Any]:
    """Choose CUDA fitting only after an identical synchronized CPU/CUDA comparison."""
    if not torch.cuda.is_available():
        return {"selected_fit_device":"cpu","cuda_available":False,"reason":"CUDA unavailable"}
    generator=torch.Generator().manual_seed(50_000)
    choices=torch.tensor([0,.25,.5,.75,1],dtype=torch.double)
    contexts=choices[torch.randint(0,5,(128,),generator=generator)]
    samples=oracle.sample_training(contexts,60_000)
    results={};parameters={}
    for device in ("cpu","cuda"):
        model=BivariateEnergyModel(True,rho=oracle.rho,l2_precision=config.l2_precision,quadrature_points=config.quadrature_points).to(device)
        fit=model.fit(samples.to(device),contexts.to(device),max_iter=config.max_iterations)
        results[device]=asdict(fit);parameters[device]=model.parameter.cpu()
    objective_difference=abs(results["cpu"]["objective"]-results["cuda"]["objective"])
    parameter_difference=float((parameters["cpu"]-parameters["cuda"]).abs().max())
    speedup=results["cpu"]["seconds"]/max(results["cuda"]["seconds"],1e-15)
    agreement=objective_difference<=1e-8 and parameter_difference<=1e-6
    selected="cuda" if agreement and speedup>=1.2 and results["cuda"]["converged"] else "cpu"
    return {"selected_fit_device":selected,"cuda_available":True,"speedup":speedup,"objective_difference":objective_difference,"parameter_max_difference":parameter_difference,"agreement":agreement,"fits":results}


def run_task04ae(config:Task04AEConfig,output_dir:Path,*,device:str="cpu")->dict[str,Any]:
    output_dir.mkdir(parents=True,exist_ok=True)
    (output_dir/"task04ae_config.json").write_text(json.dumps(asdict(config),indent=2)+"\n")
    oracle,endpoint,preflight=oracle_preflight(config);(output_dir/"oracle_preflight.json").write_text(json.dumps(preflight,indent=2)+"\n")
    if preflight["status"]!="PASS":
        gate=classify(config,preflight,[],[],[],[]);(output_dir/"gate_status.json").write_text(json.dumps(gate,indent=2)+"\n");return {"preflight":preflight,"gate":gate}
    all_metrics:list[dict[str,Any]]=[];all_marginal:list[dict[str,Any]]=[];all_panels:list[dict[str,Any]]=[];timing:list[dict[str,Any]]=[];parameters:list[dict[str,Any]]=[]
    target=torch.device(device)
    if target.type=="cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA profile requested but CUDA is unavailable")
    device_profile=profile_fit_devices(config,oracle) if config.profile=="full" and target.type=="cuda" else {"selected_fit_device":str(target),"cuda_available":torch.cuda.is_available()}
    (output_dir/"device_profile.json").write_text(json.dumps(device_profile,indent=2)+"\n")
    fit_target=torch.device(str(device_profile["selected_fit_device"]))
    for seed in config.seeds:
        generator=torch.Generator().manual_seed(50_000+seed)
        choices=torch.tensor([0,.25,.5,.75,1],dtype=torch.double)
        contexts=choices[torch.randint(0,5,(max(config.sizes),),generator=generator)]
        samples=oracle.sample_training(contexts,60_000+seed)
        for n in config.sizes:
            case_file=output_dir/f"case_seed{seed}_n{n}.json"
            if case_file.exists():
                saved=json.loads(case_file.read_text())
                all_metrics.extend(saved["metrics"]);all_marginal.extend(saved["marginal"]);all_panels.extend(saved["panels"]);timing.extend(saved["timing"]);parameters.extend(saved["parameters"])
                continue
            models={"G0":BivariateEnergyModel(False,rho=oracle.rho,l2_precision=config.l2_precision,quadrature_points=config.quadrature_points).to(target)}
            case_timing=[];case_parameters=[]
            for name,pairwise in (("U",False),("P",True)):
                model=BivariateEnergyModel(pairwise,rho=oracle.rho,l2_precision=config.l2_precision,quadrature_points=config.quadrature_points).to(fit_target)
                result=model.fit(samples[:n].to(fit_target),contexts[:n].to(fit_target),max_iter=config.max_iterations)
                model.to(target)
                models[name]=model
                row={"seed":seed,"n":n,"method":name,**asdict(result),"fit_device":str(fit_target),"evaluation_device":str(target),"peak_rss_bytes":_peak_rss_bytes(),"peak_cuda_bytes":torch.cuda.max_memory_allocated(target) if target.type=="cuda" else 0};timing.append(row);case_timing.append(row)
                additions=[{"seed":seed,"n":n,"method":name,"index":index,"value":float(value)} for index,value in enumerate(model.parameter.cpu())]
                parameters.extend(additions);case_parameters.extend(additions)
            metrics,marginal=_density_rows(oracle,endpoint,models,seed,n,config.incumbent);panels=_panel_rows(oracle,endpoint,models,seed,n,config.incumbent,config.panel_count)
            all_metrics.extend(metrics);all_marginal.extend(marginal);all_panels.extend(panels)
            case_file.write_text(json.dumps({"seed":seed,"n":n,"context_counts":{str(float(r)):int((contexts[:n]==r).sum()) for r in choices},"metrics":metrics,"marginal":marginal,"panels":panels,"timing":case_timing,"parameters":case_parameters},indent=2)+"\n")
            _write_csv(output_dir/"metrics.partial.csv",all_metrics);_write_csv(output_dir/"marginal_safety.partial.csv",all_marginal);_write_csv(output_dir/"batch_decisions.partial.csv",all_panels);_write_csv(output_dir/"timing.partial.csv",timing)
    _write_csv(output_dir/"metrics.csv",all_metrics);_write_csv(output_dir/"marginal_safety.csv",all_marginal);_write_csv(output_dir/"batch_decisions.csv",all_panels);_write_csv(output_dir/"timing.csv",timing);_write_csv(output_dir/"parameters.csv",parameters)
    gate=classify(config,preflight,all_metrics,all_marginal,all_panels,timing);(output_dir/"gate_status.json").write_text(json.dumps(gate,indent=2)+"\n")
    environment={"python":sys.version,"platform":platform.platform(),"torch":torch.__version__,"device":str(target),"cuda_device":torch.cuda.get_device_name(target) if target.type=="cuda" else None};(output_dir/"environment.json").write_text(json.dumps(environment,indent=2)+"\n")
    summary=["# Task 04A-E summary","",f"**Decision: {gate['decision']}**","","This report is generated from the bounded q=2 matched-marginal experiment.","","## Eight completion questions","","1. Oracle calibration and matched marginal checks are recorded in `oracle_preflight.json`.","2. Gaussian/t-copula q=2 contrast is reported before learning.","3. Exact G0 recovery and convex normalization are gated by pytest.","4. P-versus-U joint KL is the dependence-learning gate.","5. Explicit marginal KL, Pearson, CDF, and q=1 EI errors test marginal safety.","6. Five-context qEI curves and the paired panel test q=2 value.","7. Fit, quadrature, device, and memory instrumentation are saved separately.",f"8. The frozen classifier returns `{gate['decision']}`; it does not implement Task 04B.","","```json",json.dumps(gate,indent=2),"```",""]
    (output_dir/"TASK_04AE_SUMMARY.md").write_text("\n".join(summary));_plots(output_dir,all_metrics,all_panels)
    return {"preflight":preflight,"metrics":all_metrics,"marginal":all_marginal,"panels":all_panels,"timing":timing,"gate":gate}
