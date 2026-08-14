from __future__ import annotations

import copy

import torch

from energy_bo.experiments.task04ad import DecisionDiagnosticConfig, classify_decision_diagnostic
from energy_bo.local_process.decision import (
    active_candidate_sets,
    build_counterfactual_panel,
    counterfactual_metrics,
    pairwise_metrics,
    select_real_pairs,
)
from energy_bo.local_process.energy import LocalEnergyModel


def test_candidate_sets_and_panels_are_deterministic():
    first=active_candidate_sets(6,100,sobol_count=32,grid_size=5)
    second=active_candidate_sets(6,100,sobol_count=32,grid_size=5)
    assert torch.equal(first.construction,second.construction)
    assert torch.equal(first.verification,second.verification)
    assert first.construction.shape==(57,6) and first.verification.shape==(32,6)
    assert torch.all(first.construction[:,2:]==.5)


def test_real_pairs_use_only_reference_and_context_and_obey_constraints():
    count=200
    g0=.9+.001*(torch.arange(count,dtype=torch.double)%10)
    summary=torch.zeros(count,7,dtype=torch.double); summary[:,0]=torch.linspace(-1,1,count)
    direction=torch.zeros(7,dtype=torch.double); direction[0]=1
    pairs=select_real_pairs(g0,summary,direction,maximum_pairs=20,relative_tolerance=.01)
    changed_oracle=torch.randn(count,generator=torch.Generator().manual_seed(9),dtype=torch.double)
    assert torch.equal(pairs,select_real_pairs(g0,summary,direction,maximum_pairs=20,relative_tolerance=.01))
    assert changed_oracle.shape==g0.shape  # Oracle values are deliberately absent from the API.
    relative=(g0[pairs[:,0]]-g0[pairs[:,1]]).abs()/torch.maximum(g0[pairs[:,0]],g0[pairs[:,1]])
    score=summary@direction; low,high=torch.quantile(score,torch.tensor([.1,.9],dtype=torch.double))
    assert len(pairs)>0 and float(relative.max())<=.01+1e-15
    assert torch.all(score[pairs[:,1]]-score[pairs[:,0]]>=high-low-1e-14)


def test_counterfactual_panel_and_unary_invariance():
    g0=torch.linspace(0,1,100,dtype=torch.double)
    summary=torch.randn(100,7,generator=torch.Generator().manual_seed(4),dtype=torch.double)
    direction=torch.linspace(-1,1,7,dtype=torch.double)
    panel=build_counterfactual_panel(g0,summary,direction,count=16)
    model=LocalEnergyModel(False); model.unary=torch.linspace(-.1,.1,7,dtype=torch.double)
    mean=torch.linspace(-.5,.5,16,dtype=torch.double); scale=torch.full_like(mean,.4)
    low=model.expected_improvement(mean,scale,panel.low_summary.expand(16,-1),0.0)
    high=model.expected_improvement(mean,scale,panel.high_summary.expand(16,-1),0.0)
    assert len(panel.base_indices)==16 and torch.equal(low,high)


def test_pairwise_and_counterfactual_metrics_have_known_values():
    oracle=torch.tensor([1.,2.,2.,1.],dtype=torch.double)
    model=torch.tensor([0.,3.,0.,2.],dtype=torch.double)
    pairs=torch.tensor([[0,1],[2,3]])
    result=pairwise_metrics(model,oracle,pairs)
    assert result["choice_accuracy"]==.5
    assert result["margin_weighted_regret"]==.5
    counter=counterfactual_metrics(torch.tensor([1.,1.]),torch.tensor([2.,.5]),torch.tensor([1.,1.]),torch.tensor([2.,.5]))
    assert counter["oracle_significant_fraction"]==1 and counter["sign_accuracy"]==1
    assert counter["median_relative_contrast_error"]==0


def _synthetic_complete_result():
    metrics=[]; diagnostics=[]
    for seed in range(100,108):
        for regime,n in (("G",256),("I",128),("I",256)):
            for method in ("G0","U","P"):
                row={"seed":seed,"regime":regime,"n":n,"method":method,"conditional_kl":0.0,"average_correction_kl":0.001,"normalized_regret":0.0,"within_one_percent":True}
                if regime=="I" and n==256:
                    row["conditional_kl"]={"G0":.12,"U":.1,"P":.05}[method]
                    if method=="U": row["normalized_regret"],row["within_one_percent"]=.2,False
                    if method=="P": row["normalized_regret"]=.05
                metrics.append(row)
            diagnostic={"seed":seed,"regime":regime,"n":n,"finite":True,"converged":True,"teacher_free_pairs":True,"candidate_pool_verified":True,"real_pair_count":64 if regime=="I" else 0}
            if regime=="I":
                diagnostic.update({"counter_oracle_significant_fraction":1.0,"counter_P_sign_accuracy":.9,"counter_P_median_relative_contrast_error":.1,"real_U_choice_accuracy":.5,"real_P_choice_accuracy":.9,"real_U_margin_weighted_regret":.5,"real_P_margin_weighted_regret":.1})
            diagnostics.append(diagnostic)
    return metrics,diagnostics


def test_frozen_outcome_classifier():
    config=DecisionDiagnosticConfig.local(); metrics,diagnostics=_synthetic_complete_result()
    assert classify_decision_diagnostic(config,metrics,diagnostics)["decision"]=="LOCAL_GO"
    failed=copy.deepcopy(diagnostics)
    for row in failed:
        if row["regime"]=="I" and row["n"]==256: row["counter_oracle_significant_fraction"]=0.0
    assert classify_decision_diagnostic(config,metrics,failed)["decision"]=="ORACLE_NO_GO"
    invalid=copy.deepcopy(diagnostics); invalid[0]["candidate_pool_verified"]=False
    assert classify_decision_diagnostic(config,metrics,invalid)["decision"]=="INVALID"
