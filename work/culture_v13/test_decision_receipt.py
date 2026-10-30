"""Decision-invariance and evidence-direction tests; no API calls."""
import copy
import pytest
from decision_receipt import build_receipt, fusion_account
from inference_v13 import predict

def record(p):
    return {'output':{'ranking':['C01','C02'] if p>=50 else ['C02','C01'], 'probability_c01':p,'explanation':'test'}}

def setup(fallback=False):
    cfg={'selected_agent':'a','selected_qloo':'q','models':{
        'a':{'features':['glm_log_odds','mf_32_10','hier_agent'],'model':{'weights':[.5,.7,.2],'scale':[1,2,1]}},
        'q':{'features':['glm_log_odds','mf_32_10','bernoulli_k12_qloo','hier_qloo'],'model':{'weights':[.5,.7,.3,.2],'scale':[1,2,1,1]}}}}
    features={'mf_32_10':.5,'bernoulli_k12_qloo':.7};initial=record(70);reviews={'agent':record(60),'qloo':record(80)}
    result=predict(features,initial,cfg,hierarchical=reviews,qloo_available=not fallback)
    response={'recommendations':{'agent':{'candidate_id':'C01'}},'qloo_evidence':{'qloo_group_evidence':[
        {'direction':'liked','seed_history_ids':['H01','H02'],'candidate_affinity':{'C01':.9,'C02':.5}},
        {'direction':'disliked','seed_history_ids':['H06','H07','H08'],'candidate_affinity':{'C01':.8,'C02':.4}}]}}
    return response,result,cfg,features,initial,{},reviews

def test_exact_fusion_readonly_and_cultural_counterevidence():
    args=setup();before=copy.deepcopy(args);r=build_receipt(*args)
    assert args==before
    assert r['fusion']['verified']
    assert r['fusion']['reconstructed_score']==pytest.approx(args[1]['probabilities']['agent_qloo'],abs=1e-12)
    assert r['supporting_groups']==1 and r['caution_groups']==1 and r['mixed_evidence']
    assert r['choice_changed'] is False
    assert {t['id'] for t in r['fusion']['terms']}=={'initial','ratings','cultural_groups','review'}

def test_zero_is_valid_missing_is_not_support():
    args=list(setup());g=args[0]['qloo_evidence']['qloo_group_evidence']
    g[0]['candidate_affinity']={'C01':0.,'C02':0.}
    g[1]['candidate_affinity']={'C01':.7}
    r=build_receipt(*args)
    assert [x['interpretation'] for x in r['group_ledger']]==['neutral','unavailable']
    assert r['valid_groups']==1

def test_invalid_seed_direction_cannot_be_evidence():
    args=list(setup());g=args[0]['qloo_evidence']['qloo_group_evidence']
    g[0]['seed_history_ids']=['H06'];g[1]['candidate_affinity']['C01']=True
    r=build_receipt(*args)
    assert r['valid_groups']==0


def test_fallback_uses_actual_degraded_fusion_not_control_review():
    r=build_receipt(*setup(True))
    assert r['fusion']['verified'] and not r['qloo_used_for_decision']
    assert r['choice_changed'] is None
    assert 'cultural_groups' not in {t['id'] for t in r['fusion']['terms']}
    assert 'unavailable' in r['summary'].lower()


def test_receipt_never_claims_counterfactual_if_control_not_run():
    args=list(setup());args[0]['recommendations']={};r=build_receipt(*args)
    assert not r['comparison_available'] and r['choice_changed'] is None


def test_unverified_decomposition_fails_closed():
    args=list(setup());args[1]['probabilities']['agent_qloo']=.51
    assert not build_receipt(*args)['fusion']['available']


def test_selected_second_candidate_reverses_evidence_interpretation():
    args=list(setup());args[1]['choices']['agent_qloo']=1
    r=build_receipt(*args)
    assert r['selected_candidate']=='C02'
    assert [x['interpretation'] for x in r['group_ledger']]==['caution','support']
    assert r['choice_changed'] is True
