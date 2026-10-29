"""Synthetic-only tests for the preregistered, recalibrated Qloo ablation."""
import copy
import json

import numpy as np
import pytest

import matched_control as selection
import evaluate_confirmation as evaluation


@pytest.fixture(autouse=True)
def reject_network(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail('Matched-control tests must never call an API')
    monkeypatch.setattr('httpx.Client.request', fail)


def model(names, weights=None, scale=None, qloo=False):
    return {'features':list(names),
            'model':{'scale':scale or [1.] * len(names),
                     'weights':weights or [.2] * len(names)},
            'selection':{'name':'synthetic','features':list(names),'uses_qloo':qloo,
                         'inner_ridges':[10.] * 5,'correct':1,'total':2,'log_loss':.6}}


def used(p=60, tokens=100):
    return {'output':{'ranking':['C01','C02'] if p>=50 else ['C02','C01'],
                      'probability_c01':p},
            'metrics':{'input_tokens':tokens,'output_tokens':0},
            'physical_requests':1,
            'calls':[{'usage':{'prompt_tokens':tokens,'completion_tokens':0}}]}


def test_final_selected_family_has_an_exact_information_matched_control():
    from pathlib import Path
    config=json.loads((Path(__file__).parent/'agent_config.json').read_text())
    names=config['models'][config['selected_qloo']]['features']
    control=config['models'][config['selected_agent']]['features']
    mapped,mapping=selection.matched_feature_names(names)
    assert mapped==control
    assert len(mapping)==len(names)
    assert not any('qloo' in n or n.startswith(('qmeta_','qgraph_','qgroup_')) for n in mapped)


def test_mapping_retains_cf_removes_exclusive_evidence_and_deduplicates():
    names=selection.BASE+selection.CF+selection.MIXED+selection.AFFINITY+selection.GRAPH+[
        'qmeta_curated_ridge_0.1','qgroup_pooled_contrast','hier_qloo','judge_qloo',
        'bernoulli_k12_qloo','bernoulli_k24_qloo']
    original=copy.deepcopy(names)
    mapped,mapping=selection.matched_feature_names(names)
    assert mapped==selection.BASE+selection.CF+['hier_agent','judge_agent']+selection.MCF
    assert names==original
    assert sum(x['action']=='remove' for x in mapping)==len(selection.AFFINITY+selection.GRAPH)+2
    with pytest.raises(ValueError,match='Unregistered'):
        selection.matched_feature_names(selection.BASE+['future_hidden_cultural_signal'])


def test_exact_control_reuse_cannot_change_main_selection_or_models(monkeypatch):
    models={'a':model(selection.BASE),
            'q':model(selection.BASE+selection.AFFINITY,qloo=True)}
    oofs={'a':{'D0':.7,'D1':.4},'q':{'D0':.9,'D1':.8}}
    original=copy.deepcopy((models,oofs))
    monkeypatch.setattr(selection,'nested_family',lambda *x:pytest.fail('Exact control must reuse its fit'))
    matched,probabilities=selection.build_matched_control(
        'q',models,oofs,['D0','D1'],{},np.array([1.,0.]),np.array([0,1]),{})
    assert matched['reused_from']=='a'
    assert matched['model']==models['a']['model'] and probabilities==oofs['a']
    assert matched['selection']['conditional_on_selected_qloo_family']
    assert not matched['selection']['confirmation_used']
    assert (models,oofs)==original
    matched['model']['scale'][0]=999.
    probabilities['D0']=0.
    assert (models,oofs)==original


def test_new_matched_mechanism_uses_same_nested_rule_on_only_supplied_development(monkeypatch):
    # A valid combination with no exact pre-existing family exercises refitting.
    names=selection.BASE+['judge_qloo','interaction_mixed_ridge_1']
    models={'q':model(names,qloo=True)}
    expected=selection.BASE+['judge_agent','interaction_cf_ridge_1']
    values={ident:{name:float(i+j+1) for j,name in enumerate(expected)}
            for i,ident in enumerate(['D0','D1'])}
    labels=np.array([1.,0.]);folds=np.array([0,1]);seen=[]
    def nested(X,y,fold):
        seen.append((X.copy(),y.copy(),fold.copy()))
        return np.array([.7,.4]),{'scale':[2.]*5,'weights':[.3]*5},[1.,10.]
    monkeypatch.setattr(selection,'nested_family',nested)
    matched,p=selection.build_matched_control('q',models,{},['D0','D1'],values,labels,folds,
        {'D0':{'base':100,'short_agent':200},'D1':{'base':100,'short_agent':300}})
    assert matched['reused_from'] is None and matched['features']==expected
    assert len(seen)==1 and np.array_equal(seen[0][1],labels) and np.array_equal(seen[0][2],folds)
    assert np.array_equal(seen[0][0],[[values[i][n] for n in expected] for i in ['D0','D1']])
    assert matched['model']['scale']==[2.]*5
    assert matched['selection']['aggregate_token_upper_ratio']==2.5
    assert p=={'D0':.7,'D1':.4}


def test_matched_prediction_uses_own_scaling_and_agent_review_without_qloo_anchor():
    names=['glm_log_odds','judge_agent']
    control=model(names,weights=[0.,1.],scale=[1.,2.])
    cfg={'matched_no_qloo':{**control,'source_family':'q','reused_from':'a'}}
    before=copy.deepcopy(cfg)
    answer=evaluation.predict_matched_control({'qmeta_irrelevant':1e20},used(80),cfg,
        {'agent':used(20),'qloo':used(99)}, {})
    assert answer['choice']==1 and answer['probability']<.5
    assert not answer['review_fallbacks']['short']
    # An absent/invalid no-Qloo review falls back to the initial result, with no
    # borrowing of a valid Qloo review or any new API call.
    fallback=evaluation.predict_matched_control({},used(80),cfg,{'qloo':used(1)}, {})
    assert fallback['choice']==0 and fallback['review_fallbacks']['short']
    assert cfg==before


def test_infer_saves_retrained_and_secondary_decisions_without_opening_answers(tmp_path,monkeypatch):
    root=tmp_path/'root';private=tmp_path/'private'
    root.mkdir();private.mkdir()
    monkeypatch.setattr(evaluation,'ROOT',root)
    monkeypatch.setattr(evaluation,'PRIVATE',private)
    monkeypatch.setattr(evaluation,'verify_lock',lambda:None)
    visible={'id':'T01','likes':[[i,5] for i in range(1,6)],
             'dislikes':[[i,1] for i in range(6,9)],'candidates':[9,10]}
    monkeypatch.setattr(evaluation,'visible_cases',lambda:[visible])
    names=['glm_log_odds','interaction_cf_ridge_1','judge_agent']
    matched={**model(names,weights=[0.,-2.,0.],scale=[1.,2.,1.]),
             'source_family':'q','reused_from':'matching_control'}
    cfg={'selected_agent':'a','selected_qloo':'q','judge_version':'mock',
         'models':{'a':model(['glm_log_odds'],weights=[1.]),
                   'q':model(['glm_log_odds','interaction_mixed_ridge_1','judge_qloo'],
                             weights=[0.,2.,0.],scale=[1.,.01,1.],qloo=True)},
         'matched_no_qloo':matched}
    def write(path,value):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value))
    write(root/'agent_config.json',cfg)
    write(private/'predictions/T01.json',used(tokens=100))
    write(private/'judges/mock/agent/T01.json',used(20,tokens=300))
    write(private/'judges/mock/qloo/T01.json',{
        **used(80,tokens=700),'request_data':{'qloo_cultural_evidence':{'source':'MOCK'}}})
    calls=[]
    class Tools:
        def build(self,h,c,initial,evidence,*,required_features,group_record):
            calls.append(set(required_features))
            return {'values':{'glm_log_odds':.4,'interaction_cf_ridge_1':.1,
                              'interaction_mixed_ridge_1':.02},
                    'trace':{'source_availability':{'required_tools_valid':True}}}
    monkeypatch.setattr(evaluation,'FeatureTools',Tools)
    def guarded_read(path):
        assert path.name!='confirmation_cases.json','Inference opened the answer key'
        assert path.is_relative_to(tmp_path),'Synthetic test touched real private data'
        return json.loads(path.read_text())
    monkeypatch.setattr(evaluation,'read',guarded_read)
    evaluation.infer()
    result=json.loads((private/'confirmation_decisions/T01.json').read_text())
    assert len(calls)==1 and 'interaction_cf_ridge_1' in calls[0]
    assert result['ablation']['choice']==1
    assert result['secondary_fixed_weight_ablation']['choice']==0
    assert set(result['decision']['choices'])=={'glm','agent','agent_qloo'}
    assert result['ablation']['cost']['tokens_reported']==400
    assert result['ablation']['cost']['tokens_reported_complete']
    assert result['cost']['agent_qloo']['tokens_reported']==800
    assert result['cost']['agent']['tokens_reported']==100


def test_score_keeps_primary_retrained_and_secondary_fixed_weight_apart(tmp_path,monkeypatch):
    # This in-memory fixture is invented. No real confirmation files are read.
    monkeypatch.setattr(evaluation,'ROOT',tmp_path/'root')
    monkeypatch.setattr(evaluation,'PRIVATE',tmp_path/'private')
    monkeypatch.setattr(evaluation,'verify_lock',lambda:None)
    monkeypatch.setattr(evaluation,'sha',lambda p:'synthetic-hash')
    costs={'tokens_reported':100,'tokens_reported_complete':True,'tokens_upper_bound':100}
    cfg={'selected_agent':'a','selected_qloo':'q',
         'models':{'a':model(selection.BASE),'q':model(selection.BASE+selection.AFFINITY,qloo=True)},
         'matched_no_qloo':{**model(selection.BASE),'source_family':'q','reused_from':'a'}}
    records={
        'agent_config.json':cfg,
        'confirmation_cases.json':[{'id':'SYNTHETIC','candidates':[1,2],'preferred':1,'difficulty':'ordinary'}],
        'SYNTHETIC.json':{
            'decision':{'choices':{'glm':0,'agent':0,'agent_qloo':0},'qloo_fallback':False},
            'cost':{name:costs for name in ['glm','agent','agent_qloo']},
            'ablation':{'choice':1,'description':'RETRAINED MOCK','cost':costs},
            'secondary_fixed_weight_ablation':{'choice':0,'description':'FIXED MOCK','cost':costs}},
        'formal_lock.json':{'source_sha':{}},'final_development_selection.json':{'mock':True}}
    monkeypatch.setattr(evaluation,'read',lambda p:records[p.name])
    summarized=[];saved={}
    def summarize(rows,**kwargs):
        summarized.append((copy.deepcopy(rows),kwargs))
        return {'methods':{},'comparisons':[]}
    monkeypatch.setattr(evaluation,'summarize_predictions',summarize)
    monkeypatch.setattr(evaluation,'save',lambda p,v:saved.update({p.name:v}))
    evaluation.score()
    assert len(summarized)==3
    assert summarized[0][1]['methods']==['glm','agent','agent_qloo']
    primary=next(row for row in summarized[1][0] if row['method']=='without_qloo')
    secondary=next(row for row in summarized[2][0] if row['method']=='without_qloo')
    assert primary['correct'] is False and secondary['correct'] is True
    assert primary['tokens_reported']==100 and primary['tokens_reported_complete']
    report=saved['confirmation_report.json']
    assert report['ablation']['kind']=='development_recalibrated_matched_no_qloo'
    assert report['secondary_fixed_weight_ablation']['kind']=='fixed_weight_distribution_shift_diagnostic'
    assert any('背景3640' in item for item in report['metadata']['limitations'])
