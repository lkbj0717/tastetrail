"""Bounded response recovery: synthetic records and MockTransport only."""
import copy
import json
from types import SimpleNamespace

import httpx
import pytest

import hierarchical_agent as base
import hierarchical_repair as repair
from common import token_count,usage_complete,usage_upper_bound
from evidence_cards import EvidenceCardBuilder

H=[(1,5),(2,4),(3,5),(4,4),(5,4),(6,1),(7,2),(8,2)]
C=[9,10]
PLAN={'hypotheses':[{'claim':'May enjoy restrained stories.','support':['H01'],
                    'counterevidence':['H06'],'unknown':'Exact motivations remain unknown.'}],
      'liked_groups':[['H01','H02'],['H03','H04','H05']],
      'disliked_group':['H06','H07','H08']}
ANSWER={'ranking':['C01','C02'],'probability_c01':65,'explanation':'H01 supports C01; H06 is contrary evidence.'}


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    monkeypatch.setattr('tastetrail.model_client.settings',lambda:{'GLM_API_KEY':'synthetic-test-only','GLM_BASE_URL':'https://unit.test/v1'})
    monkeypatch.setattr('tastetrail.model_client.time.sleep',lambda n:None)
    monkeypatch.setattr('httpx.HTTPTransport.handle_request',lambda *a,**k:pytest.fail('Real network is forbidden'))


def used(tokens,output=None):
    return {'error':None,'fallback':False,'output':output,
            'metrics':{'input_tokens':tokens,'output_tokens':0,'api_calls':1},
            'calls':[{'http_status':200,'finish_reason':'stop',
                      'usage':{'prompt_tokens':tokens,'completion_tokens':0}}],
            'physical_requests':1,'reported_tokens':tokens,'accounted_tokens':tokens,
            'usage_is_upper_bound':False}


def records(mode='agent'):
    initial=used(1000,{'ranking':['C02','C01'],'probability_c01':20,
                       'explanation':'BASELINE_ANCHOR_NEVER_IN_REPAIR'})
    initial['preferred']='HIDDEN_SENTINEL'
    plan=used(500,copy.deepcopy(PLAN))
    first=used(8692,copy.deepcopy(initial['output']))
    first.update(error='ModelError: Model response exceeded token budget.',fallback=True,
                 fallback_source='initial_glm',request_fingerprint='first-fingerprint',
                 request_data={'history':[{'history_id':f'H{i:02d}','rating':r} for i,(_,r) in enumerate(H,1)],
                               'candidates':[{'candidate_id':'C01'},{'candidate_id':'C02'}],
                               'preference_plan':copy.deepcopy(PLAN),'community_tools':{'margin':.2}})
    first['calls'][0]['finish_reason']='length'
    if mode=='qloo':first['request_data'].update(qloo_cultural_evidence={'source':'MOCK'},qloo_group_evidence=[{'source':'MOCK'}])
    return first,plan,initial


def transport_factory(responses,captured):
    queue=iter(responses)
    def factory():
        def handler(request):
            captured.append(json.loads(request.content))
            item=next(queue)
            status=item.get('status',200)
            value={'model':'mock','choices':[{'finish_reason':item.get('finish','stop'),
                  'message':{'content':item.get('content',json.dumps(ANSWER))}}]}
            if item.get('usage',True):value['usage']=item.get('tokens',{'prompt_tokens':200,'completion_tokens':100})
            return httpx.Response(status,json=value)
        return base.SingleAttemptPacedTransport(inner=httpx.MockTransport(handler),interval=0)
    return factory


@pytest.mark.parametrize('mode',['agent','qloo'])
def test_truncation_repair_uses_identical_policy_no_anchor_and_keeps_original_cost(mode):
    first,plan,initial=records(mode);original=copy.deepcopy(first);captured=[]
    result=repair.apply_repair(first,plan,initial,mode,transport_factory=transport_factory([{}],captured))
    assert result['repair_attempted'] and result['repair_succeeded'] and not result['error']
    assert result['output']['ranking']==ANSWER['ranking']
    assert result['physical_requests']==2 and len(captured)==1
    assert captured[0]['reasoning_effort']=='low' and captured[0]['max_tokens']==2048
    assert result['reported_tokens']==8692+300
    assert result['production_accounted_tokens']==1000+500+8692+300
    assert usage_complete(result) and usage_upper_bound(result)==8992
    assert result['production_token_ratio_upper_bound']<=18
    text=json.dumps(captured)
    assert 'BASELINE_ANCHOR_NEVER_IN_REPAIR' not in text and 'HIDDEN_SENTINEL' not in text
    assert ('qloo_cultural_evidence' in text)==(mode=='qloo')
    assert first==original


def test_valid_original_is_reused_even_when_disagreeing_with_baseline():
    first,plan,initial=records();first.update(error=None,fallback=False,output=copy.deepcopy(ANSWER),response_object=copy.deepcopy(ANSWER))
    result=repair.apply_repair(first,plan,initial,'agent',transport_factory=lambda:pytest.fail('Valid response retried'))
    assert result['output']==first['output'] and result['calls']==first['calls']
    assert not result['repair_attempted'] and result['repair_reason']=='original_success'
    assert result['reported_tokens']==first['reported_tokens']


@pytest.mark.parametrize('kind,reason',[
    ('planner','planner_failed'),('qloo','qloo_evidence_unavailable'),
    ('unknown','unknown_cost_upper_bound'),('baseline','initial_usage_unavailable'),
    ('budget','original_budget_violation'),('transport','not_an_objective_format_or_truncation_failure')])
def test_fail_closed_conditions_do_not_call_repair(kind,reason):
    first,plan,initial=records('qloo')
    if kind=='planner':plan['error']='planner_parse';plan['fallback']=True
    if kind=='qloo':first['request_data'].pop('qloo_cultural_evidence')
    if kind=='unknown':
        first.pop('accounted_tokens');first['calls'][0].pop('usage');first['usage_is_upper_bound']=True
    if kind=='baseline':initial['calls'][0].pop('usage')
    if kind=='budget':first['budget_violation']=True
    if kind=='transport':first['calls'][0].update(http_status=503,finish_reason=None)
    result=repair.apply_repair(first,plan,initial,'qloo',transport_factory=lambda:pytest.fail('Ineligible recovery called'))
    assert not result['repair_attempted'] and result['repair_reason']==reason
    if kind=='unknown':assert result['accounted_tokens'] is None and result['production_accounted_tokens'] is None


def test_invalid_original_fallback_output_does_not_hide_schema_failure():
    first,plan,initial=records()
    first['error']='JSONDecodeError: Expecting value';first['calls'][0].update(finish_reason='stop',response_message={'content':'not JSON'})
    assert first['output']['ranking']==['C02','C01']
    out=repair.apply_repair(first,plan,initial,'agent',transport_factory=transport_factory([{}],[]))
    assert out['repair_succeeded'] and out['repair_reason']=='response_schema_invalid'


def test_insufficient_residual_budget_skips_without_erasing_first_charge():
    first,plan,initial=records();plan=used(7800,PLAN)
    out=repair.apply_repair(first,plan,initial,'agent',transport_factory=lambda:pytest.fail('Insufficient budget'))
    assert not out['repair_attempted'] and out['repair_reason']=='repair_budget_preflight_skip'
    assert out['production_accounted_tokens']==1000+7800+8692
    assert out['output']==first['output']


def test_failed_repair_is_one_physical_call_and_unknown_usage_is_reserved():
    first,plan,initial=records();captured=[]
    out=repair.apply_repair(first,plan,initial,'agent',
        transport_factory=transport_factory([{'status':503,'usage':False}],captured))
    assert len(captured)==1 and out['physical_requests']==2
    assert out['repair_attempted'] and not out['repair_succeeded']
    assert out['output']['ranking']==initial['output']['ranking'] and out['fallback']
    stage=out['repair_record']
    assert stage['accounted_tokens']==stage['prompt_utf8_upper_bound']+stage['max_completion_tokens']
    assert out['accounted_tokens']==8692+stage['accounted_tokens']
    assert not usage_complete(out) and usage_upper_bound(out)==out['accounted_tokens']
    assert out['production_accounted_tokens']<=18000


def test_provider_overuse_preserves_cost_and_refuses_repair_output():
    first,plan,initial=records()
    out=repair.apply_repair(first,plan,initial,'agent',transport_factory=transport_factory([
        {'tokens':{'prompt_tokens':100000,'completion_tokens':100000}}],[]))
    assert not out['repair_succeeded'] and out['budget_violation']
    assert out['production_accounted_tokens']==1000+500+8692+200000
    assert out['output']['ranking']==initial['output']['ranking']


def test_wrapper_cache_preserves_source_hash_and_never_retries_twice(tmp_path):
    first,plan,initial=records();captured=[]
    out=repair.wrap_review_record(first,plan,initial,'agent','TEST01',private_path=tmp_path,
         transport_factory=transport_factory([{}],captured))
    again=repair.wrap_review_record(first,plan,initial,'agent','TEST01',private_path=tmp_path,
         transport_factory=lambda:pytest.fail('Cached recovery repeated'))
    assert out==again and out['source_record_sha256']==base._hash(first)
    assert out['source_record_path']=='hierarchical_v2/reviews/agent/TEST01.json'
    altered=copy.deepcopy(first);altered['request_data']['community_tools']['margin']=.8
    with pytest.raises(ValueError,match='different'):
        repair.wrap_review_record(altered,plan,initial,'agent','TEST01',private_path=tmp_path)


def test_new_wrapper_reuses_immutable_v2_plan_and_first_review(tmp_path):
    metadata={str(i):{'title':f'Film {i}','year':1990+i,'genres':['Drama']} for i in range(1,11)}
    builder=EvidenceCardBuilder(tmp_path,movie_metadata=metadata)
    initial=used(1000,{'ranking':['C02','C01'],'probability_c01':20})
    class Community:
        def evaluate(self,*args):return {'community_margin':.2,'mf_32_10':.1,'neighbors_100':.3},{},{}
    captures=[]
    plan=repair.run_plan(H,C,'TEST02',1000,private_path=tmp_path,builder=builder,
        transport_factory=transport_factory([{'content':json.dumps(PLAN)}],captures))
    old_plan=(tmp_path/base.VERSION/'plans/TEST02.json').read_bytes()
    result=repair.run_review(H,C,'TEST02','agent',plan,initial,private_path=tmp_path,builder=builder,
        community_tools=Community(),transport_factory=transport_factory([
        {'finish':'length','tokens':{'prompt_tokens':300,'completion_tokens':8192}},{}],captures))
    source_path=tmp_path/base.VERSION/'reviews/agent/TEST02.json';old_review=source_path.read_bytes()
    source=json.loads(old_review)
    assert source['error'] and result['repair_succeeded'] and result['source_record_sha256']==base._hash(source)
    assert (tmp_path/base.VERSION/'plans/TEST02.json').read_bytes()==old_plan
    repair.run_review(H,C,'TEST02','agent',plan,initial,private_path=tmp_path,builder=builder,
        community_tools=Community(),transport_factory=lambda:pytest.fail('Successful wrapper replay called API'))
    assert source_path.read_bytes()==old_review and len(captures)==3


def test_version_router_is_explicit():
    from hierarchy_backend import get_backend
    assert get_backend()==base and get_backend(base.VERSION)==base
    assert get_backend(repair.VERSION)==repair
    with pytest.raises(ValueError,match='Unsupported'):get_backend('invented_version')


def test_confirmation_routes_selected_version_without_private_or_answer_reads(tmp_path,monkeypatch):
    import run_confirmation as runner
    monkeypatch.setattr(runner,'ROOT',tmp_path/'root');monkeypatch.setattr(runner,'PRIVATE',tmp_path/'private')
    calls=[];plan=used(200,PLAN)
    def read(path):
        assert path.is_relative_to(tmp_path)
        if path.name=='agent_config.json':return {'hierarchical_version':repair.VERSION}
        assert path.name=='T01.json'
        return used(1000,ANSWER)
    monkeypatch.setattr(runner,'read',read)
    backend=SimpleNamespace(VERSION=repair.VERSION,
        run_plan=lambda *a,**k:plan,
        run_review=lambda *a,**k:calls.append(a[3]) or {'error':None})
    def get(version):assert version==repair.VERSION;return backend
    monkeypatch.setattr('hierarchy_backend.get_backend',get)
    out=runner.hierarchical_one({'id':'T01','likes':H[:5],'dislikes':H[5:],'candidates':C},None,None,None,None)
    assert set(calls)=={'agent','qloo'} and out['plan_error'] is None


def test_service_routes_planner_and_review_through_same_registered_wrapper(tmp_path,monkeypatch):
    import service_v13 as svc
    metadata={str(i):{'title':f'Film {i}','year':1990+i,'genres':['Drama']} for i in range(1,11)}
    config={'selected_agent':'a','selected_qloo':'q','hierarchical_version':repair.VERSION,
            'judge_version':'cards2_paced','models':{
        'a':{'features':['hier_agent'],'model':{'scale':[1.],'weights':[1.]}},
        'q':{'features':['hier_qloo'],'model':{'scale':[1.],'weights':[1.]}}}}
    calls=[];plan=used(200,PLAN)
    def planner(*args,**kwargs):calls.append(('planner',kwargs['private_path']));return plan
    def reviewer(*args,**kwargs):calls.append((args[3],kwargs['private_path']));return used(300,ANSWER)
    backend=SimpleNamespace(VERSION=repair.VERSION,run_plan=planner,run_review=reviewer)
    def get(version):assert version==repair.VERSION;return backend
    monkeypatch.setattr('hierarchy_backend.get_backend',get)
    service=svc.RecommendationService(private_path=tmp_path,config=config,metadata=metadata,
                                      community=object(),hierarchy_culture_builder=object())
    assert service._planner(H,C,'TEST03',used(1000,ANSWER))[0] is plan
    service._hierarchy_review(H,C,'TEST03','qloo',plan,used(1000,ANSWER))
    assert calls==[('planner',tmp_path),('qloo',tmp_path)]
