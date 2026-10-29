"""Frozen three-condition confirmation. Collection never reads the answer key."""
import hashlib
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor,as_completed
from common import ROOT,PRIVATE,OLD,SEED,read,save,sha,parse_ranking,token_count
from baseline_snapshot import PROMPT
from tastetrail.model_client import ModelClient
from tastetrail.qloo_http import QlooClient
from tastetrail.fusion import qloo_features
from paced_transport import PacedTransport


def verify_lock():
    lock=read(ROOT/'research/formal_lock.json')
    assert sha(ROOT/'agent_config.json')==lock['configuration_sha']
    assert sha(PRIVATE/'confirmation_cases.json')==lock['cases_sha']
    assert sha(PRIVATE/'confirmation_inputs.json')==lock['inputs_sha']
    for name,digest in lock['source_sha'].items():assert sha(ROOT/name)==digest,name
    for path,digest in lock['private_tool_sha'].items():assert sha(path)==digest,path
    for path,digest in lock.get('dependency_source_sha',{}).items():assert sha(path)==digest,path


def visible_cases():
    # Data boundary: return only public titles/ratings through movie IDs.
    return read(PRIVATE/'confirmation_inputs.json')


def baseline_one(c):
    path=PRIVATE/'predictions'/f"{c['id']}.json"
    if path.exists():return read(path)
    meta=read(OLD/'metadata.json')
    payload={'liked_movies':[{**meta[str(i)],'rating':r} for i,r in c['likes']],
             'disliked_movies':[{**meta[str(i)],'rating':r} for i,r in c['dislikes']],
             'candidates':[{'id':f'C{j+1:02}',**meta[str(i)]} for j,i in enumerate(c['candidates'])]}
    from hierarchical_agent import SingleAttemptPacedTransport
    transport=SingleAttemptPacedTransport()
    client=ModelClient(provider='glm',max_tokens=16384,transport=transport)
    out={'id':c['id'],'error':None};start=time.monotonic()
    try:
        msg=client.complete([{'role':'system','content':PROMPT},
            {'role':'user','content':json.dumps(payload,ensure_ascii=False)}],json_output=True)
        out['output']=parse_ranking(msg.get('content',''))
    except Exception as e:out['error']=type(e).__name__+': '+str(e)[:180]
    out.update(seconds=time.monotonic()-start,calls=client.calls,metrics=client.metrics())
    out['transport_events']=transport.events
    out['physical_requests']=transport.physical_requests
    usage_known=all(isinstance(call.get('usage',{}).get('prompt_tokens'),(int,float)) and
                    isinstance(call.get('usage',{}).get('completion_tokens'),(int,float))
                    for call in client.calls) and bool(client.calls)
    bound=len((PROMPT+json.dumps(payload,ensure_ascii=False)).encode())+256+16384
    out['usage_is_upper_bound']=bool(transport.physical_requests and not usage_known)
    out['accounted_tokens']=max(token_count(out),bound if out['usage_is_upper_bound'] else 0)
    transport.close()
    save(path,out);return out


def qloo_one(c):
    path=PRIVATE/'evidence'/f"{c['id']}.json"
    if path.exists():return read(path)
    catalog=read(OLD/'catalog.json');want=[catalog[str(i)] for i in c['candidates']]
    history=c['likes']+c['dislikes']
    signals=[[catalog[str(i)] for i,r in c['likes']]]+[[catalog[str(i)]] for i,r in history]
    client=QlooClient(ttl=30*86400);out={'id':c['id'],'error':None,'rows':[]};start=time.monotonic()
    try:
        for signal in signals:
            data=client.get('/v2/insights',{'filter.type':'urn:entity:movie',
                 'signal.interests.entities':','.join(signal),
                 'filter.results.entities':','.join(want),'take':2})
            es=data.get('results',{}).get('entities',[])
            if len(es)!=2 or {e['entity_id'] for e in es}!=set(want):
                raise ValueError('Exact candidate pair unavailable')
            scores={e['entity_id']:e.get('query',{}).get('affinity') for e in es}
            out['rows'].append([scores[q] for q in want])
        out['features']=qloo_features(out['rows'],[r for i,r in history])
    except Exception as e:out['error']=type(e).__name__+': '+str(e)[:180]
    out.update(seconds=time.monotonic()-start,calls=client.calls)
    save(path,out);return out


def judge_one(c,mode):
    from compact_cards import EvidenceCardBuilder,JUDGE_SYSTEM_PROMPT
    cfg=read(ROOT/'agent_config.json');version=cfg['judge_version'] or 'cards2'
    path=PRIVATE/'judges'/version/mode/f"{c['id']}.json"
    if path.exists():return read(path)
    assert version in {'cards2','cards2_paced'}
    b=read(PRIVATE/'predictions'/f"{c['id']}.json");bt=token_count(b)
    data=EvidenceCardBuilder().build(c['likes']+c['dislikes'],c['candidates'],include_qloo=mode=='qloo')
    user=json.dumps(data,ensure_ascii=False,separators=(',',':'))
    bound=len((JUDGE_SYSTEM_PROMPT+user).encode())+256
    limit=min(4096,int(9*bt-bound))
    out={'id':c['id'],'mode':mode,'version':version,'error':None,'baseline_tokens':bt,
         'prompt_utf8_upper_bound':bound,'max_completion_tokens':limit,'request_data':data,
         'reasoning_effort_requested':'low'}
    if limit<768:
        out.update(error='budget_preflight_skip',calls=[],metrics={},seconds=0,physical_requests=0,accounted_tokens=0)
        save(path,out);return out
    from hierarchical_agent import SingleAttemptPacedTransport
    transport=SingleAttemptPacedTransport()
    client=ModelClient(provider='glm',max_tokens=limit,reasoning_effort='low',transport=transport);start=time.monotonic()
    try:
        msg=client.complete([{'role':'system','content':JUDGE_SYSTEM_PROMPT},
                             {'role':'user','content':user}],json_output=True)
        out['output']=parse_ranking(msg.get('content',''))
    except Exception as e:out['error']=type(e).__name__+': '+str(e)[:180]
    out.update(calls=client.calls,metrics=client.metrics(),seconds=time.monotonic()-start)
    out['transport_events']=transport.events
    out['physical_requests']=transport.physical_requests
    known=all(isinstance(call.get('usage',{}).get('prompt_tokens'),(int,float)) and
              isinstance(call.get('usage',{}).get('completion_tokens'),(int,float))
              for call in client.calls) and bool(client.calls)
    out['usage_is_upper_bound']=bool(transport.physical_requests and not known)
    out['reported_tokens']=token_count(out)
    out['accounted_tokens']=max(token_count(out),bound+limit if out['usage_is_upper_bound'] else 0)
    transport.close()
    out['total_token_ratio']=(bt+token_count(out))/bt if bt else None
    save(path,out);return out


def hierarchical_one(c,builder,culture,community,catalog):
    from hierarchy_backend import get_backend
    cfg=read(ROOT/'agent_config.json')
    backend=get_backend(cfg.get('hierarchical_version'))
    VERSION=backend.VERSION
    initial=read(PRIVATE/'predictions'/f"{c['id']}.json")
    # An absent initial usage cannot define a relative budget: fail closed.
    if token_count(initial)<=0:
        skipped={'id':c['id'],'error':'initial_usage_unavailable_no_extra_model_calls',
                 'calls':[],'metrics':{},'physical_requests':0,'accounted_tokens':0,
                 'reported_tokens':0,'fallback':True,'output':None}
        save(PRIVATE/VERSION/'plans'/f"{c['id']}.json",skipped)
        for mode in ['agent','qloo']:
            save(PRIVATE/VERSION/'reviews'/mode/f"{c['id']}.json",{**skipped,'mode':mode})
        return skipped
    h=c['likes']+c['dislikes'];ids=c['candidates']
    plan=backend.run_plan(h,ids,c['id'],token_count(initial),builder=builder)
    modes=['agent','qloo'];random.Random(SEED+int(c['id'][1:])).shuffle(modes)
    errors={}
    for mode in modes:
        r=backend.run_review(h,ids,c['id'],mode,plan,initial,builder=builder,
                     culture_builder=culture,community_tools=community,catalog=catalog)
        errors[mode]=r['error']
    return {'id':c['id'],'error':errors,'plan_error':plan['error']}


def run(kind):
    verify_lock();cs=visible_cases();random.Random(SEED).shuffle(cs)
    if kind=='baseline':jobs=[(baseline_one,c) for c in cs];workers=2
    elif kind=='qloo':jobs=[(qloo_one,c) for c in cs];workers=4
    elif kind=='groups':
        from grouped_qloo import query_history
        jobs=[(lambda c:query_history(c['likes']+c['dislikes'],c['candidates'],case_id=c['id']),c) for c in cs];workers=4
    elif kind=='hierarchical':
        import numpy as np
        from evidence_cards import EvidenceCardBuilder
        from tastetrail.cluster_tools import TasteTools
        builder=EvidenceCardBuilder();culture=EvidenceCardBuilder()
        community=TasteTools(dict(np.load(OLD/'tools.npz')));catalog=read(OLD/'catalog.json')
        jobs=[(lambda c:hierarchical_one(c,builder,culture,community,catalog),c) for c in cs];workers=2
    else:
        cfg=read(ROOT/'agent_config.json')
        modes=[]
        for mode,name in [('agent',cfg['selected_agent']),('qloo',cfg['selected_qloo'])]:
            if 'judge_'+mode in cfg['models'][name]['features']:modes.append(mode)
        # If selected Qloo uses an added judge, collect matched no-Qloo judge
        # too, even when the stronger no-Qloo model does not use one.
        if 'qloo' in modes and 'agent' not in modes:modes.append('agent')
        jobs=[(lambda c,m=m:judge_one(c,m),c) for c in cs for m in modes];workers=2
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures=[ex.submit(fn,c) for fn,c in jobs]
        for j,f in enumerate(as_completed(futures),1):
            r=f.result();print(json.dumps({'kind':kind,'completed':j,'total':len(jobs),
                                           'id':r.get('id'),'error':r.get('error')}),flush=True)
