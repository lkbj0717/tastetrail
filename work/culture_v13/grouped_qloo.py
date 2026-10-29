"""Candidate-independent cultural subgroups, queried jointly against real Qloo.

Exactly three additional logical queries: 2 liked movies, 3 liked movies,
and all 3 disliked movies. Personal rating values never leave this process.
"""
from pathlib import Path
import collections,hashlib,itertools,json,math,random,re,sys,time
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'culture_v11/src'))
from audit_features import CulturalMetadataTools,PRIVATE,OLD,save
from graph_models import repaired_model,BASE,QLOO
from tastetrail.config import settings
from tastetrail.fusion import fit,probability
DEST=PRIVATE/'grouped_qloo';BASE_URL='https://hackathon.api.qloo.com'

def dev_cases():
    return [c for c in json.loads((OLD/'cases.json').read_text()) if c['split']=='dev']

def partition(likes,kernel,index):
    """Ten fixed 2/3 partitions. Tie-break uses public movie IDs only."""
    likes=sorted((int(i),int(r)) for i,r in likes);assert len(likes)==5
    trials=[]
    for chosen in itertools.combinations(range(5),2):
        groups=[[likes[j] for j in chosen],[likes[j] for j in range(5) if j not in chosen]]
        coherence=[];within=[]
        for g in groups:
            values=[float(kernel[index[a[0]],index[b[0]]]) for a,b in itertools.combinations(g,2)]
            coherence.append(float(np.mean(values)));within.extend(values)
        trials.append({'groups':groups,'coherence':coherence,'partition_score':float(np.mean(within)),
                       'tie_key':tuple(i for i,r in groups[0])})
    selected=min(trials,key=lambda x:(-x['partition_score'],x['tie_key']))
    strengths=np.array([sum(r-3 for i,r in g) for g in selected['groups']],float)
    weights=strengths*np.maximum(selected['coherence'],1e-6);weights/=weights.sum()
    return {**selected,'group_weights':weights.tolist(),'group_strengths':strengths.tolist()}

def request(params,key):
    """Bounded single attempt; cache + operational failures retained explicitly."""
    import httpx
    signature=json.dumps([BASE_URL,'/v2/insights',params],sort_keys=True);digest=hashlib.sha256(signature.encode()).hexdigest()
    own=DEST/'cache'/f'{digest}.json';existing=PRIVATE.parent/'qloo_cache'/f'{digest}.json'
    for p in [own,existing]:
        if p.exists():
            try:cached=json.loads(p.read_text())
            except (ValueError,OSError):continue
            if 'data' in cached and cached.get('http_status',200)==200 and time.time()-cached.get('fetched_at',0)<30*86400:
                return cached['data'],{'cache_hit':True,'digest':digest,'seconds':0,'fetched_at':cached['fetched_at']}
    started=time.monotonic();record={'cache_hit':False,'digest':digest,'attempt':1};data=None
    try:
        with httpx.Client(timeout=httpx.Timeout(40,connect=12),trust_env=False) as client:
            response=client.get(BASE_URL+'/v2/insights',params=params,headers={'X-Api-Key':key})
        record['http_status']=response.status_code
        record['quota_headers']={k:v for k,v in response.headers.items() if 'ratelimit' in k.lower() or 'rate-limit' in k.lower() or k.lower()=='retry-after'}
        record['response_bytes_decoded']=len(response.content)
        try:data=response.json()
        except ValueError:data={'error_text':response.text[:600]}
        record['fetched_at']=time.time()
        temporary=own.with_suffix(f'.{time.time_ns()}.tmp')
        save(temporary,{'fetched_at':record['fetched_at'],'data':data,'http_status':response.status_code,'quota_headers':record['quota_headers']});temporary.replace(own)
        if response.status_code!=200:raise ValueError(f'Qloo HTTP {response.status_code}')
        if not isinstance(data,dict) or not data.get('success',True) or 'results' not in data:raise ValueError('Invalid Qloo response')
    except (httpx.HTTPError,ValueError) as e:record['error']=type(e).__name__+': '+str(e)[:180]
    record['seconds']=time.monotonic()-started
    return data,record

def new_features(rows,weights,pooled_like_ratio):
    a=np.asarray(rows,float)
    if a.shape!=(3,2) or not np.isfinite(a).all() or np.any(a<=0):raise ValueError('Incomplete group affinities')
    ratios=np.log(a[:,0])-np.log(a[:,1]);w=np.asarray(weights,float)
    return {'qgroup_weighted_contrast':float(w@ratios[:2]-ratios[2]),
            'qgroup_pooled_contrast':float(pooled_like_ratio-ratios[2])}

def query_history(history,candidates,*,case_id=None,cultural=None,catalog=None,key=None):
    """Public tool API: visible history + candidates only, never hidden labels."""
    history=sorted((int(i),int(r)) for i,r in history);candidates=list(map(int,candidates))
    likes=[x for x in history if x[1]>=4];dislikes=[x for x in history if x[1]<=2]
    if len(likes)!=5 or len(dislikes)!=3 or any(r not in [1,2,4,5] for i,r in history) or len(candidates)!=2 or len(set(i for i,r in history)|set(candidates))!=10:
        raise ValueError('Five likes, three dislikes, and two distinct unseen candidates required')
    fingerprint=hashlib.sha256(json.dumps([history,candidates]).encode()).hexdigest()
    case_id=case_id or 'LIVE_'+fingerprint[:20]
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}',case_id):raise ValueError('Unsafe case identifier')
    path=DEST/'cases'/f'{case_id}.json'
    if path.exists():
        prior=json.loads(path.read_text())
        if prior.get('request_fingerprint')!=fingerprint:raise ValueError('Cache case ID refers to different input')
        return prior
    cultural=cultural or CulturalMetadataTools();catalog=catalog or json.loads((OLD/'catalog.json').read_text());key=key or settings()['QLOO_API_KEY']
    start=time.monotonic();plan=partition(likes,cultural.kernels['curated'],cultural.index)
    groups=plan['groups']+[dislikes]
    want=[catalog[str(i)] for i in candidates]
    result={'id':case_id,'history':history,'candidates':candidates,'request_fingerprint':fingerprint,
            'plan':plan,'error':None,'rows':[],'calls':[],'requests':[]}
    for group in groups:
        params={'filter.type':'urn:entity:movie','signal.interests.entities':','.join(catalog[str(i)] for i,r in group),
                'filter.results.entities':','.join(want),'take':2}
        data,call=request(params,key);result['calls'].append(call);result['requests'].append(params)
        if call.get('error'):result['error']=call['error'];break
        try:
            es=data['results']['entities'];scores={e['entity_id']:e.get('query',{}).get('affinity') for e in es}
            if len(es)!=2 or set(scores)!=set(want):raise ValueError('Exact two candidates not returned')
            row=[scores[i] for i in want]
            if not all(isinstance(v,(int,float)) and math.isfinite(v) and v>0 for v in row):raise ValueError('Missing or invalid affinity')
            result['rows'].append(row)
        except (KeyError,TypeError,ValueError) as e:result['error']=str(e);break
    result['seconds']=time.monotonic()-start;save(path,result);return result

def features_from_rows(history,candidates,group_record,existing_rows):
    """Pure antisymmetric features; existing rows use explicit order or record order.

    existing_rows may be {rows: [...], candidates: [movie_id,movie_id]}, or a
    plain 9x2 list in group_record['candidates'] order. The first row is pooled
    likes. No labels or external I/O are accepted/performed.
    """
    if group_record.get('error'):raise ValueError('Grouped Qloo evidence unavailable')
    candidates=list(map(int,candidates));record_order=list(map(int,group_record['candidates']))
    if set(candidates)!=set(record_order):raise ValueError('Evidence belongs to another candidate pair')
    if isinstance(existing_rows,dict):
        existing_order=list(map(int,existing_rows.get('candidates',record_order)));previous=existing_rows['rows']
    else:existing_order=record_order;previous=existing_rows
    if set(existing_order)!=set(candidates):raise ValueError('Pooled evidence belongs to another pair')
    first=previous[0]
    if not all(isinstance(x,(int,float)) and math.isfinite(x) and x>0 for x in first):raise ValueError('Missing pooled-likes affinity')
    pool=[first[existing_order.index(i)] for i in candidates];pooled=math.log(pool[0])-math.log(pool[1])
    positions=[record_order.index(i) for i in candidates];rows=[[r[j] for j in positions] for r in group_record['rows']]
    ratings={int(i):int(r) for i,r in history};plan=group_record['plan']
    if {i for i,r in group_record['history']}!=set(ratings):raise ValueError('Evidence belongs to another history')
    strengths=np.array([sum(ratings[int(i)]-3 for i,r in g) for g in plan['groups']],float)
    weights=strengths*np.maximum(plan['coherence'],1e-6);weights/=weights.sum()
    return new_features(rows,weights,pooled)

def one(c,cultural,catalog,key):
    return query_history(c['likes']+c['dislikes'],c['candidates'],case_id=c['id'],cultural=cultural,catalog=catalog,key=key)

def joint_vs_single(c,group_record):
    """Compare public-query algebra only; this routine never inspects labels."""
    if group_record['error']:return []
    previous=json.loads((OLD/'evidence'/f"{c['id']}.json").read_text())['rows']
    single={int(i):previous[j+1] for j,(i,r) in enumerate(c['likes']+c['dislikes'])}
    groups=group_record['plan']['groups']+[sorted((int(i),int(r)) for i,r in c['dislikes'])];out=[]
    for j,g in enumerate(groups):
        a=np.array([single[i] for i,r in g],float)
        if not np.isfinite(a).all() or np.any(a<=0):continue
        actual=math.log(group_record['rows'][j][0])-math.log(group_record['rows'][j][1])
        mean_log=float(np.mean(np.log(a[:,0])-np.log(a[:,1])))
        log_mean=float(math.log(a[:,0].mean())-math.log(a[:,1].mean()))
        out.append({'case_id':c['id'],'group_index':j,'group_movie_ids':[i for i,r in g],
          'actual_joint_logratio':actual,'mean_single_logratio':mean_log,'logratio_of_mean_affinities':log_mean,
          'difference_vs_mean_log':actual-mean_log,'difference_vs_log_mean':actual-log_mean,
          'direction_change_vs_mean_log':bool((actual>=0)!=(mean_log>=0)),
          'direction_change_vs_log_mean':bool((actual>=0)!=(log_mean>=0))})
    return out

def collect(stage):
    cs=dev_cases();cultural=CulturalMetadataTools();catalog=json.loads((OLD/'catalog.json').read_text());key=settings()['QLOO_API_KEY']
    if stage=='pilot':cs=[c for c in cs if c['id'] in ['D001','D002']]
    else:
        pilot=json.loads((ROOT/'research/grouped_qloo_pilot.json').read_text())
        if not pilot['batch_allowed']:raise RuntimeError('Pilot did not permit batch')
    DEST.mkdir(parents=True,exist_ok=True,mode=0o700);rows=[]
    with ThreadPoolExecutor(max_workers=2 if stage=='pilot' else 4) as pool:
        futures={pool.submit(one,c,cultural,catalog,key):c['id'] for c in cs}
        for j,f in enumerate(as_completed(futures),1):
            r=f.result();rows.append(r);print(json.dumps({'stage':stage,'completed':j,'total':len(cs),'id':r['id'],'error':r['error'],'seconds':r['seconds']}),flush=True)
    summary={'stage':stage,'cases':len(rows),'success':sum(not r['error'] for r in rows),
       'failed':[{k:r[k] for k in ['id','error','seconds']} for r in rows if r['error']],
       'network_calls':sum(not call['cache_hit'] for r in rows for call in r['calls']),
       'cache_hits':sum(call['cache_hit'] for r in rows for call in r['calls']),
       'case_seconds_mean':float(np.mean([r['seconds'] for r in rows])),
       'case_seconds_max':float(max(r['seconds'] for r in rows)),
       'request_seconds_mean':float(np.mean([call['seconds'] for r in rows for call in r['calls']])),
       'quota_headers':[call['quota_headers'] for r in rows for call in r['calls'] if call.get('quota_headers')],
       'records_file':'private grouped_qloo/cases/*.json','candidate_labels_sent':False}
    if stage=='pilot':
        byid={c['id']:c for c in cs};algebra=[r for row in rows for r in joint_vs_single(byid[row['id']],row)]
        summary['joint_vs_single']=algebra
        summary['max_difference_vs_mean_log']=max((abs(r['difference_vs_mean_log']) for r in algebra),default=None)
        summary['max_difference_vs_log_mean']=max((abs(r['difference_vs_log_mean']) for r in algebra),default=None)
        summary['joint_matches_normalized_linear_pool']=len(algebra)==6 and summary['max_difference_vs_log_mean']<1e-5
        remaining=[]
        for headers in summary['quota_headers']:
            for name,value in headers.items():
                if 'month' in name.lower() and 'remaining' in name.lower():
                    try:remaining.append(int(value))
                    except ValueError:pass
        needed=3*(len(dev_cases())-len(rows));summary['estimated_remaining_requests']=needed
        summary['observed_month_quota_remaining']=min(remaining) if remaining else None
        summary['batch_allowed']=summary['success']==len(rows) and summary['case_seconds_max']<=30 and (not remaining or min(remaining)>=needed) and not summary['joint_matches_normalized_linear_pool']
    save(ROOT/f'research/grouped_qloo_{stage}.json',summary);print(json.dumps(summary),flush=True)

def development():
    cs=dev_cases();old=json.loads((OLD/'features_dev.json').read_text());rows={};valid={};maxerr=0;errors=[]
    for c in cs:
        ident=c['id'];v=dict(old[ident]['values']);raw=json.loads((OLD/'predictions'/f'{ident}.json').read_text());v['glm_log_odds']=repaired_model(raw)['glm_log_odds']
        for n in QLOO:v.setdefault(n,0.)
        grouped=json.loads((DEST/'cases'/f'{ident}.json').read_text());valid[ident]=False
        if not grouped['error']:
            try:
                previous=json.loads((OLD/'evidence'/f'{ident}.json').read_text())['rows']
                extra=features_from_rows(c['likes']+c['dislikes'],c['candidates'],grouped,previous)
                rev=features_from_rows(c['likes']+c['dislikes'],list(reversed(c['candidates'])),grouped,previous)
                maxerr=max(maxerr,max(abs(extra[n]+rev[n]) for n in extra));v.update(extra);valid[ident]=True
            except (ValueError,KeyError,IndexError,TypeError) as e:errors.append({'id':ident,'error':str(e)})
        else:errors.append({'id':ident,'error':grouped['error']})
        rows[ident]=v
    save(PRIVATE/'grouped_qloo_features_dev.json',{'values':rows,'group_valid':valid})
    G=['qgroup_weighted_contrast','qgroup_pooled_contrast']
    families={'base':BASE,'old_qloo':BASE+QLOO,'grouped_qloo':BASE+G,'combined_qloo':BASE+QLOO+G}
    folds={}
    for difficulty in ['ordinary','shared_genre','fine_taste']:
        subset=[c for c in cs if c['difficulty']==difficulty];random.Random(20261025).shuffle(subset)
        for j,c in enumerate(subset):folds[c['id']]=j%5
    y=np.array([c['candidates'][0]==c['preferred'] for c in cs],float);fold=np.array([folds[c['id']] for c in cs]);mask=np.array([valid[c['id']] for c in cs]);trials=[];models={};oofs={}
    XB=np.array([[rows[c['id']][n] for n in BASE+QLOO] for c in cs])
    for name,names in families.items():
        X=np.array([[rows[c['id']].get(n,0.) for n in names] for c in cs]);uses_group=name in ['grouped_qloo','combined_qloo'];ok=mask if uses_group else np.ones(len(cs),bool)
        for ridge in [1.,10.]:
            pred=np.zeros(len(cs))
            for k in range(5):
                tr=fold!=k;te=fold==k;pred[te&ok]=probability(fit(X[tr&ok],y[tr&ok],ridge),X[te&ok])
                if np.any(te&~ok):pred[te&~ok]=probability(fit(XB[tr],y[tr],ridge),XB[te&~ok])
            loss=float(np.mean(-y*np.log(np.maximum(pred,1e-9))-(1-y)*np.log(np.maximum(1-pred,1e-9))))
            row={'name':name,'ridge':ridge,'features':names,'correct':int(((pred>=.5)==y).sum()),'total':len(cs),'log_loss':loss,'group_fallbacks':int((~ok).sum())};trials.append(row)
            key=f'{name}:{ridge:g}';models[key]={'features':names,'model':fit(X[ok],y[ok],ridge),'selection':row};oofs[key]={c['id']:float(p) for c,p in zip(cs,pred)}
    best={name:min([r for r in trials if r['name']==name],key=lambda r:(r['log_loss'],-r['correct'],-r['ridge'])) for name in families}
    save(ROOT/'research/grouped_qloo_development.json',{'development_only':True,'configuration_count':8,'candidate_swap_max_error':maxerr,'group_errors':errors,'trials':trials,'best_by_family':best})
    save(ROOT/'grouped_qloo_candidate_config.json',{'models':{name:models[f"{r['name']}:{r['ridge']:g}"] for name,r in best.items()},'fallback':'same-ridge old_qloo for unavailable group evidence'})
    save(PRIVATE/'grouped_qloo_dev_oof.json',oofs);print(json.dumps({'best_by_family':best,'candidate_swap_max_error':maxerr}),flush=True)

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1] in ['pilot','batch']:collect(sys.argv[1])
    else:development()
