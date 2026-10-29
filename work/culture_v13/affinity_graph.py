"""Audit cached public movie-to-movie Qloo affinities; no preference labels.

Only catalog.json and raw Qloo cache responses are read. Existing cache keys
encode requests as SHA256; known single-seed request signatures reconstruct
their public query context without loading any user case files.
"""
from pathlib import Path
import collections,hashlib,json,sys,time,math
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'culture_v11/src'))
from tastetrail.config import settings
from audit_features import PRIVATE,OLD,save
BASE='https://hackathon.api.qloo.com';CACHE=PRIVATE.parent/'qloo_cache';DEST=PRIVATE/'affinity_graph'

def signature(params):
    return hashlib.sha256(json.dumps([BASE,'/v2/insights',params],sort_keys=True).encode()).hexdigest()

def entities(data):
    r=data.get('results',{})
    return r.get('entities',[]) if isinstance(r,dict) else []

def cache_audit():
    catalog=json.loads((OLD/'catalog.json').read_text());ids=sorted(map(int,catalog));qids=[catalog[str(i)] for i in ids];qidx={q:i for i,q in enumerate(qids)}
    groups=collections.defaultdict(dict);counts=collections.Counter();edge_samples=collections.defaultdict(list);provenance=[]
    for p in sorted(CACHE.glob('*.json')):
        counts['files']+=1
        try:o=json.loads(p.read_text());es=entities(o.get('data',{}))
        except (ValueError,TypeError):continue
        if not es:continue
        if len(es)!=2:counts['not_two_entity_results']+=1;continue
        pair=tuple(sorted(e.get('entity_id','') for e in es))
        if len(set(pair))!=2 or not all(q in qidx for q in pair):continue
        if not all(isinstance(e.get('query',{}).get('affinity'),(float,int)) and math.isfinite(e['query']['affinity']) for e in es):
            counts['two_catalog_items_without_finite_affinity']+=1;continue
        counts['two_catalog_items_with_affinity']+=1
        groups[pair][p.stem]={'fetched_at':o.get('fetched_at',0),'scores':{e['entity_id']:e['query']['affinity'] for e in es}}
    counts['unique_two_candidate_sets']=len(groups)
    for pair,records in groups.items():
        for order in [pair,pair[::-1]]:
            params={'filter.type':'urn:entity:movie','filter.results.entities':','.join(order),'take':2}
            for seed in qids:
                params['signal.interests.entities']=seed;digest=signature(params)
                if digest not in records:continue
                row=records[digest];counts['verified_single_seed_cache_files']+=1
                for target,value in row['scores'].items():edge_samples[qidx[seed],qidx[target]].append((row['fetched_at'],float(value)))
                provenance.append({'cache_digest':digest,'seed_movie_id':ids[qidx[seed]],'target_movie_ids':[ids[qidx[q]] for q in order]})
    # New probes retain explicit public request metadata and are safe to merge.
    probe_file=DEST/'probe_requests.json'
    if probe_file.exists():
        for probe in json.loads(probe_file.read_text()):
            if probe['status']!=200:continue
            response=json.loads((DEST/probe['response_file']).read_text());seed=probe['params']['signal.interests.entities']
            for e in entities(response):
                target=e.get('entity_id');value=e.get('query',{}).get('affinity')
                if seed in qidx and target in qidx and isinstance(value,(int,float)) and math.isfinite(value):
                    edge_samples[qidx[seed],qidx[target]].append((probe['fetched_at'],float(value)))
    A=np.full((len(ids),len(ids)),np.nan)
    for (i,j),values in edge_samples.items():A[i,j]=max(values,key=lambda x:x[0])[1]
    observed=np.isfinite(A);off=~np.eye(len(ids),dtype=bool);directed=int((observed&off).sum());symmetric=(observed|observed.T)&off
    reciprocal=observed&observed.T&off;dup=[max(v for t,v in row)-min(v for t,v in row) for row in edge_samples.values() if len(row)>1]
    reverse=np.abs(A-A.T)[reciprocal]
    stats={'catalog_items':len(ids),'cache_counts':dict(counts),
       'directed_observed_off_diagonal':directed,'directed_possible_off_diagonal':len(ids)*(len(ids)-1),
       'directed_coverage':directed/(len(ids)*(len(ids)-1)),
       'undirected_observed':int(symmetric.sum()/2),'undirected_possible':len(ids)*(len(ids)-1)//2,
       'undirected_coverage':float(symmetric.mean()*len(ids)/(len(ids)-1)),
       'out_degree_quantiles':np.quantile((observed&off).sum(axis=1),[0,.25,.5,.75,1]).tolist(),
       'zero_degree_items':int(((observed&off).sum(axis=1)==0).sum()),
       'reciprocal_pairs':int(reciprocal.sum()/2),'reverse_difference_max':float(reverse.max()) if len(reverse) else None,
       'reverse_difference_median':float(np.median(reverse)) if len(reverse) else None,
       'repeat_edge_count':len(dup),'repeat_edge_difference_max':float(max(dup)) if dup else None,
       'repeat_edge_difference_median':float(np.median(dup)) if dup else None,
       'observed_diagonal':int(np.diag(observed).sum()),
       'notes':['No user cases or ratings read.','Only cache requests whose exact public signature is reconstructed contribute edges.','Missing edges remain NaN, never zero affinity.','Unknown multi-seed responses are excluded.']}
    DEST.mkdir(parents=True,exist_ok=True,mode=0o700)
    np.savez_compressed(DEST/'recovered_affinity_graph.npz',movie_ids=np.array(ids),affinity=A);(DEST/'recovered_affinity_graph.npz').chmod(0o600)
    save(DEST/'cache_edge_provenance.json',provenance);save(ROOT/'research/affinity_graph_cache_audit.json',stats)
    print(json.dumps(stats),flush=True)

def probes():
    import httpx
    DEST.mkdir(parents=True,exist_ok=True,mode=0o700);ledger=DEST/'probe_requests.json'
    if ledger.exists():raise RuntimeError('Two-probe audit already executed; refuse extra calls')
    catalog=json.loads((OLD/'catalog.json').read_text());ids=sorted(map(int,catalog));qids=[catalog[str(i)] for i in ids]
    seed=catalog['1'];key=settings()['QLOO_API_KEY'];rows=[]
    # Probe 1 is exactly the large request under consideration; probe 2 uses
    # documented 50-result bounds and short URLs if the first is rejected.
    first={'filter.type':'urn:entity:movie','signal.interests.entities':seed,'filter.results.entities':','.join(qids),'take':473}
    for j in [0,1]:
        if j==0:params=first
        elif rows[0]['status']==200:
            params={**first,'take':50}
        else:
            params={'filter.type':'urn:entity:movie','signal.interests.entities':seed,'filter.results.entities':','.join(qids[:50]),'take':50}
        started=time.monotonic();name=f'probe_{j+1}_response.json'
        with httpx.Client(timeout=httpx.Timeout(60,connect=12),trust_env=False) as client:
            response=client.get(BASE+'/v2/insights',params=params,headers={'X-Api-Key':key})
        try:data=response.json()
        except ValueError:data={'response_text':response.text[:2000]}
        save(DEST/name,data)
        es=entities(data);returned={e.get('entity_id') for e in es};scores=[e.get('query',{}).get('affinity') for e in es]
        row={'index':j+1,'params':params,'status':response.status_code,'seconds':time.monotonic()-started,
             'fetched_at':time.time(),'response_file':name,'requested_count':len(params['filter.results.entities'].split(',')),
             'returned_count':len(es),'unique_returned_count':len(returned),
             'self_returned':seed in returned,'self_affinity':next((e.get('query',{}).get('affinity') for e in es if e.get('entity_id')==seed),None),
             'finite_affinities':sum(isinstance(v,(int,float)) and math.isfinite(v) for v in scores),
             'top_level_response_keys':list(data) if isinstance(data,dict) else [],
             'error_excerpt':str(data)[:600] if response.status_code!=200 else None}
        rows.append(row);save(ledger,rows)
    public=[{k:v for k,v in row.items() if k not in ['params','response_file']} for row in rows]
    save(ROOT/'research/affinity_graph_probe_summary.json',{'probes':public,
      'official_docs_take_limit':50,'official_reference':'https://github.com/qloo/docs-public/blob/main/reference/insights-api-deep-dive.md',
      'no_bulk_collection_performed':True,'no_personal_information_sent':True})
    print(json.dumps(public),flush=True)

def edge_loss_gradient(params,edge_i,edge_j,y):
    E,b,mu=params
    residual=np.sum(E[edge_i]*E[edge_j],axis=1)+b[edge_i]+b[edge_j]+mu[0]-y
    g=2*residual/len(y);gE=np.zeros_like(E);gb=np.zeros_like(b)
    np.add.at(gE,edge_i,g[:,None]*E[edge_j]);np.add.at(gE,edge_j,g[:,None]*E[edge_i])
    np.add.at(gb,edge_i,g);np.add.at(gb,edge_j,g)
    loss=float(np.mean(residual**2))
    return loss,[gE,gb,np.array([g.sum()])]

def _edge_train(n,d,ii,jj,y,train,validation,steps=1500,early_stop=True):
    rng=np.random.default_rng(20261031+d)
    params=[rng.normal(0,.03,(n,d)),np.zeros(n),np.array([y[train].mean()])]
    mom=[np.zeros_like(p) for p in params];var=[np.zeros_like(p) for p in params]
    best=None;log=[]
    for step in range(1,steps+1):
        chosen=rng.choice(train,size=min(512,len(train)),replace=False)
        loss,grad=edge_loss_gradient(params,ii[chosen],jj[chosen],y[chosen])
        grad[0]+=.02*params[0]/params[0].size;grad[1]+=.02*params[1]/len(params[1])
        for k,(p,g) in enumerate(zip(params,grad)):
            mom[k]=.9*mom[k]+.1*g;var[k]=.999*var[k]+.001*g*g
            p-=.008*(mom[k]/(1-.9**step))/(np.sqrt(var[k]/(1-.999**step))+1e-8)
        if step%100==0 or step==steps:
            idx=validation if len(validation) else train
            val,_=edge_loss_gradient(params,ii[idx],jj[idx],y[idx]);row={'step':step,'edge_validation_mse':val,'training_batch_mse':loss};log.append(row)
            if best is None or val<best[0]:best=(val,[p.copy() for p in params],step)
            if early_stop and step-best[2]>=400:break
    return (best[1] if early_stop else params),log,(best[2] if early_stop else steps),best[0]

def train_sparse():
    """Observed-only completion. No rating/target labels enter graph training."""
    obj=np.load(DEST/'recovered_affinity_graph.npz');ids=obj['movie_ids'];A=obj['affinity'];n=len(ids)
    known=np.isfinite(A);counts=known.astype(float)+known.T.astype(float)
    sym=(np.nan_to_num(A,nan=0)+np.nan_to_num(A.T,nan=0))/np.maximum(counts,1)
    ii,jj=np.where(np.triu(counts>0,1));y=np.log(np.maximum(sym[ii,jj],1e-9))
    assert np.all(np.isfinite(y));assert len(y)>n*4
    rng=np.random.default_rng(20261031);order=rng.permutation(len(y));cut=int(.85*len(y));train=order[:cut].tolist();val=order[cut:].tolist()
    # Every observable movie must have at least one train edge.
    seen=set(ii[train])|set(jj[train])
    for k in val.copy():
        if ii[k] not in seen or jj[k] not in seen:train.append(k);val.remove(k);seen.update([ii[k],jj[k]])
    train=np.array(train);val=np.array(val);records={};latent={};started=time.monotonic()
    for d in [8,16]:
        p,log,step,mse=_edge_train(n,d,ii,jj,y,train,val)
        # Refit with the chosen step count on all public edges; no preference labels.
        full,_,_,_=_edge_train(n,d,ii,jj,y,np.arange(len(y)),np.array([],int),steps=step,early_stop=False)
        E,b,mu=full;active=np.bincount(np.r_[ii,jj],minlength=n)>0;E[~active]=0
        path=DEST/f'completion_{d}.npz';np.savez_compressed(path,movie_ids=ids,E=E,b=b,mu=mu,active=active);path.chmod(0o600)
        latent[d]=(E,active)
        records[str(d)]={'dimension':d,'selected_step':step,'held_out_public_edge_mse':mse,'validation_history':log,
                        'training_edges':len(train),'validation_edges':len(val),'refit_public_edges':len(y),
                        'weights_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    selected=min([8,16],key=lambda d:(records[str(d)]['held_out_public_edge_mse'],d));E,active=latent[selected]
    # Kernel construction itself supplies no synthetic observed affinity labels.
    Z=E-E[active].mean(axis=0);Z[~active]=0;Z/=np.maximum(np.linalg.norm(Z,axis=1,keepdims=True),1e-12)
    K=Z@Z.T
    arr=np.load(OLD/'tools.npz');assert np.array_equal(arr['ids'],ids);raw=arr['raw'];bias=arr['item_bias'];means=arr['user_means'];U=np.zeros((len(raw),selected))
    for j,row in enumerate(raw):
        m=(row>0)&active;H=Z[m];target=row[m]-means[j]-bias[m]
        U[j]=np.linalg.solve(H.T@H+10*np.eye(selected),H.T@target)
    U-=U.mean(axis=0);cov=U.T@U/len(U);cov/=max(np.trace(cov)/selected,1e-12);cov=.75*cov+.25*np.eye(selected)
    bg=Z@cov@Z.T;norm=np.sqrt(np.maximum(np.diag(bg),1e-12));bg=bg/norm[:,None]/norm[None,:]
    path=DEST/'sparse_graph_tools.npz';np.savez_compressed(path,movie_ids=ids,kernel=K,background_kernel=bg,active=active,item_bias=bias);path.chmod(0o600)
    summary={'source':'exact-hash-recovered Qloo single-public-movie affinity responses','dimensions_attempted':[8,16],
      'selected_on_public_edge_validation_only':selected,'models':records,'public_nodes':n,'active_nodes':int(active.sum()),
      'observed_undirected_edges':len(y),'seconds':time.monotonic()-started,
      'graph_training_uses_ratings':False,'background_covariance_users':len(raw),
      'missing_edges_are_training_targets':False,'unknown_nodes_have_random_features':False,
      'recovered_graph_sha256':hashlib.sha256((DEST/'recovered_affinity_graph.npz').read_bytes()).hexdigest(),
      'tools_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
      'notes':['Only finite observed off-diagonal public edges enter completion loss.','No unobserved edge is labeled zero or disliked.','Unobserved candidate nodes force the non-graph fallback.','Completed entries and kernel diagonals are model estimates, not Qloo API observations.','Background covariance correction uses the existing disjoint 3640-user background only.']}
    save(ROOT/'research/affinity_graph_training.json',summary);print(json.dumps({k:v for k,v in summary.items() if k!='models'}),flush=True)

class SparseAffinityTools:
    def __init__(self,path=None):
        self.a=dict(np.load(path or DEST/'sparse_graph_tools.npz'));self.idx={int(i):j for j,i in enumerate(self.a['movie_ids'])}
    def evaluate(self,history,candidates):
        h=np.array([self.idx[int(i)] for i,r in history]);r=np.array([r for i,r in history],float);c=np.array([self.idx[int(i)] for i in candidates]);active=self.a['active'];keep=active[h];h=h[keep];r=r[keep]
        if not np.all(active[c]) or len(h)<4 or not np.any(r>=4) or not np.any(r<=2):return None
        bias=self.a['item_bias'];y=r-r.mean()-bias[h];out={}
        for key,name in [('kernel','qaff_sparse_margin'),('background_kernel','qaff_background_margin')]:
            K=self.a[key];alpha=np.linalg.solve(K[np.ix_(h,h)]+np.eye(len(h)),y);score=bias[c]+K[np.ix_(h,c)].T@alpha
            out[name]=float(score[0]-score[1])
        return out

def development_sparse():
    import random
    from graph_models import repaired_model,BASE as B,QLOO as Q
    from tastetrail.fusion import fit,probability
    cases=[c for c in json.loads((OLD/'cases.json').read_text()) if c['split']=='dev'];old=json.loads((OLD/'features_dev.json').read_text());tool=SparseAffinityTools();rows={};valid={};maxerr=0.
    for c in cases:
        ident=c['id'];v=dict(old[ident]['values']);v['glm_log_odds']=repaired_model(json.loads((OLD/'predictions'/f'{ident}.json').read_text()))['glm_log_odds']
        for name in Q:v.setdefault(name,0.)
        extra=tool.evaluate(c['likes']+c['dislikes'],c['candidates']);valid[ident]=extra is not None
        if extra is not None:
            rev=tool.evaluate(c['likes']+c['dislikes'],list(reversed(c['candidates'])));maxerr=max(maxerr,max(abs(extra[n]+rev[n]) for n in extra));v.update(extra)
        rows[ident]=v
    save(PRIVATE/'affinity_graph_features_dev.json',{'values':rows,'graph_valid':valid})
    families={'base':B,'old_qloo':B+Q,'sparse_qloo':B+Q+['qaff_sparse_margin'],'sparse_background_qloo':B+Q+['qaff_sparse_margin','qaff_background_margin']}
    folds={}
    for difficulty in ['ordinary','shared_genre','fine_taste']:
        subset=[c for c in cases if c['difficulty']==difficulty];random.Random(20261025).shuffle(subset)
        for j,c in enumerate(subset):folds[c['id']]=j%5
    y=np.array([c['candidates'][0]==c['preferred'] for c in cases],float);fold=np.array([folds[c['id']] for c in cases]);mask=np.array([valid[c['id']] for c in cases]);trials=[];models={};oofs={}
    for family,names in families.items():
        X=np.array([[rows[c['id']].get(n,0.) for n in names] for c in cases]);uses_graph=family.startswith('sparse_')
        for ridge in [1.,10.]:
            pred=np.zeros(len(cases))
            for k in range(5):
                tr=fold!=k;te=fold==k;ok=mask if uses_graph else np.ones(len(cases),bool)
                pred[te&ok]=probability(fit(X[tr&ok],y[tr&ok],ridge),X[te&ok])
                if np.any(te&~ok):
                    baseline=X[:,:len(B+Q)];pred[te&~ok]=probability(fit(baseline[tr],y[tr],ridge),baseline[te&~ok])
            loss=float(np.mean(-y*np.log(np.maximum(pred,1e-9))-(1-y)*np.log(np.maximum(1-pred,1e-9))))
            row={'name':family,'ridge':ridge,'features':names,'correct':int(((pred>=.5)==y).sum()),'total':len(cases),'log_loss':loss,'graph_fallbacks':int((~mask).sum()) if uses_graph else 0};trials.append(row)
            key=f'{family}:{ridge:g}';keep=mask if uses_graph else np.ones(len(cases),bool);models[key]={'features':names,'model':fit(X[keep],y[keep],ridge),'selection':row};oofs[key]={c['id']:float(p) for c,p in zip(cases,pred)}
    best={name:min([r for r in trials if r['name']==name],key=lambda r:(r['log_loss'],-r['correct'],-r['ridge'])) for name in families}
    save(ROOT/'research/affinity_graph_development.json',{'development_only':True,'configuration_count':8,'candidate_swap_max_error':maxerr,'trials':trials,'best_by_family':best})
    save(ROOT/'affinity_graph_candidate_config.json',{'models':{name:models[f"{r['name']}:{r['ridge']:g}"] for name,r in best.items()},'graph_dimension_selection':'public Qloo edge validation only','fallback':'If graph coverage unavailable, use same-ridge old_qloo model.'})
    save(PRIVATE/'affinity_graph_dev_oof.json',oofs);print(json.dumps({'best_by_family':best,'swap_error':maxerr}),flush=True)

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='probe':probes()
    elif len(sys.argv)>1 and sys.argv[1]=='train':train_sparse()
    elif len(sys.argv)>1 and sys.argv[1]=='dev':development_sparse()
    else:cache_audit()
