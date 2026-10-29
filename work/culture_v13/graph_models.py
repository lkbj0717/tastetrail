"""Culture-kernel preference graphs, fitted only on background ratings and D/E.

All candidate scoring is antisymmetric. The only local targets are the eight
visible history ratings. Candidate ratings never enter feature construction.
"""
from pathlib import Path
import json,math,random,sys,hashlib,time
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'culture_v11/src'))
from audit_features import CulturalMetadataTools,PRIVATE,OLD,save
from tastetrail.fusion import fit,probability

BASE=['glm_log_odds','mf_32_10','community_margin']
QLOO=['signed_history_log_ratio','mf_weighted_qloo']
FAMILIES={
 'base':BASE,
 'old_affinity':BASE+QLOO,
 'curated_metadata':BASE+QLOO+['qmeta_curated_ridge_0.1'],
 'local_graph':BASE+QLOO+['qgraph_loo','qgraph_asymmetric'],
 'background_graph':BASE+QLOO+['qgraph_background','qgraph_background_loo'],
 'combined_graph':BASE+QLOO+['qgraph_loo','qgraph_asymmetric','qgraph_background','qgraph_background_loo'],
}

def repaired_model(row):
    """One predeclared rule for all systems: ranking determines direction.

    Probability magnitude is retained but reflected around .5 if inconsistent.
    Original files and response text remain untouched.
    """
    o=row.get('output')
    if not o:
        for call in reversed(row.get('calls',[])):
            text=(call.get('response_message') or {}).get('content','')
            try:o=json.loads(text)
            except (TypeError,ValueError):continue
            if isinstance(o,dict):break
    if not isinstance(o,dict):raise ValueError('No parseable JSON output')
    ranking=o.get('ranking');p=float(o.get('probability_c01'))/100
    if not isinstance(ranking,list) or len(ranking)!=2 or set(ranking)!={'C01','C02'}:
        raise ValueError('No valid complete ranking')
    if not np.isfinite(p) or not 0<=p<=1:raise ValueError('Invalid confidence')
    p=float(np.clip(p,1e-6,1-1e-6));magnitude=min(5.,abs(math.log(p/(1-p))))
    magnitude=max(magnitude,.01);logit=magnitude if ranking[0]=='C01' else -magnitude
    return {'ranking':ranking,'glm_log_odds':logit,'probability_c01':100/(1+math.exp(-logit)),
            'repair_applied':bool(row.get('error')) or ((p>=.5)!=(ranking[0]=='C01') and p!=.5)}

def _center(K):
    K=K-K.mean(axis=1,keepdims=True)-K.mean(axis=0,keepdims=True)+K.mean()
    scale=np.sqrt(np.maximum(np.diag(K),1e-12));K=K/scale[:,None]/scale[None,:]
    return (K+K.T)/2

def build_tools():
    cultural=CulturalMetadataTools();arr=np.load(OLD/'tools.npz')
    assert len(arr['raw'])==3640
    assert list(map(int,arr['ids']))==cultural.ids
    # A fixed blend of high-level culture and fine-grained topic neighborhoods.
    K=_center(.75*cultural.kernels['curated']+.25*cultural.kernels['tags'])
    ev,V=np.linalg.eigh(K);take=np.argsort(ev)[-32:];Z=V[:,take]*np.sqrt(np.maximum(ev[take],0))
    Z/=np.sqrt(np.maximum(np.mean(np.sum(Z*Z,axis=1)),1e-12))
    raw=arr['raw'];mask=arr['mask'];bias=arr['item_bias'];means=arr['user_means']
    U=np.zeros((len(raw),32))
    for j in range(len(raw)):
        observed=mask[j].astype(bool);H=Z[observed]
        y=raw[j,observed]-means[j]-bias[observed]
        U[j]=np.linalg.solve(H.T@H+10*np.eye(32),H.T@y)
    # Background users tell us which cultural axes distinguish preferences;
    # centering removes axes explaining a preference shared by nearly everyone.
    U-=U.mean(axis=0);cov=U.T@U/len(U);cov/=max(np.trace(cov)/32,1e-12)
    cov=.75*cov+.25*np.eye(32)
    KG=_center(Z@cov@Z.T)
    path=PRIVATE/'qloo_graph_tools.npz';np.savez_compressed(path,ids=arr['ids'],kernel=K,background_kernel=KG,item_bias=bias)
    path.chmod(0o600)
    save(ROOT/'research/graph_tools_provenance.json',{
        'background_users':3640,'culture_basis':32,'background_ridge':10,
        'covariance_shrinkage_to_identity':.25,'metadata_mix_curated':.75,
        'source_tools_sha256':hashlib.sha256((OLD/'tools.npz').read_bytes()).hexdigest(),
        'qloo_metadata_sha256':hashlib.sha256((PRIVATE/'qloo_metadata.json').read_bytes()).hexdigest(),
        'tool_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'uses_target_user_ids':False,'uses_candidate_ratings':False})

class GraphTools:
    def __init__(self,path=None):
        self.a=dict(np.load(path or PRIVATE/'qloo_graph_tools.npz'))
        self.ids=list(map(int,self.a['ids']));self.idx={v:i for i,v in enumerate(self.ids)}

    @staticmethod
    def predict(K,h,c,ratings,bias,mode):
        y=ratings-ratings.mean()-bias[h]
        H=K[np.ix_(h,h)];B=K[np.ix_(h,c)];ridge=1.
        weights=np.ones(len(h));trace={}
        if mode=='loo':
            A=H+ridge*np.eye(len(h));inv=np.linalg.inv(A);alpha=inv@y
            loo_residual=alpha/np.maximum(np.diag(inv),1e-12)
            weights=np.maximum(.1,1/(1+loo_residual**2))
            trace['history_loo_reliability']=weights.tolist()
        elif mode=='asymmetric':
            # Local model choice uses only known history, not candidate quality.
            trials=[]
            for nw in [.25,1.,4.]:
                w=np.where(ratings<=2,nw,1.)
                A=H+ridge*np.diag(1/w);inv=np.linalg.inv(A);alpha=inv@y
                residual=alpha/np.maximum(np.diag(inv),1e-12)
                trials.append((float(np.mean(np.minimum(residual**2,16))),abs(math.log(nw)),nw))
            nw=min(trials)[2];weights=np.where(ratings<=2,nw,1.)
            trace['negative_history_weight']=nw
        alpha=np.linalg.solve(H+ridge*np.diag(1/weights),y)
        score=bias[c]+B.T@alpha
        return float(score[0]-score[1]),trace

    def evaluate(self,history,candidates):
        h=np.array([self.idx[int(i)] for i,r in history]);c=np.array([self.idx[int(i)] for i in candidates]);r=np.array([r for i,r in history],float)
        values={};traces={}
        for name,key,mode in [('qgraph_loo','kernel','loo'),('qgraph_asymmetric','kernel','asymmetric'),
            ('qgraph_background','background_kernel','plain'),('qgraph_background_loo','background_kernel','loo')]:
            values[name],traces[name]=self.predict(self.a[key],h,c,r,self.a['item_bias'],mode)
        return values,traces

def development():
    # Filter roles immediately: C/F/R labels are never evaluated or selected.
    cases=[c for c in json.loads((OLD/'cases.json').read_text()) if c['split']=='dev']
    old=json.loads((OLD/'features_dev.json').read_text());meta=json.loads((PRIVATE/'qloo_metadata_features_dev.json').read_text())
    tool=GraphTools();features={};repairs=[];swaperr=0.;traces={}
    for c in cases:
        ident=c['id'];raw=json.loads((OLD/'predictions'/f'{ident}.json').read_text());model=repaired_model(raw)
        if model['repair_applied']:repairs.append(ident)
        g,trace=tool.evaluate(c['likes']+c['dislikes'],c['candidates'])
        rev,_=tool.evaluate(c['likes']+c['dislikes'],list(reversed(c['candidates'])))
        swaperr=max(swaperr,max(abs(g[n]+rev[n]) for n in g))
        v={**old[ident]['values'],**meta[ident],**g,'glm_log_odds':model['glm_log_odds']}
        # Neutral signed margin for unavailable affinity, not a fabricated score.
        # Metadata remains available even when the affinity route omits an item.
        for n in QLOO:v.setdefault(n,0.)
        features[ident]=v;traces[ident]=trace
    save(PRIVATE/'graph_features_dev.json',features);save(PRIVATE/'graph_traces_dev.json',traces)
    folds={}
    for difficulty in ['ordinary','shared_genre','fine_taste']:
        rows=[c for c in cases if c['difficulty']==difficulty];random.Random(20261025).shuffle(rows)
        for j,c in enumerate(rows):folds[c['id']]=j%5
    y=np.array([c['candidates'][0]==c['preferred'] for c in cases],float);f=np.array([folds[c['id']] for c in cases]);results=[];models={};oofs={}
    for name,names in FAMILIES.items():
        X=np.array([[features[c['id']][n] for n in names] for c in cases])
        for ridge in [.1,1.,10.,100.]:
            pred=np.zeros(len(cases))
            for k in range(5):
                tr=f!=k;te=f==k;pred[te]=probability(fit(X[tr],y[tr],ridge),X[te])
            correct=int(((pred>=.5)==y).sum());loss=float(np.mean(-y*np.log(np.maximum(pred,1e-9))-(1-y)*np.log(np.maximum(1-pred,1e-9))))
            key=f'{name}:{ridge:g}';results.append({'name':name,'ridge':ridge,'features':names,'correct':correct,'total':len(cases),'accuracy':correct/len(cases),'log_loss':loss})
            oofs[key]={c['id']:float(p) for c,p in zip(cases,pred)};models[key]={'features':names,'model':fit(X,y,ridge)}
    selection=lambda r:(r['log_loss'],-r['correct'],len(r['features']),-r['ridge'])
    byfamily={name:min([r for r in results if r['name']==name],key=selection) for name in FAMILIES}
    final=min([r for r in results if r['name']!='base'],key=selection)
    out={'development_only':True,'fold_seed':20261025,'cases':len(cases),'repaired_baseline_cases':repairs,
      'candidate_swap_max_error':swaperr,'configuration_count':len(results),'trials':results,'best_by_family':byfamily,'selected':final,
      'notes':['All variants share ranking-priority parsing for every baseline answer.','Missing old affinity margins are neutral zero; cached Qloo metadata remains usable.','No C/F/R outcome comparison performed.','Internal asymmetry adaptation uses only eight observed ratings.','Hyperparameters are exploratory; an untouched evaluation is required.']}
    save(ROOT/'research/graph_dev_results.json',out);save(PRIVATE/'graph_dev_oof.json',oofs)
    chosen={name:{**models[f"{r['name']}:{r['ridge']:g}"],'selection':r} for name,r in byfamily.items()}
    save(ROOT/'graph_candidate_config.json',{'selected':final,'models':chosen,'protocol':out['notes']})
    print(json.dumps({'best_by_family':byfamily,'selected':final,'repairs':repairs,'swap_error':swaperr},ensure_ascii=False))

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='build':build_tools()
    else:development()
