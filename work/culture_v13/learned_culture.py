"""A frozen cultural representation with learned asymmetric taste interactions.

Four fixed background-training settings: cultural/MF representations x 32/64.
Movie vectors are never free parameters. Only two small metrics, a history gate,
and a shared quality coefficient are learned from explicit background ratings.
"""
from pathlib import Path
import json,random,math,sys,collections,hashlib,time
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'culture_v11/src'))
from audit_features import CulturalMetadataTools,PRIVATE,OLD,save
from graph_models import repaired_model,BASE,QLOO
from tastetrail.fusion import fit,probability
DEST=PRIVATE/'learned_culture'
SEED=20261030

def sigmoid(x):return 1/(1+np.exp(-np.clip(x,-30,30)))

def prepare_context(Z,bias,history,ratings,candidates):
    e=Z[history];t=Z[candidates]
    dots=np.einsum('bhd,bcd->bch',e,t)
    contexts=[]
    for sl in [slice(0,5),slice(5,8)]:
        logits=4*dots[:,:,sl]
        weights=np.exp(logits-logits.max(axis=2,keepdims=True))
        weights*=np.abs(ratings[:,None,sl]-3)
        weights/=np.maximum(weights.sum(axis=2,keepdims=True),1e-12)
        contexts.append(np.einsum('bch,bhd->bcd',weights,e[:,sl]))
    P=e[:,:5].mean(axis=1);N=e[:,5:].mean(axis=1)
    coherence_pos=(np.sum(P*P,axis=1)*25-5)/20
    coherence_neg=(np.sum(N*N,axis=1)*9-3)/6
    between=np.sum(P*N,axis=1)
    G=np.stack([np.ones(len(e)),coherence_pos,coherence_neg,between],axis=1)
    return contexts[0],contexts[1],t,G,bias[candidates]

def forward(params,context,gradient=False):
    A,B,g,beta=params;P,N,T,G,bias=context
    pos=np.einsum('bcd,df,bcf->bc',P,A,T)
    neg=np.einsum('bcd,df,bcf->bc',N,B,T)
    s=sigmoid(G@g);gate=2*s
    score=pos-gate[:,None]*neg+float(beta[0])*bias
    delta=score[:,0]-score[:,1]
    if not gradient:return delta
    dd=-sigmoid(-delta)/len(delta);ds=np.stack([dd,-dd],axis=1)
    dA=np.einsum('bc,bcd,bcf->df',ds,P,T)
    dB=np.einsum('bc,bcd,bcf->df',-ds*gate[:,None],N,T)
    dg=G.T@((-ds*neg).sum(axis=1)*2*s*(1-s))
    db=np.array([np.sum(ds*bias)])
    return delta,[dA,dB,dg,db]

def regularization(params):
    A,B,g,beta=params;d=len(A)
    # Identity-centered shrinkage preserves the semantic prior under little data.
    loss=.001*(np.sum((A-np.eye(d))**2)+np.sum((B-np.eye(d))**2))+.005*np.sum(g*g)
    grads=[.002*(A-np.eye(d)),.002*(B-np.eye(d)),.01*g,np.zeros_like(beta)]
    return float(loss),grads

def _draw(profiles,n,rng,genre,quality):
    hs=[];rs=[];cs=[]
    for j in rng.integers(len(profiles),size=n):
        p=profiles[int(j)];tp=p['target_pos'];tn=p['target_neg']
        x=int(rng.choice(tp));eligible=tn[(genre[x,tn]>0)&(np.abs(quality[x]-quality[tn])<=.4)]
        y=int(rng.choice(eligible if len(eligible) and rng.random()<.8 else tn))
        hp=rng.choice(p['history_pos'],5,replace=False);hn=rng.choice(p['history_neg'],3,replace=False);h=np.r_[hp,hn]
        hs.append(h);rs.append([p['ratings'][int(i)] for i in h]);cs.append([x,y])
    return np.asarray(hs,int),np.asarray(rs,float),np.asarray(cs,int)

def background_data():
    arr=np.load(OLD/'tools.npz');ids=list(map(int,arr['ids']));idx={i:j for j,i in enumerate(ids)}
    split=json.loads((OLD/'split_private.json').read_text());bg=split['background_users'];assert len(bg)==3640
    # Partition before any supervised representation fitting, including the MF control.
    shuffled=bg.copy();random.Random(SEED).shuffle(shuffled);cut=int(.8*len(shuffled))
    train_users=set(shuffled[:cut]);valid_users=set(shuffled[cut:])
    seq=collections.defaultdict(list)
    ratings_path=OLD.parent/'culture_v9/ratings.dat'
    for line in ratings_path.read_text().splitlines():
        u,i,r,t=map(int,line.split('::'))
        if u in train_users or u in valid_users:seq[u].append((i,r,t))
    pools={'train':[],'validation':[]}
    for u,rows in seq.items():
        rows.sort(key=lambda x:(x[2],x[0]));cut=int(.8*len(rows));h,t=rows[:cut],rows[cut:]
        hp=[idx[i] for i,r,ts in h if i in idx and r>=4];hn=[idx[i] for i,r,ts in h if i in idx and r<=2]
        tp=[idx[i] for i,r,ts in t if i in idx and r>=4];tn=[idx[i] for i,r,ts in t if i in idx and r<=2]
        if len(hp)<5 or len(hn)<3 or not tp or not tn:continue
        assert set(hp+hn).isdisjoint(tp+tn)
        p={'user_private':u,'history_pos':np.array(hp),'history_neg':np.array(hn),
           'target_pos':np.array(tp),'target_neg':np.array(tn),
           'ratings':{idx[i]:r for i,r,ts in rows if i in idx}}
        pools['train' if u in train_users else 'validation'].append(p)
    bgindex={u:j for j,u in enumerate(bg)};raw=arr['raw'][[bgindex[u] for u in sorted(train_users)]]
    means=arr['user_means'][[bgindex[u] for u in sorted(train_users)]];mask=raw>0
    bias=((raw-means[:,None])*mask).sum(axis=0)/(mask.sum(axis=0)+20)
    residual=(raw-means[:,None]-bias[None,:])*mask
    cultural=CulturalMetadataTools();assert cultural.ids==ids
    K=.75*cultural.kernels['curated']+.25*cultural.kernels['tags']
    K=K-K.mean(axis=0,keepdims=True)-K.mean(axis=1,keepdims=True)+K.mean()
    ev,V=np.linalg.eigh(K);order=np.argsort(ev)[::-1]
    _,s,vt=np.linalg.svd(residual,full_matrices=False)
    embeddings={}
    for d in [32,64]:
        for name,Z in [('qloo',V[:,order[:d]]*np.sqrt(np.maximum(ev[order[:d]],0))),
                       ('mf',vt[:d].T*np.sqrt(s[:d]))]:
            Z=Z/np.maximum(np.linalg.norm(Z,axis=1,keepdims=True),1e-12)
            embeddings[f'{name}_{d}']=Z
    genre=arr['genres']@arr['genres'].T
    # Fixed validation tasks, never refreshed according to candidate performance.
    validation=_draw(pools['validation'],1000,np.random.default_rng(SEED+1),genre,bias)
    provenance={'seed':SEED,'background_total':3640,'representation_train_users':len(train_users),
      'representation_validation_users':len(valid_users),'eligible_training_profiles':len(pools['train']),
      'eligible_validation_profiles':len(pools['validation']),'validation_tasks':1000,
      'history_size':8,'positive_history':5,'negative_history':3,'temporal_history_fraction':.8,
      'missing_rating_is_negative':False,'target_users_used':False,
      'train_tools_sha256':hashlib.sha256((OLD/'tools.npz').read_bytes()).hexdigest(),
      'metadata_sha256':hashlib.sha256((PRIVATE/'qloo_metadata.json').read_bytes()).hexdigest()}
    DEST.mkdir(parents=True,exist_ok=True,mode=0o700)
    save(DEST/'background_user_split.json',{'train':sorted(train_users),'validation':sorted(valid_users)})
    save(ROOT/'research/learned_culture_training_protocol.json',provenance)
    return embeddings,bias,pools['train'],validation,genre,ids,provenance

def train_one(name,Z,bias,profiles,validation,genre,steps=1200):
    d=Z.shape[1];params=[np.eye(d),np.eye(d),np.zeros(4),np.ones(1)]
    momentum=[np.zeros_like(x) for x in params];variance=[np.zeros_like(x) for x in params]
    rng=np.random.default_rng(SEED+2);vc=prepare_context(Z,bias,*validation)
    best=None;log=[];started=time.monotonic()
    for step in range(1,steps+1):
        batch=_draw(profiles,192,rng,genre,bias);context=prepare_context(Z,bias,*batch)
        delta,grads=forward(params,context,True);reg,reggrads=regularization(params)
        for j,(p,g,rg) in enumerate(zip(params,grads,reggrads)):
            g+=rg;momentum[j]=.9*momentum[j]+.1*g;variance[j]=.999*variance[j]+.001*g*g
            p-=.006*(momentum[j]/(1-.9**step))/(np.sqrt(variance[j]/(1-.999**step))+1e-8)
        if step%100==0:
            pred=forward(params,vc);loss=float(np.mean(np.logaddexp(0,-pred)))
            row={'step':step,'validation_loss':loss,'validation_accuracy':float(np.mean(pred>=0)),
                 'training_batch_loss':float(np.mean(np.logaddexp(0,-delta))),'regularization':reg}
            log.append(row)
            if best is None or loss<best[0]:best=(loss,[p.copy() for p in params],step)
            # User-split early stopping, maximum 1200 fixed steps, patience 400.
            if step-best[2]>=400:break
    A,B,g,beta=best[1]
    path=DEST/f'{name}.npz';np.savez_compressed(path,Z=Z,bias=bias,A=A,B=B,g=g,beta=beta);path.chmod(0o600)
    return {'name':name,'dimension':d,'selected_step':best[2],'validation_loss':best[0],
            'validation_accuracy':next(r['validation_accuracy'] for r in log if r['step']==best[2]),
            'seconds':time.monotonic()-started,'learned_parameter_count':sum(p.size for p in params),
            'movie_embeddings_fixed':True,'history':log,'weights_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}

def train_all():
    embeddings,bias,profiles,validation,genre,ids,provenance=background_data()
    rows={}
    for name in ['mf_32','qloo_32','mf_64','qloo_64']:
        rows[name]=train_one(name,embeddings[name],bias,profiles,validation,genre)
        print(json.dumps({k:v for k,v in rows[name].items() if k!='history'}),flush=True)
    selected={source:min([r for n,r in rows.items() if n.startswith(source+'_')],key=lambda r:(r['validation_loss'],r['dimension']))['name'] for source in ['mf','qloo']}
    save(ROOT/'research/learned_culture_training.json',{'provenance':provenance,'models':rows,'selected_on_background_only':selected,'ids':ids})

class LearnedCultureTools:
    def __init__(self,training=None,weights_dir=None):
        report=json.loads((Path(training) if training else ROOT/'research/learned_culture_training.json').read_text())
        self.ids=list(map(int,report['ids']));self.idx={i:j for j,i in enumerate(self.ids)};self.models={}
        for source,name in report['selected_on_background_only'].items():
            self.models[source]=dict(np.load((Path(weights_dir) if weights_dir else DEST)/f'{name}.npz'))
    def evaluate(self,history,candidates):
        # Consistent sign groups independent of caller ordering within each group.
        history=sorted(history,key=lambda x:(x[1]<=2,int(x[0])))
        assert len(history)==8 and sum(r>=4 for i,r in history)==5 and sum(r<=2 for i,r in history)==3
        h=np.array([[self.idx[int(i)] for i,r in history]]);r=np.array([[r for i,r in history]],float);c=np.array([[self.idx[int(i)] for i in candidates]])
        result={}
        for source,m in self.models.items():
            context=prepare_context(m['Z'],m['bias'],h,r,c)
            result[f'learned_{source}_margin']=float(forward([m[n] for n in ['A','B','g','beta']],context)[0])
        return result

def development():
    cases=[c for c in json.loads((OLD/'cases.json').read_text()) if c['split']=='dev']
    old=json.loads((OLD/'features_dev.json').read_text());tool=LearnedCultureTools();features={};swaperr=0
    for c in cases:
        ident=c['id'];v=dict(old[ident]['values']);raw=json.loads((OLD/'predictions'/f'{ident}.json').read_text())
        v['glm_log_odds']=repaired_model(raw)['glm_log_odds'];extra=tool.evaluate(c['likes']+c['dislikes'],c['candidates'])
        rev=tool.evaluate(c['likes']+c['dislikes'],list(reversed(c['candidates'])));swaperr=max(swaperr,max(abs(extra[n]+rev[n]) for n in extra))
        v.update(extra)
        for n in QLOO:v.setdefault(n,0.)
        features[ident]=v
    save(PRIVATE/'learned_culture_features_dev.json',features)
    families={'base':BASE,'old_qloo':BASE+QLOO,
      'learned_agent':BASE+['learned_mf_margin'],
      'learned_qloo':BASE+['learned_mf_margin']+QLOO+['learned_qloo_margin']}
    folds={}
    for difficulty in ['ordinary','shared_genre','fine_taste']:
        subset=[c for c in cases if c['difficulty']==difficulty];random.Random(20261025).shuffle(subset)
        for j,c in enumerate(subset):folds[c['id']]=j%5
    y=np.array([c['candidates'][0]==c['preferred'] for c in cases],float);fold=np.array([folds[c['id']] for c in cases]);trials=[];models={};oofs={}
    for name,names in families.items():
        X=np.array([[features[c['id']][n] for n in names] for c in cases])
        for ridge in [1.,10.]:
            pred=np.zeros(len(cases))
            for k in range(5):pred[fold==k]=probability(fit(X[fold!=k],y[fold!=k],ridge),X[fold==k])
            loss=float(np.mean(-y*np.log(np.maximum(pred,1e-9))-(1-y)*np.log(np.maximum(1-pred,1e-9))))
            row={'name':name,'ridge':ridge,'features':names,'correct':int(((pred>=.5)==y).sum()),'total':len(cases),'log_loss':loss};trials.append(row)
            key=f'{name}:{ridge:g}';models[key]={'features':names,'model':fit(X,y,ridge),'selection':row};oofs[key]={c['id']:float(p) for c,p in zip(cases,pred)}
    best={name:min([r for r in trials if r['name']==name],key=lambda r:(r['log_loss'],-r['correct'],-r['ridge'])) for name in families}
    save(ROOT/'research/learned_culture_development.json',{'development_only':True,'configuration_count':8,'candidate_swap_max_error':swaperr,'trials':trials,'best_by_family':best})
    save(ROOT/'learned_culture_candidate_config.json',{'models':{name:models[f"{r['name']}:{r['ridge']:g}"] for name,r in best.items()},'learned_representation_selection':'background validation only'})
    save(PRIVATE/'learned_culture_dev_oof.json',oofs)
    print(json.dumps({'best_by_family':best,'swap_error':swaperr}),flush=True)

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='train':train_all()
    else:development()
