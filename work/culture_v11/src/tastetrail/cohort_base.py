"""Background-only preference tools; no LLM tokens or heldout labels needed."""
import numpy as np

def normalize(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-9)
def softmax(x):
    x=x-np.max(x);a=np.exp(x);return a/a.sum()

def spherical_kmeans(X,k,seed=20261025,iterations=60):
    X=normalize(np.asarray(X,float));rng=np.random.default_rng(seed)
    centers=[X[rng.integers(len(X))]]
    for _ in range(1,k):
        dist=np.maximum(0,1-(X@np.array(centers).T).max(axis=1))
        centers.append(X[rng.choice(len(X),p=dist/dist.sum())] if dist.sum()>0 else X[len(centers)%len(X)])
    C=np.array(centers);last=None
    for _ in range(iterations):
        score=X@C.T;labels=score.argmax(axis=1)
        if last is not None and np.array_equal(last,labels):break
        last=labels.copy()
        C=normalize(np.array([X[labels==i].mean(axis=0) if np.any(labels==i) else X[score.max(axis=1).argmin()] for i in range(k)]))
    return C,labels

def interest_groups(embeddings):
    """Deterministic two-interest partition, separate for likes and dislikes."""
    X=normalize(embeddings)
    if len(X)<=2:return np.arange(len(X))
    similarity=X@X.T;np.fill_diagonal(similarity,np.inf)
    a,b=np.unravel_index(np.argmin(similarity),similarity.shape)
    if np.allclose(X[a],X[b]):return np.zeros(len(X),int)
    C=X[[a,b]];last=None
    for _ in range(15):
        labels=(X@C.T).argmax(axis=1)
        if last is not None and np.array_equal(labels,last):break
        if len(set(labels))<2:break
        last=labels.copy();C=normalize(np.array([X[labels==j].mean(axis=0) for j in range(2)]))
    return labels

def build(raw,user_means,genre_matrix):
    raw=np.asarray(raw,float);mask=(raw>0).astype(float)
    global_mean=float(raw.sum()/mask.sum());item_bias=(raw.sum(axis=0)+20*global_mean)/(mask.sum(axis=0)+20)-global_mean
    centered=(raw-user_means[:,None])*mask
    norms=np.maximum(np.linalg.norm(centered,axis=0),1e-9)
    counts=mask.T@mask
    corr=(centered.T@centered)/(norms[:,None]*norms[None,:])*counts/(counts+20)
    np.fill_diagonal(corr,0.)
    residual=(centered-item_bias)*mask
    # Item factors and user cohorts fit ONLY on independent background people.
    _,sv,Vt=np.linalg.svd(residual,full_matrices=False)
    item_factors=Vt[:48].T*np.sqrt(sv[:48])
    user_latent=residual@Vt[:48].T
    cohort_centers,cohort_labels=spherical_kmeans(user_latent,32)
    prototypes=[];sizes=[]
    for k in range(32):
        group=cohort_labels==k;sizes.append(int(group.sum()))
        prototypes.append((centered[group].sum(axis=0)+10*item_bias)/(mask[group].sum(axis=0)+10))
    return dict(raw=raw,mask=mask,user_means=user_means,item_bias=item_bias,correlation=corr,
                residual=residual,item_factors=item_factors,genres=genre_matrix,
                cohort_prototypes=np.array(prototypes),cohort_sizes=np.array(sizes),
                cohort_centers=cohort_centers,cohort_labels=cohort_labels)

class TasteTools:
    def __init__(self,arrays):
        self.a=arrays;self.idx={int(i):j for j,i in enumerate(arrays['ids'])}

    def evaluate(self,history,candidates,qrows=None):
        a=self.a;hi=[self.idx[i] for i,r in history];ci=[self.idx[i] for i in candidates]
        ratings=np.array([r for i,r in history],float);signed=ratings-3;likes=ratings>=4;neg=ratings<=2
        query_mean=ratings.mean();res=ratings-query_mean-a['item_bias'][hi]
        corr=a['correlation'][np.ix_(ci,hi)];gs=a['genres'][ci]@a['genres'][hi].T
        item_score=(corr@signed)/np.maximum(np.abs(corr).sum(axis=1),1e-8)
        genre_score=(gs@signed)/np.maximum(np.abs(gs).sum(axis=1),1e-8)
        values={'community_margin':float(item_score[0]-item_score[1]),
                'genre_margin':float(genre_score[0]-genre_score[1]),
                'quality_margin':float(a['item_bias'][ci[0]]-a['item_bias'][ci[1]])}
        # Retrieve real people with similar observed taste, never query candidates
        # when computing neighbor weights or overlap counts.
        observed=a['mask'][:,hi];overlap=observed.sum(axis=1)
        background=a['residual'][:,hi]
        similarity=(background@res)/np.maximum(np.sqrt((background*background).sum(axis=1)*(observed@(res*res))),1e-8)
        similarity*=overlap/(overlap+3);similarity[overlap<3]=0;similarity=np.maximum(similarity,0.)
        order=np.argsort(-similarity,kind='stable')
        neighbor_meta={}
        for k in [30,100]:
            take=order[:k];w=similarity[take]**2
            scores=a['item_bias'][ci]+(w@a['residual'][np.ix_(take,ci)])/(w@a['mask'][np.ix_(take,ci)]+1.)
            values[f'neighbors_{k}']=float(scores[0]-scores[1]);neighbor_meta[str(k)]={'positive_weight_neighbors':int(np.sum(w>0))}
        # Fold in this new user's 8 ratings; no retraining or extra API call.
        H=a['item_factors'][hi];latent=np.linalg.solve(H.T@H+10*np.eye(H.shape[1]),H.T@res)
        score=a['item_bias'][ci]+a['item_factors'][ci]@latent
        values['latent_margin']=float(score[0]-score[1])
        # Soft memberships avoid forcing a mixed-taste person into one group.
        proto=a['cohort_prototypes'];distance=((proto[:,hi]-(ratings-query_mean))**2).mean(axis=1)
        posterior=softmax(-distance/.35)
        score=posterior@proto[:,ci];values['cohort_margin']=float(score[0]-score[1])
        entropy=float(-np.sum(posterior*np.log(np.maximum(posterior,1e-12)))/np.log(len(posterior)))
        values['cohort_entropy']=entropy
        # Group positive and negative histories independently.
        E=normalize(a['item_factors']);groups=[];group_sign=[]
        for direction,selector in [(1,likes),(-1,neg)]:
            positions=np.flatnonzero(selector);labels=interest_groups(E[np.array(hi)[positions]])
            for label in sorted(set(labels)):
                positions_in_group=positions[labels==label];groups.append(positions_in_group);group_sign.append(direction)
        centroids=normalize(np.array([E[np.array(hi)[g]].mean(axis=0) for g in groups]))
        positive=[j for j,s in enumerate(group_sign) if s==1];negative=[j for j,s in enumerate(group_sign) if s==-1]
        def clustered_scores(matrix,conditioned):
            scores=[]
            for cand in range(2):
                group_values=np.array([np.average(matrix[cand,g],weights=np.abs(signed[g])) for g in groups])
                total=0.
                for direction,indices in [(1,positive),(-1,negative)]:
                    w=softmax((centroids[indices]@E[ci[cand]])/.25) if conditioned else np.ones(len(indices))/len(indices)
                    total+=direction*float(w@group_values[indices])
                scores.append(total)
            return scores[0]-scores[1]
        values['balanced_interest_cf']=float(clustered_scores(corr,False))
        values['target_interest_cf']=float(clustered_scores(corr,True))
        qfeatures={}
        if qrows is not None:
            arr=np.asarray(qrows,float)
            if arr.shape!=(len(history)+1,2) or not np.isfinite(arr).all() or np.any(arr<0):raise ValueError('Incomplete Qloo evidence')
            logs=np.log(np.maximum(arr[1:],1e-9));centered_logs=(logs-logs.mean(axis=1,keepdims=True)).T
            qfeatures['balanced_interest_qloo']=float(clustered_scores(centered_logs,False))
            qfeatures['target_interest_qloo']=float(clustered_scores(centered_logs,True))
            agreement=posterior@proto[:,hi]
            reliability=np.maximum(.05,1/(1+(agreement-(ratings-query_mean))**2))
            ratios=logs[:,0]-logs[:,1]
            qfeatures['cohort_weighted_qloo']=float(np.average(ratios[likes],weights=reliability[likes]*signed[likes])-np.average(ratios[neg],weights=reliability[neg]*(-signed[neg])))
        order_cohort=np.argsort(-posterior)[:3]
        trace={'interest_groups':[{'direction':'like' if s==1 else 'dislike','history_positions':g.tolist()} for g,s in zip(groups,group_sign)],
               'top_cohorts':[{'cohort':int(k),'weight':float(posterior[k]),'background_members':int(a['cohort_sizes'][k])} for k in order_cohort],
               'cohort_entropy':entropy,'neighbors':neighbor_meta}
        return values,qfeatures,trace
