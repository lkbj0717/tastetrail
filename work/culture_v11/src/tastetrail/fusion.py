"""Small auditable pairwise fusion; all fitting belongs to development only."""
import math
import numpy as np

BASE_FEATURES = ['glm_log_odds', 'community_margin', 'genre_margin']
QLOO_FEATURES = ['pooled_log_ratio', 'signed_history_log_ratio', 'best_like_log_ratio', 'signed_agreement']

def sigmoid(x):
    return 1/(1+np.exp(-np.clip(x, -35, 35)))

def fit(X, y, ridge):
    X, y = np.asarray(X, float), np.asarray(y, float)
    # Pair reversal augmentation implies zero mean. Scale without subtracting a
    # mean so reversing candidates must reverse the final probability exactly.
    scale = np.sqrt(np.mean(X*X, axis=0)); scale[scale < 1e-8] = 1.
    Z = X/scale
    w = np.zeros(Z.shape[1])
    for _ in range(80):
        p = sigmoid(Z@w)
        grad = Z.T@(p-y) + ridge*w
        hess = (Z.T*(p*(1-p)))@Z + ridge*np.eye(len(w))
        step = np.linalg.solve(hess, grad)
        w -= step
        if np.max(np.abs(step)) < 1e-8: break
    return {'scale': scale.tolist(), 'weights': w.tolist(), 'ridge': ridge}

def probability(model, X):
    X = np.asarray(X, float)
    return sigmoid((X/np.asarray(model['scale']))@np.asarray(model['weights']))

def select(X, y, folds):
    X, y, folds = np.asarray(X), np.asarray(y), np.asarray(folds)
    rows=[]
    for ridge in [.1, 1., 10.]:
        out=np.zeros(len(y))
        for f in range(5):
            train=folds!=f; test=~train
            model=fit(X[train], y[train], ridge)
            out[test]=probability(model, X[test])
        loss=float(np.mean(-y*np.log(np.maximum(out, 1e-9))-(1-y)*np.log(np.maximum(1-out, 1e-9))))
        rows.append({'ridge':ridge, 'cross_validation_log_loss':loss,
                     'cross_validation_accuracy':float(np.mean((out>=.5)==y))})
    best=min(rows,key=lambda r:(r['cross_validation_log_loss'],-r['ridge']))
    return {'model':fit(X,y,best['ridge']), 'selection':rows, 'selected':best,
            'fit_count':len(y)}

def model_feature(p, first_choice):
    p=float(p)/100
    if p == .5: return .01 if first_choice else -.01
    return float(np.clip(math.log(p/(1-p)), -5, 5))

def qloo_features(rows, ratings):
    """Each row has [affinity(C01), affinity(C02)], pooled first, then history."""
    a=np.asarray(rows,float)
    if a.shape != (len(ratings)+1,2) or not np.all(np.isfinite(a)) or np.any(a<0):
        raise ValueError('Incomplete or invalid exact-candidate Qloo evidence')
    logs=np.log(np.maximum(a,1e-9)); ratio=np.clip(logs[:,0]-logs[:,1],-10,10)
    likes=np.array(ratings)>=4; dislikes=np.array(ratings)<=2
    if not likes.any() or not dislikes.any(): raise ValueError('Both taste directions required')
    weights=np.abs(np.asarray(ratings)-3)
    pos=float(np.average(ratio[1:][likes],weights=weights[likes]))
    neg=float(np.average(ratio[1:][dislikes],weights=weights[dislikes]))
    maxima=logs[1:][likes].max(axis=0)
    agreement=float(np.mean(np.sign(ratio[1:][likes]))-np.mean(np.sign(ratio[1:][dislikes])))
    return [float(ratio[0]),pos-neg,float(np.clip(maxima[0]-maxima[1],-10,10)),agreement]

def decide(base_features, qfeatures, config):
    p_agent=float(probability(config['agent']['model'],base_features))
    fallback=qfeatures is None
    p_final=p_agent if fallback else float(probability(config['agent_qloo']['model'],base_features+qfeatures))
    return {'agent_probability':p_agent,'agent_qloo_probability':p_final,
            'agent_choice':0 if p_agent>=.5 else 1,
            'agent_qloo_choice':0 if p_final>=.5 else 1,
            'qloo_fallback':fallback}
