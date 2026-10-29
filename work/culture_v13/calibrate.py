"""Development-only selection. No confirmation cases or labels are loaded."""
import collections
import json
import math
import random
import numpy as np
from common import ROOT,PRIVATE,OLD,read,save,dev_cases,recovered_output,token_count
from tastetrail.fusion import fit,probability

BASE=['glm_log_odds','mf_32_10','community_margin']
AFFINITY=['signed_history_log_ratio','mf_weighted_qloo']
GRAPH=['qgraph_background','qgraph_background_loo']


def loss(y,p):
    p=np.clip(p,1e-9,1-1e-9)
    return float(np.mean(-y*np.log(p)-(1-y)*np.log(1-p)))


def folds_for(cs):
    fold={}
    for group in ['ordinary','shared_genre','fine_taste']:
        rows=[c for c in cs if c['difficulty']==group]
        random.Random(20261025).shuffle(rows)
        for i,c in enumerate(rows):fold[c['id']]=i%5
    return np.array([fold[c['id']] for c in cs])


def nested_family(X,y,fold):
    """Choose ridge using inner CV, not the held-out outer-fold labels."""
    predictions=np.zeros(len(y));chosen=[]
    for outer in range(5):
        train=fold!=outer;test=~train;trials=[]
        for ridge in [1.,10.,100.]:
            inner=np.zeros(train.sum());XF=X[train];yf=y[train];ff=fold[train]
            for k in sorted(set(ff)):
                a=ff!=k;b=~a;inner[b]=probability(fit(XF[a],yf[a],ridge),XF[b])
            trials.append((loss(yf,inner),-ridge,ridge))
        ridge=min(trials)[2];chosen.append(ridge)
        predictions[test]=probability(fit(X[train],y[train],ridge),X[test])
    final_trials=[]
    for ridge in [1.,10.,100.]:
        p=np.zeros(len(y))
        for k in range(5):
            a=fold!=k;b=~a;p[b]=probability(fit(X[a],y[a],ridge),X[b])
        final_trials.append((loss(y,p),-ridge,ridge))
    ridge=min(final_trials)[2]
    return predictions,fit(X,y,ridge),chosen


def run(version='cards2',include_judges=True):
    cs=dev_cases();values=read(PRIVATE/'graph_features_dev.json')
    y=np.array([c['candidates'][0]==c['preferred'] for c in cs],float)
    fold=folds_for(cs)
    families={
      'agent_numeric':BASE,
      'qloo_affinity':BASE+AFFINITY,
      'qloo_metadata':BASE+AFFINITY+['qmeta_curated_ridge_0.1'],
      'qloo_graph':BASE+AFFINITY+GRAPH,
    }
    if include_judges:
        families.update({
          'agent_judge':BASE+['judge_agent'],
          'qloo_judge':BASE+AFFINITY+['judge_qloo'],
          'qloo_judge_semantic':BASE+['judge_qloo'],
          'qloo_judge_graph':BASE+AFFINITY+GRAPH+['judge_qloo'],
        })
    usage={};failures=collections.defaultdict(list)
    for c in cs:
        ident=c['id'];b=read(OLD/'predictions'/f'{ident}.json');o=recovered_output(b)
        values[ident]['glm_log_odds']=o['log_odds'] if o else 0.
        usage[ident]={'base':token_count(b)}
        if include_judges:
            for mode in ['agent','qloo']:
                path=PRIVATE/'judges'/version/mode/f'{ident}.json'
                assert path.exists(),f'Missing prespecified development call: {path.name}'
                record=read(path);j=recovered_output(record)
                # A failed extra call retains the initial opinion; no retry
                # merely because the output was wrong or malformed.
                values[ident]['judge_'+mode]=j['log_odds'] if j else values[ident]['glm_log_odds']
                usage[ident][mode]=token_count(record)
                if not j:failures[mode].append(ident)
    models={};trials=[];oofs={}
    for name,names in families.items():
        X=np.array([[values[c['id']][n] for n in names] for c in cs])
        p,m,inner=nested_family(X,y,fold)
        r={'name':name,'features':names,'log_loss':loss(y,p),
           'correct':int(((p>=.5)==y).sum()),'total':len(cs),
           'inner_selected_ridges':inner,'final_ridge':m['ridge'],
           'uses_qloo':name.startswith('qloo'),'uses_judge':'judge' in name}
        trials.append(r);models[name]={'features':names,'model':m,'selection':r}
        oofs[name]={c['id']:float(v) for c,v in zip(cs,p)}
    key=lambda r:(r['log_loss'],-r['correct'],len(r['features']),r['name'])
    best_a=min([r for r in trials if not r['uses_qloo']],key=key)
    best_q=min([r for r in trials if r['uses_qloo']],key=key)
    config={'selected_agent':best_a['name'],'selected_qloo':best_q['name'],
            'judge_version':version if include_judges else None,'models':models,
            'selection':'nested ridge CV, family by outer log loss; development only',
            'fallback':'failed extra judge uses initial GLM margin; unavailable Qloo uses selected no-Qloo model'}
    report={'n':len(cs),'trials':trials,'selected_agent':best_a,'selected_qloo':best_q,
            'failures':dict(failures),'cost':{}}
    for mode in ['agent','qloo'] if include_judges else []:
        report['cost'][mode]={'aggregate_ratio':1+sum(u[mode] for u in usage.values())/sum(u['base'] for u in usage.values()),
            'max_ratio':max((u['base']+u[mode])/u['base'] for u in usage.values() if u['base']),
            'count_over_3x':sum((u['base']+u[mode])>3*u['base'] for u in usage.values()),
            'count_over_10x':sum((u['base']+u[mode])>10*u['base'] for u in usage.values())}
    save(ROOT/'agent_config.json',config);save(ROOT/'research/development_selection.json',report)
    save(PRIVATE/'development_features_final.json',values);save(PRIVATE/'development_oof_final.json',oofs)
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    import sys
    run(include_judges='--numeric-only' not in sys.argv)
