"""Matched Qloo control construction for the final method evaluation.

This module preserves the registered feature mapping and development-only
recalibration functions. It does not search alternative method families.
"""
import copy
import numpy as np
from calibrate import BASE, AFFINITY, GRAPH, nested_family, loss

CF=['interaction_cf_ridge_1','interaction_cf_ridge_10']
MIXED=['interaction_mixed_ridge_1','interaction_mixed_ridge_10']
GROUP=['qgroup_weighted_contrast','qgroup_pooled_contrast']
MPLAIN=['bernoulli_k12_plain','bernoulli_k24_plain']
MCF=['bernoulli_k12_cf','bernoulli_k24_cf']
MQ=['bernoulli_k12_qloo','bernoulli_k24_qloo']


def matched_feature_names(names):
    """Remove Qloo information while preserving each available CF mechanism.

    This mapping is registered before confirmation. Unknown features fail
    closed, so adding a new cultural feature cannot silently contaminate the
    control. Shared feature order is preserved and duplicate columns removed.
    """
    keep=set(BASE+CF+MPLAIN+MCF+['judge_agent','hier_agent'])
    result=[];mapping=[]
    for source in names:
        if source=='hier_qloo':target='hier_agent'
        elif source=='judge_qloo':target='judge_agent'
        elif source in MIXED:target=source.replace('interaction_mixed_','interaction_cf_')
        elif source in MQ:target=source[:-5]+'_cf'
        elif source in AFFINITY+GRAPH or source.startswith(('qmeta_','qgraph_','qgroup_')):
            target=None
        elif source in keep:target=source
        else:raise ValueError('Unregistered matched-control feature: '+source)
        mapping.append({'source':source,'target':target,
                        'action':'remove' if target is None else 'keep' if target==source else 'replace'})
        if target is not None and target not in result:result.append(target)
    if not result:raise ValueError('Matched control must retain shared information')
    return result,mapping


def build_matched_control(source_name,models,oofs,case_ids,values,y,fold,costs):
    """Refit the selected mechanism without Qloo using development data only.

    Exact existing controls are reused, including their OOF and full-data fit.
    Otherwise the same nested ridge procedure is applied once to the mapped
    mechanism. This is not another family search and cannot change the three
    main methods. Callers must supply D/E development data, never confirmation.
    """
    names,mapping=matched_feature_names(models[source_name]['features'])
    matches=sorted(name for name,row in models.items()
                   if not row['selection']['uses_qloo'] and row['features']==names)
    reused=matches[0] if matches else None
    if reused:
        entry=copy.deepcopy(models[reused]);predictions=copy.deepcopy(oofs[reused])
        selection=entry['selection'];model=entry['model']
    else:
        X=np.array([[values[ident][name] for name in names] for ident in case_ids])
        p,model,inner=nested_family(X,y,fold)
        predictions={ident:float(pi) for ident,pi in zip(case_ids,p)}
        kind=('hier_agent' if 'hier_agent' in names else
              'short_agent' if 'judge_agent' in names else 'base')
        known=all(row[kind] is not None and row['base']>0 for row in costs.values())
        ratios=[row[kind]/row['base'] for row in costs.values()
                if row[kind] is not None and row['base']>0]
        selection={'correct':int(((p>=.5)==y).sum()),'total':len(case_ids),
                   'accuracy':float(((p>=.5)==y).mean()),'log_loss':loss(y,p),
                   'inner_ridges':inner,'uses_qloo':False,'uses_judge':kind.startswith('short'),
                   'uses_hierarchical':kind.startswith('hier'),
                   'aggregate_token_upper_ratio':sum(row[kind] for row in costs.values())/
                       sum(row['base'] for row in costs.values()) if known else None,
                   'maximum_token_upper_ratio':max(ratios) if known else None,
                   'budget_eligible':known and max(ratios)<20}
    selection.update(name='matched_no_qloo',features=list(names),
                     conditional_on_selected_qloo_family=True,confirmation_used=False)
    return {'name':'matched_no_qloo','features':names,'model':model,'selection':selection,
            'source_family':source_name,'source_features':list(models[source_name]['features']),
            'feature_mapping':mapping,'reused_from':reused,
            'calibration':'Same D/E-only nested ridge rule and fresh no-Qloo feature scales; exact matching Agent fits reused.',
            'development_status':'Exploratory: mechanism conditional on the development-selected Qloo family.'},predictions
