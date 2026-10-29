"""Frozen numerical decision: no dataset IDs, labels or network calls."""
import math
from common import recovered_output
from tastetrail.fusion import probability


def qloo_review_evidence_available(record,*,hierarchical=False):
    data=record.get('request_data',{})
    cultural=bool(data.get('qloo_cultural_evidence'))
    return cultural and (bool(data.get('qloo_group_evidence')) if hierarchical else True)


def predict(features,initial,config,judges=None,*,hierarchical=None,qloo_available=True,groups_available=True):
    values=dict(features);judges=judges or {}
    first=recovered_output(initial)
    values['glm_log_odds']=first['log_odds'] if first else 0.
    fallbacks={}
    for mode in ['agent','qloo']:
        output=recovered_output(judges.get(mode,{}))
        values['judge_'+mode]=output['log_odds'] if output else values['glm_log_odds']
        fallbacks[mode]=output is None or bool(judges.get(mode,{}).get('error'))
    hierarchy_fallbacks={}
    for mode in ['agent','qloo']:
        output=recovered_output((hierarchical or {}).get(mode,{}))
        values['hier_'+mode]=output['log_odds'] if output else values['glm_log_odds']
        hierarchy_fallbacks[mode]=output is None or bool((hierarchical or {}).get(mode,{}).get('error'))
    def score(name,overrides=None):
        model=config['models'][name]
        source={**values,**(overrides or {})}
        x=[source[n] for n in model['features']]
        if not all(math.isfinite(float(v)) for v in x):raise ValueError('Non-finite tool feature')
        return float(probability(model['model'],x))
    agent=score(config['selected_agent'])
    model=config['models'][config['selected_qloo']]
    needs_groups=any(n.startswith('qgroup_') for n in model['features'])
    fallback=not qloo_available or (needs_groups and not groups_available)
    # Production fallback cannot borrow the independently evaluated control's
    # extra model calls. Retain local tools and initial opinion, making no new
    # paid call after a failed Qloo attempt.
    qloo=score(config['selected_agent'],{'judge_agent':values['glm_log_odds'],
               'hier_agent':values['glm_log_odds']}) if fallback else score(config['selected_qloo'])
    probabilities={'glm':first['probability_c01']/100 if first else None,
                   'agent':agent,'agent_qloo':qloo}
    choices={k:(None if v is None else (0 if v>=.5 else 1)) for k,v in probabilities.items()}
    if first:choices['glm']=0 if first['ranking'][0]=='C01' else 1
    return {'choices':choices,'probabilities':probabilities,'qloo_fallback':fallback,
            'qloo_fallback_policy':'degraded_agent_initial_review_no_extra_call' if fallback else None,
            'judge_fallbacks':fallbacks,'hierarchical_fallbacks':hierarchy_fallbacks,
            'initial_valid':first is not None}
