"""Read-only explanation of a completed decision; no APIs or decision changes.

Group associations are observations, not causes or personal probabilities.
The additive account reconstructs the deployed fusion, including fallback.
"""
from __future__ import annotations
import math
from common import recovered_output

LABELS = {
    'initial': 'Initial model judgment',
    'ratings': 'Patterns in background ratings',
    'cultural_groups': 'Qloo-informed taste groups',
    'review': 'Review of cultural support and counterevidence',
    'independent_review': 'Review without Qloo',
}


def _number(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def fusion_account(config, features, initial, judges, hierarchical, result):
    """Reconstruct a score already produced by inference_v13, never set a choice."""
    if not config.get('models') or result['choices'].get('agent_qloo') not in (0, 1):
        return {'available': False, 'reason': 'No numerical decision available.'}
    values = dict(features)
    first = recovered_output(initial)
    values['glm_log_odds'] = first['log_odds'] if first else 0.
    for mode in ('agent', 'qloo'):
        for prefix, records in [('judge_', judges or {}), ('hier_', hierarchical or {})]:
            output = recovered_output(records.get(mode, {}))
            values[prefix + mode] = output['log_odds'] if output else values['glm_log_odds']
    fallback = result.get('qloo_fallback') is True
    name = config['selected_agent' if fallback else 'selected_qloo']
    if fallback:
        values.update(judge_agent=values['glm_log_odds'], hier_agent=values['glm_log_odds'])
    entry = config['models'][name];model = entry['model'];buckets = {}
    if not (len(entry['features']) == len(model['weights']) == len(model['scale'])):
        return {'available': False, 'reason': 'Incomplete numerical account.'}
    for feature, weight, scale in zip(entry['features'], model['weights'], model['scale']):
        v = values.get(feature)
        if not all(_number(x) for x in (v, weight, scale)) or scale <= 0:
            return {'available': False, 'reason': 'Incomplete numerical account.'}
        if feature == 'glm_log_odds':group = 'initial'
        elif feature.endswith('_qloo') and feature.startswith('bernoulli_'):group = 'cultural_groups'
        elif feature in ('judge_qloo', 'hier_qloo'):group = 'review'
        elif feature in ('judge_agent', 'hier_agent'):group = 'independent_review'
        else:group = 'ratings'
        buckets[group] = buckets.get(group, 0.) + float(v * weight / scale)
    total = sum(buckets.values())
    reconstructed = 1 / (1 + math.exp(-max(-35., min(35., total))))
    expected = result.get('probabilities', {}).get('agent_qloo')
    if not _number(expected) or abs(reconstructed - expected) > 1e-10:
        return {'available': False, 'reason': 'Numerical account could not be verified.'}
    direction = 1 if result['choices']['agent_qloo'] == 0 else -1
    return {'available': True, 'model': name, 'verified': True,
            'score_for_first_candidate': total, 'reconstructed_score': reconstructed,
            'terms': [{'id': k, 'label': LABELS[k], 'toward_selected': direction * v}
                      for k, v in buckets.items()],
            'meaning': 'Additive contributions to the frozen model score, oriented toward the selected film. Not percentages, personal liking probabilities, or causal effects of Qloo.'}


def build_receipt(response, result, config, features, initial, judges=None, hierarchical=None):
    choice = result.get('choices', {}).get('agent_qloo')
    selected = f'C{choice + 1:02}' if choice in (0, 1) else None
    other = ('C02' if selected == 'C01' else 'C01') if selected else None
    fallback = result.get('qloo_fallback') is True
    recommendations = response.get('recommendations', {})
    control = recommendations.get('agent')
    control_id = control.get('candidate_id') if isinstance(control, dict) else None
    changed = (control_id != selected) if control_id in ('C01', 'C02') and selected and not fallback else None
    ledger = []
    groups = response.get('qloo_evidence', {}).get('qloo_group_evidence', [])
    for n, row in enumerate(groups if isinstance(groups, list) else []):
        if not isinstance(row, dict):continue
        direction = row.get('direction');seeds = row.get('seed_history_ids');affinity = row.get('candidate_affinity', {})
        if not isinstance(affinity, dict):affinity = {}
        valid_seeds = isinstance(seeds, list) and len(seeds) > 0 and len(set(str(s) for s in seeds)) == len(seeds)
        allowed = {'H01','H02','H03','H04','H05'} if direction == 'liked' else {'H06','H07','H08'} if direction == 'disliked' else set()
        valid_seeds = valid_seeds and all(isinstance(s, str) and s in allowed for s in seeds)
        valid = bool(selected and direction in ('liked', 'disliked') and valid_seeds and
                     all(_number(affinity.get(c)) and affinity[c] >= 0 for c in ('C01', 'C02')))
        item = {'id': f'Q{n+1:02}', 'source': 'Qloo group affinity', 'direction': direction if direction in ('liked','disliked') else 'unknown',
                'seed_history_ids': seeds if valid_seeds else [], 'available': valid,
                'candidate_affinity': {c: affinity.get(c) if _number(affinity.get(c)) and affinity[c] >= 0 else None for c in ('C01','C02')}}
        if valid:
            margin = affinity[selected] - affinity[other]
            if direction == 'disliked':margin = -margin
            item['interpretation'] = 'support' if margin > 1e-12 else 'caution' if margin < -1e-12 else 'neutral'
        else:item['interpretation'] = 'unavailable'
        ledger.append(item)
    support = sum(r['interpretation'] == 'support' for r in ledger)
    caution = sum(r['interpretation'] == 'caution' for r in ledger)
    valid_count = sum(r['available'] for r in ledger)
    conflict = bool(support and caution)
    if fallback:summary = 'Qloo was unavailable for the final decision. No cultural improvement is claimed.'
    elif not valid_count:summary = 'Group evidence is incomplete. Inspect the available cultural descriptions and execution record.'
    elif conflict:summary = 'Cultural signals are mixed: some support this choice and others caution against it.'
    elif caution:summary = 'The available group associations caution against this choice; other model signals outweighed them.'
    elif support:summary = 'Available group associations support this choice. They are not a guarantee of personal enjoyment.'
    else:summary = 'The available group associations do not distinguish the two films.'
    fusion = fusion_account(config, features, initial, judges, hierarchical, result)
    return {'schema_version': 1, 'selected_candidate': selected, 'agent_candidate': control_id,
            'qloo_used_for_decision': not fallback, 'comparison_available': changed is not None,
            'choice_changed': changed, 'summary': summary, 'supporting_groups': support,
            'caution_groups': caution, 'valid_groups': valid_count, 'mixed_evidence': conflict,
            'group_ledger': ledger, 'fusion': fusion,
            'limits': ['Association is not causation or an individual satisfaction probability.',
                       'Agent and Agent + Qloo use separately calibrated models. A choice difference is a method comparison, not proof of the causal contribution of one API call.',
                       'The receipt explains the completed prediction. It does not modify it or call a model.']}
