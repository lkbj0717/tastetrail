"""Versioned research utilities; predictions never receive held-out ratings."""
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
V11 = ROOT.parent / 'culture_v11'
for p in [V11, V11 / 'src']:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
PRIVATE = Path.home() / '.local/share/tastetrail/culture_v13'
OLD = PRIVATE.parent / 'culture_v11'
SEED = 20261028


def read(path):
    return json.loads(Path(path).read_text())


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2))
    tmp.replace(path)
    if PRIVATE in path.parents:
        path.chmod(0o600)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_ranking(text):
    """Ranking is the answer; inconsistent probability only affects confidence.

    Applied identically to baseline and both agent conditions. Original records
    remain immutable. No preferred item or hidden rating is accepted here.
    """
    t = text.strip()
    if t.startswith('```'):
        t = t.split('\n', 1)[1].rsplit('```', 1)[0].strip()
    result = json.loads(t)
    ranking = result.get('ranking')
    if ranking not in [['C01', 'C02'], ['C02', 'C01']]:
        raise ValueError('Expected exact two-candidate permutation')
    p = result.get('probability_c01')
    flags = []
    if isinstance(p, bool) or not isinstance(p, (int, float)) or not 1 <= p <= 99:
        p = 51 if ranking[0] == 'C01' else 49
        flags.append('missing_or_invalid_confidence_weak_fallback')
    magnitude = max(.01, abs(math.log(p / (100-p))))
    if (p > 50 and ranking[0] == 'C02') or (p < 50 and ranking[0] == 'C01'):
        flags.append('probability_direction_normalized_to_ranking')
    margin = magnitude if ranking[0] == 'C01' else -magnitude
    explanation = result.get('explanation', result.get('evidence', ''))
    return {'ranking': ranking, 'probability_c01': 100/(1+math.exp(-margin)),
            'log_odds': margin, 'explanation': str(explanation)[:2000],
            'parser_flags': flags}


def recovered_output(record):
    if record.get('output'):
        return parse_ranking(json.dumps(record['output']))
    for call in reversed(record.get('calls', [])):
        text = call.get('response_message', {}).get('content')
        if text:
            try:
                return parse_ranking(text)
            except (ValueError, TypeError, KeyError):
                continue
    return None


def token_count(record):
    m = record.get('metrics', {})
    return int(m.get('input_tokens', 0) + m.get('output_tokens', 0))


def usage_complete(record):
    if not record or record.get('usage_is_upper_bound'):return False
    calls=record.get('calls',[])
    if not calls:return record.get('physical_requests')==0 and token_count(record)==0
    return all(isinstance(c.get('usage',{}).get('prompt_tokens'),(int,float)) and
               isinstance(c.get('usage',{}).get('completion_tokens'),(int,float)) for c in calls)


def usage_upper_bound(record):
    if usage_complete(record):return token_count(record)
    value=record.get('accounted_tokens')
    if isinstance(value,(int,float)) and value>=token_count(record):return int(value)
    calls=record.get('calls',[]);bound=record.get('prompt_utf8_upper_bound')
    if not calls or not isinstance(bound,(int,float)):return None
    upper=0
    for call in calls:
        usage=call.get('usage',{})
        if all(isinstance(usage.get(k),(int,float)) for k in ['prompt_tokens','completion_tokens']):
            upper+=usage['prompt_tokens']+usage['completion_tokens']
        else:
            cap=call.get('max_tokens',record.get('max_completion_tokens'))
            if not isinstance(cap,(int,float)) or cap<0:return None
            upper+=bound+cap
    return int(max(upper,token_count(record)))


def dev_cases():
    return [c for c in read(OLD/'cases.json') if c['split'] == 'dev']
