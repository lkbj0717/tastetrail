"""Pure statistics for a long prediction table; no files or network are read.

Each row has case_id, method, difficulty, correct (bool or None), optional valid,
tokens_reported (nonnegative integer or None), tokens_reported_complete (bool),
and tokens_upper_bound (nonnegative integer or None). Tokens refer to the whole
production branch, INCLUDING its one-shot baseline and shared planner allocation.
A partial usage sum must set tokens_reported_complete=False. An absent usage is
unknown, never zero. For completely reported usage the exact sum is also a valid
upper bound when no separate bound is supplied.

summarize_predictions(rows, baseline_method='glm', expected_cases=None,
                      methods=None, comparisons=None)

expected_cases optionally provides [{case_id, difficulty}, ...] so even cases
missing from EVERY method remain in the full denominator. Missing/invalid answers
count as incorrect; missing cost remains unknown. All ratios use the same case's
completely reported one-shot baseline tokens as denominator.
"""
from __future__ import annotations

import itertools
import math
from collections import defaultdict
from typing import Iterable, Mapping, Sequence

import numpy as np

DEFAULT_SEED = 20261028
DEFAULT_BOOTSTRAP_SAMPLES = 10_000
_Z95 = 1.959963984540054


def wilson_interval(correct: int, total: int) -> list[float] | None:
    """Two-sided 95% Wilson interval, as proportions; undefined for n=0."""
    if isinstance(correct, bool) or isinstance(total, bool):
        raise ValueError('Counts must be integers, not booleans')
    if not isinstance(correct, int) or not isinstance(total, int) or not 0 <= correct <= total:
        raise ValueError('Require integer counts 0 <= correct <= total')
    if total == 0:
        return None
    p = correct / total
    z2 = _Z95 * _Z95
    denominator = 1 + z2 / total
    middle = (p + z2 / (2 * total)) / denominator
    half = _Z95 * math.sqrt(p * (1-p) / total + z2 / (4*total*total)) / denominator
    return [max(0., middle-half), min(1., middle+half)]


def exact_mcnemar(helped: int, hurt: int) -> float:
    """Exact two-sided binomial p-value on discordant correctness pairs."""
    if any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in (helped, hurt)):
        raise ValueError('Discordance counts must be nonnegative integers')
    n = helped + hurt
    if not n:
        return 1.
    # Python integer arithmetic keeps this exact until the final division.
    numerator = 2 * sum(math.comb(n, k) for k in range(min(helped, hurt) + 1))
    return min(1., numerator / (1 << n))


def paired_bootstrap_interval(differences: Sequence[int], *, seed: int = DEFAULT_SEED,
                              samples: int = DEFAULT_BOOTSTRAP_SAMPLES) -> list[float] | None:
    """Percentile 95% CI of mean paired correctness differences (B - A).

    Multinomial sampling of {-1, 0, 1} counts is distributionally identical to
    sampling the paired case rows with replacement. It avoids a samples-by-n
    array and is invariant to row ordering. It does not sample the two methods
    independently, which would lose pairing.
    """
    if isinstance(samples, bool) or not isinstance(samples, int) or samples <= 0:
        raise ValueError('samples must be a positive integer')
    d = list(differences)
    if any(isinstance(x, bool) or not isinstance(x, (int, np.integer)) or x not in (-1, 0, 1) for x in d):
        raise ValueError('Paired correctness differences must be -1, 0 or 1')
    if not d:
        return None
    n = len(d)
    probs = np.array([d.count(-1), d.count(0), d.count(1)], dtype=float) / n
    counts = np.random.default_rng(seed).multinomial(n, probs, size=samples)
    delta = (counts[:, 2] - counts[:, 0]) / n
    return [float(x) for x in np.quantile(delta, [.025, .975])]


def _nonnegative_int(value, name):
    if value is not None and (isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0):
        raise ValueError(f'{name} must be a nonnegative integer or None')
    return None if value is None else int(value)


def _prepare(rows: Iterable[Mapping], expected_cases, methods):
    catalog = {}
    if expected_cases is not None:
        for c in expected_cases:
            ident, difficulty = c['case_id'], c['difficulty']
            if not isinstance(ident, str) or not ident or not isinstance(difficulty, str) or not difficulty:
                raise ValueError('Expected case_id and difficulty must be nonempty strings')
            if ident in catalog:
                raise ValueError(f'Duplicate expected case: {ident}')
            catalog[ident] = difficulty
    table = {}
    discovered = set()
    for source in rows:
        r = dict(source)
        ident, method, difficulty = r['case_id'], r['method'], r['difficulty']
        if any(not isinstance(x, str) or not x for x in (ident, method, difficulty)):
            raise ValueError('case_id, method and difficulty must be nonempty strings')
        if expected_cases is not None and ident not in catalog:
            raise ValueError(f'Unexpected case: {ident}')
        if ident in catalog and catalog[ident] != difficulty:
            raise ValueError(f'Conflicting difficulty for case {ident}')
        catalog[ident] = difficulty
        key = (ident, method)
        if key in table:
            raise ValueError(f'Duplicate prediction row: {key}')
        correct = r.get('correct')
        if correct is not None and not isinstance(correct, bool):
            raise ValueError('correct must be bool or None; do not pass probabilities')
        valid = r.get('valid', correct is not None)
        if not isinstance(valid, bool) or (valid and correct is None):
            raise ValueError('A valid answer needs a known boolean correctness')
        if not valid and correct is True:
            raise ValueError('An invalid answer cannot be marked correct')
        reported = _nonnegative_int(r.get('tokens_reported'), 'tokens_reported')
        complete = r.get('tokens_reported_complete', reported is not None)
        if not isinstance(complete, bool) or (complete and reported is None):
            raise ValueError('Complete usage requires a reported token count')
        upper = _nonnegative_int(r.get('tokens_upper_bound'), 'tokens_upper_bound')
        if upper is None and complete:
            upper = reported
        if upper is not None and reported is not None and upper < reported:
            raise ValueError('Upper token bound cannot be below known reported tokens')
        r.update(correct=bool(correct and valid), valid=valid,
                 tokens_reported=reported, tokens_reported_complete=complete,
                 tokens_upper_bound=upper)
        table[key] = r
        discovered.add(method)
    if methods is None:
        names = sorted(discovered)
    else:
        names = list(methods)
        if len(names) != len(set(names)) or any(not isinstance(x, str) or not x for x in names):
            raise ValueError('methods must be unique nonempty strings')
        if not discovered <= set(names):
            raise ValueError('Prediction table contains a method excluded from methods')
    return catalog, table, names


def _accuracy(case_ids, table, method):
    rows = [table.get((i, method)) for i in case_ids]
    n = len(rows)
    correct = sum(bool(r and r['correct']) for r in rows)
    return {'correct': correct, 'total': n, 'accuracy': correct/n if n else None,
            'wilson_95': wilson_interval(correct, n),
            'valid': sum(bool(r and r['valid']) for r in rows),
            'missing_rows': sum(r is None for r in rows),
            'invalid_answers': sum(bool(r is not None and not r['valid']) for r in rows)}


def _ratio_fields(pairs):
    # Pairs passed here have complete numerator/baseline counts and positive
    # per-case denominator; no missing response silently contributes zero.
    if not pairs:
        return {'aggregate_ratio': None, 'max_ratio': None,
                'count_over_3x': 0, 'count_over_10x': 0, 'count_at_least_20x': 0}
    ratios = [a/b for a,b in pairs]
    return {'aggregate_ratio': sum(a for a,b in pairs)/sum(b for a,b in pairs),
            'max_ratio': max(ratios),
            'count_over_3x': sum(a > 3*b for a,b in pairs),
            'count_over_10x': sum(a > 10*b for a,b in pairs),
            'count_at_least_20x': sum(a >= 20*b for a,b in pairs)}


def _token_summary(ids, table, method, baseline_method, upper=False):
    costs, baselines, all_pairs = [], [], []
    observed_sum = 0
    for ident in ids:
        r, b = table.get((ident, method)), table.get((ident, baseline_method))
        if r and r['tokens_reported'] is not None:
            observed_sum += r['tokens_reported']
        if upper:
            a = r['tokens_upper_bound'] if r else None
        else:
            a = r['tokens_reported'] if r and r['tokens_reported_complete'] else None
        bt = b['tokens_reported'] if b and b['tokens_reported_complete'] else None
        costs.append(a); baselines.append(bt)
        if a is not None and bt is not None and bt > 0:
            all_pairs.append((a, bt))
    n = len(ids)
    all_costs_known = n > 0 and all(x is not None for x in costs)
    all_baselines_known = n > 0 and all(x is not None for x in baselines)
    aggregate_ready = all_costs_known and all_baselines_known and sum(baselines) > 0
    individual_ready = n > 0 and len(all_pairs) == n
    paired = _ratio_fields(all_pairs)
    return {
        'kind': 'conservative_upper_bound' if upper else 'completely_reported',
        'total_cases': n,
        'known_numerator_cases': sum(x is not None for x in costs),
        'unknown_numerator_cases': sum(x is None for x in costs),
        'known_baseline_cases': sum(x is not None for x in baselines),
        'zero_baseline_cases': sum(x == 0 for x in baselines if x is not None),
        'total_tokens': sum(costs) if all_costs_known else None,
        'sum_known_token_counts': sum(x for x in costs if x is not None),
        'observed_reported_tokens_including_partial': observed_sum,
        'full_baseline_reported_tokens': sum(baselines) if all_baselines_known else None,
        'aggregate_ratio': sum(costs)/sum(baselines) if aggregate_ready else None,
        'max_ratio': paired['max_ratio'] if individual_ready else None,
        'count_over_3x': paired['count_over_3x'] if individual_ready else None,
        'count_over_10x': paired['count_over_10x'] if individual_ready else None,
        'count_at_least_20x': paired['count_at_least_20x'] if individual_ready else None,
        'known_pairs_only': {'n': len(all_pairs), **paired},
        'interpretation': ('Upper bounds include reserved allowances for unknown usage; not actual billable cost.'
                           if upper else 'Incomplete usage is unknown. Known-pairs statistics do not represent all cases.')
    }


def summarize_predictions(rows: Iterable[Mapping], *, baseline_method: str = 'glm',
                          expected_cases: Iterable[Mapping] | None = None,
                          methods: Sequence[str] | None = None,
                          comparisons: Sequence[tuple[str, str]] | None = None,
                          seed: int = DEFAULT_SEED,
                          bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES) -> dict:
    """Produce JSON-serializable full-denominator metrics and paired comparisons.

    comparisons contains (before_method, after_method); helped means after was
    correct and before was wrong. Default: all unordered method pairs in the
    supplied methods order (or alphabetical order). No p-value adjustment for
    multiple pairs is silently applied; select primary contrasts beforehand.
    """
    catalog, table, names = _prepare(rows, expected_cases, methods)
    if baseline_method not in names:
        raise ValueError('baseline_method must be included in methods/prediction rows')
    ids = sorted(catalog)
    grouped = defaultdict(list)
    for ident in ids:
        grouped[catalog[ident]].append(ident)
    report = {'total_cases': len(ids), 'methods': {}, 'comparisons': [],
              'confidence_level': .95, 'bootstrap_seed': seed,
              'bootstrap_samples': bootstrap_samples,
              'cost_baseline_method': baseline_method,
              'full_denominator_policy': 'missing and invalid answers are incorrect; unknown token usage is not zero',
              'inference_limitations': ['Wilson/paired bootstrap assume case-level independent users.',
                                       'No multiple-comparison correction; primary comparisons must be prespecified.']}
    for method in names:
        report['methods'][method] = {
            **_accuracy(ids, table, method),
            'by_difficulty': {g: _accuracy(ci, table, method) for g,ci in sorted(grouped.items())},
            'tokens': {'reported': _token_summary(ids, table, method, baseline_method),
                       'upper_bound': _token_summary(ids, table, method, baseline_method, upper=True)}}
    pairs = list(comparisons) if comparisons is not None else list(itertools.combinations(names, 2))
    for before, after in pairs:
        if before not in names or after not in names or before == after:
            raise ValueError('Comparisons require two distinct existing methods')
        differences = [int(bool(table.get((i,after),{}).get('correct',False))) -
                       int(bool(table.get((i,before),{}).get('correct',False))) for i in ids]
        helped, hurt = differences.count(1), differences.count(-1)
        interval = paired_bootstrap_interval(differences, seed=seed, samples=bootstrap_samples)
        report['comparisons'].append({
            'before': before, 'after': after, 'total': len(ids),
            'helped': helped, 'hurt': hurt, 'unchanged': len(ids)-helped-hurt,
            'accuracy_difference': (helped-hurt)/len(ids) if ids else None,
            'difference_percentage_points': 100*(helped-hurt)/len(ids) if ids else None,
            'exact_mcnemar_p_two_sided': exact_mcnemar(helped,hurt),
            'paired_bootstrap_95': interval,
            'paired_bootstrap_95_percentage_points': [100*x for x in interval] if interval is not None else None})
    return report
