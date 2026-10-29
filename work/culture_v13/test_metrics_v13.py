"""Synthetic tests only: never read target cases, labels, saved results or APIs."""
import json
import unittest

from metrics_v13 import (wilson_interval, exact_mcnemar, paired_bootstrap_interval,
                        summarize_predictions)


def row(i, m, correct, tokens=100, **extra):
    return {'case_id':str(i), 'method':m, 'difficulty':'hard' if i % 2 else 'easy',
            'correct':correct, 'tokens_reported':tokens, **extra}


class StatisticalMetricsTests(unittest.TestCase):
    def test_wilson_known_endpoints_and_center(self):
        self.assertAlmostEqual(wilson_interval(0,100)[1], .03699349820698568)
        self.assertAlmostEqual(wilson_interval(100,100)[0], .9630065017930143)
        lo,hi = wilson_interval(50,100)
        self.assertAlmostEqual(lo,1-hi)
        self.assertAlmostEqual(lo,.4038315303659957)
        self.assertIsNone(wilson_interval(0,0))
        with self.assertRaises(ValueError): wilson_interval(2,1)

    def test_exact_discordance_test(self):
        self.assertEqual(exact_mcnemar(0,0),1)
        self.assertEqual(exact_mcnemar(4,4),1)
        self.assertEqual(exact_mcnemar(10,0),2/1024)
        self.assertEqual(exact_mcnemar(9,1),22/1024)
        self.assertEqual(exact_mcnemar(1,9),22/1024)
        with self.assertRaises(ValueError): exact_mcnemar(-1,0)

    def test_paired_bootstrap_determinism_and_pairing(self):
        d=[1]*8+[-1]*2+[0]*10
        self.assertEqual(paired_bootstrap_interval(d),paired_bootstrap_interval(list(reversed(d))))
        self.assertEqual(paired_bootstrap_interval([0]*12),[0.,0.])
        self.assertEqual(paired_bootstrap_interval([1]*12),[1.,1.])
        self.assertIsNone(paired_bootstrap_interval([]))
        ci=paired_bootstrap_interval(d)
        self.assertLessEqual(ci[0],.3)
        self.assertGreaterEqual(ci[1],.3)

    def test_full_denominator_missing_answers_and_missing_entire_case(self):
        data=[row(0,'glm',True),row(0,'qloo',True,200),row(1,'glm',False),
              row(1,'qloo',None,None,valid=False)]
        cases=[{'case_id':str(i),'difficulty':'hard' if i%2 else 'easy'} for i in range(3)]
        r=summarize_predictions(data,expected_cases=cases,methods=['glm','qloo'],comparisons=[('glm','qloo')])
        self.assertEqual(r['methods']['qloo']['total'],3)
        self.assertEqual(r['methods']['qloo']['correct'],1)
        self.assertEqual(r['methods']['qloo']['missing_rows'],1)
        self.assertEqual(r['methods']['qloo']['invalid_answers'],1)
        self.assertEqual(r['methods']['qloo']['by_difficulty']['easy']['total'],2)
        self.assertEqual(r['comparisons'][0]['unchanged'],3)
        json.dumps(r,allow_nan=False)

    def test_paired_help_hurt_and_percentage_points(self):
        before=[True,False,False,True,False]
        after =[False,True,True,True,False]
        data=[row(i,m,c) for m,x in [('glm',before),('qloo',after)] for i,c in enumerate(x)]
        p=summarize_predictions(data,methods=['glm','qloo'])['comparisons'][0]
        self.assertEqual((p['helped'],p['hurt'],p['unchanged']),(2,1,2))
        self.assertEqual(p['difference_percentage_points'],20)
        self.assertEqual(p['exact_mcnemar_p_two_sided'],1)

    def test_unknown_usage_does_not_become_free(self):
        data=[row(0,'glm',True,100),row(1,'glm',True,100),
              row(0,'qloo',True,200),
              row(1,'qloo',False,150,tokens_reported_complete=False,tokens_upper_bound=1800)]
        t=summarize_predictions(data)['methods']['qloo']['tokens']
        self.assertIsNone(t['reported']['aggregate_ratio'])
        self.assertIsNone(t['reported']['max_ratio'])
        self.assertIsNone(t['reported']['count_over_3x'])
        self.assertEqual(t['reported']['known_pairs_only']['aggregate_ratio'],2)
        self.assertEqual(t['reported']['observed_reported_tokens_including_partial'],350)
        self.assertEqual(t['reported']['unknown_numerator_cases'],1)
        self.assertEqual(t['upper_bound']['aggregate_ratio'],10)
        self.assertEqual(t['upper_bound']['max_ratio'],18)
        self.assertEqual(t['upper_bound']['count_over_10x'],1)

    def test_thresholds_strict_3_10_and_inclusive_20(self):
        amounts=[300,301,1000,1001,2000]
        data=[row(i,'glm',True,100) for i in range(5)]+[row(i,'qloo',True,x) for i,x in enumerate(amounts)]
        t=summarize_predictions(data)['methods']['qloo']['tokens']['reported']
        self.assertEqual(t['aggregate_ratio'],sum(amounts)/500)
        self.assertEqual(t['max_ratio'],20)
        self.assertEqual(t['count_over_3x'],4)
        self.assertEqual(t['count_over_10x'],2)
        self.assertEqual(t['count_at_least_20x'],1)

    def test_unknown_or_zero_baseline_not_divided(self):
        for baseline in [None,0]:
            t=summarize_predictions([row(0,'glm',False,baseline),row(0,'qloo',True,100)])['methods']['qloo']['tokens']['reported']
            self.assertIsNone(t['aggregate_ratio'])
            self.assertIsNone(t['max_ratio'])
            self.assertEqual(t['known_pairs_only']['n'],0)

    def test_reject_inconsistent_or_duplicate_rows(self):
        badsets=[
            [row(0,'glm',True),row(0,'glm',False)],
            [row(0,'glm',True),row(0,'qloo',True,difficulty='different')],
            [row(0,'glm',True,valid=False)],
            [row(0,'glm',True,tokens_upper_bound=99)],
            [row(0,'glm',True,None,tokens_reported_complete=True)],
            [row(0,'glm',True,-1)],
        ]
        for rows in badsets:
            with self.assertRaises(ValueError): summarize_predictions(rows)

    def test_all_missing_method_keeps_denominator_and_unknown_cost(self):
        r=summarize_predictions([row(0,'glm',True)],methods=['glm','qloo'])
        self.assertEqual(r['methods']['qloo']['accuracy'],0)
        self.assertEqual(r['methods']['qloo']['missing_rows'],1)
        self.assertIsNone(r['methods']['qloo']['tokens']['upper_bound']['aggregate_ratio'])


if __name__=='__main__':unittest.main()
