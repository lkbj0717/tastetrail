import unittest
from common import parse_ranking, recovered_output


class RankingTests(unittest.TestCase):
    def test_direction_conflict_does_not_discard_valid_choice(self):
        r = parse_ranking('{"ranking":["C02","C01"],"probability_c01":82}')
        self.assertEqual(r['ranking'][0], 'C02')
        self.assertLess(r['log_odds'], 0)
        self.assertAlmostEqual(r['probability_c01'], 18)
        self.assertTrue(r['parser_flags'])

    def test_recovery_uses_actual_answer_not_error_or_label(self):
        r = recovered_output({'error':'old parser rejected', 'calls':[
            {'response_message':{'content':'{"ranking":["C02","C01"],"probability_c01":82}'}}]})
        self.assertEqual(r['ranking'], ['C02','C01'])
        self.assertIsNone(recovered_output({'error':'HTTP503','calls':[]}))

    def test_swapping_candidate_aliases_reverses_margin(self):
        a=parse_ranking('{"ranking":["C02","C01"],"probability_c01":82}')
        b=parse_ranking('{"ranking":["C01","C02"],"probability_c01":18}')
        self.assertAlmostEqual(a['log_odds'], -b['log_odds'])

    def test_invalid_confidence_is_weak_not_strong(self):
        r=parse_ranking('{"ranking":["C01","C02"],"probability_c01":"99"}')
        self.assertAlmostEqual(r['probability_c01'],51)
        with self.assertRaises(ValueError):
            parse_ranking('{"ranking":["C01","C01"]}')


if __name__=='__main__': unittest.main()
