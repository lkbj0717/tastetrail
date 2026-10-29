import unittest
from inference_v13 import predict


class DecisionTests(unittest.TestCase):
    def setUp(self):
        self.config={'selected_agent':'a','selected_qloo':'q','models':{
            'a':{'features':['glm_log_odds','community_margin'],
                 'model':{'scale':[1,1],'weights':[1,1]}},
            'q':{'features':['glm_log_odds','community_margin','qgroup_signed'],
                 'model':{'scale':[1,1,1],'weights':[1,1,2]}}}}
        self.initial={'output':{'ranking':['C01','C02'],'probability_c01':60}}

    def test_missing_qloo_has_exact_strong_control_fallback(self):
        r=predict({'community_margin':.2,'qgroup_signed':-1},self.initial,self.config,
                  qloo_available=False)
        self.assertTrue(r['qloo_fallback'])
        self.assertEqual(r['probabilities']['agent_qloo'],r['probabilities']['agent'])

    def test_candidate_swap_reverses_final_choice_and_probability(self):
        a=predict({'community_margin':.2,'qgroup_signed':-1},self.initial,self.config)
        other={'output':{'ranking':['C02','C01'],'probability_c01':40}}
        b=predict({'community_margin':-.2,'qgroup_signed':1},other,self.config)
        for key in a['probabilities']:
            self.assertAlmostEqual(a['probabilities'][key]+b['probabilities'][key],1)
            self.assertNotEqual(a['choices'][key],b['choices'][key])

    def test_missing_group_cannot_be_replaced_with_fabricated_association(self):
        r=predict({'community_margin':.2},self.initial,self.config,groups_available=False)
        self.assertTrue(r['qloo_fallback'])

    def test_failed_initial_can_use_local_tools_but_is_not_reported_as_llm_success(self):
        r=predict({'community_margin':.2,'qgroup_signed':-1},{'error':'HTTP503'},self.config)
        self.assertFalse(r['initial_valid'])
        self.assertIsNone(r['choices']['glm'])
        self.assertIsNotNone(r['choices']['agent'])

    def test_failed_qloo_does_not_borrow_paid_control_review(self):
        self.config['models']['a']['features']=['glm_log_odds','hier_agent']
        h={'agent':{'output':{'ranking':['C02','C01'],'probability_c01':1}}}
        r=predict({'community_margin':.2},self.initial,self.config,
                  hierarchical=h,qloo_available=False)
        self.assertNotEqual(r['choices']['agent_qloo'],r['choices']['agent'])
        self.assertEqual(r['choices']['agent_qloo'],r['choices']['glm'])
        self.assertEqual(r['qloo_fallback_policy'],'degraded_agent_initial_review_no_extra_call')


if __name__=='__main__':unittest.main()
