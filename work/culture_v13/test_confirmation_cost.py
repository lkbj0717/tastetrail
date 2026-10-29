import pytest
from evaluate_confirmation import branch_cost,complete_usage
from inference_v13 import qloo_review_evidence_available


def used(n=100):
    return {'metrics':{'input_tokens':n,'output_tokens':n},
            'calls':[{'usage':{'prompt_tokens':n,'completion_tokens':n}}]}


def test_unreported_failure_is_not_zero_upper_bound():
    bad={'metrics':{},'calls':[{'http_status':503}]}
    cost=branch_cost([],bad,{}, {},{},'agent')
    assert cost['tokens_upper_bound'] is None
    assert not cost['tokens_reported_complete']


def test_missing_plan_cannot_be_silently_charged_as_zero():
    with pytest.raises(ValueError,match='planner'):
        branch_cost(['hier_agent'],used(),{}, {'agent':used()},{},'agent')


def test_known_skipped_stage_costs_zero_but_empty_record_is_unknown():
    assert not complete_usage({})
    assert complete_usage({'physical_requests':0,'calls':[],'metrics':{},'accounted_tokens':0})


def test_partial_cultural_evidence_is_not_complete_hierarchical_evidence():
    row={'request_data':{'qloo_group_evidence':[{'source':'mock'}]}}
    assert not qloo_review_evidence_available(row,hierarchical=True)
    row['request_data']['qloo_cultural_evidence']={'source':'mock'}
    row['error']='model_parse_failure'
    assert qloo_review_evidence_available(row,hierarchical=True)


def test_short_confirmation_uses_one_attempt_and_charges_unknown_usage(tmp_path,monkeypatch):
    import json
    import compact_cards
    import run_confirmation as runner
    from hierarchical_agent import SingleAttemptPacedTransport
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    monkeypatch.setattr(runner,'PRIVATE',tmp_path/'private')
    (tmp_path/'agent_config.json').write_text(json.dumps({'judge_version':'cards2_paced'}))
    (tmp_path/'private/predictions').mkdir(parents=True)
    (tmp_path/'private/predictions/T01.json').write_text(json.dumps(used(500)))
    class Builder:
        def build(self,*args,**kwargs):return {'history':[],'candidates':[]}
    class Client:
        def __init__(self,*,transport,**kwargs):
            assert isinstance(transport,SingleAttemptPacedTransport)
            self.transport=transport
            self.calls=[]
        def complete(self,*args,**kwargs):
            self.transport.physical_requests=1
            self.calls=[{'http_status':503}]
            raise RuntimeError('mock unavailable response')
        def metrics(self):return {}
    monkeypatch.setattr(compact_cards,'EvidenceCardBuilder',Builder)
    monkeypatch.setattr(runner,'ModelClient',Client)
    result=runner.judge_one({'id':'T01','likes':[],'dislikes':[],'candidates':[]},'agent')
    assert result['physical_requests']==1
    assert result['usage_is_upper_bound']
    assert result['accounted_tokens']==result['prompt_utf8_upper_bound']+result['max_completion_tokens']
