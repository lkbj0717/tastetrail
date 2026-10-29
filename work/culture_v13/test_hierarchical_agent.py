import copy
import json
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest

import hierarchical_agent as ha
from evidence_cards import EvidenceCardBuilder


H = [(1, 5), (2, 4), (3, 5), (4, 4), (5, 4), (6, 1), (7, 2), (8, 2)]
C = [9, 10]
PLAN = {
    "hypotheses": [{"claim": "May favor restrained crime stories.", "support": ["H01", "H03"],
                    "counterevidence": ["H07"], "unknown": "Ratings do not reveal a specific reason."}],
    "liked_groups": [["H01", "H03"], ["H02", "H04", "H05"]],
    "disliked_group": ["H06", "H07", "H08"],
}
ANSWER = {"ranking": ["C01", "C02"], "probability_c01": 65,
          "explanation": "H01 and H03 support the restrained style; H07 is meaningful contrary evidence."}


class Community:
    def evaluate(self, history, candidates, qrows=None):
        assert qrows is None
        return {"community_margin": .1, "mf_32_10": -.2, "neighbors_100": .3}, {}, {}


class Culture:
    def build(self, history, candidates, include_qloo=True):
        assert include_qloo
        return {"qloo_cultural_evidence": {"source": "mock Qloo", "tags": {"C01": ["Restrained"], "C02": ["Kinetic"]}}}


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr("tastetrail.model_client.settings", lambda: {"GLM_API_KEY": "unit-test-not-real", "GLM_BASE_URL": "https://unit.test/v1"})
    # Operational retry delays are not useful in transport-only unit tests.
    monkeypatch.setattr("tastetrail.model_client.time.sleep", lambda _: None)
    metadata = {str(i): {"title": f"Public film {i}", "year": 1990 + i, "genres": ["Drama"]} for i in range(1, 11)}
    builder = EvidenceCardBuilder(tmp_path, movie_metadata=metadata)
    initial = {"metrics": {"input_tokens": 600, "output_tokens": 400},
               "output": {"ranking": ["C02", "C01"], "probability_c01": 20,
                          "explanation": "BASELINE_SECRET_ANCHOR"}, "preferred": "HIDDEN_SENTINEL"}
    return tmp_path, builder, initial


def model_transport(objects, captured=None, status=200, usage=True):
    queue = iter(objects)
    captured = captured if captured is not None else []
    def factory():
        def handle(request):
            payload = json.loads(request.content)
            captured.append(payload)
            content = next(queue)
            response = {"model": "mock-glm", "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": content if isinstance(content, str) else json.dumps(content)}}]}
            if usage:
                response["usage"] = {"prompt_tokens": 250, "completion_tokens": 200}
            return httpx.Response(status, json=response)
        return ha.SingleAttemptPacedTransport(inner=httpx.MockTransport(handle), interval=0)
    return factory


def test_planner_schema_rejects_bad_ids_groups_and_extra_answers():
    assert ha.validate_plan(copy.deepcopy(PLAN)) == PLAN
    for mutation in [lambda x: x["liked_groups"][0].append("H06"),
                     lambda x: x["disliked_group"].append("H01"),
                     lambda x: x["hypotheses"][0]["support"].append("H99"),
                     lambda x: x.update(ranking=["C01", "C02"]),
                     lambda x: x["liked_groups"][1].append("H01")]:
        bad = copy.deepcopy(PLAN); mutation(bad)
        with pytest.raises(ValueError): ha.validate_plan(bad)


def test_shared_plan_no_initial_anchor_and_no_qloo_control(setup):
    root, builder, initial = setup
    captured = []
    plan = ha.run_plan(H, C, "TEST01", 1000, private_path=root, builder=builder,
                       transport_factory=model_transport([PLAN], captured))
    assert not plan["fallback"]
    again = ha.run_plan(H, C, "TEST01", 1000, private_path=root, builder=builder,
                        transport_factory=lambda: pytest.fail("Shared planner was called twice"))
    assert plan == again
    agent = ha.run_review(H, C, "TEST01", "agent", plan, initial, private_path=root, builder=builder,
                          community_tools=Community(), transport_factory=model_transport([ANSWER], captured),
                          qloo_factory=lambda: pytest.fail("No-Qloo branch accessed Qloo"),
                          culture_builder=object())
    assert not agent["fallback"] and agent["output"]["ranking"] == ANSWER["ranking"]
    assert agent["output"]["explanation"] == ANSWER["explanation"]
    assert agent["shared_planner_tokens"] == plan["accounted_tokens"] == 450
    assert agent["production_reported_tokens"] == 1900
    sent = json.dumps(captured)
    assert "BASELINE_SECRET_ANCHOR" not in sent and "HIDDEN_SENTINEL" not in sent
    assert "qloo_group_evidence" not in agent["request_data"]
    assert "qloo_cultural_evidence" not in agent["request_data"]
    assert all(p["max_tokens"] <= 8192 for p in captured)
    assert all("reasoning_effort" not in p for p in captured)
    assert "do not rank candidates" in json.loads(captured[0]["messages"][1]["content"])["task"]


def test_qloo_only_sends_public_uuids_and_both_branches_charge_shared_plan(setup):
    root, builder, initial = setup
    plan = ha.run_plan(H, C, "TEST02", 1000, private_path=root, builder=builder,
                       transport_factory=model_transport([PLAN]))
    params_seen = []
    class Qloo:
        def __init__(self): self.calls = []
        def get(self, endpoint, params):
            assert endpoint == "/v2/insights";params_seen.append(params)
            self.calls.append({"endpoint": endpoint, "cache_hit": False})
            wanted = params["filter.results.entities"].split(",")
            return {"results": {"entities": [{"entity_id": u, "query": {"affinity": s}} for u, s in zip(wanted, [.6, .3])]}}
    catalog = {str(i): str(uuid.UUID(int=i)).upper() for i in range(1, 11)}
    out = ha.run_review(H, C, "TEST02", "qloo", plan, initial, private_path=root, builder=builder,
                        community_tools=Community(), culture_builder=Culture(), catalog=catalog,
                        qloo_factory=Qloo, transport_factory=model_transport([ANSWER]))
    assert not out["fallback"] and len(params_seen) == 3
    assert out["shared_planner_tokens"] == 450 and out["production_token_ratio_reported"] == 1.9
    assert "qloo_cultural_evidence" in out["request_data"]
    assert out["request_data"]["community_tools"]["matrix_factorization_margin"] == -.2
    for p in params_seen:
        assert set(p) == {"filter.type", "signal.interests.entities", "filter.results.entities", "take"}
        for key in ("signal.interests.entities", "filter.results.entities"):
            for item in p[key].split(","): uuid.UUID(item)
        assert "rating" not in json.dumps(p) and "TEST02" not in json.dumps(p)


def test_plan_failure_fixed_groups_and_review_failure_reverts_to_initial(setup):
    root, builder, initial = setup
    plan = ha.run_plan(H, C, "TEST03", 1000, private_path=root, builder=builder,
                       transport_factory=model_transport([{"bad_schema": True}]))
    assert plan["fallback"] and plan["output"] == ha.fallback_plan(H)
    reversed_candidates = ha.run_plan(H, C[::-1], "TEST03_REVERSED", 1000,
        private_path=root, builder=builder, transport_factory=model_transport([{"bad_schema": True}]))
    assert plan["output"] == reversed_candidates["output"]
    review = ha.run_review(H, C, "TEST03", "agent", plan, initial, private_path=root, builder=builder,
                           community_tools=Community(), transport_factory=model_transport(["not json"]))
    assert review["fallback"] and review["fallback_source"] == "initial_glm"
    assert review["output"]["ranking"] == ["C02", "C01"]
    assert review["reported_tokens"] == 450  # Failed parsing is still charged.
    assert review["production_token_ratio_upper_bound"] <= 18


def test_budget_skip_no_request_and_unknown_failure_conservatively_charged(setup):
    root, builder, initial = setup
    low = ha.run_plan(H, C, "LOW", 10, private_path=root, builder=builder,
                      transport_factory=lambda: pytest.fail("Budget guard did not skip"))
    assert low["error"] == "budget_preflight_skip" and low["physical_requests"] == 0
    captured=[]
    failed = ha.run_plan(H, C, "HTTPFAIL", 1000, private_path=root, builder=builder,
                         transport_factory=model_transport([{}], captured, status=503, usage=False))
    assert len(captured) == 1 and failed["physical_requests"] == 1
    assert failed["usage_is_upper_bound"] and failed["accounted_tokens"] > 8192
    assert failed["accounted_tokens"] <= 17000
    review=ha.run_review(H,C,"HTTPFAIL","agent",failed,initial,private_path=root,builder=builder,
                         community_tools=Community(),transport_factory=model_transport([ANSWER]))
    assert review["production_accounted_tokens"] <= 18000
    assert review["usage_is_upper_bound"]


def test_concurrent_model_stages_are_limited_to_two(setup):
    _, _, _ = setup
    counts={"active":0,"maximum":0};lock=threading.Lock();wait=threading.Event()
    def factory():
        def handle(request):
            with lock:
                counts["active"]+=1;counts["maximum"]=max(counts["maximum"],counts["active"])
            wait.wait(.03)
            with lock:counts["active"]-=1
            return httpx.Response(200,json={"choices":[{"finish_reason":"stop","message":{"content":"{}"}}],
                    "usage":{"prompt_tokens":10,"completion_tokens":10}})
        return ha.SingleAttemptPacedTransport(inner=httpx.MockTransport(handle),interval=0)
    with ThreadPoolExecutor(max_workers=6) as pool:
        results=list(pool.map(lambda _:ha._model_stage("x",{},2000,transport_factory=factory),range(6)))
    assert counts["maximum"]==2 and all(r["physical_requests"]==1 for r in results)


def test_review_uses_same_ranking_first_policy_as_baseline():
    bad=copy.deepcopy(ANSWER);bad["probability_c01"]=20
    out=ha.validate_review(bad)
    assert out['ranking']==ANSWER['ranking'] and out['probability_c01']>50
    assert 'probability_direction_normalized_to_ranking' in out['parser_flags']
    bad=copy.deepcopy(ANSWER);bad["explanation"]="word "*101
    assert len(ha.validate_review(bad)['explanation'].split())==100
    bad=copy.deepcopy(ANSWER);bad['probability_c01_note']=None
    assert ha.validate_review(bad)['ranking']==ANSWER['ranking']


def test_group_object_compatibility_does_not_guess_or_change_ids():
    p=copy.deepcopy(PLAN)
    p['liked_groups']=[{'ids':g,'label':'comment'} for g in p['liked_groups']]
    p['disliked_group']={'ids':p['disliked_group'],'theme':'comment'}
    assert ha.validate_plan(p)==PLAN
    p['liked_groups'][0]['ids'].append('H08')
    with pytest.raises(ValueError):ha.validate_plan(p)


def test_group_contrast_cancels_candidate_general_affinity_and_reverses():
    rows=[{'seed_history_ids':g,'candidate_affinity':{'C01':.6,'C02':.3}}
          for g in PLAN['liked_groups']+[PLAN['disliked_group']]]
    assert ha.group_contrast(H,rows)['signed_log_affinity_contrast']==0
    rows[0]['candidate_affinity']['C01']=.9
    a=ha.group_contrast(H,rows)['signed_log_affinity_contrast']
    for r in rows:r['candidate_affinity']={'C01':r['candidate_affinity']['C02'],'C02':r['candidate_affinity']['C01']}
    assert ha.group_contrast(H,rows)['signed_log_affinity_contrast']==-a


def test_review_refuses_modified_plan_and_token_accounting(setup):
    root,builder,initial=setup
    plan=ha.run_plan(H,C,"TAMPER",1000,private_path=root,builder=builder,
        transport_factory=model_transport([PLAN]))
    def no_request(): pytest.fail("Invalid plan reached model transport")
    bad=copy.deepcopy(plan);bad["output"]["hypotheses"][0]["claim"]="A changed hypothesis"
    with pytest.raises(ValueError,match="digest"):
        ha.run_review(H,C,"TAMPER","agent",bad,initial,private_path=root,builder=builder,
            community_tools=Community(),transport_factory=no_request)
    bad=copy.deepcopy(plan);bad["accounted_tokens"]=-1
    with pytest.raises(ValueError,match="accounting"):
        ha.run_review(H,C,"TAMPER","agent",bad,initial,private_path=root,builder=builder,
            community_tools=Community(),transport_factory=no_request)
