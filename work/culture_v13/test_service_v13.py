"""Service integration tests: synthetic inputs, injected tools, no paid calls."""
import copy
import json

import numpy as np
import pytest

import service_v13 as svc


META = {str(i): {"title": f"Public film {i}", "year": 1990 + i, "genres": ["Drama"]}
        for i in range(1, 11)}
H = [(1, 5), (2, 4), (3, 5), (4, 4), (5, 4), (6, 1), (7, 2), (8, 2)]
C = [9, 10]
BODY = {"likes": [{**{k: META[str(i)][k] for k in ("title", "year")}, "rating": r} for i, r in H[:5]],
        "dislikes": [{**{k: META[str(i)][k] for k in ("title", "year")}, "rating": r} for i, r in H[5:]],
        "candidates": [{k: META[str(i)][k] for k in ("title", "year")} for i in C]}


def record(p=20, tokens=1000, explanation="BASELINE_ANCHOR_NOT_FOR_JUDGES"):
    return {"output": {"ranking": ["C01", "C02"] if p >= 50 else ["C02", "C01"],
                       "probability_c01": p, "explanation": explanation},
            "metrics": {"input_tokens": tokens // 2, "output_tokens": tokens - tokens // 2},
            "reported_tokens": tokens, "accounted_tokens": tokens, "physical_requests": 1,
            "calls": [{"http_status": 200, "usage": {"prompt_tokens": tokens // 2, "completion_tokens": tokens - tokens // 2}}], "error": None}


class Community:
    a = {"ids": np.arange(1, 11)}
    def evaluate(self, history, candidates, qrows=None):
        assert history == H and candidates == C and qrows is None
        return {"community_margin": .3, "mf_32_10": .7, "neighbors_100": .4, "cohort_entropy": .1}, {}, {"source": "MOCK_ONLY"}


def config(agent_features=None, qloo_features=None, **service):
    a = agent_features or ["glm_log_odds", "mf_32_10", "judge_agent"]
    q = qloo_features or ["glm_log_odds", "mf_32_10", "judge_qloo"]
    def model(features):
        return {"features": features, "model": {"scale": [1.] * len(features), "weights": [.2] * len(features)}}
    return {"selected_agent": "a", "selected_qloo": "q", "judge_version": "cards2_paced",
            "models": {"a": model(a), "q": model(q)}, "service": service}


@pytest.fixture(autouse=True)
def reject_network(monkeypatch):
    def fail(*args, **kwargs): pytest.fail("A service unit test attempted a real HTTP request")
    monkeypatch.setattr("httpx.Client.request", fail)


def test_catalog_validation_rejects_unknown_and_hidden_fields_before_calls(tmp_path):
    service = svc.RecommendationService(private_path=tmp_path, config=config(), metadata=META,
                                       baseline=lambda *x: pytest.fail("Invalid input reached API"))
    assert service.resolve(BODY) == (H, C)
    bad = copy.deepcopy(BODY); bad["preferred"] = 9
    with pytest.raises(ValueError): service.recommend(bad)
    bad = copy.deepcopy(BODY); bad["candidates"][0]["title"] = "Unknown"
    with pytest.raises(ValueError, match="Unsupported"): service.recommend(bad)
    bad = copy.deepcopy(BODY); bad["likes"][0]["rating"] = 2
    with pytest.raises(ValueError): service.recommend(bad)


def test_hierarchy_comparison_shares_initial_and_plan_and_accounts_once(tmp_path):
    calls = []
    plan = {"output": {"hypotheses": []}, "metrics": {"input_tokens": 100, "output_tokens": 200},
            "reported_tokens": 300, "accounted_tokens": 300, "physical_requests": 1, "calls": [{"usage": {"prompt_tokens": 100, "completion_tokens": 200}}], "fallback": False}
    def baseline(h, c, ident): calls.append("initial"); return record()
    def planner(h, c, ident, tokens):
        assert tokens == 1000 and h == H and c == C
        calls.append("planner"); return plan
    def reviewer(h, c, ident, mode, shared, initial):
        assert shared is plan and initial["output"]["ranking"][0] == "C02"
        calls.append(mode)
        row = record(60 if mode == "agent" else 75, 400 if mode == "agent" else 500, "A comparison from observed ratings.")
        if mode == "qloo":
            row["request_data"] = {"qloo_cultural_evidence": {"source": "MOCK_ONLY", "tags": {"C01": ["Restrained"]}},"qloo_group_evidence":[{"source":"MOCK_ONLY"}]}
            row["qloo_calls"] = [{"cache_hit": True}] * 3
        return row
    service = svc.RecommendationService(private_path=tmp_path, metadata=META,
        config=config(["glm_log_odds", "hier_agent"], ["glm_log_odds", "hier_qloo"]),
        community=Community(), baseline=baseline, planner=planner, hierarchy_review=reviewer,
        cards_review=lambda *x: pytest.fail("Short reviewer was unnecessarily called"))
    result = service.recommend(BODY)
    assert calls == ["initial", "planner", "agent", "qloo"]
    assert set(result["recommendations"]) == {"glm", "agent", "agent_qloo"}
    assert result["costs"]["agent"]["reported_tokens"] == 1700
    assert result["costs"]["agent_qloo"]["reported_tokens"] == 1800
    assert result["costs"]["comparison"]["reported_tokens"] == 2200
    assert result["costs"]["comparison"]["logical_qloo_queries"] == 3
    assert result["warnings"] and not result["configuration_frozen"]
    assert result["scope"]["city_itinerary_validated"] is False


def test_short_calls_have_no_anchor_and_qloo_payload_is_separate(tmp_path):
    class Culture:
        def build(self, h, c, include_qloo):
            assert include_qloo
            return {"task": "compare", "qloo_cultural_evidence": {"source": "MOCK_ONLY", "tags": {"C01": ["Quiet"]}}}
    service = svc.RecommendationService(private_path=tmp_path, config=config(), metadata=META,
                                       community=Community(), baseline=lambda *x: record(), culture_builder=Culture())
    seen = []
    def call(prompt, payload, **kwargs):
        seen.append(payload)
        assert "BASELINE_ANCHOR_NOT_FOR_JUDGES" not in json.dumps(payload)
        return {**record(60, 500, "Visible ratings support this choice."), "request_data": payload}
    service._call_model = call
    result = service.recommend(BODY)
    assert len(seen) == 2
    assert "qloo_cultural_evidence" not in seen[0]
    assert "qloo_cultural_evidence" in seen[1]
    assert result["costs"]["agent_qloo"]["token_ratio"] == 1.5
    result2 = service.recommend(BODY)
    assert len(seen) == 2 and result2["costs"]["agent_qloo"]["short_review_cache_hit"]


def test_numeric_qloo_uses_numeric_features_without_changing_control(tmp_path):
    seen = []
    class Features:
        def build(self, h, c, initial, evidence, **kwargs):
            seen.append(evidence)
            return {"values": {"community_margin": 999., "mf_32_10": 999., "signed_history_log_ratio": .4}}
    cfg = config(["glm_log_odds", "mf_32_10"], ["glm_log_odds", "mf_32_10", "signed_history_log_ratio"])
    service = svc.RecommendationService(private_path=tmp_path, config=cfg, metadata=META,
        community=Community(), feature_builder=Features(), baseline=lambda *x: record(),
        affinity=lambda *x: {"error": None, "rows": [[.4, .2]] * 9, "calls": [{"cache_hit": True}] * 9},
        cards_review=lambda *x: pytest.fail("No judge feature selected"))
    result = service.recommend(BODY)
    assert len(seen) == 1 and result["costs"]["comparison"]["logical_qloo_queries"] == 9
    assert result["community_evidence"]["margins"]["mf_32_10"] == .7
    assert result["costs"]["agent_qloo"]["reported_tokens"] == 1000


def test_hierarchy_failure_falls_back_without_unbounded_second_review(tmp_path):
    calls = []
    plan = {"output": {}, "reported_tokens": 300, "accounted_tokens": 300, "metrics": {"input_tokens": 100, "output_tokens": 200}, "physical_requests": 1, "calls": [{"usage": {"prompt_tokens": 100, "completion_tokens": 200}}], "fallback": False}
    def review(h, c, ident, mode, shared, initial):
        calls.append(mode)
        return {"error": "simulated failure", "metrics": {}, "calls": [], "accounted_tokens": 5000,
                "reported_tokens": 0, "usage_is_upper_bound": True}
    cfg = config(judge_backend="hierarchical", decision_backend="judge")
    service = svc.RecommendationService(private_path=tmp_path, config=cfg, metadata=META,
        community=Community(), baseline=lambda *x: record(), planner=lambda *x: plan, hierarchy_review=review)
    result = service.recommend(BODY, compare=False)
    assert calls == ["qloo"]
    assert result["qloo_fallback"] and result["recommendations"]["agent_qloo"]["candidate_id"] == "C02"
    assert result["costs"]["agent_qloo"]["accounted_tokens_upper_bound"] == 6300
    assert result["costs"]["agent_qloo"]["within_20x_budget"]
    assert "agent" not in result["recommendations"]


def test_rejects_mixed_short_hierarchical_branch_before_requests(tmp_path):
    with pytest.raises(ValueError, match="cannot combine"):
        svc.RecommendationService(private_path=tmp_path, metadata=META,
            config=config(qloo_features=["glm_log_odds", "judge_qloo", "hier_qloo"]))


def test_numeric_control_does_not_run_or_pay_for_qloo_planner(tmp_path):
    calls=[]
    plan={"output": {}, "metrics": {"input_tokens": 100, "output_tokens": 200},
          "reported_tokens": 300, "accounted_tokens": 300, "physical_requests": 1, "calls": [{"usage": {"prompt_tokens": 100, "completion_tokens": 200}}], "fallback": False}
    def review(h,c,ident,mode,shared,initial):
        calls.append(mode)
        return {**record(70,500,"The Qloo comparison."),
                "request_data": {"qloo_cultural_evidence": {"source": "MOCK_ONLY", "tags": {}},"qloo_group_evidence":[{"source":"MOCK_ONLY"}]}}
    service=svc.RecommendationService(private_path=tmp_path,metadata=META,
        config=config(["glm_log_odds","mf_32_10"],["glm_log_odds","hier_qloo"]),
        community=Community(),baseline=lambda *x:record(),planner=lambda *x:plan,
        hierarchy_review=review,cards_review=lambda *x:pytest.fail("Unselected short judge called"))
    result=service.recommend(BODY)
    assert calls==["qloo"]
    assert result["costs"]["agent"]["reported_tokens"]==1000
    assert result["costs"]["agent"]["shared_planner_tokens"]==0
    assert result["costs"]["agent_qloo"]["reported_tokens"]==1800


@pytest.mark.parametrize("evidence_returned", [True, False])
def test_failed_hierarchy_matches_shared_inference_without_control_borrowing(tmp_path, evidence_returned):
    from inference_v13 import predict
    plan={"output": {}, "metrics": {"input_tokens": 100, "output_tokens": 200},
          "reported_tokens": 300, "accounted_tokens": 300, "physical_requests": 1, "calls": [{"usage": {"prompt_tokens": 100, "completion_tokens": 200}}], "fallback": False}
    rows={"agent":record(99,400,"A strong control review."),
          "qloo":{**record(20,500,"The initial fallback."),"error":"simulated review failure"}}
    if evidence_returned:
        rows["qloo"]["request_data"]={"qloo_group_evidence":[{"direction":"liked","candidate_affinity":{"C01":.4,"C02":.3}}],"qloo_cultural_evidence":{"source":"MOCK_ONLY"}}
    cfg=config(["glm_log_odds","hier_agent"],["glm_log_odds","hier_qloo"])
    service=svc.RecommendationService(private_path=tmp_path,metadata=META,config=cfg,
        community=Community(),baseline=lambda *x:record(),planner=lambda *x:plan,
        hierarchy_review=lambda h,c,ident,mode,shared,initial:rows[mode])
    result=service.recommend(BODY)
    direct=predict({"glm_log_odds":0},record(),cfg,hierarchical=rows,qloo_available=evidence_returned)
    assert result["probabilities"]==direct["probabilities"]
    assert result["qloo_fallback"] is (not evidence_returned)
    assert result["recommendations"]["agent"]["candidate_id"]=="C01"
    assert result["recommendations"]["agent_qloo"]["candidate_id"]=="C02"
    assert result["fallbacks"]["hierarchical_reviews"]["qloo"]
    assert result["costs"]["agent_qloo"]["reported_tokens"]==1800


def test_cf_interaction_matches_existing_formula_without_culture(tmp_path):
    from interaction_tools import InteractionTools, normalized_gram
    community = Community()
    rng = np.random.default_rng(45)
    community.a = {"ids": np.arange(1, 11), "mf_32_10": rng.normal(size=(10, 4)), "item_bias": rng.normal(size=10)}
    community.idx = {i: i - 1 for i in range(1, 11)}
    service = svc.RecommendationService(private_path=tmp_path, config=config(), metadata=META, community=community)
    values, _ = service.local_values(H, C, record(), ["interaction_cf_ridge_1", "interaction_cf_ridge_10"])
    reference = InteractionTools(range(1, 11), normalized_gram(community.a["mf_32_10"]), np.eye(10), community.a["item_bias"]).evaluate(H, C)
    for name in ("interaction_cf_ridge_1", "interaction_cf_ridge_10"):
        assert values[name] == pytest.approx(reference[name], abs=1e-12)
    assert service.culture_builder is None and service.feature_builder is None


def test_baseline_cache_reuses_same_prompt_across_config_versions(tmp_path):
    service = svc.RecommendationService(private_path=tmp_path, config=config(), metadata=META)
    calls = []
    def call(prompt, payload, **kwargs):
        calls.append((prompt, payload, kwargs))
        return record()
    service._call_model = call
    first, cached = service._baseline(H, C, "FIRST")
    assert not cached
    second, cached = service._baseline(H, C, "SECOND")
    assert cached and first == second and len(calls) == 1
    assert calls[0][2]["max_tokens"] == 16384
    assert {c["id"] for c in calls[0][1]["candidates"]} == {"C01", "C02"}


def test_app_endpoint_integrates_service_response(tmp_path, monkeypatch):
    import app_v13
    service = svc.RecommendationService(private_path=tmp_path, config=config(), metadata=META,
        community=Community(), baseline=lambda *x: record(),
        cards_review=lambda h, c, ident, mode, initial: {**record(60, 400), "request_data": {}})
    monkeypatch.setattr(svc, "_SERVICE", service)
    result = app_v13.endpoint(app_v13.Request.model_validate(BODY), True)
    assert set(result["recommendations"]) == {"glm", "agent", "agent_qloo"}
    assert result["seconds"] >= 0 and result["selected_version"]


def test_invalid_initial_answer_keeps_glm_unavailable_but_numeric_agent_recovers(tmp_path):
    from inference_v13 import predict
    initial = record(tokens=1000)
    initial.pop('output')
    initial['error'] = 'invalid JSON, known model usage'
    cfg = config(['glm_log_odds','mf_32_10'], ['glm_log_odds','mf_32_10'])
    service = svc.RecommendationService(private_path=tmp_path, metadata=META, config=cfg,
        community=Community(), baseline=lambda *args:initial,
        planner=lambda *args:pytest.fail('Numeric model requested planner'),
        cards_review=lambda *args:pytest.fail('Numeric model requested reviewer'))
    result = service.recommend(BODY)
    reference = predict({'mf_32_10':.7},initial,cfg)
    assert result['probabilities'] == reference['probabilities']
    assert result['recommendations']['glm'] is None
    assert 'No valid model ranking' in result['explanations']['glm']
    assert result['recommendations']['agent']['candidate_id'] == 'C01'
    assert result['costs']['glm']['reported_tokens'] == 1000
    assert result['trace'][1]['status'] == 'failed'


@pytest.mark.parametrize('valid_ranking', [False,True])
def test_zero_or_unknown_initial_usage_skips_all_additional_models(tmp_path,valid_ranking):
    from inference_v13 import predict
    initial = {'error':'HTTP503 unknown usage','calls':[{'http_status':503}], 'metrics':{},
               'physical_requests':1,'usage_is_upper_bound':True,'accounted_tokens':18000}
    if valid_ranking:
        initial['output'] = record()['output']
    cfg = config(['glm_log_odds','mf_32_10','hier_agent'], ['glm_log_odds','mf_32_10','hier_qloo'])
    def forbidden(*args):pytest.fail('Unknown initial usage reached paid planner/reviewer')
    service = svc.RecommendationService(private_path=tmp_path, metadata=META, config=cfg,
        community=Community(), baseline=lambda *args:initial, planner=forbidden,
        hierarchy_review=forbidden,cards_review=forbidden)
    result = service.recommend(BODY)
    expected = predict({'mf_32_10':.7}, initial,cfg,qloo_available=False)
    assert result['probabilities'] == expected['probabilities']
    assert result['qloo_fallback']
    assert result['costs']['agent_qloo']['reported_tokens'] is None
    assert result['costs']['agent_qloo']['observed_reported_tokens'] == 0
    assert result['costs']['agent_qloo']['accounted_tokens_upper_bound'] == 18000
    assert result['costs']['agent_qloo']['token_ratio'] is None
    assert result['costs']['agent_qloo']['token_ratio_upper_bound'] is None
    assert result['costs']['agent_qloo']['within_20x_budget'] is None


def test_final_only_numeric_branch_does_not_run_control_only_planner(tmp_path):
    cfg = config(['glm_log_odds','mf_32_10','hier_agent'],['glm_log_odds','mf_32_10'])
    service = svc.RecommendationService(private_path=tmp_path,metadata=META,config=cfg,
        community=Community(),baseline=lambda *args:record(),
        planner=lambda *args:pytest.fail('Control-only planner was called for final-only request'),
        hierarchy_review=lambda *args:pytest.fail('Control-only review was called'))
    result = service.recommend(BODY,compare=False)
    assert 'agent' not in result['recommendations'] and 'agent' not in result['costs']
    assert result['costs']['agent_qloo']['reported_tokens'] == 1000
    assert result['costs']['comparison']['reported_tokens'] == 1000
    assert result['costs']['agent_qloo']['shared_planner_tokens'] == 0


def test_caught_review_exception_preserves_unknown_actual_and_bound(tmp_path):
    def failing(*args):raise OSError('simulated record storage failure after unobserved request')
    service = svc.RecommendationService(private_path=tmp_path,metadata=META,config=config(),
        community=Community(),baseline=lambda *args:record(),cards_review=failing)
    result = service.recommend(BODY,compare=False)
    cost = result['costs']['agent_qloo']
    assert cost['reported_tokens'] is None
    assert cost['accounted_tokens_upper_bound'] is None
    assert cost['observed_reported_tokens'] == 1000
    assert cost['token_ratio'] is None and cost['token_ratio_upper_bound'] is None
    assert cost['within_20x_budget'] is None
    assert result['costs']['comparison']['accounted_tokens_upper_bound'] is None
    assert result['qloo_fallback']


def test_bounded_unknown_review_matches_confirmation_accounting(tmp_path):
    from evaluate_confirmation import branch_cost
    initial=record()
    review={'error':'HTTP503','metrics':{},'calls':[{'http_status':503}],
            'physical_requests':1,'accounted_tokens':5000,'usage_is_upper_bound':True}
    service=svc.RecommendationService(private_path=tmp_path,metadata=META,config=config(),
        community=Community(),baseline=lambda *args:initial,cards_review=lambda *args:review)
    result=service.recommend(BODY,compare=False)
    offline=branch_cost(['judge_qloo'],initial,{'qloo':review},{},{},'qloo')
    online=result['costs']['agent_qloo']
    assert online['observed_reported_tokens']==offline['tokens_reported']==1000
    assert online['reported_tokens_complete']==offline['tokens_reported_complete']==False
    assert online['accounted_tokens_upper_bound']==offline['tokens_upper_bound']==6000
    assert online['reported_tokens'] is None and online['token_ratio'] is None
    assert online['token_ratio_upper_bound']==6


def test_empty_record_is_unknown_but_deliberate_skip_is_known_zero():
    unknown=svc._record_cost({})
    assert unknown['reported_tokens'] is None and unknown['accounted_tokens'] is None
    assert unknown['usage_unknown']
    skip=svc._record_cost(svc._not_called_record('budget_preflight_skip'))
    assert skip['reported_tokens']==skip['accounted_tokens']==0
    assert skip['reported_tokens_complete'] and not skip['usage_unknown']
