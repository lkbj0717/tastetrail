import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from evidence_cards import EvidenceCardBuilder, JUDGE_SYSTEM_PROMPT


HISTORY = [(1, 5), (2, 4), (3, 5), (4, 4), (5, 4), (6, 1), (7, 2), (8, 2)]


@pytest.fixture
def builder(tmp_path):
    public = {str(i): {"title": f"Public movie {i}", "year": 1990 + i,
                       "genres": ["Drama"], "hidden_rating": 5}
              for i in range(1, 11)}
    (tmp_path / "metadata.json").write_text(json.dumps(public))
    text = " ".join(["Sparse melancholy recollections challenge the familiar story"] * 50)
    movies = {str(i): {"properties": {"plot_themes_description": text,
              "style_description": text, "emotional_tone_description": text},
              "preferred": True} for i in range(1, 11)}
    (tmp_path / "qloo_metadata.json").write_text(json.dumps({"movies": movies}))
    k = np.eye(10)
    # Discrimination times rating strength: liked 3,1; disliked 6,8.
    for j, (a, b) in enumerate([(0.3, 0.1), (0.9, 0.6), (0.8, 0.1),
                               (0.1, 0.1), (0.1, 0.2), (0.1, 0.8),
                               (0.3, 0.4), (0.9, 0.6)]):
        k[j, 8], k[j, 9] = a, b
    tools = SimpleNamespace(index={i: i - 1 for i in range(1, 11)}, kernels={"curated": k})
    return EvidenceCardBuilder(tmp_path, culture_tools=tools)


def test_candidate_swap_keeps_same_history_and_swaps_evidence(builder):
    left = builder.build(HISTORY, [9, 10])
    right = builder.build(HISTORY, [10, 9])
    assert left["history"] == right["history"]
    lq, rq = left["qloo_cultural_evidence"], right["qloo_cultural_evidence"]
    assert [x["history_id"] for x in lq["history"]] == ["H01", "H03", "H06", "H08"]
    assert [x["history_id"] for x in lq["history"]] == [x["history_id"] for x in rq["history"]]
    for a, b in zip(lq["history"], rq["history"]):
        assert a["similarity"]["C01"] == b["similarity"]["C02"]
        assert a["similarity"]["C02"] == b["similarity"]["C01"]
    assert left["candidates"][0]["title"] == right["candidates"][1]["title"]


def test_control_contains_same_basic_inputs_without_loading_qloo(builder, monkeypatch):
    monkeypatch.setattr(builder, "_load_qloo", lambda: pytest.fail("control loaded Qloo"))
    payload = builder.build(HISTORY, [9, 10], include_qloo=False)
    assert len(payload["history"]) == 8
    assert "qloo_cultural_evidence" not in payload
    assert "hidden_rating" not in json.dumps(payload)
    assert "preferred" not in json.dumps(payload)
    assert builder.messages(HISTORY, [9, 10], include_qloo=False)[0]["content"] == JUDGE_SYSTEM_PROMPT


def test_compact_cultural_block_has_no_baseline_or_labels(builder):
    payload = builder.build(HISTORY, [9, 10])
    evidence = payload["qloo_cultural_evidence"]
    assert len(json.dumps(evidence, ensure_ascii=False)) < 4200
    text = json.dumps(payload)
    for prohibited in ("hidden_rating", "preferred", "baseline", "probability_c01"):
        assert prohibited not in text
    for item in evidence["candidates"] + evidence["history"]:
        for field in ("theme", "style", "tone"):
            assert len(item[field]) <= 109


def test_identical_similarity_does_not_add_arbitrary_history(builder):
    builder._tools.kernels["curated"][:, 9] = builder._tools.kernels["curated"][:, 8]
    payload = builder.build(HISTORY, [9, 10])
    assert payload["qloo_cultural_evidence"]["history"] == []


def test_missing_descriptions_are_unknown_and_candidate_schemas_match(builder):
    builder._load_qloo()
    builder._qloo["9"]["properties"] = {}
    payload = builder.build(HISTORY, [9, 10])
    a, b = payload["qloo_cultural_evidence"]["candidates"]
    assert a.keys() == b.keys()
    assert a["theme"] is None and a["style"] is None and a["tone"] is None


def test_rejects_whole_case_objects_and_duplicate_candidates(builder):
    with pytest.raises(ValueError):
        builder.build({"history": HISTORY, "preferred": 9}, [9, 10])
    with pytest.raises(ValueError):
        builder.build(HISTORY, [9, 9])
    with pytest.raises(ValueError):
        builder.build(HISTORY, [1, 10])
