"""Numerical parity/lazy-source tests use a single D development input only."""
import json
from pathlib import Path

import numpy as np
import pytest

from feature_tools_v13 import (FeatureTools, COMMUNITY, AFFINITY, METADATA, GRAPH,
                               INTERACTION_CF, INTERACTION_MIXED, MIXTURE, GROUPS, requires_qloo)
from common import OLD, PRIVATE
from tastetrail.cluster_tools import TasteTools


@pytest.fixture(scope="module")
def development_input():
    # The group cache holds public visible input, not target-user hidden ratings.
    path = PRIVATE / "grouped_qloo/cases/D001.json"
    if not path.exists():
        pytest.skip("Local private development fixtures are not installed")
    group = json.loads(path.read_text())
    initial = json.loads((OLD / "predictions/D001.json").read_text())
    evidence = json.loads((OLD / "evidence/D001.json").read_text())
    with np.load(OLD / "tools.npz") as stored:
        community = TasteTools({k: stored[k] for k in stored.files})
    return group["history"], group["candidates"], initial, evidence, group, community


def test_constructor_and_unknown_features_perform_no_io(monkeypatch):
    def prohibited(*a, **k):
        raise AssertionError("Unexpected file read")
    monkeypatch.setattr(np, "load", prohibited)
    monkeypatch.setattr(Path, "read_text", prohibited)
    tool = FeatureTools()
    with pytest.raises(ValueError, match="Unsupported"):
        tool.build([], [], {}, required_features={"not_a_real_feature"})


def test_cf_path_never_reads_qloo_data(development_input, monkeypatch):
    history, candidates, initial, evidence, group, community = development_input
    real_load, real_read = np.load, Path.read_text
    accessed = []

    def guarded_load(path, *a, **k):
        accessed.append(str(path))
        assert "qloo" not in Path(path).name
        assert "qloo_metadata" not in str(path)
        return real_load(path, *a, **k)

    def guarded_read(path, *a, **k):
        assert "qloo" not in str(path)
        accessed.append(str(path))
        return real_read(path, *a, **k)

    monkeypatch.setattr(np, "load", guarded_load)
    monkeypatch.setattr(Path, "read_text", guarded_read)
    tool = FeatureTools(community=community)
    names = {"mf_32_10", "community_margin", "bernoulli_k12_cf", "bernoulli_k24_plain"} | INTERACTION_CF
    first = tool.build(history, candidates, initial, required_features=names)
    # Supplied Qloo data cannot accidentally influence a no-Qloo branch.
    second = tool.build(history, candidates, initial, evidence=evidence, required_features=names)
    for name in names:
        assert first["values"][name] == second["values"][name]
    assert tool._culture is None and tool._graph is None
    assert not first["trace"]["feature_sources"]["uses_qloo"]
    assert len(accessed) == 2  # precisely two selected mixture files


def test_original_featurebuilder_exact_parity(development_input):
    from numeric_features import FeatureBuilder
    history, candidates, initial, evidence, group, community = development_input
    expected = FeatureBuilder().build(history, candidates, initial, evidence)
    names = set(expected["values"])
    actual = FeatureTools(community=community).build(history, candidates, initial, evidence,
                                                   required_features=names)
    assert actual["initial_valid"] == expected["initial_valid"]
    assert actual["affinity_valid"] == expected["affinity_valid"]
    for name in names:
        assert actual["values"][name] == expected["values"][name], name


def test_interaction_and_mixture_and_groups_reuse_original_formulas(development_input):
    from interaction_tools import InteractionTools
    from mixture_tools import MixtureTools
    from grouped_qloo import features_from_rows
    history, candidates, initial, evidence, group, community = development_input
    names = INTERACTION_CF | INTERACTION_MIXED | MIXTURE | GROUPS
    actual = FeatureTools(community=community).build(history, candidates, initial, evidence,
                                                   required_features=names, group_record=group)["values"]
    expected = InteractionTools.from_existing().evaluate(history, candidates)
    for name in MIXTURE:
        with np.load(PRIVATE / "mixture_tools" / (name + ".npz")) as data:
            tool = MixtureTools(data["ids"], data["probability"], data["prior"])
            expected[name] = tool.evaluate(history, candidates)[0]
    expected.update(features_from_rows(history, candidates, group, evidence["rows"]))
    for name in names:
        assert actual[name] == expected[name], name


def test_candidate_swap_all_selected_signed_features(development_input):
    history, candidates, initial, evidence, group, community = development_input
    names = ((COMMUNITY - {"cohort_entropy"}) | AFFINITY | METADATA | GRAPH
             | INTERACTION_CF | INTERACTION_MIXED | MIXTURE | GROUPS)
    original = initial["output"]
    flipped = {"output": {**original, "ranking": ["C02" if x == "C01" else "C01" for x in original["ranking"]],
                           "probability_c01": 100 - original["probability_c01"]}}
    # Explicit source order prevents a column mismatch when candidates reverse.
    evidence = {**evidence, "candidates": candidates}
    tool = FeatureTools(community=community)
    a = tool.build(history, candidates, initial, evidence, names, group)
    b = tool.build(history, candidates[::-1], flipped, evidence, names, group)
    for name in names:
        assert a["values"][name] == pytest.approx(-b["values"][name], abs=1e-10), name


def test_missing_affinity_groups_and_external_judge_features(development_input):
    history, candidates, initial, evidence, group, community = development_input
    names = {"signed_history_log_ratio", "mf_weighted_qloo", "qgroup_pooled_contrast",
             "judge_agent", "hier_qloo"}
    result = FeatureTools(community=community).build(history, candidates, initial, required_features=names)
    assert not result["affinity_valid"]
    assert not result["trace"]["groups"]["valid"]
    assert result["values"]["qgroup_pooled_contrast"] == 0.
    assert result["values"]["signed_history_log_ratio"] == 0.
    assert result["values"]["mf_weighted_qloo"] == 0.
    assert not result["trace"]["source_availability"]["required_tools_valid"]
    assert "judge_agent" not in result["values"] and "hier_qloo" not in result["values"]
    assert requires_qloo({"qmeta_curated_ridge_0.1"})
    assert requires_qloo({"bernoulli_k12_qloo"})
    assert not requires_qloo({"interaction_cf_ridge_1", "bernoulli_k12_cf"})


def test_metadata_only_success_does_not_require_affinity_response(development_input):
    history, candidates, initial, evidence, group, community = development_input
    result = FeatureTools(community=community).build(
        history, candidates, initial,
        required_features={"qmeta_curated_ridge_0.1", "bernoulli_k12_qloo"})
    assert not result["affinity_valid"]
    assert result["trace"]["source_availability"]["required_tools_valid"]
    assert result["trace"]["source_availability"]["metadata"]


def test_legacy_rows_use_supplied_candidate_order_for_group_features(development_input):
    history, candidates, initial, evidence, group, community = development_input
    tool = FeatureTools(community=community)
    a = tool.build(history, candidates, initial, evidence, GROUPS, group)
    reverse_evidence = {**evidence, "rows": [row[::-1] for row in evidence["rows"]]}
    b = tool.build(history, candidates[::-1], initial, reverse_evidence, GROUPS, group)
    for name in GROUPS:
        assert a["values"][name] == pytest.approx(-b["values"][name], abs=1e-12)
