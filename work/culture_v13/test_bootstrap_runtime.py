"""Deployment rebuild tests use synthetic background data and mocked HTTP only."""
import hashlib
import json
from pathlib import Path
import uuid

import httpx
import numpy as np
import pytest

import bootstrap_runtime as runtime
from audit_features import CulturalMetadataTools, FIELDS
from tastetrail.qloo_http import QlooClient


def catalog(n):
    return {str(i): str(uuid.UUID(int=i)) for i in range(1, n + 1)}


def entity(qid):
    i = uuid.UUID(qid).int
    return {"entity_id": qid, "name": f"Public Movie {i}",
            "properties": {"release_year": 1990, "audience_identity": f"culture group {i % 3}",
                           "collaborators": [f"Public Artist {i % 4}"], "imdb_rating": 9.9,
                           "user_id": "excluded", "affinity": 777},
            "query": {"affinity": .98}, "user_rating": 5,
            "tags": [{"tag_id": f"urn:tag:genre:qloo:group{i % 3}", "name": "Drama",
                      "type": "urn:tag:genre", "query": {"affinity": .9}}]}


def cache_document(directory, mapping):
    directory.mkdir(parents=True)
    (directory / "public.json").write_text(json.dumps({
        "fetched_at": 123, "data": {"results": {"entities": [entity(q) for q in mapping.values()]}}}))


def synthetic_old(directory, n=12):
    directory.mkdir(parents=True)
    mapping = catalog(n)
    (directory / "catalog.json").write_text(json.dumps(mapping))
    (directory / "metadata.json").write_text(json.dumps({mid: {"title": "Public", "year": 1990,
                                                               "genres": ["Drama"]} for mid in mapping}))
    rng = np.random.default_rng(71)
    raw = rng.integers(0, 6, size=(3640, n)).astype(float)
    raw[:, 0], raw[:, 1] = 4, 2
    np.savez_compressed(directory / "tools.npz", ids=np.arange(1, n + 1), raw=raw,
                        mask=raw > 0, user_means=np.full(3640, 3.0), item_bias=np.zeros(n),
                        mf_32_10=rng.normal(size=(n, 32)))
    return mapping


def test_readiness_missing_is_read_only_and_never_creates_directories(tmp_path, monkeypatch):
    monkeypatch.setattr(QlooClient, "__init__", lambda *a, **kw: pytest.fail("Read-only check opened a client"))
    report = runtime.check_runtime(tmp_path / "old", tmp_path / "private", [1, 2])
    assert not report["ready"] and report["read_only"] and report["network_requests"] == 0
    assert list(tmp_path.iterdir()) == []


def test_exact_public_uuid_queries_exclude_self_and_drop_scores(tmp_path):
    mapping = catalog(57)  # Requires bounded chunks and a second seed for self.
    requests = []

    def mocked(request):
        assert request.url.host == "hackathon.api.qloo.com"
        assert request.url.path == "/v2/insights"
        params = dict(request.url.params)
        assert set(params) == {"filter.type", "signal.interests.entities", "filter.results.entities", "take"}
        assert params["filter.type"] == "urn:entity:movie"
        targets = params["filter.results.entities"].split(",")
        assert 1 <= len(targets) <= 50
        assert int(params["take"]) == len(targets)
        assert params["signal.interests.entities"] not in targets
        assert params["signal.interests.entities"] in list(mapping.values())[:2]
        assert set(targets) <= set(mapping.values())
        requests.append(params)
        return httpx.Response(200, json={"results": {"entities": [entity(q) for q in targets]}})

    client = QlooClient(key="mock-only-not-a-real-key", cache_dir=tmp_path / "cache",
                        transport=httpx.MockTransport(mocked))
    document, stats = runtime.collect_metadata(mapping, tmp_path / "cache", client)
    runtime.validate_metadata(document, mapping)
    assert document["movie_count"] == 57 and not stats["missing_movie_ids"]
    assert len(requests) == stats["logical_metadata_queries"] == 3
    for row in document["movies"].values():
        assert set(row["properties"]) <= set(FIELDS)
        assert set(row) == {"entity_id", "name", "release_year", "properties", "tags", "source_cache_digest", "fetched_at"}
        assert set(row["tags"][0]) == {"id", "name", "type"}
    assert "imdb_rating" not in json.dumps(document)
    assert "user_rating" not in json.dumps(document)


def test_query_rejects_profile_text_duplicates_and_self():
    qids = list(catalog(2).values())
    for seed, target in [(qids[0], [qids[0]]), (qids[0], ["my email@example.com"]),
                         (qids[0], [qids[1], qids[1]]), ("private-user", [qids[1]])]:
        with pytest.raises(ValueError):
            runtime.public_query(seed, target)


def test_qloo_uppercase_uuid_representation_is_valid():
    mapping = {"1": "745F8813-3667-4D09-B210-BC085AF583E4",
               "2": "5CD8A2EA-A92E-4E2D-A5EB-9467A6720C29"}
    runtime.validate_catalog(mapping, [1, 2])
    params = runtime.public_query(mapping["1"], [mapping["2"]])
    assert params["signal.interests.entities"] == mapping["1"]
    with pytest.raises(ValueError):
        runtime.public_query(mapping["1"], [mapping["1"].lower()])


def test_cache_metadata_reusable_without_network_and_missing_is_explicit(tmp_path):
    mapping = catalog(3)
    cache_document(tmp_path / "cache", {k: v for k, v in mapping.items() if k != "3"})
    doc, summary = runtime.collect_metadata(mapping, tmp_path / "cache")
    assert summary["logical_metadata_queries"] == 0
    assert summary["missing_movie_ids"] == ["3"]
    with pytest.raises(ValueError, match="partial runtime"):
        runtime.validate_metadata(doc, mapping)


def test_search_only_cache_gets_insights_enrichment_when_network_is_opted_in(tmp_path):
    mapping = catalog(3)
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "search.json").write_text(json.dumps({"fetched_at": 123, "data": {"results": [
        {"entity_id": q, "name": "Public title", "properties": {"release_year": 1990}, "tags": []}
        for q in mapping.values()]}}))

    class MockClient:
        def get(self, endpoint, params):
            return {"results": {"entities": [entity(q) for q in params["filter.results.entities"].split(",")]}}

    result, stats = runtime.collect_metadata(mapping, cache, MockClient())
    assert stats["cached_movies"] == 3 and stats["logical_metadata_queries"] == 2
    assert stats["rich_description_movies"] == 3
    runtime.validate_metadata(result, mapping)


def test_473_basic_cached_movies_are_enriched_with_two_explicit_sparse_exceptions(tmp_path):
    mapping = catalog(473)
    cache = tmp_path / "cache"
    cache.mkdir()
    original_entities = []
    for qid in mapping.values():
        original_entities.append({"entity_id": qid, "name": "Public title",
                                  "properties": {"release_year": 1990, "genres": ["Drama"],
                                                 "duration": 100, "content_rating": "PG",
                                                 "release_country": "US"},
                                  "tags": [{"id": "urn:tag:genre:qloo:drama", "name": "Drama"}]})
    (cache / "search.json").write_text(json.dumps({"fetched_at": 123,
                                                  "data": {"results": original_entities}}))
    queries = []

    class MockClient:
        def get(self, endpoint, params):
            assert endpoint == "/v2/insights"
            targets = params["filter.results.entities"].split(",")
            assert len(targets) <= 50 and params["signal.interests.entities"] not in targets
            queries.append(targets)
            result = []
            for qid in targets:
                mid = uuid.UUID(qid).int
                # Most responses have one useful prose field but fewer total
                # properties than search: rich prose must still take precedence.
                props = {"style_description": "A quiet reflective style."} if mid <= 471 else {}
                result.append({"entity_id": qid, "name": "Public title", "properties": props, "tags": []})
            return {"results": {"entities": result}}

    document, stats = runtime.collect_metadata(mapping, cache, MockClient())
    runtime.validate_metadata(document, mapping)
    assert stats["cached_movies"] == 473
    assert stats["rich_description_movies"] == 471
    assert stats["basic_only_movie_ids"] == ["472", "473"]
    assert not stats["missing_movie_ids"]
    assert len(queries) == stats["logical_metadata_queries"] == 11
    assert set(qid for chunk in queries for qid in chunk) == set(mapping.values())
    for mid in ("1", "250", "471"):
        assert document["movies"][mid]["properties"]["style_description"]
    for mid in ("472", "473"):
        row = document["movies"][mid]
        assert row["properties"]["duration"] == 100
        assert row["tags"][0]["name"] == "Drama"
        assert row["source_cache_digest"] == "search"  # Empty reply never downgraded the base record.


def test_atomic_write_refuses_overwrite(tmp_path):
    target = tmp_path / "asset.npz"
    target.write_bytes(b"frozen")
    with pytest.raises(FileExistsError):
        runtime.write_new(target, b"different")
    assert target.read_bytes() == b"frozen"
    assert not list(tmp_path.glob(".runtime-*"))


def test_background_only_rebuild_six_weights_then_preserves_assets(tmp_path, monkeypatch):
    old, private, cache = tmp_path / "culture_v11", tmp_path / "culture_v13", tmp_path / "cache"
    mapping = synthetic_old(old)
    cache_document(cache, mapping)
    original_read = Path.read_text

    def read_guard(path, *args, **kwargs):
        assert path.name not in {"cases.json", "split_private.json", "features_dev.json"}
        assert "confirmation" not in path.parts and "predictions" not in path.parts
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_guard)
    monkeypatch.setattr(QlooClient, "__init__", lambda *a, **kw: pytest.fail("Offline build opened a client"))
    ids = sorted(map(int, mapping))
    report = runtime.build_runtime(old, private, cache, ids, with_graph=True)
    assert report["check"]["ready"] and len(report["mixtures"]["created"]) == 6
    assert len(report["mixtures"]["training"]) == 2
    assets = [private / "qloo_metadata.json", private / "qloo_graph_tools.npz",
              *sorted((private / "mixture_tools").glob("*.npz"))]
    hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in assets}
    assert all(p.stat().st_mode & 0o777 == 0o600 for p in assets)
    monkeypatch.setattr(runtime, "fit_mixture", lambda *a, **kw: pytest.fail("Existing model was retrained"))
    monkeypatch.setattr(runtime, "graph_arrays", lambda *a, **kw: pytest.fail("Existing graph was retrained"))
    second = runtime.build_runtime(old, private, cache, ids, with_graph=True)
    assert second["check"]["ready"] and second["mixtures"]["created"] == []
    assert len(second["mixtures"]["preserved"]) == 6
    assert hashes == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in assets}


def test_invalid_existing_metadata_is_not_overwritten(tmp_path):
    old, private = tmp_path / "old", tmp_path / "new"
    mapping = synthetic_old(old)
    private.mkdir()
    path = private / "qloo_metadata.json"
    path.write_text('{"movie_count":0,"movies":{}}')
    initial = path.read_bytes()
    with pytest.raises(ValueError, match="partial runtime"):
        runtime.build_runtime(old, private, tmp_path / "cache", sorted(map(int, mapping)))
    assert path.read_bytes() == initial
