"""Rebuild v13 deployment assets from public entities + disjoint background data.

Default is a read-only readiness check. --build writes missing files only;
--allow-network separately opts in to public-UUID Qloo metadata requests.
This module never opens evaluation cases, labels, predictions, or API secrets
at import time. It does not refit the frozen decision/fusion configuration.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parent
V11 = ROOT.parent / "culture_v11"
sys.path.insert(0, str(V11 / "src"))
from audit_features import CulturalMetadataTools, FIELDS, TEXT_FIELDS
from graph_models import _center
from mixture_tools import CONFIGS, MixtureTools, SETTINGS, fit_mixture, neighbor_kernel, with_graph_prior

DEFAULT_PARENT = Path.home() / ".local/share/tastetrail"
BACKGROUND_USERS = 3640
MAX_TARGETS = 50


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, content):
    """Publish a complete mode-0600 file atomically, never replacing a target."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".runtime-", delete=False) as handle:
            temporary = Path(handle.name)
            os.chmod(temporary, 0o600)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)  # Atomic no-clobber publication, including races.
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_json_new(path, obj):
    write_new(path, (json.dumps(obj, ensure_ascii=False, indent=2) + "\n").encode())


def write_npz_new(path, **arrays):
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    write_new(path, stream.getvalue())


def validate_catalog(catalog, expected_ids):
    expected = list(map(int, expected_ids))
    if expected != sorted(set(expected)) or sorted(map(int, catalog)) != expected:
        raise ValueError("Catalog must match the frozen public MovieLens movie ID list")
    for value in catalog.values():
        if not isinstance(value, str) or str(uuid.UUID(value)) != value.lower():
            raise ValueError("Catalog must contain hyphenated public entity UUIDs only")
    if len({value.lower() for value in catalog.values()}) != len(catalog):
        raise ValueError("Duplicate public entity identities")
    if len(expected) < 2:
        raise ValueError("At least two public seed entities are required")


def validate_background(arrays, ids, expected_users=BACKGROUND_USERS):
    n = len(ids)
    if list(map(int, arrays["ids"])) != list(ids):
        raise ValueError("Background item ordering differs from the public catalog")
    raw = arrays["raw"]
    if raw.shape != (expected_users, n) or not np.isin(raw, [0, 1, 2, 3, 4, 5]).all():
        raise ValueError("Expected the fixed disjoint background rating matrix")
    for name, shape in [("mask", raw.shape), ("user_means", (expected_users,)),
                        ("item_bias", (n,)), ("mf_32_10", (n, 32))]:
        if arrays[name].shape != shape or not np.isfinite(arrays[name]).all():
            raise ValueError("Invalid background asset: " + name)
    if not np.array_equal(arrays["mask"].astype(bool), raw > 0):
        raise ValueError("Observed-rating mask is inconsistent")


def load_background(old):
    with np.load(Path(old) / "tools.npz", allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def sanitize_entity(entity, source_digest=None, fetched_at=None):
    """Keep public movie culture only; affinity and external ratings are dropped."""
    props = entity.get("properties") or {}
    tags = entity.get("tags") or []
    return {
        "entity_id": entity["entity_id"], "name": entity.get("name"),
        "release_year": props.get("release_year"),
        "properties": {field: props[field] for field in FIELDS if props.get(field) is not None},
        "tags": [{"id": tag.get("id") or tag.get("tag_id"),
                  **{key: tag[key] for key in ("name", "type") if key in tag}}
                 for tag in tags if isinstance(tag, dict)],
        "source_cache_digest": source_digest, "fetched_at": fetched_at,
    }


def _entities(data):
    results = data.get("results", {})
    return results if isinstance(results, list) else results.get("entities", [])


def _rich_description_count(entry):
    """Audience/style/theme/tone/genre prose, distinct from basic search facts."""
    return sum(bool(entry["properties"].get(field)) for field in TEXT_FIELDS)


def _quality(entry):
    # A response with useful culture prose outranks a basic search record,
    # even when the search record contains more duration/country/etc. fields.
    return (_rich_description_count(entry),
            sum(bool(entry["properties"].get(field)) for field in FIELDS),
            len(entry["tags"]), entry.get("fetched_at") or 0)


def read_cached_metadata(catalog, cache_dir):
    """Read responses only. Query inputs/target labels are never inspected."""
    reverse = {value.lower(): key for key, value in catalog.items()}
    found = {}
    for path in sorted(Path(cache_dir).glob("*.json")):
        try:
            obj = read(path)
            for entity in _entities(obj.get("data", {})):
                mid = reverse.get(str(entity.get("entity_id", "")).lower())
                if mid is None:
                    continue
                entry = sanitize_entity(entity, path.stem, obj.get("fetched_at"))
                entry["entity_id"] = catalog[mid]
                if mid not in found or _quality(entry) > _quality(found[mid]):
                    found[mid] = entry
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            continue
    return found


def public_query(seed, targets):
    """Bounded exact movie filters; no case ID, profile, rating, or dialogue."""
    if not targets or len(targets) > MAX_TARGETS:
        raise ValueError("Expected 1..50 distinct public target UUIDs")
    for value in [seed, *targets]:
        if not isinstance(value, str) or str(uuid.UUID(value)) != value.lower():
            raise ValueError("Only hyphenated public entity UUIDs may leave the server")
    if len({value.lower() for value in targets}) != len(targets):
        raise ValueError("Duplicate target entity UUIDs")
    if seed.lower() in {value.lower() for value in targets}:
        raise ValueError("The fixed public seed must not be a requested target")
    return {"filter.type": "urn:entity:movie", "signal.interests.entities": seed,
            "filter.results.entities": ",".join(targets), "take": len(targets)}


def collect_metadata(catalog, cache_dir, client=None):
    found = read_cached_metadata(catalog, cache_dir)
    initial_count = len(found)
    seeds = [catalog[mid] for mid in sorted(catalog, key=int)[:2]]
    requested = 0
    if client is not None:
        # A fixed second public seed handles the first seed's self-exclusion
        # and any omitted targets. No outcome-dependent search or retries.
        for seed in seeds:
            pending = [mid for mid in sorted(catalog, key=int)
                       if (mid not in found or _rich_description_count(found[mid]) == 0)
                       and catalog[mid] != seed]
            for start in range(0, len(pending), MAX_TARGETS):
                mids = pending[start:start + MAX_TARGETS]
                targets = [catalog[mid] for mid in mids]
                params = public_query(seed, targets)
                data = client.get("/v2/insights", params)
                requested += 1
                reverse = {catalog[mid].lower(): mid for mid in mids}
                digest = hashlib.sha256(json.dumps(
                    ["https://hackathon.api.qloo.com", "/v2/insights", params],
                    sort_keys=True).encode()).hexdigest()
                for entity in _entities(data):
                    mid = reverse.get(str(entity.get("entity_id", "")).lower())
                    if mid is not None:
                        entry = sanitize_entity(entity, digest, time.time())
                        entry["entity_id"] = catalog[mid]
                        if mid not in found or _quality(entry) > _quality(found[mid]):
                            found[mid] = entry
    missing = sorted(set(catalog) - set(found), key=int)
    document = {"schema_version": 1, "source": "Qloo public movie entity metadata; deployment rebuild",
                "excluded": ["affinity", "hidden_ratings", "user IDs", "external aggregate ratings"],
                "movie_count": len(found), "missing_movie_ids": missing, "movies": found}
    return document, {"cached_movies": initial_count, "logical_metadata_queries": requested,
                      "missing_movie_ids": missing,
                      "rich_description_movies": sum(_rich_description_count(row) > 0 for row in found.values()),
                      "basic_only_movie_ids": sorted([mid for mid, row in found.items()
                                                      if _rich_description_count(row) == 0], key=int)}


def validate_metadata(document, catalog):
    entries = document["movies"]
    if set(entries) != set(catalog) or document.get("movie_count") != len(catalog):
        raise ValueError("Metadata lacks catalog movies; no partial runtime is accepted")
    allowed = {"entity_id", "name", "release_year", "properties", "tags", "source_cache_digest", "fetched_at"}
    for mid, entry in entries.items():
        if entry["entity_id"] != catalog[mid] or set(entry) - allowed or set(entry["properties"]) - set(FIELDS):
            raise ValueError("Metadata identity or public-field allowlist mismatch")
        if any(set(tag) - {"id", "name", "type"} for tag in entry["tags"]):
            raise ValueError("Unexpected tag fields")


def graph_arrays(original, cultural):
    """Exact graph_models.build_tools formula, without its global output path."""
    if list(map(int, original["ids"])) != cultural.ids:
        raise ValueError("Culture/background item order mismatch")
    kernel = _center(.75 * cultural.kernels["curated"] + .25 * cultural.kernels["tags"])
    eigenvalues, vectors = np.linalg.eigh(kernel)
    take = np.argsort(eigenvalues)[-32:]
    features = vectors[:, take] * np.sqrt(np.maximum(eigenvalues[take], 0))
    features /= np.sqrt(np.maximum(np.mean(np.sum(features * features, axis=1)), 1e-12))
    rank = features.shape[1]
    users = np.zeros((len(original["raw"]), rank))
    for j, ratings in enumerate(original["raw"]):
        observed = original["mask"][j].astype(bool)
        h = features[observed]
        target = ratings[observed] - original["user_means"][j] - original["item_bias"][observed]
        users[j] = np.linalg.solve(h.T @ h + 10 * np.eye(rank), h.T @ target)
    users -= users.mean(axis=0)
    covariance = users.T @ users / len(users)
    covariance /= max(np.trace(covariance) / rank, 1e-12)
    covariance = .75 * covariance + .25 * np.eye(rank)
    return {"ids": original["ids"], "kernel": kernel,
            "background_kernel": _center(features @ covariance @ features.T),
            "item_bias": original["item_bias"]}


def build_mixtures(original, cultural, destination):
    """Only background observations train six fixed Bernoulli model assets."""
    ids = list(map(int, original["ids"]))
    if ids != cultural.ids:
        raise ValueError("Culture/background item order mismatch")
    cf = original["mf_32_10"].astype(float)
    cf /= np.maximum(np.linalg.norm(cf, axis=1, keepdims=True), 1e-12)
    weights = {"plain": None, "cf": neighbor_kernel(cf @ cf.T),
               "qloo": neighbor_kernel(cultural.kernels["curated"])}
    training, created, preserved = [], [], []
    for k in SETTINGS["K"]:
        paths = {kind: Path(destination) / f"bernoulli_k{k}_{kind}.npz" for kind in weights}
        absent = [kind for kind, path in paths.items() if not path.exists()]
        for kind, path in paths.items():
            if path.exists():
                validate_mixture(path, ids, k)
                preserved.append(path.name)
        if not absent:
            continue
        arrays, metrics = fit_mixture(original["raw"], k)
        training.append(metrics)
        for kind in absent:
            probability = with_graph_prior(arrays, weights[kind])
            MixtureTools(ids, probability, arrays["prior"])
            write_npz_new(paths[kind], ids=np.array(ids), probability=probability, prior=arrays["prior"])
            created.append(paths[kind].name)
    return {"training": training, "created": created, "preserved": preserved}


def validate_mixture(path, ids, k):
    with np.load(path, allow_pickle=False) as data:
        if list(map(int, data["ids"])) != list(ids) or len(data["prior"]) != k:
            raise ValueError("Mixture item order or group count mismatch")
        MixtureTools(data["ids"], data["probability"], data["prior"])


def validate_graph(path, ids):
    with np.load(path, allow_pickle=False) as data:
        n = len(ids)
        if list(map(int, data["ids"])) != list(ids):
            raise ValueError("Graph item ordering mismatch")
        for key, shape in [("kernel", (n, n)), ("background_kernel", (n, n)), ("item_bias", (n,))]:
            if data[key].shape != shape or not np.isfinite(data[key]).all():
                raise ValueError("Invalid graph array: " + key)


def check_runtime(old, private, expected_ids, with_graph=False):
    """Read-only validation. No mkdir, credential read, cache scan, or API call."""
    old, private = Path(old), Path(private)
    assets = {"catalog": old / "catalog.json", "movie_metadata": old / "metadata.json",
              "background_tools": old / "tools.npz", "culture_metadata": private / "qloo_metadata.json"}
    assets.update({f"bernoulli_k{k}_{kind}": private / "mixture_tools" / f"bernoulli_k{k}_{kind}.npz"
                   for k, kind in CONFIGS})
    if with_graph:
        assets["culture_graph"] = private / "qloo_graph_tools.npz"
    report = {"ready": False, "network_requests": 0, "read_only": True,
              "resources": {name: {"exists": path.is_file()} for name, path in assets.items()}}
    if not all(path.is_file() for path in assets.values()):
        report["missing"] = [str(path) for path in assets.values() if not path.is_file()]
        return report
    try:
        catalog = read(assets["catalog"])
        validate_catalog(catalog, expected_ids)
        ids = list(map(int, expected_ids))
        if sorted(map(int, read(assets["movie_metadata"]))) != ids:
            raise ValueError("Movie title metadata/catalog mismatch")
        validate_background(load_background(old), ids)
        metadata = read(assets["culture_metadata"])
        validate_metadata(metadata, catalog)
        for k, kind in CONFIGS:
            validate_mixture(assets[f"bernoulli_k{k}_{kind}"], ids, k)
        if with_graph:
            validate_graph(assets["culture_graph"], ids)
        report.update(ready=True, movies=len(ids), background_users=BACKGROUND_USERS,
                      rich_description_movies=sum(_rich_description_count(row) > 0
                                                  for row in metadata["movies"].values()),
                      basic_only_movie_ids=sorted([mid for mid, row in metadata["movies"].items()
                                                 if _rich_description_count(row) == 0], key=int))
    except (OSError, ValueError, KeyError, TypeError) as error:
        report["error"] = str(error)
    return report


def build_runtime(old, private, cache_dir, expected_ids, *, with_graph=False, client=None):
    old, private = Path(old), Path(private)
    required_old = [old / name for name in ("catalog.json", "metadata.json", "tools.npz")]
    if not all(path.is_file() for path in required_old):
        raise ValueError("v11 runtime missing: run culture_v11/download.py then culture_v11/bootstrap_runtime.py first")
    catalog = read(old / "catalog.json")
    validate_catalog(catalog, expected_ids)
    original = load_background(old)
    validate_background(original, list(map(int, expected_ids)))
    if sorted(map(int, read(old / "metadata.json"))) != list(map(int, expected_ids)):
        raise ValueError("Movie title metadata/catalog mismatch")
    metadata_path = private / "qloo_metadata.json"
    report = {"uses_development_or_evaluation_labels": False, "created": [], "preserved": []}
    if metadata_path.exists():
        validate_metadata(read(metadata_path), catalog)
        report["preserved"].append(metadata_path.name)
    else:
        document, summary = collect_metadata(catalog, cache_dir, client)
        report["metadata_collection"] = summary
        if summary["missing_movie_ids"]:
            raise ValueError(f"Metadata missing for {len(summary['missing_movie_ids'])} movies; no partial asset written. "
                             "Re-run with --allow-network, or restore a complete private metadata snapshot.")
        validate_metadata(document, catalog)
        write_json_new(metadata_path, document)
        report["created"].append(metadata_path.name)
    cultural = CulturalMetadataTools(metadata_path)
    graph_path = private / "qloo_graph_tools.npz"
    if with_graph:
        if graph_path.exists():
            validate_graph(graph_path, cultural.ids)
            report["preserved"].append(graph_path.name)
        else:
            write_npz_new(graph_path, **graph_arrays(original, cultural))
            report["created"].append(graph_path.name)
    report["mixtures"] = build_mixtures(original, cultural, private / "mixture_tools")
    report["inputs_sha256"] = {"background_tools": sha(old / "tools.npz"),
                              "catalog": sha(old / "catalog.json"), "culture_metadata": sha(metadata_path)}
    report["training_settings"] = SETTINGS
    # Each invocation has its own provenance; frozen files are never overwritten.
    provenance = private / "runtime_builds" / f"{time.time_ns()}.json"
    write_json_new(provenance, report)
    report["check"] = check_runtime(old, private, expected_ids, with_graph)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Read-only readiness check (default)")
    mode.add_argument("--build", action="store_true", help="Build missing private assets only")
    parser.add_argument("--allow-network", action="store_true", help="Allow public-UUID Qloo metadata retrieval during --build")
    parser.add_argument("--with-graph", action="store_true", help="Also build/check the optional qgraph numerical asset")
    parser.add_argument("--v11-private", type=Path, default=Path(os.environ.get("V11_PRIVATE", DEFAULT_PARENT / "culture_v11")))
    parser.add_argument("--private", type=Path, default=Path(os.environ.get("V13_PRIVATE", DEFAULT_PARENT / "culture_v13")))
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_PARENT / "qloo_cache")
    args = parser.parse_args(argv)
    if args.allow_network and not args.build:
        parser.error("--allow-network requires --build; --check is always offline")
    expected = read(V11 / "catalog_movie_ids.json")["movie_ids"]
    try:
        if not args.build:
            result = check_runtime(args.v11_private, args.private, expected, args.with_graph)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if result["ready"] else 1
        client = None
        if args.allow_network and not (args.private / "qloo_metadata.json").exists():
            from tastetrail.qloo_http import QlooClient
            client = QlooClient(cache_dir=args.cache_dir, ttl=30 * 86400)
        result = build_runtime(args.v11_private, args.private, args.cache_dir, expected,
                               with_graph=args.with_graph, client=client)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["check"]["ready"] else 1
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        print(json.dumps({"ready": False, "error": str(error)}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
