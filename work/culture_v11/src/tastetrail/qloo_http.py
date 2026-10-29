"""Documented event REST endpoints. Private cache; no silent synthetic fallback."""
import hashlib
import json
import os
import time
from pathlib import Path
import httpx
from .config import private_dir, settings


class QlooError(RuntimeError):
    pass


class QlooClient:
    BASE = "https://hackathon.api.qloo.com"

    def __init__(self, key=None, *, cache_dir=None, transport=None, ttl=86400):
        self.key = key or settings().get("QLOO_API_KEY", "")
        self.cache_dir = Path(cache_dir) if cache_dir else private_dir() / "qloo_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.transport, self.ttl, self.calls = transport, ttl, []

    def get(self, endpoint, params):
        if endpoint not in {"/search", "/v2/insights", "/v2/tags"}:
            raise ValueError("Endpoint is not allowlisted")
        if not self.key:
            raise QlooError("Qloo key is not configured.")
        signature = json.dumps([self.BASE, endpoint, params], sort_keys=True)
        digest = hashlib.sha256(signature.encode()).hexdigest()
        path = self.cache_dir / (digest + ".json")
        if path.exists():
            cached = json.loads(path.read_text())
            if time.time() - cached["fetched_at"] < self.ttl:
                self.calls.append({"endpoint": endpoint, "cache_hit": True, "digest": digest,
                                   "fetched_at": cached["fetched_at"]})
                return cached["data"]
        for attempt in range(2):
            started = time.monotonic()
            record = {"endpoint": endpoint, "cache_hit": False, "digest": digest, "attempt": attempt + 1}
            try:
                with httpx.Client(timeout=httpx.Timeout(40, connect=12), trust_env=False, transport=self.transport) as http:
                    response = http.get(self.BASE + endpoint, params=params, headers={"X-Api-Key": self.key})
                record["http_status"] = response.status_code
                # Search's documented no-results response is distinct from authentication/service failure.
                if endpoint == "/search" and response.status_code == 404:
                    return {"results": []}
                if response.status_code != 200:
                    raise QlooError(f"Qloo HTTP {response.status_code}")
                data = response.json()
                if not isinstance(data, dict) or "results" not in data or data.get("success") is False:
                    raise QlooError("Unexpected Qloo response schema")
                record["fetched_at"] = time.time()
                tmp = path.with_suffix(f".{os.getpid()}.{time.time_ns()}.tmp")
                with open(tmp, "w") as f:
                    os.chmod(tmp, 0o600)
                    json.dump({"fetched_at": record["fetched_at"], "data": data}, f)
                tmp.replace(path)
                return data
            except (httpx.HTTPError, ValueError, QlooError) as error:
                record["error"] = str(error) if isinstance(error, QlooError) else type(error).__name__
                retry = isinstance(error, httpx.TransportError) or record.get("http_status") in {429, 500, 502, 503, 504}
                if not retry or attempt:
                    raise QlooError(record["error"]) from None
                time.sleep(1)
            finally:
                record["seconds"] = round(time.monotonic() - started, 3)
                self.calls.append(record)

    def search(self, name):
        # Only a short public work/artist/entity name is sent, never the full user dialogue.
        if not isinstance(name, str) or not 1 <= len(name) <= 120 or "@" in name:
            raise ValueError("Provide a public cultural entity name, not personal information.")
        return self.get("/search", {"query": name, "take": 6})

    def recommend(self, entity_ids, city, *, take=16):
        if not entity_ids or len(entity_ids) > 4 or not 1 <= len(city) <= 80 or "@" in city:
            raise ValueError("Invalid bounded recommendation request")
        return self.get("/v2/insights", {"filter.type": "urn:entity:place",
                        "signal.interests.entities": ",".join(entity_ids), "filter.location.query": city,
                        "filter.location.radius": 0, "feature.explainability": "true", "take": min(take, 20)})
