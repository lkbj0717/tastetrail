"""Offline transport tests: no provider calls, assets, or real credentials."""
import asyncio
import json
from pathlib import Path
import sys

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT, ROOT / "work/culture_v13", ROOT / "work/culture_v11", ROOT / "work/culture_v11/src"):
    sys.path.insert(0, str(folder))

from deploy.gateway import Gateway, with_cors


class Echo:
    def __init__(self):
        self.calls = []

    async def __call__(self, scope, receive, send):
        body = b""
        if scope["method"] == "POST":
            body = (await receive())["body"]
        self.calls.append((scope["path"], body))
        await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"application/json")]})
        await send({"type": "http.response.body", "body": body or b'{"original":true}'})


def request(gateway, method="GET", path="/api/catalog", **kwargs):
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=gateway), base_url="http://test") as client:
            return await client.request(method, path, **kwargs)
    return asyncio.run(run())


@pytest.fixture(autouse=True)
def no_real_network(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("Gateway tests must not invoke real HTTP providers.")
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", fail)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", fail)


def test_auth_applies_to_all_public_data_and_paid_api_aliases():
    echo = Echo()
    gateway = Gateway(echo, token="test-server-secret", readiness=lambda: True)
    for path in ("/api/catalog", "/catalog", "/api/example", "/example", "/api/research"):
        assert request(gateway, path=path).status_code == 401
        assert request(gateway, path=path, headers={"Authorization": "Bearer wrong"}).status_code == 401
        assert request(gateway, path=path, headers={"Authorization": "Bearer test-server-secret"}).status_code == 200
    for path in ("/api/recommend", "/api/compare"):
        assert request(gateway, "POST", path, json={}).status_code == 401
    assert len(echo.calls) == 5


def test_health_is_free_minimal_and_root_never_contains_gateway_token():
    echo = Echo()
    gateway = Gateway(echo, token="not-for-browser", readiness=lambda: False)
    for path in ("/health", "/api/health"):
        response = request(gateway, path=path)
        assert response.status_code == 503 and response.json() == {"ready": False}
    gateway.readiness = lambda: True
    assert request(gateway, path="/health").json() == {"ready": True}
    assert request(gateway, "HEAD", "/health").content == b""
    assert echo.calls == []
    response = request(gateway, path="/")
    assert response.status_code == 200 and "not-for-browser" not in response.text


def test_undeclared_routes_and_methods_do_not_reach_frozen_app():
    echo = Echo()
    gateway = Gateway(echo)
    for method, path in (("GET", "/docs"), ("GET", "/openapi.json"), ("GET", "/private/credentials.json"),
                         ("POST", "/"), ("GET", "/api/recommend"), ("DELETE", "/api/catalog")):
        assert request(gateway, method, path).status_code == 404
    assert echo.calls == []


def test_body_content_type_and_syntax_are_checked_before_inference():
    echo = Echo()
    gateway = Gateway(echo, max_body=1024)
    assert request(gateway, "POST", "/api/recommend", content="{}").status_code == 415
    assert request(gateway, "POST", "/api/recommend", content="{", headers={"Content-Type": "application/json"}).status_code == 400
    assert request(gateway, "POST", "/api/recommend", json=[]).status_code == 400
    assert request(gateway, "POST", "/api/recommend", json={"oversize": "x" * 2000}).status_code == 413
    assert echo.calls == []


def test_chunked_body_cannot_bypass_size_limit():
    async def run():
        echo = Echo()
        gateway = Gateway(echo, max_body=1024)
        events = iter([{"type": "http.request", "body": b"x" * 600, "more_body": True},
                       {"type": "http.request", "body": b"x" * 600, "more_body": False}])
        sent = []
        async def receive(): return next(events)
        async def send(event): sent.append(event)
        await gateway({"type": "http", "method": "POST", "path": "/api/recommend",
                       "headers": [(b"content-type", b"application/json")]}, receive, send)
        assert sent[0]["status"] == 413 and echo.calls == []
        assert gateway.slots.acquire(blocking=False)
    asyncio.run(run())


def test_accepted_body_is_forwarded_without_prompt_or_input_changes():
    echo = Echo()
    gateway = Gateway(echo)
    body = b'{ "likes" : ["a"], "literal": "kept" }'
    response = request(gateway, "POST", "/api/compare", content=body, headers={"Content-Type": "application/json; charset=utf-8"})
    assert response.content == body
    assert echo.calls == [("/api/compare", body)]


def test_busy_request_is_rejected_without_waiting_for_original_lock():
    async def run():
        entered, release = asyncio.Event(), asyncio.Event()
        echo = Echo()
        async def slow(scope, receive, send):
            entered.set()
            await release.wait()
            await echo(scope, receive, send)
        gateway = Gateway(slow)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=gateway), base_url="http://test") as client:
            first = asyncio.create_task(client.post("/api/recommend", json={}))
            await entered.wait()
            second = await asyncio.wait_for(client.post("/api/recommend", json={}), 1)
            assert second.status_code == 429 and second.headers["retry-after"] == "10"
            release.set()
            assert (await first).status_code == 200
            assert (await client.post("/api/recommend", json={})).status_code == 200
        assert len(echo.calls) == 2
    asyncio.run(run())


def test_hourly_allowance_expires_and_invalid_json_does_not_consume_it():
    now = [100.0]
    echo = Echo()
    gateway = Gateway(echo, hourly_limit=1, clock=lambda: now[0])
    assert request(gateway, "POST", "/api/recommend", content="!", headers={"Content-Type": "application/json"}).status_code == 400
    assert request(gateway, "POST", "/api/recommend", json={}).status_code == 200
    assert request(gateway, "POST", "/api/recommend", json={}).status_code == 429
    now[0] += 3601
    assert request(gateway, "POST", "/api/recommend", json={}).status_code == 200
    assert len(echo.calls) == 2


def test_slow_upload_releases_the_inference_slot():
    async def run():
        echo = Echo()
        gateway = Gateway(echo, body_timeout=0.01)
        sent = []
        async def receive(): await asyncio.sleep(1)
        async def send(event): sent.append(event)
        await gateway({"type": "http", "method": "POST", "path": "/api/recommend",
                       "headers": [(b"content-type", b"application/json")]}, receive, send)
        assert sent[0]["status"] == 408 and echo.calls == []
        assert gateway.slots.acquire(blocking=False)
    asyncio.run(run())


def test_original_request_validation_and_compare_flag_are_preserved(monkeypatch):
    import app_v13
    calls = []
    def mock_endpoint(body, compare):
        calls.append((body.model_dump(), compare))
        return {"mock_only": True, "compare": compare}
    monkeypatch.setattr(app_v13, "endpoint", mock_endpoint)
    gateway = Gateway(app_v13.app)
    body = {"likes": [{"title": f"Film {n}", "year": 2000, "rating": 5} for n in range(5)],
            "dislikes": [{"title": f"Film {n}", "year": 2000, "rating": 1} for n in range(5, 8)],
            "candidates": [{"title": f"Film {n}", "year": 2000} for n in range(8, 10)]}
    assert request(gateway, "POST", "/api/recommend", json=body).json() == {"mock_only": True, "compare": False}
    assert request(gateway, "POST", "/api/compare", json=body).json() == {"mock_only": True, "compare": True}
    assert calls == [(body, False), (body, True)]
    body["hidden_answer"] = "C01"
    assert request(gateway, "POST", "/api/recommend", json=body).status_code == 422
    assert len(calls) == 2


def test_invalid_limits_fail_at_startup():
    for options in ({"max_concurrent": 2}, {"hourly_limit": 0}, {"max_body": 10}):
        with pytest.raises(ValueError):
            Gateway(Echo(), **options)


def test_exact_origin_preflight_allows_json_post_without_credentials():
    echo = Echo()
    application = with_cors(Gateway(echo), "https://demo.example.com")
    response = request(application, "OPTIONS", "/api/compare", headers={
        "Origin": "https://demo.example.com", "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "content-type"})
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "https://demo.example.com"
    assert response.headers["access-control-max-age"] == "600"
    assert "access-control-allow-credentials" not in response.headers
    assert echo.calls == []


def test_unapproved_origin_and_browser_authorization_preflights_are_denied():
    echo = Echo()
    application = with_cors(Gateway(echo), "https://demo.example.com")
    response = request(application, "OPTIONS", "/api/compare", headers={
        "Origin": "https://other.example.com", "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "content-type"})
    assert response.status_code == 400
    assert "access-control-allow-origin" not in response.headers
    response = request(application, "OPTIONS", "/api/compare", headers={
        "Origin": "https://demo.example.com", "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "authorization"})
    assert response.status_code == 400
    assert echo.calls == []


def test_cors_headers_and_retry_after_survive_hourly_rejection():
    echo = Echo()
    application = with_cors(Gateway(echo, hourly_limit=1), "https://demo.example.com")
    headers = {"Origin": "https://demo.example.com"}
    assert request(application, "POST", "/api/recommend", json={}, headers=headers).status_code == 200
    response = request(application, "POST", "/api/recommend", json={}, headers=headers)
    assert response.status_code == 429
    assert response.headers["access-control-allow-origin"] == "https://demo.example.com"
    assert response.headers["access-control-expose-headers"] == "Retry-After"
    assert int(response.headers["retry-after"]) > 0
    assert len(echo.calls) == 1


def test_cors_is_disabled_by_default_and_does_not_bypass_gateway_auth():
    gateway = Gateway(Echo(), token="server-only")
    assert with_cors(gateway, " , ") is gateway
    application = with_cors(gateway, "https://demo.example.com")
    response = request(application, path="/api/catalog", headers={"Origin": "https://demo.example.com"})
    assert response.status_code == 401
    assert response.headers["access-control-allow-origin"] == "https://demo.example.com"


def test_cors_configuration_requires_exact_secure_origins():
    for value in ("*", "https://*.example.com", "http://example.com", "https://example.com/path",
                  "https://user:password@example.com", "https://example.com?token=abc"):
        with pytest.raises(ValueError, match="exact HTTPS origins"):
            with_cors(Gateway(Echo()), value)
    application = with_cors(Gateway(Echo()), "https://demo.example.com/, http://localhost:3000")
    response = request(application, path="/api/catalog", headers={"Origin": "http://localhost:3000"})
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"
