"""Small ASGI boundary around the original v13 FastAPI application.

This module changes admission and transport only. Accepted requests are passed to
the frozen application byte-for-byte, without changing prompts or predictions.
Use one Uvicorn worker: concurrency and hourly admissions are process-local.
"""
from __future__ import annotations

import asyncio
from collections import deque
import hmac
import json
import os
from pathlib import Path
import threading
import time
from urllib.parse import urlsplit

from starlette.middleware.cors import CORSMiddleware


READ_PATHS = {"/", "/api/catalog", "/catalog", "/api/example", "/example", "/api/research"}
HEALTH_PATHS = {"/health", "/api/health"}
PAID_PATHS = {"/api/recommend", "/api/compare"}


def runtime_ready():
    """Cheap local readiness only; never checks an API or exposes file paths."""
    private = Path(os.environ.get("V13_PRIVATE", str(Path.home() / ".local/share/tastetrail/culture_v13")))
    old = private.parent / "culture_v11"
    required = [old / name for name in ("metadata.json", "catalog.json", "tools.npz")]
    required += [private / "qloo_metadata.json"]
    required += [private / "mixture_tools" / f"bernoulli_k{k}_{kind}.npz"
                 for k in (12, 24) for kind in ("plain", "cf", "qloo")]
    if not all(path.is_file() for path in required):
        return False
    try:
        credentials = Path(os.environ.get("TASTETRAIL_CREDENTIALS", "~/.config/tastetrail/credentials.json")).expanduser()
        settings = json.loads(credentials.read_text()) if credentials.is_file() else {}
        settings.update(os.environ)
        provider = settings.get("TASTETRAIL_MODEL_PROVIDER", "glm").upper()
        return bool(settings.get("QLOO_API_KEY") and settings.get(f"{provider}_API_KEY"))
    except (OSError, ValueError, TypeError, AttributeError):
        return False


async def respond(send, status, detail, *, retry_after=None, head=False):
    body = json.dumps(detail, separators=(",", ":")).encode()
    headers = [(b"content-type", b"application/json"), (b"cache-control", b"no-store"),
               (b"content-length", str(len(body)).encode()), (b"x-content-type-options", b"nosniff")]
    if retry_after is not None:
        headers.append((b"retry-after", str(retry_after).encode()))
    if status == 401:
        headers.append((b"www-authenticate", b"Bearer"))
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": b"" if head else body})


class Gateway:
    def __init__(self, inner, *, token="", max_body=16384, max_concurrent=1,
                 hourly_limit=30, body_timeout=10, readiness=runtime_ready,
                 clock=time.monotonic):
        if max_body < 1024 or max_concurrent != 1 or hourly_limit < 1 or body_timeout <= 0:
            raise ValueError("Require a >=1024-byte body limit, one inference slot, and positive rate/time limits.")
        self.inner = inner
        self.token = token.encode()
        self.max_body = max_body
        self.body_timeout = body_timeout
        self.hourly_limit = hourly_limit
        self.readiness = readiness
        self.clock = clock
        self.slots = threading.BoundedSemaphore(max_concurrent)
        self.admissions = deque()
        self.admission_lock = threading.Lock()

    def _authorized(self, headers):
        if not self.token:
            return True
        values = [value for key, value in headers if key.lower() == b"authorization"]
        if len(values) != 1:
            return False
        parts = values[0].split(b" ", 1)
        return (len(parts) == 2 and parts[0].lower() == b"bearer"
                and hmac.compare_digest(parts[1], self.token))

    def _admit(self):
        with self.admission_lock:
            now = self.clock()
            while self.admissions and self.admissions[0] <= now - 3600:
                self.admissions.popleft()
            if len(self.admissions) >= self.hourly_limit:
                return max(1, int(3600 - (now - self.admissions[0])) + 1)
            self.admissions.append(now)
            return None

    async def __call__(self, scope, receive, send):
        if scope["type"] == "lifespan":
            return await self.inner(scope, receive, send)
        if scope["type"] != "http":
            if scope["type"] == "websocket":
                await send({"type": "websocket.close", "code": 1008})
            return
        method, path = scope["method"], scope["path"]
        if path in HEALTH_PATHS and method in {"GET", "HEAD"}:
            ready = bool(self.readiness())
            return await respond(send, 200 if ready else 503, {"ready": ready}, head=method == "HEAD")
        allowed = (method == "GET" and path in READ_PATHS) or (method == "POST" and path in PAID_PATHS)
        if not allowed:
            return await respond(send, 404, {"detail": "Not found"})
        headers = scope.get("headers", [])
        if path != "/" and not self._authorized(headers):
            return await respond(send, 401, {"detail": "A server gateway token is required."})
        if method == "GET":
            return await self.inner(scope, receive, send)
        content_types = [v.split(b";", 1)[0].strip().lower() for k, v in headers if k.lower() == b"content-type"]
        if content_types != [b"application/json"]:
            return await respond(send, 415, {"detail": "Send application/json."})
        lengths = [v for k, v in headers if k.lower() == b"content-length"]
        if len(lengths) > 1:
            return await respond(send, 400, {"detail": "Invalid content length."})
        if lengths:
            try:
                length = int(lengths[0])
            except ValueError:
                length = -1
            if length < 0:
                return await respond(send, 400, {"detail": "Invalid content length."})
            if length > self.max_body:
                return await respond(send, 413, {"detail": "Request body is too large."})
        # The original inference service serializes calls. Reject here instead
        # of filling its thread pool with requests waiting on that lock.
        if not self.slots.acquire(blocking=False):
            return await respond(send, 429, {"detail": "The demo is busy. Please retry shortly."}, retry_after=10)
        try:
            chunks = bytearray()
            deadline = asyncio.get_running_loop().time() + self.body_timeout
            while True:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    raise TimeoutError
                event = await asyncio.wait_for(receive(), remaining)
                if event["type"] == "http.disconnect":
                    return
                if event["type"] != "http.request":
                    return await respond(send, 400, {"detail": "Invalid request body."})
                chunks.extend(event.get("body", b""))
                if len(chunks) > self.max_body:
                    return await respond(send, 413, {"detail": "Request body is too large."})
                if not event.get("more_body", False):
                    break
            # Validate cheap syntax before charging an admission. Full schema
            # validation stays in the original application.
            try:
                parsed = json.loads(chunks)
                if not isinstance(parsed, dict):
                    raise ValueError
            except (ValueError, UnicodeError):
                return await respond(send, 400, {"detail": "Expected a JSON object."})
            retry = self._admit()
            if retry is not None:
                return await respond(send, 429, {"detail": "The demo's hourly request allowance is exhausted."}, retry_after=retry)
            delivered = False

            async def replay_body():
                nonlocal delivered
                if delivered:
                    return {"type": "http.disconnect"}
                delivered = True
                return {"type": "http.request", "body": bytes(chunks), "more_body": False}

            return await self.inner(scope, replay_body, send)
        except TimeoutError:
            return await respond(send, 408, {"detail": "Request body timed out."})
        finally:
            self.slots.release()


def with_cors(application, configured_origins):
    """Opt in to exact browser origins; never make a wildcard public policy."""
    origins = []
    for value in configured_origins.split(","):
        value = value.strip()
        if not value:
            continue
        try:
            parsed = urlsplit(value)
            # Accessing port also validates malformed numeric ports.
            parsed.port
            local_http = parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1", "::1"}
            valid = (bool(parsed.hostname) and (parsed.scheme == "https" or local_http)
                     and "*" not in value and not parsed.username and not parsed.password
                     and parsed.path in {"", "/"} and not parsed.query and not parsed.fragment)
        except ValueError:
            valid = False
        if not valid:
            raise ValueError("TASTETRAIL_ALLOWED_ORIGINS requires exact HTTPS origins (or explicitly configured HTTP localhost), without wildcards or paths.")
        origin = f"{parsed.scheme}://{parsed.netloc.lower()}"
        if origin not in origins:
            origins.append(origin)
    if not origins:
        return application
    # This outer layer also decorates Gateway's 401/429/error responses so the
    # approved browser can show their real status instead of a generic CORS error.
    return CORSMiddleware(application, allow_origins=origins,
                          allow_methods=["GET", "POST"], allow_headers=["Content-Type"],
                          expose_headers=["Retry-After"], max_age=600,
                          allow_credentials=False)


def create_app():
    from app_v13 import app as frozen_app
    gateway = Gateway(frozen_app,
                      token=os.environ.get("TASTETRAIL_GATEWAY_TOKEN", ""),
                      max_body=int(os.environ.get("TASTETRAIL_MAX_BODY_BYTES", "16384")),
                      hourly_limit=int(os.environ.get("TASTETRAIL_REQUESTS_PER_HOUR", "30")))
    return with_cors(gateway, os.environ.get("TASTETRAIL_ALLOWED_ORIGINS", ""))


app = create_app()
