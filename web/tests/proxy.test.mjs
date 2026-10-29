import { test } from "node:test";
import assert from "node:assert/strict";
import { MAX_BODY_BYTES, proxyRequest, runtimeConfiguration } from "../worker/proxy.mjs";

const env = { TASTETRAIL_BACKEND_URL: "https://backend.example", TASTETRAIL_GATEWAY_TOKEN: "test-only-token" };
const request = (path, init) => new Request(`https://tastetrail.example${path}`, init);
const post = (body = "{}", headers = {}) => ({ method: "POST", headers: { "Content-Type": "application/json", ...headers }, body });
const unused = () => { throw new Error("Must not contact backend"); };

test("runtime configuration accepts plain Worker bindings and prefers them over process values", async () => {
  assert.deepEqual(await runtimeConfiguration(env, { TASTETRAIL_BACKEND_URL: "https://stale.example", TASTETRAIL_GATEWAY_TOKEN: "stale-test-value" }), env);
});

test("runtime configuration reads account-level Secrets Store through async get()", async () => {
  const store = Object.fromEntries(Object.entries(env).map(([name, value]) => [name, { async get() { return value; } }]));
  assert.deepEqual(await runtimeConfiguration(store, {}), env);
  const response = await proxyRequest(request("/api/health"), store, async (_, init) => {
    assert.equal(init.headers.Authorization, "Bearer test-only-token");
    return Response.json({ ready: true });
  });
  assert.equal(response.status, 200);
});

test("runtime configuration falls back to server process values when bindings are absent", async () => {
  assert.deepEqual(await runtimeConfiguration({}, env), env);
});

test("secret-store failures produce only type diagnostics and a generic response", async () => {
  const output = [];
  const original = console.error;
  console.error = (...values) => output.push(JSON.stringify(values));
  try {
    const response = await proxyRequest(request("/api/health"), { ...env, TASTETRAIL_GATEWAY_TOKEN: { async get() { throw new Error("sensitive-store-detail"); } } }, unused);
    assert.equal(response.status, 503);
    assert.doesNotMatch(await response.text(), /sensitive-store-detail|test-only-token|backend\.example/);
    assert.doesNotMatch(output.join(""), /sensitive-store-detail|test-only-token|backend\.example/);
    assert.match(output.join(""), /secret-store/);
  } finally { console.error = original; }
});

test("catalog is forwarded to fixed origin with server authentication, never browser credentials", async () => {
  let observed;
  const response = await proxyRequest(request("/api/catalog?q=Drama&url=https://evil.example", { headers: { Authorization: "Bearer untrusted", Cookie: "session=private" } }), env, async (url, init) => {
    observed = { url, init };
    return new Response('[{"title":"Example","year":1995}]', { headers: { "Set-Cookie": "private", "X-Internal-Path": "/private/path" } });
  });
  assert.equal(response.status, 200);
  assert.equal(observed.url, "https://backend.example/api/catalog?q=Drama&url=https://evil.example");
  assert.equal(observed.init.headers.Authorization, "Bearer test-only-token");
  assert.equal(observed.init.headers.Cookie, undefined);
  assert.equal(observed.init.redirect, "error");
  assert.equal(response.headers.get("Set-Cookie"), null);
  assert.equal(response.headers.get("X-Internal-Path"), null);
  assert.equal(response.headers.get("Cache-Control"), "no-store");
});

test("POST JSON is forwarded without prediction changes", async () => {
  const payload = '{"likes":[{"title":"Film","year":1999,"rating":5}]}';
  const response = await proxyRequest(request("/api/compare", post(payload)), env, async (_, init) => {
    assert.equal(new TextDecoder().decode(init.body), payload);
    return Response.json({ recommendations: { glm: "C01", agent: "C02", agent_qloo: "C02" } });
  });
  assert.equal((await response.json()).recommendations.agent_qloo, "C02");
});

test("unknown routes, methods, query paths and unsafe origins do not call backend", async () => {
  for (const [path, init, status] of [["/api/private", undefined, 404], ["/api/research", { method: "DELETE" }, 404], ["/api/compare?target=evil", post(), 400]]) {
    assert.equal((await proxyRequest(request(path, init), env, unused)).status, status);
  }
  for (const value of ["http://backend.example", "https://user:password@backend.example", "https://backend.example/api", "https://backend.example?target=evil"]) {
    assert.equal((await proxyRequest(request("/api/health"), { ...env, TASTETRAIL_BACKEND_URL: value }, unused)).status, 503);
  }
});

test("missing token fails closed", async () => {
  assert.equal((await proxyRequest(request("/api/health"), { TASTETRAIL_BACKEND_URL: env.TASTETRAIL_BACKEND_URL }, unused)).status, 503);
});

test("request stream limit applies even to a misleading content-length", async () => {
  const stream = new ReadableStream({ start(controller) { controller.enqueue(new Uint8Array(MAX_BODY_BYTES)); controller.enqueue(new Uint8Array(1)); controller.close(); } });
  const req = request("/api/compare", { ...post(stream, { "Content-Length": "2" }), duplex: "half" });
  assert.equal((await proxyRequest(req, env, unused)).status, 413);
});

test("invalid JSON, media type, and cross-origin browser submissions are rejected", async () => {
  for (const [init, status] of [[post("[]"), 400], [post("not json"), 400], [post("{}", { "Content-Type": "text/plain" }), 415], [post("{}", { Origin: "https://unrelated.example" }), 403]]) {
    assert.equal((await proxyRequest(request("/api/compare", init), env, unused)).status, status);
  }
});

test("upstream internal errors and tunnel HTML do not leak", async () => {
  for (const upstream of [new Response("/private/path secret=hidden", { status: 500 }), new Response("<html>Temporary tunnel</html>")]) {
    const response = await proxyRequest(request("/api/example"), env, async () => upstream);
    assert.equal(response.status, 503);
    assert.doesNotMatch(await response.text(), /private|hidden|<html>|test-only-token/);
  }
});

test("upstream backpressure retains a bounded retry interval", async () => {
  const response = await proxyRequest(request("/api/compare", post()), env, async () => new Response("internal detail", { status: 429, headers: { "Retry-After": "99999" } }));
  assert.equal(response.status, 429);
  assert.equal(response.headers.get("Retry-After"), "3600");
  assert.doesNotMatch(await response.text(), /internal detail/);
});

test("timeout produces a friendly 504", async () => {
  const response = await proxyRequest(request("/api/compare", post()), env, (_, init) => new Promise((_, reject) => init.signal.addEventListener("abort", () => reject(new DOMException("upstream detail", "AbortError")), { once: true })), 10);
  assert.equal(response.status, 504);
  assert.doesNotMatch(await response.text(), /upstream detail/);
});

test("slow request bodies also time out without making upstream calls", async () => {
  const stream = new ReadableStream({ start() {} });
  const response = await proxyRequest(request("/api/compare", { ...post(stream), duplex: "half" }), env, unused, 10);
  assert.equal(response.status, 504);
});
