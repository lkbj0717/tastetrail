// Transport boundary only: prediction and prompts stay in the frozen Python service.
export const MAX_BODY_BYTES = 16 * 1024;
const MAX_RESPONSE_BYTES = 2 * 1024 * 1024;
const GET_PATHS = new Set(["/health", "/api/health", "/api/catalog", "/api/example", "/api/research"]);
const POST_PATHS = new Set(["/api/recommend", "/api/compare"]);
const SAFE_HEADERS = {
  "Content-Type": "application/json; charset=utf-8",
  "Cache-Control": "no-store",
  "X-Content-Type-Options": "nosniff",
  "Referrer-Policy": "no-referrer",
};

function error(status, detail, retryAfter) {
  const headers = { ...SAFE_HEADERS };
  if (retryAfter) headers["Retry-After"] = String(retryAfter);
  return new Response(JSON.stringify({ detail }), { status, headers });
}

async function readLimited(stream, limit, signal) {
  if (!stream) return new Uint8Array();
  const reader = stream.getReader();
  const chunks = [];
  let length = 0;
  let abortListener;
  const aborted = new Promise((_, reject) => {
    abortListener = () => reject(new DOMException("Timed out", "AbortError"));
    signal.addEventListener("abort", abortListener, { once: true });
    if (signal.aborted) abortListener();
  });
  try {
    while (true) {
      const { done, value } = await Promise.race([reader.read(), aborted]);
      if (done) break;
      length += value.byteLength;
      if (length > limit) throw new RangeError("Body limit exceeded");
      chunks.push(value);
    }
    const bytes = new Uint8Array(length);
    let offset = 0;
    for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.byteLength; }
    return bytes;
  } finally {
    signal.removeEventListener("abort", abortListener);
    reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}

function backendOrigin(value) {
  try {
    const url = new URL(value);
    if (url.protocol !== "https:" || url.username || url.password || url.search || url.hash || url.pathname !== "/") return null;
    return url.origin;
  } catch { return null; }
}

function processValues() {
  // Runtime-only server lookup. These values are never imported by browser code.
  return typeof process !== "undefined" ? process.env : {};
}

async function bindingValue(value) {
  if (typeof value === "string") return value;
  // Cloudflare account-level Secrets Store exposes a binding with async get().
  // Ordinary Worker secrets and local .dev.vars expose a string instead.
  if (value && typeof value.get === "function") {
    const result = await value.get();
    return typeof result === "string" ? result : undefined;
  }
  return undefined;
}

export async function runtimeConfiguration(env = {}, fallback = processValues()) {
  const names = ["TASTETRAIL_BACKEND_URL", "TASTETRAIL_GATEWAY_TOKEN"];
  const values = await Promise.all(names.map(async name => {
    const binding = await bindingValue(env?.[name]);
    return typeof binding === "string" && binding.trim() ? binding : await bindingValue(fallback?.[name]);
  }));
  return Object.fromEntries(names.map((name, index) => [name, values[index]]));
}

function configurationShape(env) {
  // Diagnostics are types/booleans only: never keys, origins, contents, or errors.
  const fallback = processValues();
  const shape = value => typeof value === "string" ? "string" : value && typeof value.get === "function" ? "secret-store" : typeof value;
  return {
    backendBindingType: shape(env?.TASTETRAIL_BACKEND_URL),
    tokenBindingType: shape(env?.TASTETRAIL_GATEWAY_TOKEN),
    backendProcessValuePresent: typeof fallback?.TASTETRAIL_BACKEND_URL === "string",
    tokenProcessValuePresent: typeof fallback?.TASTETRAIL_GATEWAY_TOKEN === "string",
  };
}

export async function proxyRequest(request, env, fetcher = fetch, timeoutMs = 110_000) {
  const url = new URL(request.url);
  const isGet = request.method === "GET" && GET_PATHS.has(url.pathname);
  const isPost = request.method === "POST" && POST_PATHS.has(url.pathname);
  if (!isGet && !isPost) return error(404, "Not found.");
  if (url.search && url.pathname !== "/api/catalog") return error(400, "This endpoint does not accept query parameters.");
  if (url.search.length > 1024) return error(400, "Query is too long.");

  let configuration;
  let configurationTimer;
  try {
    configuration = await Promise.race([
      runtimeConfiguration(env),
      new Promise((_, reject) => { configurationTimer = setTimeout(() => reject(new Error("Configuration timeout")), 5000); }),
    ]);
  } catch {
    console.error("tastetrail_runtime_configuration_unavailable", configurationShape(env));
    return error(503, "The recommendation service is being connected. Please try again shortly.");
  } finally { clearTimeout(configurationTimer); }
  const origin = backendOrigin(configuration.TASTETRAIL_BACKEND_URL);
  const token = configuration.TASTETRAIL_GATEWAY_TOKEN;
  if (!origin || typeof token !== "string" || !token.trim()) {
    console.error("tastetrail_runtime_configuration_unavailable", { ...configurationShape(env), originValid: !!origin, tokenPresent: typeof token === "string" && !!token.trim() });
    return error(503, "The recommendation service is being connected. Please try again shortly.");
  }
  if (isPost) {
    const contentType = request.headers.get("Content-Type")?.split(";", 1)[0].trim().toLowerCase();
    if (contentType !== "application/json") return error(415, "Send application/json.");
    const browserOrigin = request.headers.get("Origin");
    if (browserOrigin && browserOrigin !== url.origin) return error(403, "Please submit from the TasteTrail website.");
    const declared = request.headers.get("Content-Length");
    if (declared !== null && (!/^\d+$/.test(declared) || Number(declared) > MAX_BODY_BYTES)) return error(413, "Request body is too large or has an invalid length.");
  }

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  let phase = "request";
  try {
    let body;
    if (isPost) {
      body = await readLimited(request.body, MAX_BODY_BYTES, controller.signal);
      try {
        const parsed = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(body));
        if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) throw new Error();
      } catch { return error(400, "Expected a JSON object."); }
    }
    phase = "upstream";
    const target = new URL(url.pathname, origin);
    if (url.pathname === "/api/catalog") target.search = url.search;
    const response = await fetcher(target.toString(), {
      method: request.method,
      headers: { "Accept": "application/json", "Authorization": `Bearer ${token}`, ...(isPost ? { "Content-Type": "application/json" } : {}) },
      body,
      redirect: "error",
      signal: controller.signal,
    });
    if (!response.ok) {
      response.body?.cancel().catch(() => {});
      if (response.status === 429) {
        const retry = Number(response.headers.get("Retry-After"));
        return error(429, "The demo is busy or its hourly allowance has been reached. Please retry later.", Number.isFinite(retry) && retry > 0 ? Math.min(Math.ceil(retry), 3600) : 10);
      }
      if (response.status === 422 || response.status === 400) return error(response.status, "Please check the ten distinct film titles, years, and ratings, then try again.");
      if (response.status === 413) return error(413, "Request body is too large.");
      return error(503, "The recommendation service is temporarily unavailable. Please try again shortly.");
    }
    const bytes = await readLimited(response.body, MAX_RESPONSE_BYTES, controller.signal);
    try { JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(bytes)); }
    catch { return error(503, "The recommendation service returned an incomplete response. Please try again."); }
    return new Response(bytes, { status: response.status, headers: SAFE_HEADERS });
  } catch (exception) {
    if (controller.signal.aborted || exception?.name === "AbortError") return error(504, "The comparison took too long. Please wait a moment and try again.");
    if (phase === "request" && exception instanceof RangeError) return error(413, "Request body is too large.");
    return error(503, "The recommendation service is temporarily unavailable. Please try again shortly.");
  } finally { clearTimeout(timeout); }
}
