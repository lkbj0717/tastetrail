# Deploying the TasteTrail demo

The public application compares two films using eight stated preferences. It uses
the **frozen v13 algorithm** in `work/culture_v13` and shared components in
`work/culture_v11`. The wrapper in `deploy/` adds HTTP access and admission limits;
it does not change prompts, features, ranking, or evaluation results.

## Runtime prerequisites

- Linux and Python 3.12; a CPU is sufficient. Two CPU cores and 2 GB RAM are a
  practical starting allocation. No GPU, PyTorch, or scikit-learn is needed.
- A Qloo hackathon API key and an accessible OpenAI-compatible GLM service.
  The recorded experiment used `glm-5.3-flash` at the endpoint shown in
  `.env.example`; access to that third-party deployment is not bundled.
- Private runtime assets built from MovieLens 1M and Qloo. Raw ratings, API
  responses, credentials, and background-user matrices are deliberately excluded
  from Git and Docker images. Obtain the datasets under their own terms.
- Persistent writable storage for private runtime files and response caches.

Keep this directory layout on the server:

```text
/srv/tastetrail-private/
  culture_v11/
    catalog.json
    metadata.json
    tools.npz
  culture_v13/
    qloo_metadata.json
    mixture_tools/bernoulli_k{12,24}_{plain,cf,qloo}.npz
  qloo_cache/
/srv/tastetrail-secrets/credentials.json
```

The two versioned private directories **must be siblings**. Set `V13_PRIVATE` to
the v13 directory and `V11_PRIVATE` to its neighboring `culture_v11`. Set
`TASTETRAIL_PRIVATE_DIR` to their parent for the default Qloo cache. Use absolute
paths. The live service writes new response records into the private v13 folder.
The optional `qloo_graph_tools.npz` is used by graph experiments and can be built
with `--with-graph`; it is not required by the selected live mixture method.

## Credentials

Create a server-only JSON file with mode `600`. This example contains placeholders:

```json
{
  "QLOO_API_KEY": "YOUR_QLOO_HACKATHON_KEY",
  "TASTETRAIL_MODEL_PROVIDER": "glm",
  "GLM_API_KEY": "YOUR_MODEL_KEY",
  "GLM_BASE_URL": "https://openclaw-v2.top/llm/v1",
  "GLM_MODEL": "glm-5.3-flash"
}
```

Point `TASTETRAIL_CREDENTIALS` to this file. Environment variables override JSON
values, including empty values, so do not export empty API-key placeholders when
using a credential file. The Qloo server is fixed by the research client to
`https://hackathon.api.qloo.com`. Neither provider key is returned to the browser.

## Prepare private runtime assets

From the public repository root:

```sh
python3.12 -m venv .venv
.venv/bin/pip install -r work/culture_v13/requirements-runtime-lock.txt
export PYTHONPATH="$PWD:$PWD/work/culture_v13:$PWD/work/culture_v11:$PWD/work/culture_v11/src"
export OPENBLAS_NUM_THREADS=2
export V11_PRIVATE=/srv/tastetrail-private/culture_v11
export V13_PRIVATE=/srv/tastetrail-private/culture_v13
export TASTETRAIL_PRIVATE_DIR=/srv/tastetrail-private
export TASTETRAIL_CREDENTIALS=/srv/tastetrail-secrets/credentials.json
```

If an authorized private snapshot is already available, mount or copy it into
these private locations and run the **offline** check:

```sh
.venv/bin/python work/culture_v13/bootstrap_runtime.py --check \
  --v11-private "$V11_PRIVATE" --private "$V13_PRIVATE" \
  --cache-dir "$TASTETRAIL_PRIVATE_DIR/qloo_cache"
```

For a fresh installation, explicitly run:

```sh
.venv/bin/python work/culture_v11/download.py
.venv/bin/python work/culture_v11/bootstrap_runtime.py
.venv/bin/python work/culture_v13/bootstrap_runtime.py --build --allow-network \
  --v11-private "$V11_PRIVATE" --private "$V13_PRIVATE" \
  --cache-dir "$TASTETRAIL_PRIVATE_DIR/qloo_cache"
```

These commands download MovieLens 1M from GroupLens, resolve 473 movie identities
against Qloo, build numerical tools using 3,640 separate background users, then
build six soft-cluster models and whitelisted Qloo cultural metadata. The v11
identity step typically makes roughly 473–946 Qloo search requests. The v13
metadata step typically needs about 11 batched requests, with more if results
are missing. This setup uses Qloo quota but **does not invoke the language model**.
Reuse a valid cache where possible. Existing runtime files are not overwritten;
for an interrupted partial v11 build, preserve it and select a fresh directory.

The startup script never runs these downloads or rebuilds automatically. A new
build may differ from the historical private snapshot as Qloo changes; it does
not independently reproduce the recorded 63% / 68% / 70% experiment. Exact replay
would also require the original split and API response records, which are not
published in this repository. Consult `work/culture_v13/RUN_GUIDE.txt` for further
rebuild details and dataset limitations.

## Run directly on a server

```sh
export TASTETRAIL_PYTHON="$PWD/.venv/bin/python"
export TASTETRAIL_HOST=127.0.0.1
export PORT=8773
sh deploy/run.sh
```

`run.sh` performs an offline asset check, then starts one Uvicorn worker. By
default, the application is bound only to loopback. Browse
`http://127.0.0.1:8773` locally, or use an HTTPS reverse proxy. Use a persistent
process manager such as systemd on the host for an event deployment. Keep a
generous upstream read timeout for multi-stage model calls; the original service
can spend several minutes on a request. The HTML reports provider failures
rather than synthesizing successful recommendations.

`GET /health` returns only `{"ready":true}` or `{"ready":false}` and invokes no
provider. It checks local file/credential presence; it does not guarantee that an
external provider is reachable or that its balance is sufficient. The startup
check validates asset structure. Do not treat a health response as a paid API
smoke test.

## Docker

```sh
docker build -t tastetrail:v13 .
docker run --rm --name tastetrail \
  --user "$(id -u):$(id -g)" \
  --read-only --tmpfs /tmp:rw,noexec,nosuid,size=64m \
  -p 127.0.0.1:8773:8773 \
  --mount type=bind,src=/srv/tastetrail-private,dst=/data \
  --mount type=bind,src=/srv/tastetrail-secrets/credentials.json,dst=/run/secrets/tastetrail.json,readonly \
  --env-file .env \
  tastetrail:v13
```

Copy `.env.example` to `.env` first and configure the server-only settings. The
example `--user` runs with the host owner's UID/GID, so private files with mode
600 remain readable and the private volume remains writable. Without that
override, the image runs as UID/GID 10001, which must own the mounted private
storage. The image includes source and dependencies only. Do not copy private
assets or credentials into the Docker build context. The image starts with an
offline check and exits if runtime assets are incomplete.

## Connect a public frontend through a server proxy

For a Sites Worker or other server-side reverse proxy, generate a random shared
secret and store the same value in the backend's `TASTETRAIL_GATEWAY_TOKEN` and
the proxy's secret store. The proxy attaches:

```text
Authorization: Bearer <server-only shared secret>
```

The browser sends requests to its own origin. The proxy forwards the allowlisted
paths to the backend, replacing any user-supplied Authorization header. It must
never include the shared secret in HTML, frontend JavaScript, error messages, or
public configuration. Configure HTTPS for both public access and the proxy-to-
backend connection. Set upstream timeouts appropriate for multi-stage inference.

Allowed routes are:

| Method | Routes | Gateway token when configured |
| --- | --- | --- |
| GET | `/` | No; static demonstration HTML only |
| GET / HEAD | `/health`, `/api/health` | No; local readiness boolean only |
| GET | `/api/catalog`, `/catalog`, `/api/example`, `/example`, `/api/research` | Yes |
| POST | `/api/recommend`, `/api/compare` | Yes |

All other routes, including OpenAPI documentation, return 404. CORS is disabled
unless exact origins are configured as described below. A token-protected backend's HTML needs the server proxy
to load its data; do not paste the shared secret into the browser to bypass it.
If `TASTETRAIL_GATEWAY_TOKEN` is empty, the backend can be used directly and all
listed data/recommendation routes are public. Only use that mode when you intend
to cover visitors' API costs, or keep the service behind your own access control.

## Optional direct browser access from the public Site

If the public Site cannot reach the backend through its server-side proxy, its
browser frontend can call the HTTPS backend directly. This mode uses a public
backend with `TASTETRAIL_GATEWAY_TOKEN` empty and a CORS allowlist containing the
**exact origin of the Site**, for example:

```sh
export TASTETRAIL_GATEWAY_TOKEN=
export TASTETRAIL_ALLOWED_ORIGINS=https://your-site.example.com
```

Configure the frontend's public API base URL to the HTTPS backend or tunnel URL.
Only that address is public configuration; no provider key or shared gateway
token belongs in the browser. Restart the backend after changing its allowlist.
Multiple origins can be comma-separated. Wildcards, URL paths, query strings,
embedded credentials, and non-HTTPS origins are rejected. An HTTP localhost
origin such as `http://localhost:3000` is accepted only when explicitly listed
for local development.

The middleware permits GET/POST and the JSON `Content-Type` header, handles
preflight without invoking inference, exposes `Retry-After`, and caches allowed
preflight for 600 seconds. It does not allow browser credentials. Allowed-origin
responses include CORS headers even on 401/429 errors, allowing the UI to show
authentication or quota errors accurately. The original admission limits and
optional Bearer authentication still apply.

CORS controls which browser origins can read responses; it is not authentication
or a spending limit. In this public mode, non-browser clients can also call the
allowlisted backend routes. The one-request concurrency gate and hourly allowance
remain active. Keep the public backend/tunnel alive for the judging period; if a
temporary tunnel URL changes, update the frontend API base URL. Use the server
proxy mode above when a shared gateway token is required.

## Admission limits and operation

- Exactly one inference request is admitted at a time. Concurrent paid requests
  receive HTTP 429 immediately with `Retry-After`; they do not accumulate behind
  the original service lock.
- A global rolling allowance defaults to **30 admitted POST requests per hour**.
  Adjust `TASTETRAIL_REQUESTS_PER_HOUR` for your event budget. The counter includes
  admitted requests even if downstream validation or provider calls fail.
- Requests require JSON and have a 16 KiB size limit, including chunked bodies.
  Set `TASTETRAIL_MAX_BODY_BYTES` if required. Uploading a body is limited to
  10 seconds and occupies the one admission slot.
- These controls are per process and reset on restart. Use **one worker and one
  instance**, or enforce a shared quota in the reverse proxy when scaling.
- Existing per-request model accounting and recovery limits remain in the
  frozen service. An hourly request count is not a currency budget; configure
  upstream account spending limits separately if supported by your provider.
- Keep private storage outside the web root, back it up under your own access
  controls, and avoid logging request bodies or Authorization headers. The run
  script disables access logs. Provider errors are sanitized by the frozen app.

Run offline wrapper tests with:

```sh
.venv/bin/python -m pytest -q tests/test_public_gateway.py
```

The tests use synthetic inputs and mocked application results. They verify the
authentication boundary, route allowlist, chunked-body size checks, overload
rejection, hourly admissions, and unchanged forwarding into the original input
validation. They make no calls to Qloo or the model provider.
