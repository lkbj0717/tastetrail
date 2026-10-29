# Optional hosted frontend

The active public page calls the HTTPS Python backend directly from the browser.
Set the public address in `worker/public-config.mjs` and allow the exact frontend
origin with `TASTETRAIL_ALLOWED_ORIGINS` on the backend. Provider keys stay on the
server. CORS requests omit credentials. This is the deployed transport; the
optional server proxy remains in the source for other hosting environments.

The Python server already serves its own complete UI, so this frontend is optional
for self-hosting. This portable copy excludes the hosted Site identity. Create a
new hosting manifest from `.openai/hosting.template.json` when using Sites.

## What runs where

```
Browser → frontend Worker → authenticated Python gateway → frozen v13 service
                                                       ↳ GLM and Qloo
```

`worker/assets/demo.html` is the real v13 English interface, with only local-hosting wording adapted. It requests the actual API; no recommendation responses are simulated. `worker/proxy.mjs` adds the server gateway credential and forwards only the approved endpoints. The existing Vinext template and its `sites()` integration are retained. Source licenses for bundled template dependencies are preserved.

## Run or adapt this copy

Use Node.js 22.13 or newer. No owner-specific Site identity or runtime secrets are included. First create a local hosting configuration:

```sh
cp .openai/hosting.template.json .openai/hosting.json
npm run install:ci
node --test tests/proxy.test.mjs
npm run build
```

The template omits `project_id`; it is sufficient for a local build. To publish a new Site through the Sites plugin, register your own Site and set its returned project ID in the local configuration. Do not reuse another deployment's identity. The generated `.openai/hosting.json` is ignored. Follow the hosting platform's supported source synchronization and publishing workflow.

Configure these **server-side** bindings on the frontend host:

- `TASTETRAIL_BACKEND_URL`: the Python gateway's HTTPS origin, without a path, query, fragment, or embedded credentials.
- `TASTETRAIL_GATEWAY_TOKEN`: the shared token also configured on that Python gateway.

For local Worker testing, use an ignored `.dev.vars` file with the same names. `.env.example` provides names only. GLM and Qloo API keys stay on the Python server and are never required by browser JavaScript. The frontend returns a clear 503 response until both bindings are configured.

For frontend development use `npm run dev`; for a built local Worker use `npm start`. A separately running HTTPS backend is required for real recommendations. No private model artifacts, API caches, evaluation users, or runtime credentials are distributed here.

## Transport and deployment limits

The proxy supports `GET /health`, `/api/health`, `/api/catalog`, `/api/example`, `/api/research`, and `POST /api/recommend`, `/api/compare`. It enforces a streamed 16 KiB request limit, JSON object inputs, fixed upstream origin, redirect rejection, safe error responses, and a 110-second timeout. The Python gateway enforces inference concurrency and hourly admissions. The proxy does not modify prediction inputs or outputs.

A temporary tunnel's availability depends on the gateway, tunnel process, and underlying server remaining online. Restarting a quick tunnel can change its origin and require a frontend binding update. A stable frontend URL alone does not make the backend persistent.

The research prototype supports 473 historical films. Independent 100-case confirmation produced 63% for GLM, 68% for Agent, and 70% for Agent + Qloo. The two-point Qloo improvement over Agent was not statistically significant. See the repository experiment documentation for scope and limitations.

### Runtime secret access

The server accepts ordinary Worker string bindings, asynchronous Cloudflare Secrets Store bindings through `get()`, and a server-only `process.env` fallback. Configuration failures emit only binding types and presence flags to application logs; secret values and upstream error text are never logged or sent to browsers.
