# Public demo operation

The stable public entry point is https://tastetrail-qloo-agent.shared-json.chatgpt.site.
The public source repository is https://github.com/lkbj0717/tastetrail.

The hosted page calls the Python HTTPS backend directly from the browser. The
backend explicitly allows this Site origin through CORS. Provider credentials,
private numerical assets, and API caches remain on the application server.
No provider key is shipped to the frontend.

The deployment uses two user-level systemd services:

```sh
systemctl --user status tastetrail-public-backend.service
systemctl --user status tastetrail-public-tunnel.service
```

Both are enabled at user-session startup, with restart-on-failure behavior;
user lingering is enabled on the serving machine. The active deployment permits
one inference at a time and 120 admitted POST requests per hour. GET and OPTIONS
requests do not invoke the model. The repository's generic default is 30/hour.

Keep the server and the tunnel running throughout judging. The current backend
uses a Cloudflare Quick Tunnel. It reconnects after temporary network loss, but a
new tunnel process may receive a different hostname. If that happens, update the
public backend-origin configuration in `web/worker/public-config.mjs`, synchronize
the hosted source, rebuild, and redeploy the existing Site; do not create a new
Site or change the public entry URL. Update the backend CORS allowlist only if the
frontend origin changes. A named tunnel or dedicated application host is the
longer-term replacement for this temporary backend transport.

The official judging period ends on November 5, 2026 at 11:45 PM Eastern Time
(November 6 at 12:45 PM in Beijing). Availability depends on this server, network,
and the model/Qloo services remaining operational. A successful test does not
promise uninterrupted future availability.
