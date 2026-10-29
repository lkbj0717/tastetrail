import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { validatePublicOrigin, renderDemoHtml, demoCsp } from "../worker/public-config.mjs";
test("only bare HTTPS origins are accepted", () => {
 for (const value of ["http://example.com", "https://user:pass@example.com", "https://example.com/path", "https://example.com/?q=x"]) assert.throws(() => validatePublicOrigin(value));
 assert.equal(validatePublicOrigin("https://example.com"), "https://example.com");
});
test("production UI routes every API call to the public CORS endpoint without credentials", () => {
 const template=readFileSync(new URL("../worker/assets/demo.html",import.meta.url),"utf8");
 const html=renderDemoHtml(template,"https://example.com");
 assert.ok(html.includes('const API_BASE = "https://example.com";'));
 assert.ok(!html.includes("__TASTETRAIL_PUBLIC_API_ORIGIN__"));
 const calls=html.match(/fetch\([^;]+/g);
 assert.equal(calls.length,2);
 for(const call of calls){assert.match(call,/fetch\(API_BASE\+/);assert.match(call,/credentials:'omit'/);}
 assert.ok(!html.includes("Authorization"));
 assert.match(demoCsp("https://example.com"), /connect-src 'self' https:\/\/example.com;/);
});
