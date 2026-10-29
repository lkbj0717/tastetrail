// This is the public API origin, never a provider credential.
export const PUBLIC_BACKEND_ORIGIN = "https://exploring-rebound-sharp-serum.trycloudflare.com";
export function validatePublicOrigin(value) {
  const url = new URL(value);
  if (url.protocol !== "https:" || url.username || url.password || url.pathname !== "/" || url.search || url.hash) throw new Error("Expected a public HTTPS origin");
  return url.origin;
}
export function renderDemoHtml(template, origin = PUBLIC_BACKEND_ORIGIN) {
  const placeholder = "__TASTETRAIL_PUBLIC_API_ORIGIN__";
  if (!template.includes(placeholder)) throw new Error("Missing public API configuration slot");
  return template.replace(placeholder, JSON.stringify(validatePublicOrigin(origin)));
}
export function demoCsp(origin = PUBLIC_BACKEND_ORIGIN) {
  return `default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self' ${validatePublicOrigin(origin)}; img-src 'self' data:; base-uri 'none'; form-action 'self'; frame-ancestors 'none'`;
}
