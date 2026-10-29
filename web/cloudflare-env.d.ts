declare namespace Cloudflare {
  interface Env {
    DB?: D1Database;
    BUCKET?: R2Bucket;
    TASTETRAIL_BACKEND_URL?: string | { get(): Promise<string> };
    TASTETRAIL_GATEWAY_TOKEN?: string | { get(): Promise<string> };
  }
}

declare module "*.html?raw" {
  const html: string;
  export default html;
}
