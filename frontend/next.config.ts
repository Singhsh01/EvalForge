import type { NextConfig } from "next";

// The browser talks to the Next server only; /api/* is proxied to FastAPI.
// This avoids CORS in development and keeps the backend off the public port in compose.
const API_URL = process.env.EVALFORGE_API_URL ?? "http://127.0.0.1:8000";

const config: NextConfig = {
  output: "standalone",
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${API_URL}/api/:path*` }];
  },
};

export default config;
