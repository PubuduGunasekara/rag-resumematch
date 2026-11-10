// The browser only ever talks to this Next.js server; /api/* is proxied to
// the Python API (uvicorn resumematch.api.app:app), so there's no CORS to
// configure and the API never has to be public on its own.
const API_URL = process.env.API_URL || "http://127.0.0.1:8000";

/** @type {import('next').NextConfig} */
const nextConfig = {
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${API_URL}/api/:path*` }];
  },
  // An analysis takes ~20 s warm and up to a minute cold; the default
  // proxy timeout (30 s) would cut the first one off.
  experimental: { proxyTimeout: 120_000 },
};

export default nextConfig;
