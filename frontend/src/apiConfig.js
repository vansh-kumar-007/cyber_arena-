/**
 * Resolve the API base URL once per frontend build.
 *
 * Local/test builds use localhost unless explicitly configured. Production builds
 * default to the backend advertised by the repository README instead of trying
 * to contact the end user's own machine.
 */
export function resolveApiBaseUrl({
  nodeEnv = process.env.NODE_ENV,
  configuredUrl = process.env.REACT_APP_API_URL,
} = {}) {
  const trimmed = typeof configuredUrl === "string" ? configuredUrl.trim() : "";
  const fallback =
    nodeEnv === "development" || nodeEnv === "test"
      ? "http://localhost:8000"
      : "https://cyberarena-api.onrender.com";
  return (trimmed || fallback).replace(/\/+$/, "");
}
