import { resolveApiBaseUrl } from "./apiConfig";

test("production uses the documented backend instead of a visitor localhost URL", () => {
  expect(resolveApiBaseUrl({ nodeEnv: "production", configuredUrl: "" })).toBe(
    "https://cyberarena-api.onrender.com"
  );
});

test("development and tests retain the localhost default", () => {
  expect(resolveApiBaseUrl({ nodeEnv: "development", configuredUrl: "" })).toBe(
    "http://localhost:8000"
  );
  expect(resolveApiBaseUrl({ nodeEnv: "test", configuredUrl: undefined })).toBe(
    "http://localhost:8000"
  );
});

test("explicit deployment configuration takes precedence and normalizes trailing slashes", () => {
  expect(
    resolveApiBaseUrl({
      nodeEnv: "production",
      configuredUrl: " https://api.example.test/ ",
    })
  ).toBe("https://api.example.test");
});
