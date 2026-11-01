import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError, api, getToken, getUser, setSession } from "./api";

function respond(status: number, body: unknown) {
  return vi.fn().mockResolvedValue(
    new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } }),
  );
}

describe("api client", () => {
  beforeEach(() => {
    localStorage.clear();
    window.history.replaceState(null, "", "/registry");
  });
  afterEach(() => vi.unstubAllGlobals());

  it("sends the stored bearer token", async () => {
    setSession("efk_abc", { username: "engineer", role: "engineer" });
    const f = respond(200, []);
    vi.stubGlobal("fetch", f);
    await api.datasets();
    const headers = f.mock.calls[0][1].headers as Headers;
    expect(headers.get("authorization")).toBe("Bearer efk_abc");
  });

  it("clears the session and redirects to /login on 401", async () => {
    setSession("efk_old", { username: "viewer", role: "viewer" });
    vi.stubGlobal("fetch", respond(401, { detail: "Invalid or expired token" }));
    const loc = { pathname: "/registry", href: "/registry" };
    vi.stubGlobal("location", loc);
    Object.defineProperty(window, "location", { value: loc, configurable: true });
    await expect(api.datasets()).rejects.toBeInstanceOf(ApiError);
    expect(getToken()).toBeNull();
    expect(getUser()).toBeNull();
    expect(loc.href).toBe("/login");
  });

  it("does not redirect when the login call itself returns 401", async () => {
    vi.stubGlobal("fetch", respond(401, { detail: "Invalid username or password" }));
    const loc = { pathname: "/login", href: "/login" };
    Object.defineProperty(window, "location", { value: loc, configurable: true });
    await expect(api.login("admin", "nope")).rejects.toThrow("Invalid username or password");
    expect(loc.href).toBe("/login");
  });

  it("turns FastAPI validation errors into one readable message", async () => {
    vi.stubGlobal(
      "fetch",
      respond(422, { detail: [{ msg: "Field required" }, { msg: "Input should be a number" }] }),
    );
    const err = await api.datasets().catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err.status).toBe(422);
    expect(err.message).toBe("Field required; Input should be a number");
  });

  it("surfaces 429 rate-limit errors with their status", async () => {
    vi.stubGlobal("fetch", respond(429, { detail: "Rate limit exceeded" }));
    const err = await api.datasets().catch((e) => e);
    expect(err.status).toBe(429);
    expect(err.message).toBe("Rate limit exceeded");
  });
});
