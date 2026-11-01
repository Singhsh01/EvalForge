import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

const replace = vi.fn();
vi.mock("next/navigation", () => ({ useRouter: () => ({ replace }) }));
const { default: LoginPage } = await import("./page");
const { getToken, getUser } = await import("@/lib/api");

afterEach(() => { vi.unstubAllGlobals(); localStorage.clear(); replace.mockReset(); });

function fill(user: string, pw: string) {
  fireEvent.change(screen.getByLabelText("Username"), { target: { value: user } });
  fireEvent.change(screen.getByLabelText("Password"), { target: { value: pw } });
  fireEvent.click(screen.getByRole("button", { name: "Sign in" }));
}

describe("sign-in page", () => {
  it("stores the session and goes to the overview", async () => {
    const f = vi.fn().mockResolvedValue(new Response(JSON.stringify({
      token: "efk_t", user: { username: "approver", role: "approver" } })));
    vi.stubGlobal("fetch", f);
    render(<LoginPage />);
    fill("approver", "evalforge-dev");
    await waitFor(() => expect(replace).toHaveBeenCalledWith("/"));
    expect(JSON.parse(f.mock.calls[0][1].body)).toEqual({ username: "approver", password: "evalforge-dev" });
    expect(getToken()).toBe("efk_t");
    expect(getUser()).toEqual({ username: "approver", role: "approver" });
  });

  it("shows the server's error and stays on the page", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(
      JSON.stringify({ detail: "Invalid username or password" }), { status: 401 })));
    Object.defineProperty(window, "location", { value: { pathname: "/login", href: "/login" }, configurable: true });
    render(<LoginPage />);
    fill("admin", "wrong");
    expect(await screen.findByRole("alert")).toHaveTextContent("Invalid username or password");
    expect(replace).not.toHaveBeenCalled();
    expect(getToken()).toBeNull();
  });

  it("shows the rate-limit message after too many attempts", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(
      JSON.stringify({ detail: "Rate limit exceeded" }), { status: 429 })));
    render(<LoginPage />);
    fill("admin", "x");
    expect(await screen.findByRole("alert")).toHaveTextContent("Rate limit exceeded");
  });
});
