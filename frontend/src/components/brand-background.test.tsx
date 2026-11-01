import { render, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { BrandBackground } from "./brand-background";

const ASSET = {
  width: 2560, height: 1440, dominant: "rgb(8 24 40)", placeholder: "data:image/webp;base64,AAAA",
  files: [
    { src: "/brand/hero-background-960.avif", width: 960, format: "avif" },
    { src: "/brand/hero-background-960.webp", width: 960, format: "webp" },
    { src: "/brand/hero-background-1600.webp", width: 1600, format: "webp" },
  ],
};

afterEach(() => { vi.unstubAllGlobals(); vi.unstubAllEnvs(); });

describe("optional Higgsfield background", () => {
  it("renders nothing with the default empty manifest (app works without Higgsfield)", async () => {
    const f = vi.fn().mockResolvedValue(new Response(JSON.stringify({ assets: {} })));
    vi.stubGlobal("fetch", f);
    const { container } = render(<BrandBackground />);
    await waitFor(() => expect(f).toHaveBeenCalledWith("/brand/manifest.json", expect.anything()));
    expect(container).toBeEmptyDOMElement();
  });

  it("renders nothing when the manifest is missing or the request fails", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("network")));
    const { container } = render(<BrandBackground />);
    await new Promise((r) => setTimeout(r, 10));
    expect(container).toBeEmptyDOMElement();
  });

  it("serves AVIF first, then WebP, under a readability scrim", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ assets: { "hero-background": ASSET } }))));
    const { container } = render(<BrandBackground />);
    await waitFor(() => expect(container.querySelector("picture")).not.toBeNull());
    const sources = container.querySelectorAll("source");
    expect(sources[0]).toHaveAttribute("type", "image/avif");
    expect(sources[1].getAttribute("srcset")).toContain("hero-background-1600.webp 1600w");
    expect(container.querySelector("img")).toHaveAttribute("alt", "");
    expect(container.querySelector("[class*='linear-gradient']")).not.toBeNull();
  });

  it("can be switched off entirely", async () => {
    vi.stubEnv("NEXT_PUBLIC_EVALFORGE_BRAND_ASSETS", "off");
    const f = vi.fn();
    vi.stubGlobal("fetch", f);
    render(<BrandBackground />);
    await new Promise((r) => setTimeout(r, 10));
    expect(f).not.toHaveBeenCalled();
  });
});
