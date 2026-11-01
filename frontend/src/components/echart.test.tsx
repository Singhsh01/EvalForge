import { render } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

const chart = { setOption: vi.fn(), resize: vi.fn(), dispose: vi.fn(), on: vi.fn(), off: vi.fn() };
vi.mock("echarts/core", () => ({ init: vi.fn(() => chart), use: vi.fn() }));

const { EChart } = await import("./echart");

describe("EChart wrapper", () => {
  it("applies the option with the dark theme defaults and is accessible", () => {
    const { getByRole } = render(<EChart ariaLabel="PSI heatmap" option={{ series: [{ type: "heatmap" }] }} />);
    expect(getByRole("img", { name: "PSI heatmap" })).toBeInTheDocument();
    const [opt, notMerge] = chart.setOption.mock.calls.at(-1)!;
    expect(opt).toMatchObject({ backgroundColor: "transparent", animation: true, series: [{ type: "heatmap" }] });
    expect(notMerge).toBe(true);
  });

  it("turns animation off under prefers-reduced-motion", () => {
    const mm = window.matchMedia;
    window.matchMedia = ((q: string) => ({ matches: q.includes("reduce") }) as MediaQueryList);
    render(<EChart ariaLabel="x" option={{}} />);
    expect(chart.setOption.mock.calls.at(-1)![0].animation).toBe(false);
    window.matchMedia = mm;
  });

  it("wires click handlers and disposes the chart on unmount", () => {
    const onClick = vi.fn();
    const { unmount } = render(<EChart ariaLabel="x" option={{}} onClick={onClick} />);
    expect(chart.on).toHaveBeenCalledWith("click", onClick);
    unmount();
    expect(chart.dispose).toHaveBeenCalled();
  });
});
