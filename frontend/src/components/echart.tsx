"use client";

// Thin, tree-shaken ECharts wrapper: one canvas per chart, resizes with its container,
// disposes on unmount, honours prefers-reduced-motion and follows the app's dark palette.
import { BarChart, HeatmapChart, LineChart, ScatterChart } from "echarts/charts";
import {
  DataZoomComponent, GridComponent, LegendComponent, MarkLineComponent, TooltipComponent, VisualMapComponent,
} from "echarts/components";
import * as echarts from "echarts/core";
import { CanvasRenderer } from "echarts/renderers";
import { useEffect, useRef } from "react";

echarts.use([BarChart, HeatmapChart, LineChart, ScatterChart, GridComponent, LegendComponent, TooltipComponent,
  VisualMapComponent, MarkLineComponent, DataZoomComponent, CanvasRenderer]);

export type EOption = echarts.EChartsCoreOption;

export const AXIS = {
  axisLine: { lineStyle: { color: "rgba(255,255,255,0.18)" } },
  axisLabel: { color: "rgba(255,255,255,0.55)", fontSize: 11 },
  splitLine: { lineStyle: { color: "rgba(255,255,255,0.06)" } },
  nameTextStyle: { color: "rgba(255,255,255,0.5)", fontSize: 11 },
};
export const TOOLTIP = {
  backgroundColor: "rgba(12,17,34,0.95)", borderColor: "rgba(255,255,255,0.12)",
  textStyle: { color: "#fff", fontSize: 12 }, confine: true,
};
export const LEGEND = { textStyle: { color: "rgba(255,255,255,0.7)", fontSize: 12 }, top: 0, icon: "roundRect" };
export const SERIES = ["#f5b04c", "#8fb4dc", "#5fd3a2", "#c9a7f5", "#f27a85", "#7fd6e0", "#e6e6e6", "#b9c76a"];

export function EChart({ option, height = 260, onClick, ariaLabel }: {
  option: EOption; height?: number; ariaLabel: string;
  onClick?: (p: { name?: string; data?: unknown; seriesName?: string; value?: unknown }) => void;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const chart = useRef<echarts.ECharts | null>(null);
  useEffect(() => {
    if (!ref.current) return;
    chart.current = echarts.init(ref.current, undefined, { renderer: "canvas" });
    const ro = new ResizeObserver(() => chart.current?.resize());
    ro.observe(ref.current);
    return () => { ro.disconnect(); chart.current?.dispose(); chart.current = null; };
  }, []);
  useEffect(() => {
    const reduce = typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    chart.current?.setOption({ backgroundColor: "transparent", color: SERIES, animation: !reduce,
      textStyle: { fontFamily: "IBM Plex Sans, sans-serif" }, ...option }, true);
  }, [option]);
  useEffect(() => {
    if (!onClick || !chart.current) return;
    chart.current.on("click", onClick as never);
    return () => { chart.current?.off("click"); };
  }, [onClick]);
  return <div ref={ref} role="img" aria-label={ariaLabel} style={{ height, width: "100%" }} />;
}
