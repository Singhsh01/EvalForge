import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { Dag, LineageGraph, type DagNode } from "./dag";

const NODES: DagNode[] = [
  { name: "validate_data", depends_on: [], status: "succeeded" },
  { name: "train", depends_on: ["validate_data"], status: "succeeded", attempts: 2 },
  { name: "quality_gate", depends_on: ["train"], status: "failed" },
  { name: "deploy_staging", depends_on: ["quality_gate"], status: "skipped" },
];

describe("pipeline DAG", () => {
  it("renders every step with its status and retry count", () => {
    render(<div style={{ width: 900, height: 300 }}><Dag nodes={NODES} /></div>);
    expect(screen.getByText("validate data")).toBeInTheDocument();
    expect(screen.getByText(/succeeded · 2 attempts/)).toBeInTheDocument();
    expect(screen.getByText("failed")).toBeInTheDocument();
    expect(screen.getByText("skipped")).toBeInTheDocument();
  });

  it("reports the clicked step", () => {
    const onSelect = vi.fn();
    render(<div style={{ width: 900, height: 300 }}><Dag nodes={NODES} onSelect={onSelect} /></div>);
    fireEvent.click(screen.getByText("quality gate"));
    expect(onSelect).toHaveBeenCalledWith("quality_gate");
  });
});

describe("lineage graph", () => {
  it("links each recorded artifact in provenance order", () => {
    render(
      <LineageGraph
        lineage={{
          dataset: "churn", dataset_version: 2, dataset_version_id: "dv1", dataset_sha256: "ab".repeat(32),
          run_id: "run12345abc", mlflow_run_id: "mlf0123456789", git_commit: "deadbeefcafe",
          model_version_id: "mv9", name: "churn", version: 3, approval_id: "ap7",
        }}
      />,
    );
    expect(screen.getByText("churn v2")).toBeInTheDocument();
    expect(screen.getByText(/run12345 · git deadbee/)).toBeInTheDocument();
    expect(screen.getByText("churn v3")).toBeInTheDocument();
    expect(screen.getByText("promotion approval")).toBeInTheDocument();
    expect(screen.queryByText("staging deployment")).not.toBeInTheDocument(); // not recorded
    expect(screen.getByText("training run").closest("a")).toHaveAttribute("href", "/experiments/run12345abc");
  });

  it("renders nothing when a run has no lineage yet", () => {
    const { container } = render(<LineageGraph lineage={{}} />);
    expect(container).toBeEmptyDOMElement();
  });
});
