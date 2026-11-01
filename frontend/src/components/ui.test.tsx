import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ApiError } from "@/lib/api";
import { Button, ErrorNote, JsonBlock, Status } from "./ui";

describe("ui primitives", () => {
  it("Status shows a readable label and a known tone", () => {
    render(<Status value="awaiting_approval" />);
    expect(screen.getByText("awaiting approval")).toBeInTheDocument();
  });

  it("Status falls back to a neutral style for unknown states", () => {
    render(<Status value="something_new" />);
    expect(screen.getByText("something new").className).toContain("text-white/60");
  });

  it("ErrorNote announces API errors and renders nothing without one", () => {
    const { container, rerender } = render(<ErrorNote error={null} />);
    expect(container).toBeEmptyDOMElement();
    rerender(<ErrorNote error={new ApiError(409, "Approval is already executed")} />);
    expect(screen.getByRole("alert")).toHaveTextContent("Approval is already executed");
  });

  it("Button is disabled while loading", () => {
    render(<Button loading>Approve</Button>);
    expect(screen.getByRole("button", { name: "Approve" })).toBeDisabled();
  });

  it("JsonBlock pretty-prints", () => {
    render(<JsonBlock value={{ passed: true }} />);
    expect(screen.getByText(/"passed": true/)).toBeInTheDocument();
  });
});
