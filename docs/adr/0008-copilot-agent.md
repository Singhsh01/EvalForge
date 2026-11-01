# ADR 0008 — Copilot agent: explicit state machine, evidence validation, two planners

Status: accepted (2026-11-01)

## Context
The Copilot must investigate with real data, cite evidence, never act without approval,
respect budgets, be traced end to end, and be *evaluable* — its behaviour must be a
versioned subject that the evaluation engine can score and regress.

## Decision
* A small explicit agent (`copilot/agent.py`) instead of LangGraph. The control flow is
  plan → gather (tool calls) → analyse → hypothesise → validate → report, and every
  transition is already explicit in ~600 lines. LangGraph would add a graph runtime and
  checkpointing we don't need, and make the guard rails less visible.
* **Two planners**, chosen by a versioned `AgentConfig`:
  * `rules` — deterministic playbooks per intent. Works offline, makes the agent
    reproducible, and makes regressions attributable to config changes.
  * `llm:<provider>:<model>` — the model chooses tools turn by turn from the allowed
    catalogue (JSON protocol). Falls back to `rules` when no provider is configured.
* **Evidence validation**: each tool returns the identifiers it read; findings must cite
  only those. Anything else goes to `unsupported` and is excluded from the answer. This
  applies to both planners, so an LLM cannot introduce invented run ids or metrics.
* **Guard rails**: per-config tool allow-list; read vs action tools (action tools only file
  approval requests and require `allow_actions`); step budget and wall-clock timeout capped
  by server settings; cooperative cancellation; an OpenTelemetry span per tool call.

## Consequences
* The agent is itself evaluated by an `agent` suite (tool selection, task completion,
  efficiency). Case study 2 shows a config change detected as a regression and diagnosed
  by the agent.
* Rules playbooks cover the documented question types; free-form questions outside them
  need the LLM planner.
