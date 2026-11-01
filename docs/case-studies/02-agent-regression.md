# Case study 2 — An AI agent regression found by evaluation and explained by the Copilot

**Measured in:** the same demo run ([`enterprise-demo.json`](../results/enterprise-demo.json), `case2`).

## Setup
The EvalForge Copilot is itself an agent with a **versioned configuration** (tool allow-list,
planner, budgets). It is evaluated by an `agent` suite, `copilot-investigations@v1`, with five
investigation tasks. Each case states the tools a correct investigation must use and facts the
answer must contain. Evaluators: `tool_selection` (only successful calls count),
`task_completion`, `agent_efficiency` (≤ 8 steps, no repeated identical calls).

## The change
An engineer created `copilot@v2` with the note *"trim tool list to reduce latency"*, removing
`diff_dataset_versions`.

## Detection
| Config | Pass rate | tool_selection | task_completion | p95 latency |
|---|---|---|---|---|
| copilot@v1 (baseline) | **100 %** | 0.80 | 1.00 | 149 ms |
| copilot@v2 (trimmed) | **60 %** | 0.66 | 0.80 | 58 ms |
| copilot@v3 (fix) | **100 %** | 0.80 | 1.00 | — |

The v2 run was automatically compared with the previous run of the same suite version:
`pass_rate −0.40, tool_selection −0.14, task_completion −0.20` exceeded the 0.05 tolerance,
two cases (`regression`, `dataset`) flipped from pass to fail, and an `agent_regression`
incident (severity high) was opened with both run ids.

(`tool_selection` is 0.80 even at baseline because playbooks call extra read-only tools beyond
the minimum expected set; precision is part of the score. Pass/fail requires full recall.)

## Investigation
The incident's **Investigate with Copilot** action ran the agent on its own failure. It called
`list_eval_runs → compare_eval_runs → diff_agent_configs` (3 tool calls) and reported:

* Eval run `cb291c38` (copilot@v2) pass rate 60 % vs baseline `05d7976c` (copilot@v1) 100 %;
  2 cases newly failing.
* Failing cases did not call `diff_dataset_versions` (2 cases).
* Config copilot@v1 → copilot@v2: tools removed `['diff_dataset_versions']`, nothing else changed.
* **Hypothesis (high confidence):** removing `diff_dataset_versions` caused the regression —
  every newly failing case needed it.
* **Recommendation (requires approval):** restore the tool in a new config version and re-run
  the suite before release.

Every finding cites run ids and config ids that appeared in tool results; claims citing
anything else would have been moved to `unsupported`.

## Fix and verification
`copilot@v3` restored the tool. Re-running the suite gave 100 % with no regression against
the baseline, and the incident was resolved with a note linking the verifying run.

## What this shows
* A latency optimisation silently broke two investigation types. Only a versioned eval suite
  with tool-level expectations made that visible.
* The latency gain (p95 149 → 58 ms) was real, which is exactly why the regression is easy to
  ship without evaluation.
* The investigation is attributable and reproducible because the agent's behaviour is a
  function of a hashed configuration and the rules planner is deterministic.
