# Evaluation methodology

## Suites and runs
* A **suite** is immutable and versioned per name: `kind` (prompt | rag | agent), `cases`,
  `evaluators`, `gates`, optional `corpus`; its content is SHA-256 hashed.
* A **run** applies a suite to a **subject**: a prompt version + model, a RAG configuration, or
  an agent configuration. Each case records output, scores, latency, tokens, cost and a trace id.
* Every run is compared with the previous successful run of the same suite version (or an
  explicit baseline). If any metric drops by more than `regression_tolerance` (default 0.05),
  the run is marked regressed, the cases that flipped from pass to fail are listed, and an
  incident is opened.

## Evaluators
| Evaluator | Ground truth | Score |
|---|---|---|
| `exact_match`, `contains`, `regex` | `expected`, `expected_contains`, `expected_regex` | 0/1 or fraction |
| `json_schema` | per-case or suite schema | 0/1, error message |
| `token_f1` | `reference` answer | SQuAD-style token F1 |
| `tool_selection` | `expected_tools`, `forbidden_tools` | F1 over **successful** calls; pass needs full recall |
| `tool_arguments` | `expected_args` | fraction of tools called with expected args |
| `task_completion` | `expected_answer_contains` | completed status + required facts |
| `agent_efficiency` | `max_steps` | no repeated identical calls, step budget; reports tool failures and recovery |
| `retrieval` | `relevant_ids` | recall@k (score), precision@k, MRR |
| `groundedness` | retrieved passages | share of answer sentences with ≥ 60 % content-token support in one passage; unsupported sentences listed |
| `citations` | retrieved ids | cited `[id]`s must be retrieved |
| `llm_judge` | rubric (correctness, relevance, faithfulness, completeness, instruction adherence) | 1–5 per criterion + explanation, normalised to 0–1 |

## LLM-as-judge discipline
* Judges are reported **next to** deterministic metrics, never instead of them.
* **Calibration:** cases may carry `human_scores`; `GET /api/evals/runs/{id}/judge-calibration`
  reports MAE, exact agreement, within-one agreement and bias (judge − human) per criterion.
* Judge prompts ask for JSON only and are parsed strictly; unparseable output fails the case
  instead of being guessed.

## Cost and tokens
Tokens come from provider responses. Cost is computed only from prices you configure in
`EVALFORGE_LLM_PRICING`; otherwise it is reported as unknown — EvalForge never estimates.

## What is measured in this repository
* The **agent** evaluations and the **RAG retrieval** evaluations run without external services
  and are measured (case study 2, `tests/test_llmops.py`).
* **Prompt** and **LLM-judge** evaluations are implemented and tested with a deterministic test
  double that only runs under `EVALFORGE_ENV=test`. No API key was available while this was
  built, so **no real-LLM benchmark numbers are published**. Set `EVALFORGE_ANTHROPIC_API_KEY`
  (or an OpenAI-compatible base URL, e.g. a local Ollama) to produce them.
