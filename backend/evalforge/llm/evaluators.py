"""Evaluators. Each takes ``(case, output, config)`` and returns a score record::

    {"score": float in [0, 1], "passed": bool, "detail": {...}}

Deterministic evaluators are preferred wherever ground truth exists. The LLM
judge returns structured rubric scores with explanations and is always reported
next to — never instead of — deterministic results.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Callable
from typing import Any

import jsonschema

Score = dict[str, Any]
_WORD = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    {
        "a",
        "an",
        "the",
        "of",
        "to",
        "in",
        "on",
        "for",
        "and",
        "or",
        "is",
        "are",
        "was",
        "were",
        "be",
        "by",
        "with",
        "that",
        "this",
        "it",
        "as",
        "at",
        "from",
        "has",
        "have",
        "had",
        "not",
        "no",
        "but",
        "which",
        "who",
        "what",
        "when",
        "where",
        "how",
    }
)


def _tokens(text: str) -> list[str]:
    return [t for t in _WORD.findall(str(text).lower()) if t not in _STOP]


def _norm(s: Any) -> str:
    return " ".join(str(s).strip().lower().split())


def _text(output: dict[str, Any]) -> str:
    return str(output.get("text", ""))


def exact_match(case: dict, output: dict, cfg: dict) -> Score:
    ok = _norm(_text(output)) == _norm(case.get("expected", ""))
    return {"score": float(ok), "passed": ok, "detail": {"expected": case.get("expected")}}


def contains(case: dict, output: dict, cfg: dict) -> Score:
    needles = case.get("expected_contains") or []
    text = _norm(_text(output))
    hits = [n for n in needles if _norm(n) in text]
    score = len(hits) / len(needles) if needles else 1.0
    return {
        "score": score,
        "passed": score == 1.0,
        "detail": {"missing": [n for n in needles if n not in hits]},
    }


def regex(case: dict, output: dict, cfg: dict) -> Score:
    pat = case.get("expected_regex") or cfg.get("pattern")
    ok = bool(pat and re.search(pat, _text(output), re.S))
    return {"score": float(ok), "passed": ok, "detail": {"pattern": pat}}


def json_schema(case: dict, output: dict, cfg: dict) -> Score:
    schema = case.get("schema") or cfg.get("schema")
    raw = _text(output).strip()
    m = re.search(r"\{.*\}|\[.*\]", raw, re.S)
    try:
        data = json.loads(m.group(0) if m else raw)
        if schema:
            jsonschema.validate(data, schema)
        return {"score": 1.0, "passed": True, "detail": {}}
    except (json.JSONDecodeError, jsonschema.ValidationError, TypeError) as exc:
        return {"score": 0.0, "passed": False, "detail": {"error": str(exc).splitlines()[0][:300]}}


def json_fields(case: dict, output: dict, cfg: dict) -> Score:
    """Field-level accuracy of extracted JSON against ``case["expected_json"]``.

    Values compare after whitespace/case normalisation; numbers compare numerically, so
    ``"3"`` and ``3`` match. Score = fraction of expected fields that match; passes only
    when all fields match."""
    expected = case.get("expected_json") or {}
    raw = _text(output).strip()
    m = re.search(r"\{.*\}", raw, re.S)
    try:
        got = json.loads(m.group(0) if m else raw)
    except json.JSONDecodeError:
        got = None
    if not isinstance(got, dict):
        return {"score": 0.0, "passed": False, "detail": {"error": "no JSON object in output"}}

    def same(a: Any, b: Any) -> bool:
        try:
            return float(a) == float(b)
        except (TypeError, ValueError):
            return _norm(a) == _norm(b)

    wrong = {k: got.get(k) for k, v in expected.items() if not same(got.get(k), v)}
    score = 1 - len(wrong) / len(expected) if expected else 1.0
    return {"score": round(score, 4), "passed": not wrong, "detail": {"wrong": wrong}}


def token_f1(case: dict, output: dict, cfg: dict) -> Score:
    """Reference-answer comparison (SQuAD-style token F1)."""
    ref, hyp = Counter(_tokens(case.get("reference", ""))), Counter(_tokens(_text(output)))
    overlap = sum((ref & hyp).values())
    if not ref or not hyp or not overlap:
        f1 = float(not ref and not hyp)
    else:
        p, r = overlap / sum(hyp.values()), overlap / sum(ref.values())
        f1 = 2 * p * r / (p + r)
    return {
        "score": round(f1, 4),
        "passed": f1 >= cfg.get("threshold", 0.5),
        "detail": {"threshold": cfg.get("threshold", 0.5)},
    }


# --------------------------------------------------------------------------- agent


def _calls(output: dict) -> list[dict[str, Any]]:
    return output.get("tool_calls") or []


def tool_selection(case: dict, output: dict, cfg: dict) -> Score:
    expected = set(case.get("expected_tools") or [])
    # A tool counts as used only if the call succeeded; blocked or failed calls are reported
    # separately so a refused tool cannot satisfy the expectation.
    used = {c["tool"] for c in _calls(output) if not c.get("error")}
    attempted_failed = sorted({c["tool"] for c in _calls(output) if c.get("error")})
    forbidden = set(case.get("forbidden_tools") or [])
    if not expected:
        return {"score": 1.0, "passed": not (used & forbidden), "detail": {"used": sorted(used)}}
    tp = len(expected & used)
    precision = tp / len(used) if used else 0.0
    recall = tp / len(expected)
    f1 = 2 * precision * recall / (precision + recall) if tp else 0.0
    ok = recall == 1.0 and not (used & forbidden)
    return {
        "score": round(f1, 4),
        "passed": ok,
        "detail": {
            "expected": sorted(expected),
            "used": sorted(used),
            "missing": sorted(expected - used),
            "unexpected": sorted(used - expected),
            "forbidden_used": sorted(used & forbidden),
            "attempted_but_failed": attempted_failed,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
        },
    }


def tool_arguments(case: dict, output: dict, cfg: dict) -> Score:
    expected: dict[str, dict] = case.get("expected_args") or {}
    if not expected:
        return {"score": 1.0, "passed": True, "detail": {}}
    wrong = {}
    for tool, args in expected.items():
        calls = [c for c in _calls(output) if c["tool"] == tool]
        ok = any(all(c.get("args", {}).get(k) == v for k, v in args.items()) for c in calls)
        if not ok:
            wrong[tool] = {"expected": args, "seen": [c.get("args") for c in calls]}
    score = 1 - len(wrong) / len(expected)
    return {"score": score, "passed": not wrong, "detail": {"mismatches": wrong}}


def task_completion(case: dict, output: dict, cfg: dict) -> Score:
    completed = output.get("status") == "completed"
    needles = case.get("expected_answer_contains") or []
    text = _norm(output.get("text", ""))
    hits = [n for n in needles if _norm(n) in text]
    ok = completed and len(hits) == len(needles)
    return {
        "score": float(ok),
        "passed": ok,
        "detail": {
            "status": output.get("status"),
            "missing_facts": [n for n in needles if n not in hits],
        },
    }


def agent_efficiency(case: dict, output: dict, cfg: dict) -> Score:
    calls = _calls(output)
    sigs = Counter(json.dumps([c["tool"], c.get("args", {})], sort_keys=True) for c in calls)
    repeats = sum(n - 1 for n in sigs.values() if n > 1)
    failures = sum(1 for c in calls if c.get("error"))
    recovered = output.get("status") == "completed" if failures else None
    max_steps = case.get("max_steps") or cfg.get("max_steps", 10)
    ok = repeats == 0 and len(calls) <= max_steps
    return {
        "score": 1.0 if ok else 0.0,
        "passed": ok,
        "detail": {
            "steps": len(calls),
            "max_steps": max_steps,
            "repeated_calls": repeats,
            "tool_failures": failures,
            "recovered_from_failure": recovered,
        },
    }


# --------------------------------------------------------------------------- RAG


def retrieval(case: dict, output: dict, cfg: dict) -> Score:
    relevant = set(case.get("relevant_ids") or [])
    got = list(output.get("retrieved_ids") or [])
    k = cfg.get("k") or len(got)
    top = got[:k]
    hits = len(relevant & set(top))
    precision = hits / len(top) if top else 0.0
    recall = hits / len(relevant) if relevant else 1.0
    rr = next((1 / (i + 1) for i, d in enumerate(top) if d in relevant), 0.0)
    return {
        "score": round(recall, 4),
        "passed": recall >= cfg.get("min_recall", 1.0),
        "detail": {
            "precision_at_k": round(precision, 4),
            "recall_at_k": round(recall, 4),
            "mrr": round(rr, 4),
            "k": k,
            "retrieved": top,
            "relevant": sorted(relevant),
        },
    }


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(_tokens(s)) >= 3]


def groundedness(case: dict, output: dict, cfg: dict) -> Score:
    """Evidence check: each answer sentence must be supported by the retrieved passages.

    A sentence is *supported* when at least ``min_overlap`` of its content tokens occur
    in a single retrieved passage. Unsupported sentences are reported as potential
    hallucinations with the best-matching passage, so a human can verify.
    """
    contexts: dict[str, str] = output.get("contexts") or {}
    thr = cfg.get("min_overlap", 0.6)
    sents = _sentences(_text(output))
    unsupported = []
    for s in sents:
        toks = set(_tokens(s))
        best, best_id = 0.0, None
        for cid, ctx in contexts.items():
            cov = len(toks & set(_tokens(ctx))) / len(toks) if toks else 0
            if cov > best:
                best, best_id = cov, cid
        if best < thr:
            unsupported.append({"sentence": s, "best_passage": best_id, "coverage": round(best, 3)})
    score = 1 - len(unsupported) / len(sents) if sents else 0.0
    return {
        "score": round(score, 4),
        "passed": not unsupported and bool(sents),
        "detail": {"sentences": len(sents), "unsupported": unsupported},
    }


def citations(case: dict, output: dict, cfg: dict) -> Score:
    """Citations like [doc-3] must reference retrieved passages that support the sentence."""
    contexts: dict[str, str] = output.get("contexts") or {}
    found = re.findall(r"\[([A-Za-z0-9_.:-]+)\]", _text(output))
    if not found:
        return {"score": 0.0, "passed": False, "detail": {"error": "no citations"}}
    invalid = [c for c in found if c not in contexts]
    return {
        "score": 1 - len(invalid) / len(found),
        "passed": not invalid,
        "detail": {"citations": found, "not_retrieved": invalid},
    }


# --------------------------------------------------------------------------- LLM judge

JUDGE_CRITERIA = (
    "correctness",
    "relevance",
    "faithfulness",
    "completeness",
    "instruction_adherence",
)
JUDGE_SYSTEM = (
    "You are a strict evaluation judge. Score the RESPONSE against the rubric. "
    "Use only the provided TASK, REFERENCE and CONTEXT; do not reward length. "
    'Return only JSON: {"scores": {criterion: integer 1-5}, "explanation": string}.'
)


def llm_judge(case: dict, output: dict, cfg: dict) -> Score:
    from evalforge.llm.providers import complete

    criteria = cfg.get("criteria") or list(JUDGE_CRITERIA)
    prompt = json.dumps(
        {
            "rubric": {c: "1 = very poor, 5 = excellent" for c in criteria},
            "TASK": case.get("inputs") or case.get("question"),
            "REFERENCE": case.get("reference"),
            "CONTEXT": output.get("contexts"),
            "RESPONSE": _text(output),
        },
        indent=1,
    )
    resp = complete(cfg["model"], JUDGE_SYSTEM, prompt, max_tokens=400)
    m = re.search(r"\{.*\}", resp.text, re.S)
    try:
        data = json.loads(m.group(0) if m else resp.text)
        scores = {c: int(data["scores"][c]) for c in criteria}
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        return {
            "score": 0.0,
            "passed": False,
            "detail": {"error": f"unparseable judge output: {exc}", "raw": resp.text[:500]},
        }
    norm = sum((v - 1) / 4 for v in scores.values()) / len(scores)
    return {
        "score": round(norm, 4),
        "passed": norm >= cfg.get("threshold", 0.75),
        "detail": {
            "scores": scores,
            "explanation": data.get("explanation", ""),
            "judge": cfg["model"],
        },
        "usage": {
            "input_tokens": resp.input_tokens,
            "output_tokens": resp.output_tokens,
            "cost_usd": resp.cost_usd,
            "latency_ms": resp.latency_ms,
            "model": resp.model,
            "provider": resp.provider,
        },
    }


EVALUATORS: dict[str, Callable[[dict, dict, dict], Score]] = {
    "exact_match": exact_match,
    "contains": contains,
    "regex": regex,
    "json_schema": json_schema,
    "json_fields": json_fields,
    "token_f1": token_f1,
    "tool_selection": tool_selection,
    "tool_arguments": tool_arguments,
    "task_completion": task_completion,
    "agent_efficiency": agent_efficiency,
    "retrieval": retrieval,
    "groundedness": groundedness,
    "citations": citations,
    "llm_judge": llm_judge,
}


def judge_calibration(pairs: list[tuple[dict[str, int], dict[str, int]]]) -> dict[str, Any]:
    """Agreement between LLM-judge and human rubric scores (1–5) per criterion.

    Reports mean absolute error, exact-agreement rate, within-one agreement and the
    judge's mean bias (judge − human). Positive bias = the judge is more lenient.
    """
    out: dict[str, Any] = {}
    crits = sorted({c for j, h in pairs for c in j if c in h})
    for c in crits:
        diffs = [j[c] - h[c] for j, h in pairs if c in j and c in h]
        if not diffs:
            continue
        out[c] = {
            "n": len(diffs),
            "mae": round(sum(abs(d) for d in diffs) / len(diffs), 3),
            "exact_agreement": round(sum(d == 0 for d in diffs) / len(diffs), 3),
            "within_one": round(sum(abs(d) <= 1 for d in diffs) / len(diffs), 3),
            "bias": round(sum(diffs) / len(diffs), 3),
        }
    return out
