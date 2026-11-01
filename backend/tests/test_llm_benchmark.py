"""The real-LLM benchmark's ground truth and statistics (no model involved)."""

from __future__ import annotations

from evalforge.benchmarks import llm_benchmark as B
from evalforge.llm.evaluators import json_fields


def test_generated_ground_truth_is_deterministic_and_present_in_inputs():
    a, b = B.extraction_cases(), B.extraction_cases()
    assert a == b and len(a) == 40 and len({c["id"] for c in a}) == 40
    for c in a:
        exp, text = c["expected_json"], c["inputs"]["text"]
        assert str(exp["order_id"]) in text and exp["city"] in text and exp["ship_date"] in text
    for c in B.arithmetic_cases():
        assert int(c["expected"]) > 0
    corpus, cases = B.rag_suite()
    ids = {p["id"] for p in corpus}
    for c in cases:
        (rid,) = c["relevant_ids"]
        assert rid in ids
        assert c["expected_contains"][0] in next(p["text"] for p in corpus if p["id"] == rid)


def test_arithmetic_answers_are_correct():
    for c in B.arithmetic_cases():
        nums = [int(x) for x in __import__("re").findall(r"\d+", c["inputs"]["question"])]
        a = c["inputs"]["question"]
        if a.startswith("A warehouse"):
            assert int(c["expected"]) == nums[0] + nums[1] - nums[2]
        elif a.startswith("There are"):
            assert int(c["expected"]) == nums[0] * nums[1] - nums[2]
        else:
            assert int(c["expected"]) == nums[0] * nums[1] - nums[2]


def test_wilson_interval_matches_reference_values():
    assert B.wilson(0, 10) == (0.0, 0.2775)
    assert B.wilson(10, 10) == (0.7225, 1.0)
    lo, hi = B.wilson(35, 40)
    assert lo < 35 / 40 < hi and round(lo, 3) == 0.739


def test_json_fields_evaluator():
    case = {"expected_json": {"order_id": 12345, "quantity": 3, "city": "Ulm"}}
    good = {"text": 'Here: {"order_id": "12345", "quantity": 3, "city": " ulm "}'}
    half = {"text": '{"order_id": 12345, "quantity": 30, "city": "Bonn"}'}
    assert json_fields(case, good, {})["passed"]
    r = json_fields(case, half, {})
    assert (
        not r["passed"]
        and r["score"] == round(1 / 3, 4)
        and set(r["detail"]["wrong"])
        == {
            "quantity",
            "city",
        }
    )
    assert json_fields(case, {"text": "no json"}, {})["score"] == 0.0
