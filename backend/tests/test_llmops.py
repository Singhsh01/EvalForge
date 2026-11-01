from __future__ import annotations

import json

import pytest

from evalforge.llm import evaluators as E
from evalforge.llm.providers import FixtureProvider

from .conftest import as_user
from .test_ops import approve

CAPITALS = {"France": "Paris", "Japan": "Tokyo", "Kenya": "Nairobi", "Peru": "Lima"}


@pytest.fixture()
def fixture_models():
    def good(system, user):
        country = user.split("capital of ")[1].rstrip("?")
        return json.dumps({"capital": CAPITALS.get(country, "unknown")})

    def sloppy(system, user):  # a "new model" that answers in prose half the time
        country = user.split("capital of ")[1].rstrip("?")
        if country in ("Kenya", "Peru"):
            return f"I believe it is {CAPITALS[country]}."
        return json.dumps({"capital": CAPITALS[country]})

    def judge(system, user):
        resp = json.loads(user)["RESPONSE"]
        s = 5 if "{" in resp else 2
        return json.dumps(
            {
                "scores": {"correctness": s, "relevance": 5},
                "explanation": "structured" if s == 5 else "prose",
            }
        )

    FixtureProvider.responders.update({"good": good, "sloppy": sloppy, "judge": judge})
    yield
    FixtureProvider.responders.clear()


SCHEMA = {"type": "object", "required": ["capital"], "properties": {"capital": {"type": "string"}}}


def make_suite(client, **extra):
    cases = [
        {
            "id": c.lower(),
            "inputs": {"country": c},
            "expected_regex": CAPITALS[c],
            "schema": SCHEMA,
            "human_scores": {"correctness": 5, "relevance": 5},
        }
        for c in CAPITALS
    ]
    body = {
        "name": "capitals",
        "kind": "prompt",
        "cases": cases,
        "evaluators": [{"type": "json_schema"}, {"type": "regex"}],
        "gates": {"pass_rate": 0.9, "regression_tolerance": 0.05},
    } | extra
    r = client.post("/api/evals/suites", json=body)
    assert r.status_code == 201, r.text
    return r.json()


def make_prompt(client):
    r = client.post(
        "/api/prompts",
        json={
            "name": "capital-lookup",
            "system": "Answer in JSON.",
            "template": "What is the capital of {{country}}?",
            "output_schema": SCHEMA,
        },
    )
    assert r.status_code == 201, r.text
    return r.json()


def run_eval(client, suite_id, subject):
    r = client.post("/api/evals/runs", json={"suite_id": suite_id, "subject": subject})
    assert r.status_code == 202, r.text
    return client.get(f"/api/evals/runs/{r.json()['id']}").json()


def test_prompt_registry_versioning_and_diff(client):
    pv = make_prompt(client)
    assert pv["version"] == 1 and pv["variables"] == ["country"]
    v2 = client.post(
        "/api/prompts/capital-lookup/versions",
        json={"system": "Answer in JSON only.", "template": "Capital of {{country}}? {{style}}"},
    ).json()
    assert v2["version"] == 2 and v2["variables"] == ["country", "style"]
    dup = client.post(
        "/api/prompts/capital-lookup/versions",
        json={"system": "Answer in JSON only.", "template": "Capital of {{country}}? {{style}}"},
    )
    assert dup.status_code == 409
    bad = client.post(
        "/api/prompts/capital-lookup/versions",
        json={"template": "x {{y}}", "output_schema": {"type": "nonsense"}},
    )
    assert bad.status_code == 422
    d = client.get("/api/prompts/capital-lookup/diff?a=1&b=2").json()
    assert "-What is the capital of {{country}}?" in d["diff"]


def test_prompt_eval_regression_incident_and_release(client, fixture_models):
    pv = make_prompt(client)
    suite = make_suite(client)
    base = run_eval(
        client,
        suite["id"],
        {"type": "prompt", "prompt_version_id": pv["id"], "model": "fixture:good"},
    )
    assert base["status"] == "succeeded"
    assert base["summary"]["pass_rate"] == 1.0 and base["summary"]["gates_passed"]
    assert base["summary"]["evaluator_means"] == {"json_schema": 1.0, "regex": 1.0}

    cur = run_eval(
        client,
        suite["id"],
        {"type": "prompt", "prompt_version_id": pv["id"], "model": "fixture:sloppy"},
    )
    reg = cur["regression"]
    assert cur["summary"]["pass_rate"] == 0.5 and reg["detected"]
    assert reg["deltas"]["pass_rate"] == -0.5
    assert {c["case_id"] for c in reg["newly_failing_cases"]} == {"kenya", "peru"}
    assert reg["newly_failing_cases"][0]["failed_evaluators"] == ["json_schema"]
    inc = client.get("/api/incidents").json()
    assert inc[0]["kind"] == "prompt_regression" and inc[0]["severity"] == "high"
    cmp = client.get(f"/api/evals/compare?a={base['id']}&b={cur['id']}").json()
    assert {c["case_id"] for c in cmp["cases"] if c["a_passed"] and not c["b_passed"]} == {
        "kenya",
        "peru",
    }

    # releasing requires a passing, non-regressed evaluation of this prompt version
    eng = as_user(client, "engineer")
    assert (
        client.post(
            f"/api/prompts/versions/{pv['id']}/request-release",
            json={"eval_run_id": cur["id"]},
            headers=eng,
        ).status_code
        == 409
    )
    r = client.post(
        f"/api/prompts/versions/{pv['id']}/request-release",
        json={"eval_run_id": base["id"]},
        headers=eng,
    )
    assert r.status_code == 201
    assert approve(client, r.json()["approval_id"]).json()["status"] == "executed"
    prompts = client.get("/api/prompts").json()
    assert prompts[0]["versions"][0]["status"] == "released"


def test_llm_judge_scores_and_calibration(client, fixture_models):
    pv = make_prompt(client)
    suite = make_suite(
        client,
        evaluators=[
            {
                "type": "llm_judge",
                "model": "fixture:judge",
                "criteria": ["correctness", "relevance"],
                "threshold": 0.75,
            }
        ],
    )
    run = run_eval(
        client,
        suite["id"],
        {"type": "prompt", "prompt_version_id": pv["id"], "model": "fixture:sloppy"},
    )
    by_case = {c["case_id"]: c for c in run["results"]}
    assert by_case["france"]["scores"]["llm_judge"]["detail"]["scores"]["correctness"] == 5
    assert not by_case["kenya"]["passed"]
    cal = client.get(f"/api/evals/runs/{run['id']}/judge-calibration").json()
    assert cal["pairs"] == 4
    c = cal["criteria"]["correctness"]
    assert c["exact_agreement"] == 0.5 and c["bias"] == -1.5 and c["mae"] == 1.5


def test_missing_provider_is_recorded_per_case(client):
    pv = make_prompt(client)
    suite = make_suite(client)
    run = run_eval(
        client,
        suite["id"],
        {"type": "prompt", "prompt_version_id": pv["id"], "model": "anthropic:some-model"},
    )
    assert run["status"] == "succeeded"
    assert run["summary"]["errors"] == 4 and run["summary"]["pass_rate"] == 0.0
    assert "not configured" in run["results"][0]["error"]


CORPUS = [
    {"id": "d1", "text": "The champion model alias points at the approved production version."},
    {"id": "d2", "text": "Canary deployments split traffic between stable and canary versions."},
    {"id": "d3", "text": "Rollback restores the previous stable deployment after approval."},
    {"id": "d4", "text": "Population stability index measures distribution shift of features."},
    {"id": "d5", "text": "Kolmogorov Smirnov tests compare numeric feature distributions."},
    {"id": "d6", "text": "Prompt versions are released only after a passing evaluation run."},
]


def test_rag_retrieval_suite_measures_recall(client):
    cases = [
        {"id": "q1", "question": "How does a canary split traffic?", "relevant_ids": ["d2"]},
        {
            "id": "q2",
            "question": "How is feature distribution shift measured?",
            "relevant_ids": ["d4", "d5"],
        },
        {"id": "q3", "question": "When can a prompt version be released?", "relevant_ids": ["d6"]},
    ]
    r = client.post(
        "/api/evals/suites",
        json={
            "name": "ops-docs",
            "kind": "rag",
            "cases": cases,
            "corpus": CORPUS,
            "evaluators": [{"type": "retrieval", "k": 2, "min_recall": 1.0}],
        },
    )
    assert r.status_code == 201, r.text
    run = run_eval(client, r.json()["id"], {"type": "rag", "top_k": 2})
    by = {c["case_id"]: c["scores"]["retrieval"]["detail"] for c in run["results"]}
    assert by["q1"]["retrieved"][0] == "d2" and by["q1"]["mrr"] == 1.0
    assert by["q3"]["recall_at_k"] == 1.0
    assert run["subject_key"] == "rag:k2@retrieval-only"
    assert 0 < run["summary"]["evaluator_means"]["retrieval"] <= 1


def test_suite_validation(client):
    bad = client.post(
        "/api/evals/suites",
        json={
            "name": "x1",
            "kind": "rag",
            "cases": [{"id": "a"}],
            "evaluators": [{"type": "retrieval"}],
        },
    )
    assert bad.status_code == 422 and "corpus" in bad.json()["detail"]
    bad = client.post(
        "/api/evals/suites",
        json={
            "name": "x2",
            "kind": "prompt",
            "cases": [{"id": "a"}, {"id": "a"}],
            "evaluators": [{"type": "exact_match"}],
        },
    )
    assert bad.status_code == 422


# --------------------------------------------------------------------------- evaluator units


def test_groundedness_and_citations():
    ctx = {"d1": "Canary deployments split traffic between stable and canary versions."}
    good = {
        "text": "Canary deployments split traffic between stable and canary versions [d1].",
        "contexts": ctx,
    }
    bad = {
        "text": "Canary deployments split traffic. They also encrypt every database "
        "backup nightly [d9].",
        "contexts": ctx,
    }
    assert E.groundedness({}, good, {})["passed"]
    g = E.groundedness({}, bad, {})
    assert not g["passed"] and "encrypt" in g["detail"]["unsupported"][0]["sentence"]
    assert E.citations({}, good, {})["passed"]
    assert E.citations({}, bad, {})["detail"]["not_retrieved"] == ["d9"]


def test_agent_evaluators():
    out = {
        "status": "completed",
        "text": "root cause: tool removed",
        "tool_calls": [
            {"tool": "list_eval_runs", "args": {}},
            {"tool": "list_eval_runs", "args": {}},
            {"tool": "compare_eval_runs", "args": {"baseline": "a"}},
        ],
    }
    case = {
        "expected_tools": ["compare_eval_runs", "diff_agent_configs"],
        "expected_args": {"compare_eval_runs": {"baseline": "a"}},
        "expected_answer_contains": ["root cause"],
        "max_steps": 5,
    }
    ts = E.tool_selection(case, out, {})
    assert not ts["passed"] and ts["detail"]["missing"] == ["diff_agent_configs"]
    assert E.tool_arguments(case, out, {})["passed"]
    assert E.task_completion(case, out, {})["passed"]
    eff = E.agent_efficiency(case, out, {})
    assert not eff["passed"] and eff["detail"]["repeated_calls"] == 1
    assert E.token_f1({"reference": "the capital is Paris"}, {"text": "Paris"}, {})["score"] > 0.6
