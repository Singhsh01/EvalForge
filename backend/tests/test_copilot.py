from __future__ import annotations

import json

from evalforge.copilot import agent
from evalforge.copilot.tools import ALL_TOOL_NAMES
from evalforge.llm.providers import FixtureProvider

from .conftest import mixed_frame, to_csv_bytes
from .test_ops import approve, train, upload


def seed_two_versions(client):
    ds = upload(client, to_csv_bytes(mixed_frame(240)))
    v1 = ds["versions"][0]["id"]
    r1 = train(client, v1, "logistic_regression")
    df2 = mixed_frame(240, seed=11)
    df2["income"] = df2["income"] * 1.6  # shifted feature in the newer data
    df2.loc[df2.sample(frac=0.3, random_state=1).index, "label"] = "stay"  # noisier labels
    v2 = client.post(
        f"/api/datasets/{ds['id']}/versions",
        files={"file": ("v2.csv", to_csv_bytes(df2), "text/csv")},
    ).json()["id"]
    r2 = train(client, v2, "logistic_regression")
    return ds, r1, r2


def ask(client, question, cfg=None):
    r = client.post(
        "/api/copilot/sessions", json={"question": question, "wait": True, "agent_config_id": cfg}
    )
    assert r.status_code == 201, r.text
    return r.json()


def test_model_regression_investigation_is_evidence_grounded(client):
    _, r1, r2 = seed_two_versions(client)
    s = ask(client, "Why did my latest model perform worse?")
    rep = s["report"]
    assert s["status"] == "completed" and rep["intent"] == "model_regression"
    tools = [c["tool"] for c in s["steps"]]
    assert tools[:3] == ["query_mlflow_runs", "compare_runs", "diff_dataset_versions"]
    assert any(r2[:8] in f["claim"] and r1[:8] in f["claim"] for f in rep["findings"])
    assert any("income" in h["hypothesis"] for h in rep["hypotheses"])
    assert rep["unsupported"] == []
    for f in rep["findings"]:
        assert set(f["evidence"]) <= set(rep["evidence_ids"])
    assert s["trace_id"] and s["tool_calls"] == len(s["steps"])


def test_recommendation_and_dataset_change_questions(client):
    seed_two_versions(client)
    s = ask(client, "Compare the last five experiments and recommend a candidate model.")
    assert s["report"]["intent"] == "recommend"
    assert s["report"]["recommendations"][0]["evidence"]
    d = ask(client, "Which dataset changes are responsible for this regression?")
    assert d["report"]["intent"] == "dataset_changes"
    assert any("Mean holdout F1 by dataset version" in f["claim"] for f in d["report"]["findings"])


def test_guard_rails_tool_permissions_and_budget(client):
    seed_two_versions(client)
    narrow = client.post(
        "/api/copilot/configs",
        json={
            "config": {"tools_enabled": ["query_mlflow_runs", "compare_runs"]},
            "note": "drop dataset tools to cut latency",
        },
    ).json()
    s = ask(client, "Why did my latest model perform worse?", narrow["id"])
    err = [c for c in s["steps"] if c.get("error")]
    assert err and "not enabled" in err[0]["error"]
    assert not any("income" in h["hypothesis"] for h in s["report"]["hypotheses"])

    tiny = client.post("/api/copilot/configs", json={"config": {"max_steps": 1}}).json()
    s = ask(client, "Why did my latest model perform worse?", tiny["id"])
    assert s["status"] == "budget_exceeded" and s["tool_calls"] == 1

    db_cfg = {**agent.DEFAULT_CONFIG}
    assert "request_evaluation" in ALL_TOOL_NAMES and not db_cfg["allow_actions"]
    bad = client.post("/api/copilot/configs", json={"config": {"tools_enabled": ["rm_rf"]}})
    assert bad.status_code == 422


def test_validation_drops_unsupported_claims():
    rep = {
        "findings": [
            {"claim": "ok", "evidence": ["a"]},
            {"claim": "made up", "evidence": ["zzz"]},
            {"claim": "uncited", "evidence": []},
        ],
        "hypotheses": [],
    }
    out = agent.validate(rep, {"a"})
    assert [f["claim"] for f in out["findings"]] == ["ok"]
    assert {u["claim"] for u in out["unsupported"]} == {"made up", "uncited"}


def test_llm_planner_loop_with_fixture_model(client):
    ds, *_ = seed_two_versions(client)
    script = iter(
        [
            json.dumps(
                {"action": "call", "tool": "list_dataset_versions", "args": {"dataset": ds["name"]}}
            ),
            "placeholder",
        ]
    )

    def planner(system, user):
        nxt = next(script)
        if nxt != "placeholder":
            return nxt
        vid = user.split('"id": "')[1].split('"')[0]  # cite an id the tool returned
        return json.dumps(
            {
                "action": "final",
                "findings": [
                    {"claim": "two versions", "evidence": [vid]},
                    {"claim": "invented metric 0.99", "evidence": ["not-an-id"]},
                ],
            }
        )

    FixtureProvider.responders["planner"] = planner
    try:
        cfg = client.post(
            "/api/copilot/configs", json={"config": {"planner": "llm:fixture:planner"}}
        ).json()
        s = ask(client, "What versions exist?", cfg["id"])
    finally:
        FixtureProvider.responders.clear()
    assert s["planner"] == "llm:fixture:planner" and s["status"] == "completed"
    assert [f["claim"] for f in s["report"]["findings"]] == ["two versions"]
    assert s["report"]["unsupported"][0]["claim"] == "invented metric 0.99"


AGENT_CASES = [
    {
        "id": "regression",
        "question": "Why did my latest model perform worse?",
        "expected_tools": ["query_mlflow_runs", "compare_runs", "diff_dataset_versions"],
        "expected_answer_contains": ["Data shift"],
    },
    {
        "id": "dataset",
        "question": "Which dataset changes are responsible for this regression?",
        "expected_tools": ["diff_dataset_versions"],
        "expected_answer_contains": ["Mean holdout F1 by dataset version"],
    },
    {
        "id": "recommend",
        "question": "Compare the last five experiments and recommend a candidate model.",
        "expected_tools": ["query_mlflow_runs", "recommend_candidate"],
        "expected_answer_contains": ["Pareto front"],
    },
]


def test_agent_regression_detected_investigated_fixed_and_verified(client):
    """Case study 2: an agent config change regresses the agent eval; Copilot finds why."""
    seed_two_versions(client)
    cfgs = client.get("/api/copilot/configs").json()
    v1 = cfgs[0]
    suite = client.post(
        "/api/evals/suites",
        json={
            "name": "copilot-investigations",
            "kind": "agent",
            "cases": AGENT_CASES,
            "evaluators": [
                {"type": "tool_selection"},
                {"type": "task_completion"},
                {"type": "agent_efficiency", "max_steps": 6},
            ],
            "gates": {"pass_rate": 1.0, "regression_tolerance": 0.05},
        },
    ).json()

    def run(cfg_id):
        r = client.post(
            "/api/evals/runs",
            json={"suite_id": suite["id"], "subject": {"type": "agent", "agent_config_id": cfg_id}},
        )
        return client.get(f"/api/evals/runs/{r.json()['id']}").json()

    base = run(v1["id"])
    assert base["summary"]["pass_rate"] == 1.0, base["results"]

    trimmed = [t for t in v1["config"]["tools_enabled"] if t != "diff_dataset_versions"]
    v2 = client.post(
        "/api/copilot/configs",
        json={"config": {"tools_enabled": trimmed}, "note": "trim tools to reduce latency"},
    ).json()
    bad = run(v2["id"])
    assert bad["regression"]["detected"]
    assert {c["case_id"] for c in bad["regression"]["newly_failing_cases"]} == {
        "regression",
        "dataset",
    }
    inc = next(i for i in client.get("/api/incidents").json() if i["kind"] == "agent_regression")

    res = client.post(f"/api/incidents/{inc['id']}/investigate").json()
    rep = res["report"]
    assert res["status"] == "completed" and rep["intent"] == "agent_eval_failure"
    culprit = [h for h in rep["hypotheses"] if "diff_dataset_versions" in h["hypothesis"]]
    assert culprit and culprit[0]["confidence"] == "high"
    assert any(r["requires_approval"] for r in rep["recommendations"])
    detail = client.get(f"/api/incidents/{inc['id']}").json()
    assert detail["status"] == "investigating" and detail["findings"]["hypotheses"]
    assert detail["investigations"][0]["id"] == res["id"]

    # fix: restore the tool; targeted re-evaluation runs only after approval
    v3 = client.post(
        "/api/copilot/configs",
        json={
            "config": {"tools_enabled": v1["config"]["tools_enabled"]},
            "note": "restore diff_dataset_versions",
        },
    ).json()
    act = client.post("/api/copilot/configs", json={"config": {"allow_actions": True}}).json()
    s = client.post(
        "/api/copilot/sessions",
        json={"question": "overview", "wait": True, "agent_config_id": act["id"]},
    ).json()
    assert s["status"] == "completed"
    from evalforge.copilot import tools as T
    from evalforge.db import session_scope

    with session_scope() as db:
        out = T.request_evaluation(
            db,
            suite["id"],
            json.dumps({"type": "agent", "agent_config_id": v3["id"]}),
            requested_by="engineer",
        )
    assert out["status"] == "pending_approval"
    ok = approve(client, out["approval_id"]).json()
    assert ok["status"] == "executed"
    fixed = client.get(f"/api/evals/runs/{ok['result']['eval_run_id']}").json()
    assert fixed["status"] == "succeeded" and fixed["summary"]["pass_rate"] == 1.0
    assert not fixed["regression"]["detected"]
