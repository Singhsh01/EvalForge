from __future__ import annotations

from .test_ops import train, upload


def _flush():
    from evalforge.observability import tracing

    tracing.flush()


def test_request_trace_contains_job_training_and_mlflow_spans(client, churn_csv):
    ds = upload(client, churn_csv)
    r = client.post(
        "/api/runs",
        json={"dataset_version_id": ds["versions"][0]["id"], "algorithm": "logistic_regression"},
    )
    trace_id = r.headers["x-trace-id"]
    assert len(trace_id) == 32
    _flush()
    t = client.get(f"/api/traces/{trace_id}").json()
    names = [s["name"] for s in t["spans"]]
    assert "POST /api/runs" in names
    assert any(n.startswith("job.") for n in names)
    assert "model.fit" in names and "mlflow.log_run" in names
    root = next(s for s in t["spans"] if s["parent_id"] is None)
    assert root["attributes"]["enduser.id"] == "admin"
    assert root["attributes"]["http.status_code"] == 202
    job = next(s for s in t["spans"] if s["name"].startswith("job."))
    assert any(e["name"] == "stage.training" for e in job["events"])
    assert "job.id" in t["correlations"]
    run = client.get(f"/api/runs/{r.json()['run']['id']}").json()
    assert run["job"]["result"]["trace_id"] == trace_id  # job ↔ trace correlation


def test_error_spans_and_listing(client):
    r = client.post("/api/runs", json={"dataset_version_id": "missing", "algorithm": "x"})
    assert r.status_code == 404
    _flush()
    traces = client.get("/api/traces").json()
    assert traces and all("span_count" in t for t in traces)
    assert any(t["attributes"].get("http.status_code") == 404 for t in traces)


def test_redaction_of_secrets_in_spans(client):
    from evalforge.observability import tracing

    with tracing.span(
        "custom.op",
        {
            "api_key": "sk-ant-abcdefghijklmnop",
            "note": "token efk_abcdefghijklmnopqrstu used by bob@example.com",
            "count": 3,
        },
    ):
        pass
    _flush()
    t = [x for x in client.get("/api/traces?q=custom.op").json() if x["name"] == "custom.op"][0]
    a = t["attributes"]
    assert a["api_key"] == "[REDACTED]" and a["count"] == 3
    assert "efk_" not in a["note"] and "bob@" not in a["note"]
    assert tracing.redact({"Authorization": "Bearer abcdefghijk"}) == {
        "Authorization": "[REDACTED]"
    }


def test_metrics_summary_and_audit(client, churn_csv):
    ds = upload(client, churn_csv)
    train(client, ds["versions"][0]["id"], "logistic_regression")
    m = client.get("/metrics").text
    assert "evalforge_http_requests_total" in m and "evalforge_jobs_total" in m
    _flush()
    s = client.get("/api/observability/summary").json()
    assert s["http"]["requests"] > 0 and any(r["route"] == "/api/runs" for r in s["http"]["routes"])
    assert {"kind": "train", "status": "succeeded", "count": 1} in s["jobs"]
    cp = client.get("/api/analytics/cost-performance").json()
    assert cp["training"][0]["f1"] is not None
    audit = client.get("/api/audit?action=run.").json()
    assert {e["action"] for e in audit} >= {"run.launched", "run.completed"}
    assert client.get("/api/audit", headers={"Authorization": ""}).status_code == 401
