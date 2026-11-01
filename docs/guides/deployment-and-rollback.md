# Deployment and rollback guide

## Lifecycle at a glance
```
training run ──register──► model version (MLflow registry) ──gate──► promotion request
      ──approver──► champion alias ──► staging deploy (engineer, immediate)
      ──► production deploy request ──approver──► health check ──► active
      ──► canary (1–99 %) ──► promote canary  /  rollback   (both approval-gated in production)
```

## Steps (UI: Model registry → Deployment center → Approvals)
1. **Register** a finished run (`POST /api/registry/versions {run_id}`). The model version is
   created in MLflow (`registered model = dataset name`) and tagged with the run id, dataset
   version and SHA-256. The quality gate runs immediately.
2. **Quality gate** (`POST /api/registry/versions/{id}/gate`), blocking checks:
   `min_f1` (default 0.80), `latency_p95` (single-row p95 ≤ 250 ms) and
   `no_regression_vs_champion` — the champion is **re-scored on the candidate's holdout** and the
   candidate may trail it by at most 0.01 F1. Advisory: train–test gap, leakage warnings.
   Thresholds: `EVALFORGE_GATE_*`.
3. **Promotion** (`/request-promotion`) files an approval. On acceptance the gate is re-run, the
   MLflow alias `champion` moves, and the previous champion is demoted.
4. **Staging** deploys happen immediately for engineers; **production** deploys require an
   approved version and an approval. Every deploy runs a **release health check**: the model
   loads through MLflow (skops), predicts 25 holdout rows, all labels are valid, p95 latency is
   within budget and smoke accuracy ≥ 0.5. A failing check records a `failed` deployment and
   leaves traffic untouched.
5. **Canary**: deploy with `traffic_percent < 100`. Routing hashes the request id
   (`x-request-id`), so a given request id always reaches the same version. Pin a version with
   `?version=N` for debugging.
6. **Promote canary** → canary becomes the 100 % stable. **Rollback** → if a canary is active
   it is removed; otherwise the previous stable deployment is re-deployed after a fresh health
   check. Both are approval-gated in production.

## Inference API
```bash
curl -X POST "localhost:8000/api/serve/<model>/predict?environment=production" \
  -H "Authorization: Bearer $TOKEN" -H "content-type: application/json" \
  -d '{"instances": [{"feature_a": 1.2, "feature_b": "x"}]}'
```
Missing features → 422; no deployment → 404; ≤ 1000 instances per call. Each prediction is
logged (inputs, label, probability, latency, deployment, version) and returns an id. Send
ground truth later with `POST /api/serve/feedback {"labels": {"<prediction id>": "<label>"}}`
to enable label-aware monitoring.

## Audit trail
Every registration, gate, approval request/decision, deployment, failed health check, canary
promotion and rollback is an audit event (System health & audit) and is linked to the approval
that authorised it.
