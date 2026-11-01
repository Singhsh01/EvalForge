# Case study 1 — A model from training to production, with a canary and a rollback

**Measured in:** `python -m evalforge.demo.enterprise_demo` against PostgreSQL + MLflow server +
Redis/RQ worker, 2026-11-01. Raw output: [`docs/results/enterprise-demo.json`](../results/enterprise-demo.json).
Two accounts are used throughout: `engineer` builds and requests, `approver` decides.

## The question
Can a model move from raw data to production with every step reproducible, gated by measured
quality, and reversible — without one person being able to push a change alone?

## What happened

1. **Data.** The UCI Breast Cancer Wisconsin (Diagnostic) table (569 rows, CC BY 4.0) is
   registered as an immutable, SHA-256-addressed version. All 13 quality and leakage checks
   pass; a stratified 455/114 holdout is persisted with the version.
2. **Continuous-training pipeline** (logistic regression, 5-fold CV): validate → train → evaluate
   → register → quality gate → staging → promotion request. The run finished in ~8 s and
   stopped in `awaiting_approval` — the pipeline cannot promote itself.

   | Holdout metric | Value |
   |---|---|
   | F1 (positive = malignant) | 0.951 |
   | Precision / recall | 0.975 / 0.929 |
   | ROC-AUC | 0.996 |
   | CV F1 (5-fold) | 0.972 ± 0.006 |
   | Single-row p95 latency | 3.3 ms |

3. **Approvals.** The approver accepted the promotion (MLflow alias `champion` → v1) and then
   the production deployment. The release health check loaded the model from MLflow
   (skops-only, digest-verified), predicted 25 holdout rows (smoke accuracy 0.96, p95 ≈ 3 ms)
   and passed.
4. **Challenger.** A gradient-boosting pipeline produced v2. The gate re-scored the champion on
   the same holdout: candidate F1 0.950 vs champion 0.951 (Δ −0.001, within the 0.01 tolerance),
   so it passed and was promoted for a **25 % canary**. The paired bootstrap interval for the
   difference is **[−0.054, +0.049]** on 114 holdout rows: the two models are statistically
   indistinguishable, and the gate now says so in an advisory check
   (`difference_is_distinguishable: false`). See
   [case-study-uncertainty.json](../results/case-study-uncertainty.json).
5. **Canary.** 400 requests with distinct request ids were routed by a deterministic hash:
   **101 to the canary, 299 to stable** (25.3 %). Every prediction was logged with its
   deployment id.
6. **Rollback.** Because the challenger brought no measurable gain (see the interval above) at
   higher latency, the
   engineer requested a rollback; the approver accepted; the canary ended as `rolled_back`
   and 100 % of traffic returned to v1. v1 was re-promoted as champion.

## Lineage recorded for the champion
`git_commit`, dataset name/version/SHA-256, training run id, MLflow run id, model URI,
registered version, staging deployment id and the approval id — all on the pipeline run
and visible in the Pipeline studio.

## What this shows
* Quality gates compare **like with like**: the champion is re-scored on the candidate's
  holdout, not compared with numbers from a different split.
* Production changes need two people; the approval row records who decided, why, and the
  executor's result.
* A "passing" challenger is not automatically better. The canary + rollback path is cheap,
  audited and reversible.

## Limits
114 holdout rows make F1 differences of ~0.02 statistically fragile (one malignant case moves
recall by 0.024). The gate's tolerance reflects that; it is not a significance test.
