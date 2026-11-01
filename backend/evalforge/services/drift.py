"""Data drift, prediction drift and label-aware performance monitoring.

Statistical tests (scipy) rather than a heavyweight framework — see ADR 0007:

* numeric features: two-sample Kolmogorov–Smirnov test + Population Stability Index
  over reference deciles;
* categorical features: chi-square test on the category contingency table + PSI;
* predictions: PSI and chi-square on the predicted-class distribution;
* performance: only when ground-truth labels exist for the current window
  (feedback labels on logged predictions, or a labelled dataset version).

The report keeps three conclusions separate, because they call for different actions:

* ``data_drift`` — inputs changed statistically (may or may not matter);
* ``prediction_drift`` — the model's output mix changed;
* ``performance_degradation`` — measured quality dropped (requires labels).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sqlalchemy import select
from sqlalchemy.orm import Session

from evalforge.ml import metrics as M
from evalforge.models import (
    AuditEvent,
    DatasetVersion,
    DriftReport,
    Incident,
    IncidentEvent,
    ModelVersion,
    PredictionLog,
)
from evalforge.observability import metrics, tracing
from evalforge.services.model_cache import load_cached
from evalforge.services.profiling import column_kind, load_split
from evalforge.services.storage import load_frame

DEFAULTS = {
    "psi": 0.2,
    "ks_pvalue": 0.01,
    "ks_stat": 0.1,
    "drift_share": 0.3,
    "prediction_psi": 0.2,
    "f1_drop": 0.05,
    "min_samples": 50,
}
EPS = 1e-4


class DriftError(ValueError):
    pass


def psi(ref: np.ndarray, cur: np.ndarray) -> float:
    ref, cur = np.clip(ref, EPS, None), np.clip(cur, EPS, None)
    ref, cur = ref / ref.sum(), cur / cur.sum()
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def numeric_drift(r: pd.Series, c: pd.Series, th: dict[str, float]) -> dict[str, Any]:
    r, c = r.dropna().astype(float).to_numpy(), c.dropna().astype(float).to_numpy()
    edges = np.unique(np.quantile(r, np.linspace(0, 1, 11)))
    if len(edges) < 3:
        edges = np.array([r.min() - 1, r.min(), r.max() + 1])
    edges[0], edges[-1] = -np.inf, np.inf
    rh, ch = np.histogram(r, edges)[0], np.histogram(c, edges)[0]
    ks = stats.ks_2samp(r, c)
    p = psi(rh.astype(float), ch.astype(float))
    lo, hi = float(min(r.min(), c.min())), float(max(r.max(), c.max()))
    vis = np.linspace(lo, hi, 21) if hi > lo else np.array([lo, lo + 1])
    drifted = p >= th["psi"] or (ks.pvalue < th["ks_pvalue"] and ks.statistic >= th["ks_stat"])
    return {
        "kind": "numeric",
        "test": "ks",
        "statistic": round(float(ks.statistic), 4),
        "p_value": float(ks.pvalue),
        "psi": round(p, 4),
        "drifted": bool(drifted),
        "reference_mean": round(float(r.mean()), 6),
        "current_mean": round(float(c.mean()), 6),
        "histogram": {
            "edges": [round(float(x), 6) for x in vis],
            "reference": (np.histogram(r, vis)[0] / max(len(r), 1)).round(4).tolist(),
            "current": (np.histogram(c, vis)[0] / max(len(c), 1)).round(4).tolist(),
        },
    }


def categorical_drift(r: pd.Series, c: pd.Series, th: dict[str, float]) -> dict[str, Any]:
    r, c = r.dropna().astype(str), c.dropna().astype(str)
    cats = sorted(set(r) | set(c))
    rc = r.value_counts().reindex(cats, fill_value=0).to_numpy(dtype=float)
    cc = c.value_counts().reindex(cats, fill_value=0).to_numpy(dtype=float)
    p = psi(rc, cc)
    try:
        chi = stats.chi2_contingency(np.vstack([rc + 0.5, cc + 0.5]))
        pval, stat = float(chi.pvalue), float(chi.statistic)
    except ValueError:
        pval, stat = 1.0, 0.0
    drifted = p >= th["psi"] or pval < th["ks_pvalue"]
    return {
        "kind": "categorical",
        "test": "chi2",
        "statistic": round(stat, 4),
        "p_value": pval,
        "psi": round(p, 4),
        "drifted": bool(drifted),
        "distribution": {
            "categories": cats[:30],
            "reference": (rc / max(rc.sum(), 1)).round(4)[:30].tolist(),
            "current": (cc / max(cc.sum(), 1)).round(4)[:30].tolist(),
        },
    }


def reference_frame(mv: ModelVersion) -> tuple[pd.DataFrame, pd.Series]:
    dv = mv.run.dataset_version
    df = load_frame(dv.storage_path)
    assert dv.split is not None
    train_idx, _ = load_split(dv.split["path"])
    return df.loc[train_idx], df.loc[train_idx, dv.dataset.target_column].astype(str)


def current_from_predictions(
    db: Session,
    mv: ModelVersion,
    environment: str,
    hours: int,
    limit: int,
    since: datetime | None = None,
) -> tuple[pd.DataFrame, pd.Series | None, list[str]]:
    window = datetime.now(UTC) - timedelta(hours=hours)
    if since is not None and since.tzinfo is None:
        since = since.replace(tzinfo=UTC)
    since = max(window, since) if since is not None else window
    rows = db.scalars(
        select(PredictionLog)
        .where(
            PredictionLog.model_version_id == mv.id,
            PredictionLog.environment == environment,
            PredictionLog.created_at >= since,
        )
        .order_by(PredictionLog.created_at.desc())
        .limit(limit)
    ).all()
    X = pd.DataFrame([r.inputs for r in rows])
    labels = [r.label for r in rows]
    y = pd.Series(labels, dtype=object) if all(lbl is not None for lbl in labels) and rows else None
    if y is None and rows and any(lbl is not None for lbl in labels):
        X = X[[lbl is not None for lbl in labels]].reset_index(drop=True)
        y = pd.Series([lbl for lbl in labels if lbl is not None], dtype=object)
        return X, y, [r.prediction for r in rows if r.label is not None]
    return X, y, [r.prediction for r in rows]


def build_report(
    db: Session,
    mv: ModelVersion,
    *,
    source: str,
    environment: str = "production",
    hours: int = 24,
    limit: int = 5000,
    dataset_version_id: str | None = None,
    thresholds: dict[str, float] | None = None,
    actor: str = "system",
    since: datetime | None = None,
) -> DriftReport:
    th = {**DEFAULTS, **(thresholds or {})}
    feats = mv.run.feature_columns or []
    model = load_cached(mv.source_uri, mv.artifact_sha256)
    with tracing.span(
        "drift.report", {"model.name": mv.name, "model.version": mv.version, "drift.source": source}
    ) as sp:
        ref_X, _ = reference_frame(mv)
        if source == "predictions":
            cur_X, cur_y, cur_pred = current_from_predictions(
                db, mv, environment, hours, limit, since
            )
            current_meta = {
                "environment": environment,
                "window_hours": hours,
                "since": since.isoformat() if since else None,
            }
        elif source == "dataset_version":
            dv = db.get(DatasetVersion, dataset_version_id)
            if dv is None:
                raise DriftError("Dataset version not found")
            df = load_frame(dv.storage_path)
            cur_X = df
            target = dv.dataset.target_column
            cur_y = df[target].astype(str) if target in df.columns else None
            cur_pred = None
            current_meta = {"dataset_version_id": dv.id, "version": dv.version}
        else:
            raise DriftError("source must be 'predictions' or 'dataset_version'")
        if len(cur_X) < th["min_samples"]:
            raise DriftError(
                f"Need at least {int(th['min_samples'])} current samples (have {len(cur_X)})"
            )
        missing = [f for f in feats if f not in cur_X.columns]
        if missing:
            raise DriftError(f"Current data lacks model features: {missing}")
        cur_X = cur_X[feats].copy()
        for f in feats:
            if column_kind(ref_X[f]) == "numeric":
                cur_X[f] = pd.to_numeric(cur_X[f], errors="coerce")

        results = []
        for f in feats:
            fn = (
                numeric_drift
                if column_kind(ref_X[f]) in ("numeric", "boolean")
                else categorical_drift
            )
            results.append({"feature": f, **fn(ref_X[f], cur_X[f], th)})
        results.sort(key=lambda r: -r["psi"])
        drifted = [r["feature"] for r in results if r["drifted"]]
        share = len(drifted) / max(len(results), 1)

        ref_pred = pd.Series([str(p) for p in model.predict(ref_X[feats])])
        cur_pred_s = pd.Series(
            cur_pred if cur_pred is not None else [str(p) for p in model.predict(cur_X)]
        )
        pdrift = categorical_drift(ref_pred, cur_pred_s, {**th, "psi": th["prediction_psi"]})

        perf = None
        if cur_y is not None and len(cur_y) >= th["min_samples"]:
            labels = sorted(set((mv.run.report or {}).get("labels", [])) | set(cur_y.unique()))
            cur_m, _ = M.classification_metrics(
                model, cur_X.reset_index(drop=True), cur_y.reset_index(drop=True), labels
            )
            base = mv.run.metrics or {}
            drop = round(base.get("f1", 0) - cur_m["f1"], 4)
            perf = {
                "labelled_samples": len(cur_y),
                "baseline_f1": base.get("f1"),
                "current": cur_m,
                "f1_drop": drop,
                "degraded": drop >= th["f1_drop"],
            }

        data_drift = share >= th["drift_share"]
        if perf and perf["degraded"]:
            verdict = "performance_degradation"
        elif data_drift or pdrift["drifted"]:
            verdict = (
                "statistical_drift_without_labels"
                if perf is None
                else "statistical_drift_performance_stable"
            )
        else:
            verdict = "no_drift"
        summary = {
            "data_drift": data_drift,
            "drift_share": round(share, 4),
            "drifted_features": drifted,
            "prediction_drift": pdrift["drifted"],
            "performance_degradation": perf["degraded"] if perf else None,
            "labels_available": perf is not None,
            "verdict": verdict,
            "reference_samples": len(ref_X),
            "current_samples": len(cur_X),
        }
        sp.set_attributes({"drift.verdict": verdict, "drift.share": share})
        report = DriftReport(
            model_name=mv.name,
            model_version_id=mv.id,
            source=source,
            reference={
                "dataset_version": mv.run.dataset_version.version,
                "split": "train",
                "rows": len(ref_X),
            },
            current=current_meta | {"rows": len(cur_X)},
            features=results,
            prediction_drift=pdrift,
            performance=perf,
            summary=summary,
            thresholds=th,
            created_by=actor,
        )
        db.add(report)
        db.flush()
        if verdict != "no_drift":
            metrics.DRIFT_ALERTS.labels(mv.name).inc()
            _open_incident(db, mv, report)
        return report


def _open_incident(db: Session, mv: ModelVersion, report: DriftReport) -> Incident:
    s = report.summary
    perf = report.performance
    if s["verdict"] == "performance_degradation":
        kind, sev = "performance", "high"
        title = (
            f"{mv.name} v{mv.version}: F1 dropped {perf['baseline_f1']:.3f} → "
            f"{perf['current']['f1']:.3f} on {perf['labelled_samples']} labelled samples"
        )
    else:
        kind, sev = "data_drift", "medium" if s["drift_share"] >= 0.5 else "low"
        title = (
            f"{mv.name} v{mv.version}: {len(s['drifted_features'])} feature(s) drifted "
            f"({s['drift_share']:.0%}); performance "
            f"{'unknown — no labels' if perf is None else 'stable'}"
        )
    existing = db.scalar(
        select(Incident).where(
            Incident.kind == kind,
            Incident.status.in_(["open", "investigating"]),
            Incident.details["model_version_id"].as_string() == mv.id,
        )
    )
    if existing is not None:
        existing.events.append(
            IncidentEvent(
                kind="evidence",
                message=(f"New drift report {report.id[:8]} confirms: {s['verdict']}"),
                data={"drift_report_id": report.id},
            )
        )
        return existing
    inc = Incident(
        kind=kind,
        severity=sev,
        title=title,
        source_type="drift_report",
        source_id=report.id,
        details={
            "model_name": mv.name,
            "model_version_id": mv.id,
            "model_version": mv.version,
            "drift_report_id": report.id,
            "affected_features": s["drifted_features"][:20],
            "verdict": s["verdict"],
        },
    )
    inc.events.append(
        IncidentEvent(
            kind="detected",
            message=(
                f"Drift report {report.id[:8]}: verdict {s['verdict']}, drifted features "
                f"{', '.join(s['drifted_features'][:6]) or 'none'}"
            ),
            data={"summary": s},
        )
    )
    db.add(inc)
    db.flush()
    db.add(
        AuditEvent(
            action="incident.opened",
            entity_type="incident",
            entity_id=inc.id,
            data={"kind": kind, "severity": sev},
        )
    )
    return inc


def retraining_plan(db: Session, inc: Incident, dataset_version_id: str | None) -> dict[str, Any]:
    """Decide whether retraining is justified and propose a reproducible configuration."""
    from evalforge.services import registry

    mv = db.get(ModelVersion, inc.details.get("model_version_id"))
    if mv is None:
        raise DriftError("Incident is not linked to a model version")
    report = db.get(DriftReport, inc.details.get("drift_report_id"))
    perf = report.performance if report else None
    reasons, justified = [], False
    if perf and perf.get("degraded"):
        justified = True
        reasons.append(
            f"Measured F1 drop of {perf['f1_drop']:.3f} on "
            f"{perf['labelled_samples']} labelled production samples."
        )
    elif report is not None:
        reasons.append(
            "Only statistical drift is observed; no labels confirm a quality drop. "
            "Collect labels or evaluate on a labelled version before retraining."
        )
    dataset = mv.run.dataset_version.dataset
    if dataset_version_id is None:
        latest = max(
            (v for v in dataset.versions if v.status.value == "validated"), key=lambda v: v.version
        )
        dataset_version_id = latest.id
    dv = db.get(DatasetVersion, dataset_version_id)
    if dv is None or dv.dataset_id != dataset.id:
        raise DriftError("Training data must be a version of the model's dataset")
    if dv.id == mv.run.dataset_version_id:
        justified = False
        reasons.append(
            "No newer dataset version exists: retraining on the same data would not "
            "address the shift. Upload a version that includes recent labelled data."
        )
    champ = registry.champion(db, mv.name)
    return {
        "justified": justified,
        "reasons": reasons,
        "model_version_id": mv.id,
        "champion_version": champ.version if champ else None,
        "affected_features": inc.details.get("affected_features", []),
        "config": {
            "dataset_version_id": dv.id,
            "algorithm": mv.run.algorithm,
            "hyperparameters": mv.run.hyperparameters,
            "seed": mv.run.seed,
            "cv_folds": mv.run.cv_folds or 5,
            "deploy_to_staging": True,
            "request_promotion": True,
        },
        "training_data": {"dataset_version": dv.version, "rows": dv.n_rows, "sha256": dv.sha256},
    }


# --------------------------------------------------------------------------- retraining


def _register_retraining_executor() -> None:
    from evalforge.services.auth import Principal
    from evalforge.services.governance import executor

    @executor("retraining")
    def _exec(db: Session, approval: Any, actor: Principal) -> dict[str, Any]:
        from evalforge.services import pipelines

        cfg = approval.payload["config"]
        requester = Principal("pipeline", approval.requested_by, "engineer")
        run = pipelines.create_run(db, "continuous_training", cfg, requester, trigger="incident")
        inc = db.get(Incident, approval.payload.get("incident_id"))
        if inc is not None:
            inc.status = "mitigating" if inc.status == "open" else inc.status
            inc.events.append(
                IncidentEvent(
                    kind="action",
                    actor=actor.username,
                    message=f"Retraining approved by {actor.username}; pipeline run {run.id[:8]} "
                    "started (candidate → gate vs champion → staging → promotion approval).",
                    data={"pipeline_run_id": run.id},
                )
            )
            inc.details = {**inc.details, "pipeline_run_id": run.id}
        return {"pipeline_run_id": run.id, "submit_job_id": run.job_id}


_register_retraining_executor()
