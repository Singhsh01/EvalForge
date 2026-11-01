"""Classification metrics and model diagnostics.

Only metrics that are meaningful for the task are produced: ROC-AUC/ROC curve
only when probabilities exist, binary-specific averaging for two classes and
macro averaging otherwise. Feature importance is permutation importance on the
holdout set — model-agnostic and measured, not taken from model internals.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline


def _f(x: float) -> float:
    return float(round(float(x), 6))


BOOTSTRAP_RESAMPLES = 1000


def _f1_codes(yc: np.ndarray, pc: np.ndarray, k: int, pos: int | None) -> float:
    """F1 from integer-coded labels: positive-class F1 when ``pos`` is set (binary),
    otherwise macro F1 over all k labels (matches sklearn with zero_division=0)."""
    cm = np.bincount(yc * k + pc, minlength=k * k).reshape(k, k)
    tp = np.diag(cm).astype(float)
    denom = 2 * tp + (cm.sum(0) - tp) + (cm.sum(1) - tp)
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    return float(f1[pos]) if pos is not None else float(f1.mean())


def bootstrap_f1(
    y_true: Any,
    preds: list[Any],
    labels: list[str],
    n: int = BOOTSTRAP_RESAMPLES,
    seed: int = 0,
) -> np.ndarray:
    """F1 of each prediction vector on ``n`` shared bootstrap resamples of the rows.

    Returns an array of shape (n, len(preds)). Resamples are *paired*: every model is
    scored on the same resampled rows, so differences between columns carry only the
    uncertainty of the difference, not of the shared test set."""
    idx = {lbl: i for i, lbl in enumerate(labels)}
    y = np.asarray([idx.get(str(v), -1) for v in y_true])
    ps = [np.asarray([idx.get(str(v), -1) for v in p]) for p in preds]
    keep = (y >= 0) & np.all([p >= 0 for p in ps], axis=0)
    y, ps = y[keep], [p[keep] for p in ps]
    k = len(labels)
    pos = 1 if k == 2 else None
    rng = np.random.default_rng(seed)
    out = np.empty((n, len(ps)))
    for b in range(n):
        r = rng.integers(0, len(y), len(y))
        for j, p in enumerate(ps):
            out[b, j] = _f1_codes(y[r], p[r], k, pos)
    return out


def ci95(samples: np.ndarray) -> list[float]:
    return [_f(float(np.percentile(samples, 2.5))), _f(float(np.percentile(samples, 97.5)))]


def classification_metrics(
    model: Pipeline, X: pd.DataFrame, y: pd.Series, labels: list[str]
) -> tuple[dict[str, float], dict[str, Any]]:
    y = y.astype(str)
    y_pred = pd.Series(model.predict(X)).astype(str).to_numpy()
    binary = len(labels) == 2
    avg = "binary" if binary else "macro"
    pos = labels[1] if binary else None
    kw: dict[str, Any] = {"average": avg, "zero_division": 0}
    if binary:
        kw["pos_label"] = pos

    metrics: dict[str, float] = {
        "accuracy": _f(accuracy_score(y, y_pred)),
        "balanced_accuracy": _f(balanced_accuracy_score(y, y_pred)),
        "precision": _f(precision_score(y, y_pred, **kw)),
        "recall": _f(recall_score(y, y_pred, **kw)),
        "f1": _f(f1_score(y, y_pred, **kw)),
        "f1_weighted": _f(f1_score(y, y_pred, average="weighted", zero_division=0)),
    }
    # Sampling uncertainty of F1 on this test set (percentile bootstrap, audit BEN-01):
    # small holdouts make point estimates look more precise than they are.
    if len(y) >= 10:
        lo, hi = ci95(bootstrap_f1(y.to_numpy(), [y_pred], labels)[:, 0])
        metrics["f1_ci95_low"], metrics["f1_ci95_high"] = lo, hi
    report: dict[str, Any] = {
        "labels": labels,
        "positive_label": pos,
        "averaging": avg,
        "n_samples": len(y),
        "confusion_matrix": confusion_matrix(y, y_pred, labels=labels).tolist(),
        "per_class": {
            k: {m: _f(v) for m, v in d.items()}
            for k, d in classification_report(
                y, y_pred, labels=labels, output_dict=True, zero_division=0
            ).items()
            if k in labels
        },
    }

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        classes = [str(c) for c in model.classes_]
        order = [classes.index(lbl) for lbl in labels if lbl in classes]
        proba = proba[:, order]
        present = set(y.unique())
        try:
            metrics["log_loss"] = _f(log_loss(y, proba, labels=labels))
        except ValueError:
            pass
        if binary and len(present) == 2:
            yb = (y == pos).astype(int).to_numpy()
            metrics["roc_auc"] = _f(roc_auc_score(yb, proba[:, 1]))
            fpr, tpr, _ = roc_curve(yb, proba[:, 1])
            idx = np.unique(np.linspace(0, len(fpr) - 1, min(len(fpr), 100)).astype(int))
            report["roc_curve"] = {
                "fpr": [_f(v) for v in fpr[idx]],
                "tpr": [_f(v) for v in tpr[idx]],
            }
        elif not binary and present == set(labels):
            metrics["roc_auc_ovr"] = _f(
                roc_auc_score(y, proba, multi_class="ovr", average="macro", labels=labels)
            )
    return metrics, report


def feature_importance(
    model: Pipeline, X: pd.DataFrame, y: pd.Series, seed: int, max_rows: int = 2000
) -> list[dict[str, float]]:
    if len(X) > max_rows:
        X = X.sample(max_rows, random_state=seed)
        y = y.loc[X.index]
    scoring = "f1_macro"
    r = permutation_importance(
        model, X, y.astype(str), n_repeats=5, random_state=seed, scoring=scoring, n_jobs=1
    )
    rows = [
        {"feature": c, "importance": _f(m), "std": _f(s)}
        for c, m, s in zip(X.columns, r.importances_mean, r.importances_std, strict=True)
    ]
    rows.sort(key=lambda d: d["importance"], reverse=True)
    return rows[:30]


def inference_latency(model: Pipeline, X: pd.DataFrame, n_single: int = 50) -> dict[str, float]:
    """Measured in-process latency: one batch call plus repeated single-row calls."""
    t0 = time.perf_counter()
    model.predict(X)
    batch_s = time.perf_counter() - t0
    singles = []
    sample = X.head(n_single)
    for i in range(len(sample)):
        row = sample.iloc[[i]]
        t = time.perf_counter()
        model.predict(row)
        singles.append((time.perf_counter() - t) * 1000)
    arr = np.array(singles) if singles else np.array([0.0])
    return {
        "latency_batch_ms_per_row": _f(batch_s * 1000 / max(len(X), 1)),
        "latency_single_p50_ms": _f(np.percentile(arr, 50)),
        "latency_single_p95_ms": _f(np.percentile(arr, 95)),
    }
