"""Dataset profiling, data-quality checks, leakage checks and deterministic splits.

Every check returns a structured record ``{id, title, severity, passed, message,
details}``. Checks with ``severity == "error"`` are *blocking*: a version that
fails one is stored as ``rejected`` and cannot be trained on, but stays visible
so the user can see why.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from evalforge.models import TaskType

MIN_ROWS = 30
MIN_ROWS_PER_CLASS = 5
MAX_CLASSES = 50
HIGH_MISSING = 0.4
CATEGORICAL_MAX_UNIQUE = 50
HIST_BINS = 20


@dataclass
class Check:
    id: str
    title: str
    severity: str  # "error" | "warning" | "info"
    passed: bool
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "severity": self.severity,
            "passed": self.passed,
            "message": self.message,
            "details": self.details,
        }


def _clean(v: Any) -> Any:
    """JSON-safe conversion of numpy / NaN values."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    return v


def column_kind(s: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(s):
        return "boolean"
    if pd.api.types.is_numeric_dtype(s):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    return "categorical"


def infer_schema(df: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {
            "name": c,
            "dtype": str(df[c].dtype),
            "kind": column_kind(df[c]),
            "nullable": bool(df[c].isna().any()),
        }
        for c in df.columns
    ]


def infer_task_type(target: pd.Series) -> TaskType | None:
    n = target.dropna().nunique()
    if n == 2:
        return TaskType.binary_classification
    if 2 < n <= MAX_CLASSES and (
        not pd.api.types.is_float_dtype(target)
        or bool((target.dropna() == target.dropna().round()).all())
    ):
        return TaskType.multiclass_classification
    return None


def profile_frame(df: pd.DataFrame, target: str | None) -> dict[str, Any]:
    cols: dict[str, Any] = {}
    n = len(df)
    for c in df.columns:
        s = df[c]
        kind = column_kind(s)
        missing = int(s.isna().sum())
        entry: dict[str, Any] = {
            "kind": kind,
            "missing": missing,
            "missing_ratio": _clean(missing / n if n else 0.0),
            "unique": int(s.nunique(dropna=True)),
        }
        nonnull = s.dropna()
        if kind == "numeric" and len(nonnull):
            arr = nonnull.astype(float).to_numpy()
            q = np.quantile(arr, [0.0, 0.25, 0.5, 0.75, 1.0])
            entry["stats"] = {
                "mean": _clean(arr.mean()),
                "std": _clean(arr.std(ddof=1) if len(arr) > 1 else 0.0),
                "min": _clean(q[0]),
                "p25": _clean(q[1]),
                "median": _clean(q[2]),
                "p75": _clean(q[3]),
                "max": _clean(q[4]),
                "zeros": int((arr == 0).sum()),
            }
            if q[0] == q[4]:
                counts, edges = np.array([len(arr)]), np.array([q[0], q[0]])
            else:
                counts, edges = np.histogram(arr, bins=HIST_BINS)
            entry["histogram"] = {
                "counts": [int(x) for x in counts],
                "edges": [_clean(x) for x in edges],
            }
        else:
            vc = nonnull.astype(str).value_counts().head(15)
            entry["top_values"] = [{"value": k, "count": int(v)} for k, v in vc.items()]
        cols[c] = entry

    profile: dict[str, Any] = {
        "n_rows": n,
        "n_columns": df.shape[1],
        "duplicate_rows": int(df.duplicated().sum()),
        "memory_bytes": int(df.memory_usage(deep=True).sum()),
        "columns": cols,
    }
    if target and target in df.columns:
        vc = df[target].dropna().astype(str).value_counts()
        profile["target"] = {
            "column": target,
            "classes": [{"label": k, "count": int(v)} for k, v in vc.items()],
        }
    return profile


def _id_like(s: pd.Series, n: int) -> bool:
    """Unique per row and not a continuous measurement -> almost certainly an identifier."""
    if n < MIN_ROWS or s.nunique(dropna=True) < n:
        return False
    if pd.api.types.is_float_dtype(s):
        return False
    if pd.api.types.is_integer_dtype(s):
        v = s.sort_values().to_numpy()
        return bool((np.diff(v) == 1).all())  # 0..n-1 or 1..n style row keys
    return True


def run_quality_checks(df: pd.DataFrame, target: str) -> tuple[list[Check], TaskType | None]:
    checks: list[Check] = []
    n = len(df)

    checks.append(
        Check(
            "min_rows",
            "Minimum row count",
            "error",
            n >= MIN_ROWS,
            f"{n} rows (minimum {MIN_ROWS})",
            {"rows": n},
        )
    )

    if target not in df.columns:
        checks.append(
            Check(
                "target_present",
                "Target column present",
                "error",
                False,
                f"Target column '{target}' not found",
                {"available": list(df.columns)[:50]},
            )
        )
        return checks, None
    checks.append(
        Check("target_present", "Target column present", "error", True, f"'{target}' found")
    )

    y = df[target]
    target_missing = int(y.isna().sum())
    checks.append(
        Check(
            "target_missing",
            "Target completeness",
            "warning",
            target_missing == 0,
            "No missing labels"
            if target_missing == 0
            else f"{target_missing} rows without a label will be excluded from training",
            {"missing": target_missing},
        )
    )

    task = infer_task_type(y)
    n_classes = int(y.dropna().nunique())
    checks.append(
        Check(
            "task_type",
            "Supported task type",
            "error",
            task is not None,
            f"{task.value.replace('_', ' ')} with {n_classes} classes"
            if task
            else f"Target has {n_classes} distinct values; Phase 2 supports classification "
            f"with 2–{MAX_CLASSES} classes",
            {"n_classes": n_classes},
        )
    )

    if task is not None:
        counts = y.dropna().astype(str).value_counts()
        smallest = int(counts.min())
        ratio = float(counts.min() / counts.max())
        checks.append(
            Check(
                "class_min_support",
                "Minimum examples per class",
                "error",
                smallest >= MIN_ROWS_PER_CLASS,
                f"Smallest class has {smallest} rows (minimum {MIN_ROWS_PER_CLASS})",
                {"smallest_class": counts.idxmin(), "count": smallest},
            )
        )
        checks.append(
            Check(
                "class_balance",
                "Class balance",
                "warning",
                ratio >= 0.2,
                f"Minority/majority ratio {ratio:.2f}"
                + ("" if ratio >= 0.2 else " — prefer F1/balanced accuracy over accuracy"),
                {"ratio": ratio},
            )
        )

    high_missing = {
        c: float(df[c].isna().mean())
        for c in df.columns
        if c != target and df[c].isna().mean() > HIGH_MISSING
    }
    checks.append(
        Check(
            "high_missing",
            "Columns with high missingness",
            "warning",
            not high_missing,
            "None above 40%"
            if not high_missing
            else f"{len(high_missing)} column(s) more than 40% empty",
            {"columns": high_missing},
        )
    )

    constant = [c for c in df.columns if c != target and df[c].nunique(dropna=True) <= 1]
    checks.append(
        Check(
            "constant_columns",
            "Constant columns",
            "warning",
            not constant,
            "None" if not constant else f"{len(constant)} column(s) carry no information",
            {"columns": constant},
        )
    )

    dup = int(df.duplicated().sum())
    checks.append(
        Check(
            "duplicate_rows",
            "Duplicate rows",
            "warning",
            dup == 0,
            f"{dup} exact duplicate rows" if dup else "No exact duplicates",
            {"count": dup, "ratio": dup / n if n else 0},
        )
    )

    id_cols = [c for c in df.columns if c != target and _id_like(df[c], n)]
    checks.append(
        Check(
            "identifier_columns",
            "Identifier-like columns",
            "info",
            not id_cols,
            "None"
            if not id_cols
            else f"{id_cols} look like row identifiers and will be excluded from features",
            {"columns": id_cols},
        )
    )

    high_card = [
        c
        for c in df.columns
        if c != target
        and c not in id_cols
        and column_kind(df[c]) == "categorical"
        and df[c].nunique() > CATEGORICAL_MAX_UNIQUE
    ]
    checks.append(
        Check(
            "high_cardinality",
            "High-cardinality categoricals",
            "info",
            not high_card,
            "None"
            if not high_card
            else f"{high_card}: rare categories will be grouped by the encoder",
            {"columns": high_card},
        )
    )

    if task is not None:
        checks.append(_target_leakage_check(df, target, id_cols))

    return checks, task


def _target_leakage_check(df: pd.DataFrame, target: str, skip: list[str]) -> Check:
    """Flag features that (almost) deterministically encode the label.

    Two signals: (1) a non-identifier column whose every value maps to exactly one
    class while still having multiple values; (2) a numeric column whose absolute
    correlation with the encoded label exceeds 0.98 (binary targets only).
    """
    sub = df.dropna(subset=[target])
    y = sub[target].astype(str)
    suspects: dict[str, str] = {}
    for c in sub.columns:
        if c == target or c in skip:
            continue
        x = sub[c]
        nun = x.nunique(dropna=True)
        if 1 < nun <= max(2, len(sub) // 5):
            if (pd.DataFrame({"x": x, "y": y}).dropna().groupby("x")["y"].nunique() == 1).all():
                suspects[c] = "each value maps to a single class"
                continue
        if pd.api.types.is_numeric_dtype(x) and y.nunique() == 2 and x.std(skipna=True) > 0:
            codes = (y == sorted(y.unique())[1]).astype(float)
            corr = abs(np.corrcoef(x.fillna(x.median()).astype(float), codes)[0, 1])
            if np.isfinite(corr) and corr > 0.98:
                suspects[c] = f"|corr| with label = {corr:.3f}"
    return Check(
        "target_leakage",
        "Target leakage",
        "warning",
        not suspects,
        "No feature deterministically encodes the label"
        if not suspects
        else f"{len(suspects)} feature(s) may leak the label",
        {"suspects": suspects},
    )


def make_split(
    df: pd.DataFrame,
    target: str,
    folder: Path,
    test_size: float = 0.2,
    seed: int = 42,
) -> dict[str, Any]:
    """Deterministic, stratified holdout split persisted with the version.

    Indices are stored in ``split.npz`` so every run on this version is evaluated on
    exactly the same holdout rows (required for honest run comparison).
    Also checks for train/test contamination (identical feature rows on both sides).
    """
    labelled = df.index[df[target].notna()].to_numpy()
    y = df.loc[labelled, target].astype(str)
    train_idx, test_idx = train_test_split(
        labelled, test_size=test_size, random_state=seed, stratify=y
    )
    train_idx, test_idx = np.sort(train_idx), np.sort(test_idx)
    np.savez_compressed(folder / "split.npz", train=train_idx, test=test_idx)

    feats = [c for c in df.columns if c != target]
    tr_hash = pd.util.hash_pandas_object(df.loc[train_idx, feats], index=False)
    te_hash = pd.util.hash_pandas_object(df.loc[test_idx, feats], index=False)
    overlap = int(te_hash.isin(set(tr_hash)).sum())
    return {
        "strategy": "stratified_holdout",
        "test_size": test_size,
        "seed": seed,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "path": str(folder / "split.npz"),
        "train_test_overlap_rows": overlap,
    }


def load_split(path: str) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(path)
    return z["train"], z["test"]


def split_overlap_check(split: dict[str, Any]) -> Check:
    k = split["train_test_overlap_rows"]
    return Check(
        "split_contamination",
        "Train/test contamination",
        "warning",
        k == 0,
        "No holdout row also appears in training"
        if k == 0
        else f"{k} holdout rows have an identical feature row in training — metrics may be "
        "optimistic",
        {"overlap_rows": k, "n_test": split["n_test"]},
    )
