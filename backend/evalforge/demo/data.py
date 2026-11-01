"""Reproducible demonstration data.

Sources are bundled with scikit-learn (no download, deterministic bytes):

* UCI Breast Cancer Wisconsin (Diagnostic) — Wolberg, Street, Mangasarian (1995),
  CC BY 4.0. https://archive.ics.uci.edu/dataset/17
"""

from __future__ import annotations

import io

import pandas as pd
from sklearn.datasets import load_breast_cancer


def breast_cancer_frame() -> pd.DataFrame:
    raw = load_breast_cancer(as_frame=True)
    df = raw.frame.copy()
    df.columns = [c.replace(" ", "_") for c in df.columns]
    # sklearn encodes 0 = malignant, 1 = benign; keep the human-readable label.
    df["diagnosis"] = df.pop("target").map({0: "malignant", 1: "benign"})
    return df


def breast_cancer_csv() -> bytes:
    buf = io.StringIO()
    breast_cancer_frame().to_csv(buf, index=False, float_format="%.6g")
    return buf.getvalue().encode()
