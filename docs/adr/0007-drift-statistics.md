# ADR 0007 — Drift detection with explicit statistical tests

Status: accepted (2026-11-01)

## Decision
Implement drift with scipy rather than Evidently:

* numeric features: two-sample KS test and PSI over reference deciles;
* categorical features: chi-square on the contingency table and PSI;
* predictions: PSI and chi-square on the predicted-class mix;
* performance: classification metrics on labelled current samples only.

A feature is drifted when PSI ≥ 0.2, or when KS p < 0.01 *and* the KS statistic ≥ 0.1. The
effect-size condition matters: with thousands of rows, p-values alone flag trivial shifts.

The report keeps **statistical drift**, **prediction drift** and **label-confirmed
performance degradation** apart, and the retraining plan treats only the last as
justification (statistical drift alone recommends collecting labels).

## Why not Evidently
Evidently 0.7 pulls a large dependency tree and renders its own reports; we need a small,
JSON-native result that the UI, the incident engine and the Copilot can all reason over,
with thresholds we can explain in one paragraph. The tests used are the same ones
Evidently uses by default for tabular data. Swapping in Evidently later only changes
`services/drift.py`.

## Consequences
* Reference = the model's training split, so the reference is exactly what the model saw.
* Current data = logged predictions (with an optional `since` cut-off) or a dataset version.
* Thresholds are per-report parameters, stored with the report for reproducibility.
