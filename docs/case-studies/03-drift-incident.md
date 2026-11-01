# Case study 3 — A production drift incident, diagnosed and resolved under approval

**Measured in:** the same demo run ([`enterprise-demo.json`](../results/enterprise-demo.json), `case3`).

## Scenario
The champion (logistic regression, v1) is serving in production. A scanner recalibration makes
size measurements read **35 % high** (`mean/worst radius`, `perimeter`, `area`, `area_error`).
Traffic is simulated by bootstrap-resampling the public dataset with 1 % measurement noise;
ground-truth labels arrive later as feedback. The shift is synthetic and documented; every
number below was measured on it.

## Detection
| Window | Verdict | Labelled samples | F1 | Precision | Recall |
|---|---|---|---|---|---|
| Normal traffic (300) | no drift | 300 | 0.982 | — | — |
| After recalibration (300) | **performance degradation** | 300 | **0.869** | 0.768 | 1.000 |

The drift report on the post-recalibration window (reference = the model's training split):

* 7 of 30 features drifted — exactly the seven shifted ones; PSI 2.29 (`worst_radius`),
  2.19 (`worst_perimeter`), 1.62 (`mean_radius`), 1.50 (`mean_perimeter`), 0.59 (`mean_area`).
* Drift share 23 % is below the 30 % "dataset drift" threshold, but **prediction drift** was
  detected and **labels confirmed an F1 drop of 0.082** (> 0.05 threshold) → verdict
  `performance_degradation`, high-severity incident.
* ROC-AUC stayed 0.998: the model still *ranks* cases correctly, but inflated sizes push benign
  cases over the decision threshold (precision 0.975 → 0.768). That distinction matters for the
  fix: the information is still there; the mapping to the threshold moved.

## Investigation
**Investigate with Copilot** produced 3 findings and one high-confidence hypothesis —
*covariate shift in worst_radius, worst_perimeter, mean_radius is degrading measured quality* —
citing the drift report id, and recommended (requiring approval) retraining on a dataset version
that includes recent labelled data.

## Controlled retraining
1. **Plan without new data:** not justified — retraining on the same data would not address the
   shift. (The plan also refuses on statistical drift without labels.)
2. A new dataset version **v2** (569 original + 250 labelled post-recalibration cases) was
   uploaded; the plan became **justified**: *"Measured F1 drop of 0.082 on 300 labelled
   production samples."*
3. The engineer requested retraining; nothing ran until the **approver** accepted. The approval's
   executor started the continuous-training pipeline (`trigger: incident`).
4. **Gate vs. the deployed baseline on the same holdout (v2):** candidate F1 0.949 vs champion
   v1 re-scored at **0.922** (Δ +0.027) → passed; staging deploy healthy; promotion and
   production release each required approval. **This comparison alone does not prove an
   improvement:** the paired bootstrap 95% interval for Δ is **[−0.020, +0.083]** on 164
   holdout rows, which includes 0. The evidence for the retraining is the post-release
   measurement below, on traffic from the new regime.

## Verification after release
300 new post-recalibration requests with labels, scored against the new model's own reference:

| Model | F1 | Precision | Recall | Verdict |
|---|---|---|---|---|
| v1 (before) | 0.869 | 0.768 | 1.000 | performance degradation |
| v3 (retrained, released) | **0.992** | 1.000 | 0.985 | no drift / stable |

The incident was marked resolved with the measurement in its timeline. The full event trail —
detected → investigation → findings → retraining requested → approved/pipeline started →
resolved — is stored on the incident and in the audit log.

## Caveats
* The gate comparison above is not statistically significant (interval includes 0); the
  conclusion rests on the 300-sample post-release measurement, which is itself one sample of
  a simulated regime.
* The new model's reference includes both measurement regimes, so the seven features still
  differ from *pure* recalibrated traffic (share 23 %); with performance confirmed stable and no
  prediction drift, the verdict is `no_drift` by design. A team might instead set per-feature
  alerts.
* A threshold recalibration (or a calibrated probability model) could have recovered precision
  without retraining. EvalForge does not implement that path yet.
