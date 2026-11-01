# ADR 0004 — Content-addressed dataset versions with fixed splits; DVC later

Status: accepted (2026-11-01)

## Decision
Uploads are hashed (SHA-256); the raw bytes and a canonical Parquet copy are stored under
`datasets/<dataset_id>/<sha256>/`. A version is immutable and carries its schema, profile,
quality/leakage checks and a persisted stratified holdout split. Duplicate content is
rejected with 409; schema-breaking versions with 422.

DVC is planned for Phase 3 as the *remote and pipeline* layer (pushing these
content-addressed files to object storage and versioning `dvc.yaml` training pipelines in
git). It is not introduced now because nothing in Phase 2 needs a remote, and a DVC repo
inside the API's data directory would add git coupling without benefit.

## Consequences
* Runs on the same version are always compared on identical holdout rows.
* Splits are per version; a larger v2 gets a new split — the compare view warns when runs
  span versions.
