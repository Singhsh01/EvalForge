"""Dataset ingestion: parse → hash → dedupe → profile → quality checks → split → persist."""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from evalforge.models import AuditEvent, Dataset, DatasetVersion, VersionStatus
from evalforge.services import profiling, storage


class DuplicateVersionError(Exception):
    def __init__(self, existing: DatasetVersion) -> None:
        super().__init__(f"Identical content already stored as version {existing.version}")
        self.existing = existing


class SchemaMismatchError(ValueError):
    pass


def create_dataset(
    db: Session, *, name: str, target_column: str, description: str, data: bytes, filename: str
) -> Dataset:
    ds = Dataset(name=name, target_column=target_column, description=description)
    db.add(ds)
    db.flush()
    add_version(db, ds, data=data, filename=filename, note="initial upload")
    db.add(
        AuditEvent(
            action="dataset.created",
            entity_type="dataset",
            entity_id=ds.id,
            data={"name": name, "target": target_column},
        )
    )
    return ds


def add_version(
    db: Session, ds: Dataset, *, data: bytes, filename: str, note: str = ""
) -> DatasetVersion:
    fmt = storage.detect_format(filename)
    df = storage.parse_table(data, fmt)
    digest = storage.sha256_bytes(data)

    existing = db.scalar(
        select(DatasetVersion).where(
            DatasetVersion.dataset_id == ds.id, DatasetVersion.sha256 == digest
        )
    )
    if existing is not None:
        raise DuplicateVersionError(existing)

    prev = db.scalar(
        select(DatasetVersion)
        .where(DatasetVersion.dataset_id == ds.id)
        .order_by(DatasetVersion.version.desc())
    )
    schema = profiling.infer_schema(df)
    if prev is not None:
        _check_schema_compat(prev.schema_, schema, ds.target_column)

    stored = storage.store(ds.id, data, filename, df)
    checks, task = profiling.run_quality_checks(df, ds.target_column)
    split = None
    if not any(c.severity == "error" and not c.passed for c in checks):
        split = profiling.make_split(df, ds.target_column, stored.canonical_path.parent)
        checks.append(profiling.split_overlap_check(split))
    status = (
        VersionStatus.rejected
        if any(c.severity == "error" and not c.passed for c in checks)
        else VersionStatus.validated
    )
    next_no = (
        db.scalar(
            select(func.max(DatasetVersion.version)).where(DatasetVersion.dataset_id == ds.id)
        )
        or 0
    ) + 1
    v = DatasetVersion(
        dataset_id=ds.id,
        version=next_no,
        sha256=digest,
        source_filename=filename,
        file_format=stored.file_format,
        storage_path=str(stored.canonical_path),
        size_bytes=stored.size_bytes,
        n_rows=int(df.shape[0]),
        n_columns=int(df.shape[1]),
        task_type=task,
        status=status,
        schema_=schema,
        profile=profiling.profile_frame(df, ds.target_column),
        quality_checks=[c.as_dict() for c in checks],
        split=split,
        note=note,
    )
    db.add(v)
    db.flush()
    db.add(
        AuditEvent(
            action="dataset_version.created",
            entity_type="dataset_version",
            entity_id=v.id,
            data={"version": next_no, "sha256": digest, "status": status.value},
        )
    )
    return v


def _check_schema_compat(old: list[dict], new: list[dict], target: str) -> None:
    """New versions must keep the target and every previous feature column.

    Added columns are allowed (recorded in the profile); removed or retyped
    columns would silently break models trained on earlier versions.
    """
    old_cols = {c["name"]: c["kind"] for c in old}
    new_cols = {c["name"]: c["kind"] for c in new}
    if target not in new_cols:
        raise SchemaMismatchError(f"New version is missing target column '{target}'")
    missing = sorted(set(old_cols) - set(new_cols))
    if missing:
        raise SchemaMismatchError(
            f"New version drops columns present in the previous one: {missing}"
        )
    retyped = sorted(
        c for c in old_cols if c in new_cols and old_cols[c] != new_cols[c] and c != target
    )
    if retyped:
        raise SchemaMismatchError(f"Column types changed: {retyped}")
