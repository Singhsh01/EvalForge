from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, selectinload

from evalforge.config import get_settings
from evalforge.db import get_db
from evalforge.models import Dataset, DatasetVersion
from evalforge.schemas import DatasetOut, DatasetVersionDetail, DatasetVersionSummary
from evalforge.services import datasets as svc
from evalforge.services.storage import DatasetParseError, load_frame

router = APIRouter(prefix="/api/datasets", tags=["datasets"])
NAME_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.\- ]{1,63}$"


def _counts(v: DatasetVersion) -> dict[str, int]:
    failed = sum(1 for c in v.quality_checks if c["severity"] == "error" and not c["passed"])
    warned = sum(1 for c in v.quality_checks if c["severity"] == "warning" and not c["passed"])
    return {"checks_failed": failed, "checks_warned": warned}


def summary(v: DatasetVersion) -> DatasetVersionSummary:
    return DatasetVersionSummary.model_validate(v).model_copy(update=_counts(v))


def detail(v: DatasetVersion) -> DatasetVersionDetail:
    return DatasetVersionDetail.model_validate(v).model_copy(update=_counts(v))


def dataset_out(ds: Dataset) -> DatasetOut:
    return DatasetOut(
        id=ds.id,
        name=ds.name,
        description=ds.description,
        target_column=ds.target_column,
        created_at=ds.created_at,
        versions=[summary(v) for v in ds.versions],
    )


async def _read_upload(file: UploadFile) -> bytes:
    limit = get_settings().max_upload_mb * 1024 * 1024
    data = await file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(413, f"File exceeds {get_settings().max_upload_mb} MB")
    return data


@router.get("", response_model=list[DatasetOut])
def list_datasets(db: Session = Depends(get_db)) -> list[DatasetOut]:
    rows = db.scalars(
        select(Dataset).options(selectinload(Dataset.versions)).order_by(Dataset.created_at.desc())
    ).all()
    return [dataset_out(d) for d in rows]


@router.post("", response_model=DatasetOut, status_code=status.HTTP_201_CREATED)
async def create_dataset(
    file: Annotated[UploadFile, File(description="CSV, JSON array or JSON Lines")],
    name: Annotated[str, Form(pattern=NAME_PATTERN)],
    target_column: Annotated[str, Form(min_length=1, max_length=256)],
    description: Annotated[str, Form(max_length=2000)] = "",
    db: Session = Depends(get_db),
) -> DatasetOut:
    if db.scalar(select(Dataset).where(Dataset.name == name)):
        raise HTTPException(409, f"Dataset '{name}' already exists — upload a new version instead")
    data = await _read_upload(file)
    try:
        ds = svc.create_dataset(
            db,
            name=name,
            target_column=target_column,
            description=description,
            data=data,
            filename=file.filename or "upload.csv",
        )
        db.commit()
    except DatasetParseError as exc:
        db.rollback()
        raise HTTPException(422, str(exc)) from exc
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(409, "Dataset already exists") from exc
    db.refresh(ds)
    return dataset_out(ds)


@router.get("/{dataset_id}", response_model=DatasetOut)
def get_dataset(dataset_id: str, db: Session = Depends(get_db)) -> DatasetOut:
    ds = db.get(Dataset, dataset_id)
    if ds is None:
        raise HTTPException(404, "Dataset not found")
    return dataset_out(ds)


@router.post("/{dataset_id}/versions", response_model=DatasetVersionDetail, status_code=201)
async def add_version(
    dataset_id: str,
    file: Annotated[UploadFile, File()],
    note: Annotated[str, Form(max_length=2000)] = "",
    db: Session = Depends(get_db),
) -> DatasetVersionDetail:
    ds = db.get(Dataset, dataset_id)
    if ds is None:
        raise HTTPException(404, "Dataset not found")
    data = await _read_upload(file)
    try:
        v = svc.add_version(db, ds, data=data, filename=file.filename or "upload.csv", note=note)
        db.commit()
    except svc.DuplicateVersionError as exc:
        db.rollback()
        raise HTTPException(
            409, {"message": str(exc), "existing_version": exc.existing.version}
        ) from exc
    except (DatasetParseError, svc.SchemaMismatchError) as exc:
        db.rollback()
        raise HTTPException(422, str(exc)) from exc
    return detail(v)


def _get_version(db: Session, dataset_id: str, version: int) -> DatasetVersion:
    v = db.scalar(
        select(DatasetVersion).where(
            DatasetVersion.dataset_id == dataset_id, DatasetVersion.version == version
        )
    )
    if v is None:
        raise HTTPException(404, "Dataset version not found")
    return v


@router.get("/{dataset_id}/versions/{version}", response_model=DatasetVersionDetail)
def get_version(dataset_id: str, version: int, db: Session = Depends(get_db)):
    return detail(_get_version(db, dataset_id, version))


@router.get("/{dataset_id}/versions/{version}/preview")
def preview(
    dataset_id: str,
    version: int,
    limit: int = Query(50, ge=1, le=500),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    v = _get_version(db, dataset_id, version)
    df = load_frame(v.storage_path).head(limit)
    return {
        "columns": list(df.columns),
        "rows": df.astype(object).where(df.notna(), None).values.tolist(),
    }
