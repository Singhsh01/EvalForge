"""Content-addressed dataset storage and safe tabular parsing.

Uploaded files are never executed or unpickled. CSV/JSON are parsed with pandas
under explicit size/row/column limits, then persisted as a canonical Parquet
file next to the untouched original so later reads are fast and type-stable.
"""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from evalforge.config import get_settings

ALLOWED_FORMATS = {"csv", "json"}


class DatasetParseError(ValueError):
    """Raised for malformed or out-of-policy uploads (mapped to HTTP 422)."""


@dataclass(frozen=True)
class StoredDataset:
    sha256: str
    file_format: str
    raw_path: Path
    canonical_path: Path
    size_bytes: int


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def detect_format(filename: str) -> str:
    suffix = Path(filename).suffix.lower().lstrip(".")
    if suffix in {"jsonl", "ndjson"}:
        return "json"
    if suffix not in ALLOWED_FORMATS:
        raise DatasetParseError(f"Unsupported file type '.{suffix}'. Allowed: .csv, .json, .jsonl")
    return suffix


def parse_table(data: bytes, file_format: str) -> pd.DataFrame:
    s = get_settings()
    if len(data) > s.max_upload_mb * 1024 * 1024:
        raise DatasetParseError(f"File exceeds the {s.max_upload_mb} MB upload limit")
    if not data.strip():
        raise DatasetParseError("File is empty")
    try:
        if file_format == "csv":
            df = pd.read_csv(io.BytesIO(data), low_memory=False)
        else:
            df = _parse_json(data)
    except DatasetParseError:
        raise
    except Exception as exc:  # pandas raises a zoo of exception types
        raise DatasetParseError(f"Could not parse {file_format.upper()}: {exc}") from exc

    if df.shape[1] == 0 or df.shape[0] == 0:
        raise DatasetParseError("Parsed table has no rows or no columns")
    if df.shape[0] > s.max_rows:
        raise DatasetParseError(f"Too many rows ({df.shape[0]} > {s.max_rows})")
    if df.shape[1] > s.max_columns:
        raise DatasetParseError(f"Too many columns ({df.shape[1]} > {s.max_columns})")
    df.columns = [str(c).strip() for c in df.columns]
    if df.columns.duplicated().any():
        dups = sorted(set(df.columns[df.columns.duplicated()]))
        raise DatasetParseError(f"Duplicate column names: {dups}")
    return df


def _parse_json(data: bytes) -> pd.DataFrame:
    text = data.decode("utf-8")
    stripped = text.lstrip()
    if stripped.startswith("["):
        records = json.loads(text)
        if not isinstance(records, list) or not all(isinstance(r, dict) for r in records):
            raise DatasetParseError("JSON must be an array of flat objects")
        return pd.DataFrame.from_records(records)
    if stripped.startswith("{") and "\n" in stripped.strip():
        # JSON Lines: one object per line
        return pd.read_json(io.StringIO(text), lines=True)
    if stripped.startswith("{"):
        obj = json.loads(text)
        if isinstance(obj, dict) and isinstance(obj.get("data"), list):
            return pd.DataFrame.from_records(obj["data"])
    raise DatasetParseError('JSON must be an array of objects, JSON Lines, or {"data": [...]}')


def store(dataset_id: str, data: bytes, filename: str, df: pd.DataFrame) -> StoredDataset:
    digest = sha256_bytes(data)
    fmt = detect_format(filename)
    folder = get_settings().datasets_dir / dataset_id / digest
    folder.mkdir(parents=True, exist_ok=True)
    raw_path = folder / f"original.{fmt}"
    raw_path.write_bytes(data)
    canonical = folder / "data.parquet"
    df.to_parquet(canonical, index=False)
    return StoredDataset(digest, fmt, raw_path, canonical, len(data))


def load_frame(storage_path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(storage_path)
