"""Dataset endpoints: upload, profile, preview, list."""
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from sqlalchemy.orm import Session
from typing import List
import pandas as pd
import json
import uuid
from pathlib import Path

from app.database.session import get_db
from app.database.models import (
    User, Dataset, DatasetColumn, DatasetStatus
)
from app.schemas.dataset import DatasetOut, DatasetPreview
from app.core.security import get_current_user
from app.core.config import settings

# ML Engine imports
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # add project root
from ml_engine.profilers.data_profiler import DataProfiler
from ml_engine.detectors.target_detector import TargetDetector

router = APIRouter(prefix="/api/datasets", tags=["datasets"])


def _read_dataset_file(path: Path, file_format: str) -> pd.DataFrame:
    """Read dataset file based on format."""
    if file_format == "csv":
        return pd.read_csv(path)
    if file_format in ("xlsx", "xls"):
        return pd.read_excel(path)
    if file_format == "json":
        return pd.read_json(path)
    if file_format == "parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported format: {file_format}")


@router.post("/upload", response_model=DatasetOut, status_code=status.HTTP_201_CREATED)
async def upload_dataset(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Upload a dataset file. The system will:
    1. Save the file to storage
    2. Detect format and read it
    3. Profile the data (CRISP-DM Step 2: Data Understanding)
    4. Auto-suggest target variable
    5. Save metadata to database
    """
    # Validate filename
    if not file.filename:
        raise HTTPException(400, "Filename is required")

    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in ("csv", "xlsx", "xls", "json", "parquet"):
        raise HTTPException(
            400,
            f"Unsupported file type: .{ext}. Supported: csv, xlsx, json, parquet"
        )

    # Save file
    storage_filename = f"{uuid.uuid4()}.{ext}"
    storage_path = settings.STORAGE_DIR / storage_filename
    content = await file.read()

    if len(content) > settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            413,
            f"File too large. Max: {settings.MAX_UPLOAD_SIZE_MB} MB"
        )

    with open(storage_path, "wb") as f:
        f.write(content)

    # Create dataset record
    dataset = Dataset(
        user_id=current_user.id,
        name=file.filename.rsplit(".", 1)[0],
        original_filename=file.filename,
        storage_path=str(storage_path),
        file_format=ext,
        file_size_bytes=len(content),
        status=DatasetStatus.PROFILING,
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)

    # Profile the data
    try:
        df = _read_dataset_file(storage_path, ext)
        profiler = DataProfiler()
        profile = profiler.profile(df)

        dataset.n_rows = profile["n_rows"]
        dataset.n_columns = profile["n_columns"]

        # Save column metadata
        for col_meta in profile["columns"]:
            col_record = DatasetColumn(
                dataset_id=dataset.id,
                name=col_meta["name"],
                position=col_meta["position"],
                inferred_type=col_meta["inferred_type"],
                pandas_dtype=col_meta["pandas_dtype"],
                n_missing=col_meta["n_missing"],
                n_unique=col_meta["n_unique"],
                completeness_pct=col_meta["completeness_pct"],
                mean_value=col_meta["stats"].get("mean"),
                std_value=col_meta["stats"].get("std"),
                min_value=col_meta["stats"].get("min"),
                max_value=col_meta["stats"].get("max"),
                median_value=col_meta["stats"].get("median"),
                top_values=col_meta["stats"].get("top_values"),
            )
            db.add(col_record)

        # Auto-detect target
        detector = TargetDetector()
        detection = detector.detect(profile)
        dataset.suggested_target = detection["suggested_target"]
        dataset.target_confidence = detection["confidence"]
        dataset.target_reasoning = detection["reasoning"]
        dataset.status = DatasetStatus.READY
        db.commit()
        db.refresh(dataset)
    except Exception as e:
        dataset.status = DatasetStatus.ERROR
        dataset.error_message = str(e)
        db.commit()
        raise HTTPException(500, f"Profiling failed: {e}")

    return DatasetOut.model_validate(dataset)


@router.get("/", response_model=List[DatasetOut])
def list_datasets(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List all datasets owned by current user."""
    datasets = db.query(Dataset).filter(Dataset.user_id == current_user.id).order_by(Dataset.created_at.desc()).all()
    return [DatasetOut.model_validate(d) for d in datasets]


@router.get("/{dataset_id}", response_model=DatasetOut)
def get_dataset(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get dataset detail with columns metadata."""
    ds = db.query(Dataset).filter(
        Dataset.id == dataset_id,
        Dataset.user_id == current_user.id
    ).first()
    if not ds:
        raise HTTPException(404, "Dataset not found")
    return DatasetOut.model_validate(ds)


@router.get("/{dataset_id}/preview", response_model=DatasetPreview)
def preview_dataset(
    dataset_id: int,
    n_rows: int = 10,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Preview first N rows of dataset."""
    ds = db.query(Dataset).filter(
        Dataset.id == dataset_id,
        Dataset.user_id == current_user.id
    ).first()
    if not ds:
        raise HTTPException(404, "Dataset not found")

    df = _read_dataset_file(Path(ds.storage_path), ds.file_format)
    sample = df.head(n_rows).fillna("").to_dict(orient="records")

    # Convert non-JSON-serializable values
    cleaned = []
    for row in sample:
        cleaned_row = {}
        for k, v in row.items():
            try:
                json.dumps(v)
                cleaned_row[k] = v
            except (TypeError, ValueError):
                cleaned_row[k] = str(v)
        cleaned.append(cleaned_row)

    return DatasetPreview(
        columns=list(df.columns),
        sample_rows=cleaned,
        n_total_rows=len(df),
    )


@router.delete("/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_dataset(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Delete dataset (and its file)."""
    ds = db.query(Dataset).filter(
        Dataset.id == dataset_id,
        Dataset.user_id == current_user.id
    ).first()
    if not ds:
        raise HTTPException(404, "Dataset not found")
    try:
        Path(ds.storage_path).unlink(missing_ok=True)
    except Exception:
        pass
    db.delete(ds)
    db.commit()
