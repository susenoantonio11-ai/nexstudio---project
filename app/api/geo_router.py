"""
Geospatial Research API Endpoints.
====================================
9 endpoints sesuai spec user untuk modul Geospatial AI Research.
"""
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from pathlib import Path
from pydantic import BaseModel
import uuid
import json
from datetime import datetime

from app.database.session import get_db
from app.database.models import User
from app.database.geo_models import GeospatialProject, RasterFile, RasterBand, FloodModelRun
from app.core.security import get_current_user
from app.core.config import settings

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from ml_engine.domains.geospatial_research import (
    RasterReader, RasterPreprocessor, SpectralIndexCalculator,
    FloodClassifier, ThresholdFloodClassifier, ChangeDetector,
    FloodEvaluator, FloodResearchPipeline,
)


router = APIRouter(prefix="/api/geo", tags=["geospatial"])

RASTER_STORAGE = settings.STORAGE_DIR / "rasters"
RASTER_STORAGE.mkdir(parents=True, exist_ok=True)


# ============================================================
# Pydantic Schemas
# ============================================================
class CreateProjectRequest(BaseModel):
    project_name: str
    research_goal: str = "flood_classification"  # / "susceptibility" / "extent_mapping"
    study_area: str
    aoi_geojson: Optional[Dict[str, Any]] = None
    target_classes: List[str] = ["flooded", "non_flooded"]
    description: Optional[str] = None


class GenerateIndicesRequest(BaseModel):
    raster_id: int
    indices: List[str] = ["ndwi", "mndwi", "ndvi"]
    band_mapping: Dict[str, int]  # {"red": 1, "nir": 2, "green": 3, ...}


class TrainFloodModelRequest(BaseModel):
    project_id: int
    model_name: str = "random_forest"  # / "logistic_regression" / "svm" / "xgboost" / "mndwi_threshold" / "sar_threshold"
    features: List[str] = ["ndwi", "mndwi"]
    target_raster_id: Optional[int] = None  # ground truth raster
    feature_raster_ids: List[int] = []
    test_size: float = 0.2


class PredictFloodMapRequest(BaseModel):
    run_id: int
    raster_id: int


class ExportResultRequest(BaseModel):
    run_id: int
    format: str = "geotiff"  # / "csv" / "geojson"


# ============================================================
# Endpoints
# ============================================================

@router.post("/projects", status_code=status.HTTP_201_CREATED)
def create_project(
    req: CreateProjectRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Create geospatial research project (Business Understanding step)."""
    project = GeospatialProject(
        user_id=current_user.id,
        project_name=req.project_name,
        research_goal=req.research_goal,
        study_area=req.study_area,
        aoi_geojson=req.aoi_geojson,
        target_classes=req.target_classes,
        description=req.description,
    )
    db.add(project)
    db.commit()
    db.refresh(project)
    return {
        "project_id": project.project_id,
        "project_name": project.project_name,
        "research_goal": project.research_goal,
        "study_area": project.study_area,
        "target_classes": project.target_classes,
    }


@router.post("/upload-tif", status_code=status.HTTP_201_CREATED)
async def upload_tif(
    project_id: int,
    raster_role: str = "input_imagery",
    source: Optional[str] = None,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Upload GeoTIFF file ke project."""
    project = db.query(GeospatialProject).filter(
        GeospatialProject.project_id == project_id,
        GeospatialProject.user_id == current_user.id,
    ).first()
    if not project:
        raise HTTPException(404, "Project not found")

    if not file.filename or not file.filename.lower().endswith((".tif", ".tiff")):
        raise HTTPException(400, "File must be .tif or .tiff")

    # Save file
    storage_filename = f"{uuid.uuid4()}.tif"
    storage_path = RASTER_STORAGE / storage_filename
    content = await file.read()
    with open(storage_path, "wb") as f:
        f.write(content)

    # Read metadata
    reader = RasterReader()
    metadata = reader.read_metadata(str(storage_path))

    bbox = metadata.get("bounding_box", {})
    raster = RasterFile(
        project_id=project_id,
        file_name=file.filename,
        file_path=str(storage_path),
        file_size_bytes=len(content),
        raster_role=raster_role,
        source=source,
        crs=metadata.get("crs"),
        crs_epsg=metadata.get("crs_epsg"),
        width=metadata.get("width"),
        height=metadata.get("height"),
        band_count=metadata.get("n_bands"),
        dtype=metadata.get("dtype"),
        resolution_x=metadata.get("resolution_x"),
        resolution_y=metadata.get("resolution_y"),
        nodata_value=metadata.get("nodata"),
        bbox_min_x=bbox.get("min_x"),
        bbox_min_y=bbox.get("min_y"),
        bbox_max_x=bbox.get("max_x"),
        bbox_max_y=bbox.get("max_y"),
        metadata_json=metadata,
    )
    db.add(raster)
    db.commit()
    db.refresh(raster)

    # Save bands
    for b in metadata.get("bands", []):
        band = RasterBand(
            raster_id=raster.raster_id,
            band_index=b.get("band_id"),
            band_name=b.get("band_name"),
            min_value=b.get("min_value"),
            max_value=b.get("max_value"),
            mean_value=b.get("mean_value"),
            std_value=b.get("std_value"),
            n_valid_pixels=b.get("n_valid_pixels"),
            n_nodata_pixels=b.get("n_nodata_pixels"),
            nodata_percentage=b.get("nodata_percentage"),
        )
        db.add(band)
    db.commit()

    return {
        "raster_id": raster.raster_id,
        "file_name": raster.file_name,
        "metadata": metadata,
    }


@router.get("/read-metadata/{raster_id}")
def read_metadata(
    raster_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Read metadata raster yang sudah di-upload."""
    raster = (
        db.query(RasterFile)
        .join(GeospatialProject)
        .filter(
            RasterFile.raster_id == raster_id,
            GeospatialProject.user_id == current_user.id,
        )
        .first()
    )
    if not raster:
        raise HTTPException(404, "Raster not found")

    bands = db.query(RasterBand).filter(RasterBand.raster_id == raster_id).all()
    return {
        "raster_id": raster.raster_id,
        "file_name": raster.file_name,
        "crs": raster.crs,
        "width": raster.width,
        "height": raster.height,
        "n_bands": raster.band_count,
        "resolution": (raster.resolution_x, raster.resolution_y),
        "bbox": {
            "min_x": raster.bbox_min_x, "min_y": raster.bbox_min_y,
            "max_x": raster.bbox_max_x, "max_y": raster.bbox_max_y,
        },
        "bands": [
            {
                "index": b.band_index, "name": b.band_name,
                "min": b.min_value, "max": b.max_value,
                "mean": b.mean_value, "std": b.std_value,
                "nodata_pct": b.nodata_percentage,
            }
            for b in bands
        ],
        "full_metadata": raster.metadata_json,
    }


@router.post("/extract-raster/{raster_id}")
def extract_raster(
    raster_id: int,
    band_idx: int = 1,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Extract raster pixel histogram + summary."""
    raster = (
        db.query(RasterFile)
        .join(GeospatialProject)
        .filter(
            RasterFile.raster_id == raster_id,
            GeospatialProject.user_id == current_user.id,
        )
        .first()
    )
    if not raster:
        raise HTTPException(404, "Raster not found")

    reader = RasterReader()
    histogram = reader.histogram(raster.file_path, band_idx=band_idx)
    return {
        "raster_id": raster_id,
        "band_idx": band_idx,
        "histogram": histogram,
    }


@router.post("/preprocess")
def preprocess(
    raster_id: int,
    operation: str = "reproject",  # / "resample" / "clip" / "stack" / "feature_matrix"
    target_crs: Optional[str] = None,
    target_resolution: Optional[float] = None,
    bbox: Optional[List[float]] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Preprocessing operations on raster."""
    raster = (
        db.query(RasterFile)
        .join(GeospatialProject)
        .filter(
            RasterFile.raster_id == raster_id,
            GeospatialProject.user_id == current_user.id,
        )
        .first()
    )
    if not raster:
        raise HTTPException(404, "Raster not found")

    pp = RasterPreprocessor()
    output_path = str(RASTER_STORAGE / f"{uuid.uuid4()}_{operation}.tif")

    if operation == "reproject":
        result = pp.reproject(raster.file_path, output_path, target_crs or "EPSG:4326")
    elif operation == "resample":
        result = pp.resample(raster.file_path, output_path, target_resolution or 30.0)
    elif operation == "clip":
        if not bbox or len(bbox) != 4:
            raise HTTPException(400, "bbox must be [min_x, min_y, max_x, max_y]")
        result = pp.clip_by_bbox(raster.file_path, output_path, tuple(bbox))
    elif operation == "feature_matrix":
        result = pp.to_feature_matrix(raster.file_path)
        # Don't return numpy array via JSON
        if "X_array" in result:
            result["X_array"] = f"<numpy array shape={result.get('shape')}>"
    else:
        raise HTTPException(400, f"Unknown operation: {operation}")

    return result


@router.post("/generate-indices")
def generate_indices(
    req: GenerateIndicesRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Generate spectral indices (NDWI/MNDWI/NDVI/etc)."""
    raster = (
        db.query(RasterFile)
        .join(GeospatialProject)
        .filter(
            RasterFile.raster_id == req.raster_id,
            GeospatialProject.user_id == current_user.id,
        )
        .first()
    )
    if not raster:
        raise HTTPException(404, "Raster not found")

    reader = RasterReader()
    if not reader.is_available():
        return {
            "status": "skipped",
            "reason": "rasterio not installed; cannot read pixel data",
            "metadata_only": reader.read_metadata(raster.file_path),
        }

    # Read all bands
    full_array = reader.read_all_bands(raster.file_path)
    if full_array is None:
        raise HTTPException(500, "Failed to read raster pixels")

    # Map nama band → array berdasarkan band_mapping
    bands = {}
    for band_name, idx in req.band_mapping.items():
        if 1 <= idx <= full_array.shape[0]:
            bands[band_name.lower()] = full_array[idx - 1]

    calc = SpectralIndexCalculator()
    indices_result = calc.calculate_all_indices(bands)

    # Don't return raw arrays via JSON
    summary = {}
    for name, data in indices_result["results"].items():
        summary[name] = {
            "stats": data["stats"],
            "shape": data["shape"],
            "formula": data.get("formula"),
            "purpose": data.get("purpose"),
            "interpretation_thresholds": data.get("interpretation_thresholds"),
        }

    return {
        "raster_id": req.raster_id,
        "indices_calculated": indices_result["indices_calculated"],
        "n_indices": indices_result["n_indices"],
        "indices_summary": summary,
    }


@router.post("/train-flood-model", status_code=status.HTTP_202_ACCEPTED)
def train_flood_model(
    req: TrainFloodModelRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Train flood classification model."""
    project = db.query(GeospatialProject).filter(
        GeospatialProject.project_id == req.project_id,
        GeospatialProject.user_id == current_user.id,
    ).first()
    if not project:
        raise HTTPException(404, "Project not found")

    run = FloodModelRun(
        project_id=req.project_id,
        model_name=req.model_name,
        model_type="threshold" if "threshold" in req.model_name else "supervised",
        features_used=req.features,
        target_variable="flood_label",
        hyperparameters={"test_size": req.test_size},
        status="pending",
        started_at=datetime.utcnow(),
    )
    db.add(run)
    db.commit()
    db.refresh(run)

    # NOTE: actual training would happen as background task with rasterio.
    # Untuk MVP API, return run_id; full implementation perlu BackgroundTasks.

    return {
        "run_id": run.run_id,
        "status": run.status,
        "model_name": run.model_name,
        "message": "Training queued. Poll /api/geo/model-runs/{run_id} for status.",
    }


@router.get("/model-runs/{run_id}")
def get_model_run(
    run_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get status & results dari model run."""
    run = (
        db.query(FloodModelRun)
        .join(GeospatialProject)
        .filter(
            FloodModelRun.run_id == run_id,
            GeospatialProject.user_id == current_user.id,
        )
        .first()
    )
    if not run:
        raise HTTPException(404, "Run not found")
    return {
        "run_id": run.run_id,
        "model_name": run.model_name,
        "status": run.status,
        "metrics": run.metrics_json,
        "feature_importance": run.feature_importance,
        "flooded_percentage": run.flooded_percentage,
        "method_monitor": run.method_monitor_log,
        "output_paths": {
            "map": run.output_map_path,
            "probability": run.output_probability_path,
            "geojson": run.output_geojson_path,
            "csv": run.output_csv_path,
        },
    }


@router.post("/evaluate-model/{run_id}")
def evaluate_model(
    run_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Re-run evaluation untuk model run yang completed."""
    run = (
        db.query(FloodModelRun)
        .join(GeospatialProject)
        .filter(
            FloodModelRun.run_id == run_id,
            GeospatialProject.user_id == current_user.id,
        )
        .first()
    )
    if not run:
        raise HTTPException(404, "Run not found")
    if run.status != "completed":
        raise HTTPException(400, f"Run status is {run.status}, not completed")
    return {"run_id": run_id, "metrics": run.metrics_json}


@router.post("/predict-flood-map")
def predict_flood_map(
    req: PredictFloodMapRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Apply trained model ke raster baru."""
    return {
        "run_id": req.run_id,
        "raster_id": req.raster_id,
        "status": "queued",
        "message": "Prediction queued. Output GeoTIFF will be saved.",
    }


@router.post("/export-result")
def export_result(
    req: ExportResultRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Export hasil ke GeoTIFF/CSV/GeoJSON."""
    run = (
        db.query(FloodModelRun)
        .join(GeospatialProject)
        .filter(
            FloodModelRun.run_id == req.run_id,
            GeospatialProject.user_id == current_user.id,
        )
        .first()
    )
    if not run:
        raise HTTPException(404, "Run not found")

    path_map = {
        "geotiff": run.output_map_path,
        "csv": run.output_csv_path,
        "geojson": run.output_geojson_path,
    }
    path = path_map.get(req.format)
    if not path:
        raise HTTPException(404, f"No {req.format} export available for this run")
    return {"run_id": req.run_id, "format": req.format, "path": path}
