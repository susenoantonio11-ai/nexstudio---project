"""
ComplexDataScienceAIEngine — FastAPI router
==========================================
Mounts all 11 endpoints required by the front-end Data Science module.

POST /api/ai/complex/analyze-dataset       — ComplexDatasetAnalyzer
POST /api/ai/complex/profile-large-dataset — LargeDatasetProcessingEngine (file path)
POST /api/ai/complex/detect-target         — Target detection only (lighter)
POST /api/ai/complex/detect-patterns       — Hidden patterns only (mutual info)
POST /api/ai/image/analyze                 — ImageAnalysisAIModel (multipart upload)
POST /api/ai/image/classify                — Convenience: task=classification
POST /api/ai/image/segment                 — Convenience: task=segmentation
POST /api/ai/image/detect-objects          — Convenience: task=object_detection
POST /api/ai/image/extract-features        — VisualFeatureExtractor
POST /api/ai/image/build-dataset           — ImageDatasetBuilder (folder path)
POST /api/ai/image/explain                 — ImageExplainabilityEngine

Every response uses the standard envelope:
  { status, model_name, ..., method_monitor: { method, why_used, formulas, limitations, citations } }
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.ai_models import (
    ComplexDatasetAnalyzer,
    LargeDatasetProcessingEngine,
    MultimodalDataScienceEngine,
    ImageAnalysisAIModel,
    ComputerVisionPipeline,
    ImageDatasetBuilder,
    VisualFeatureExtractor,
    ImageExplainabilityEngine,
    MultisourceFloodFusion,
    DynamicModelSelectionEngine,
)
import numpy as np

router = APIRouter(prefix="/api/ai", tags=["complex_data_science"])


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------
class AnalyzeDatasetRequest(BaseModel):
    rows: list = Field(default_factory=list)
    target: Optional[str] = None
    columns: Optional[list] = None


class LargeFileRequest(BaseModel):
    file_path: str
    chunksize: int = 50_000
    max_in_memory_rows: int = 200_000


class TargetDetectRequest(BaseModel):
    rows: list = Field(default_factory=list)


class FuseRequest(BaseModel):
    tabular_rows: list = Field(default_factory=list)
    image_features: Optional[list] = None
    texts: Optional[list] = None
    time_series: Optional[list] = None
    target: Optional[list] = None
    task: str = "auto"


class BuildImageDatasetRequest(BaseModel):
    folder: str
    blur_threshold: float = 100.0
    duplicate_threshold: int = 6


# ---------------------------------------------------------------------------
# COMPLEX DATA endpoints
# ---------------------------------------------------------------------------
@router.post("/complex/analyze-dataset")
def analyze_dataset(req: AnalyzeDatasetRequest) -> Dict[str, Any]:
    if not req.rows:
        raise HTTPException(status_code=400, detail="rows must not be empty")
    return ComplexDatasetAnalyzer().analyze({
        "rows": req.rows,
        "target": req.target,
        "columns": req.columns,
    })


@router.post("/complex/profile-large-dataset")
def profile_large_dataset(req: LargeFileRequest) -> Dict[str, Any]:
    eng = LargeDatasetProcessingEngine(
        chunksize=req.chunksize,
        max_in_memory_rows=req.max_in_memory_rows,
    )
    return eng.profile(req.file_path)


@router.post("/complex/detect-target")
def detect_target(req: TargetDetectRequest) -> Dict[str, Any]:
    res = ComplexDatasetAnalyzer().analyze({"rows": req.rows})
    return {
        "status": res.get("status", "error"),
        "model_name": "ComplexDatasetAnalyzer.detect_target",
        "detected_target_variable": res.get("detected_target_variable"),
        "target_meta": res.get("target_meta"),
        "method_monitor": {
            "method": "Heuristic ranking (name hint + low cardinality + right-edge column)",
            "why_used": "Quick suggestion; does not require ground-truth labels.",
        },
    }


@router.post("/complex/detect-patterns")
def detect_patterns(req: AnalyzeDatasetRequest) -> Dict[str, Any]:
    res = ComplexDatasetAnalyzer().analyze({"rows": req.rows, "target": req.target})
    return {
        "status": res.get("status", "error"),
        "model_name": "ComplexDatasetAnalyzer.detect_patterns",
        "hidden_patterns": res.get("hidden_patterns"),
        "relationships": res.get("dataset_profile", {}).get("relationships", []),
        "method_monitor": {
            "method": "Mutual information (Kraskov et al., 2004) + Pearson relationship graph",
        },
    }


@router.post("/complex/multimodal-fuse")
def multimodal_fuse(req: FuseRequest) -> Dict[str, Any]:
    return MultimodalDataScienceEngine().fuse(req.dict())


# ---------------------------------------------------------------------------
# IMAGE endpoints
# ---------------------------------------------------------------------------
@router.post("/image/analyze")
async def image_analyze(
    file: UploadFile = File(...),
    domain: str = Form("product"),
    task: str = Form("classification"),
) -> Dict[str, Any]:
    blob = await file.read()
    if not blob:
        raise HTTPException(status_code=400, detail="empty file")
    pipeline = ComputerVisionPipeline()
    return pipeline.run(blob, {"domain": domain, "task": task})


@router.post("/image/classify")
async def image_classify(file: UploadFile = File(...), domain: str = Form("product")) -> Dict[str, Any]:
    blob = await file.read()
    return ImageAnalysisAIModel().analyze(blob, {"domain": domain, "task": "classification"})


@router.post("/image/segment")
async def image_segment(file: UploadFile = File(...), domain: str = Form("product")) -> Dict[str, Any]:
    blob = await file.read()
    return ImageAnalysisAIModel().analyze(blob, {"domain": domain, "task": "segmentation"})


@router.post("/image/detect-objects")
async def image_detect_objects(file: UploadFile = File(...), domain: str = Form("product")) -> Dict[str, Any]:
    blob = await file.read()
    return ImageAnalysisAIModel().analyze(blob, {"domain": domain, "task": "object_detection"})


@router.post("/image/extract-features")
async def image_extract_features(file: UploadFile = File(...)) -> Dict[str, Any]:
    blob = await file.read()
    return VisualFeatureExtractor().extract_one(blob)


@router.post("/image/build-dataset")
def image_build_dataset(req: BuildImageDatasetRequest) -> Dict[str, Any]:
    return ImageDatasetBuilder().build(
        root=req.folder,
        blur_threshold=req.blur_threshold,
        duplicate_threshold=req.duplicate_threshold,
    )


@router.post("/image/explain")
async def image_explain(
    file: UploadFile = File(...),
    domain: str = Form("product"),
    predicted_label: Optional[str] = Form(None),
    confidence: Optional[float] = Form(None),
) -> Dict[str, Any]:
    blob = await file.read()
    pred = {"predicted_label": predicted_label, "confidence": confidence} if predicted_label else None
    return ImageExplainabilityEngine().explain(blob, {"domain": domain}, prediction=pred)


class MultisourceFuseRequest(BaseModel):
    """Each layer is a dict { source_id: { band: 2D-array-as-list-of-lists } }.
    All arrays must already share the same (H, W) shape."""
    layers: Dict[str, Dict[str, list]]
    target_resolution_m: int = 30
    target_crs: str = "EPSG:4326"


@router.post("/complex/multisource-flood-fuse")
def multisource_flood_fuse(req: MultisourceFuseRequest) -> Dict[str, Any]:
    """Stack multi-source rasters → train Random Forest flood classifier.
    Caller pre-resamples each band to a common grid; this endpoint does the
    feature engineering (NDWI, MNDWI, TWI, S1 rules, CN, distance-to-river)
    and trains a baseline if a `bnpb_event_mask` layer is present."""
    layers_np: Dict[str, Dict[str, Any]] = {}
    for sid, bands in (req.layers or {}).items():
        layers_np[sid] = {}
        for bname, arr in (bands or {}).items():
            layers_np[sid][bname] = np.array(arr, dtype=float)
    eng = MultisourceFloodFusion(
        target_resolution_m=req.target_resolution_m,
        target_crs=req.target_crs,
    )
    return eng.fuse(layers_np)


class ModelSelectionRequest(BaseModel):
    """Algorithm-agnostic dataset characteristics. All fields optional —
    engine fills sensible defaults via _normalize."""
    n_rows: int = 0
    n_features: int = 0
    n_numeric: int = 0
    n_categorical: int = 0
    missing_pct: float = 0.0
    imbalance_ratio: float = 1.0
    has_temporal: bool = False
    has_spatial: bool = False
    has_text: bool = False
    has_image: bool = False
    nonlinear_relationship: bool = False
    sparse_features: bool = False
    seasonal: bool = False
    multivariate: bool = False
    multimodal: bool = False
    has_static_features: bool = False
    target_type: str = "binary"
    domain_hint: str = "general"
    task: str = "classification"


@router.post("/reasoning/select-model")
def reasoning_select_model(req: ModelSelectionRequest) -> Dict[str, Any]:
    """Algorithm-agnostic, dataset-aware model recommendation with full
    reasoning timeline. Returns ranked candidates + non-prioritized list +
    confidence + Method Monitor envelope for the audit drawer."""
    return DynamicModelSelectionEngine().select(req.dict())


@router.get("/complex/health")
def complex_health() -> Dict[str, Any]:
    """Quick reachability check + dependency report."""
    deps = {}
    for mod_name in ("pandas", "sklearn", "numpy", "PIL", "skimage", "cv2", "pyarrow", "torch", "tensorflow"):
        try:
            __import__(mod_name)
            deps[mod_name] = "available"
        except ImportError:
            deps[mod_name] = "missing"
    return {
        "status": "ok",
        "engine": "ComplexDataScienceAIEngine",
        "models": [
            "ComplexDatasetAnalyzer", "LargeDatasetProcessingEngine", "MultimodalDataScienceEngine",
            "ImageAnalysisAIModel", "ComputerVisionPipeline", "ImageDatasetBuilder",
            "VisualFeatureExtractor", "ImageExplainabilityEngine",
        ],
        "dependencies": deps,
    }
