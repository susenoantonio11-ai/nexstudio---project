"""
Quality Engine API — /api/ai/quality/*
Wraps AnalysisQualityEngine + sub-validators.
"""
from __future__ import annotations
from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
from datetime import datetime
import sys, pathlib
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ml_engine.domains.quality_engine import AnalysisQualityEngine

router = APIRouter(prefix="/api/ai/quality", tags=["AnalysisQualityEngine"])
_engine = AnalysisQualityEngine()


class QualityAssessRequest(BaseModel):
    # Silence Pydantic v2 protected_namespaces warning · model_eval bukan
    # reserved attribute, kita pakai sebagai field domain quality assessment.
    model_config = {"protected_namespaces": ()}

    data: Optional[Dict[str, Any]] = None
    model_eval: Optional[Dict[str, Any]] = None
    predictions: Optional[List[Dict[str, Any]]] = None
    instance: Optional[Dict[str, float]] = None
    domain: str = "score"
    result: Optional[Dict[str, Any]] = None


@router.post("/assess", summary="AnalysisQualityEngine — full quality scorecard")
def quality_assess(payload: QualityAssessRequest):
    return _engine.assess(
        data=payload.data, model_eval=payload.model_eval, predictions=payload.predictions,
        instance=payload.instance, domain=payload.domain, result=payload.result
    )


class DatasetValidateRequest(BaseModel):
    rows: Optional[List[List[Any]]] = None
    columns: Optional[List[str]] = None
    target: Optional[str] = None
    values: Optional[List[Any]] = None


@router.post("/validate-dataset", summary="DataQualityValidator")
def validate_dataset(payload: DatasetValidateRequest):
    return _engine.data_q.assess(payload.dict(exclude_none=True))


class ModelValidateRequest(BaseModel):
    y_true: List[Any]
    y_pred: List[Any]
    y_proba: Optional[List[float]] = None
    task: str = "classification"


@router.post("/validate-model", summary="ModelQualityValidator")
def validate_model(payload: ModelValidateRequest):
    if payload.task == "regression":
        return _engine.model_q.assess_regression(payload.y_true, payload.y_pred)
    return _engine.model_q.assess_classification(payload.y_true, payload.y_pred, payload.y_proba)


class UncertaintyRequest(BaseModel):
    point_estimate: float
    std_dev: float = 0.0
    sample_size: int = 100
    confidence_level: float = 0.95


@router.post("/uncertainty", summary="UncertaintyEstimator")
def uncertainty(payload: UncertaintyRequest):
    return _engine.unc.estimate(
        point_estimate=payload.point_estimate, std_dev=payload.std_dev,
        sample_size=payload.sample_size, confidence_level=payload.confidence_level
    )


class ExplainRequest(BaseModel):
    instance: Dict[str, float]
    base_value: float = 0.5


@router.post("/explain", summary="ExplainabilityChecker — permutation importance")
def explain(payload: ExplainRequest):
    return _engine.exp.check(payload.instance, base_value=payload.base_value)


@router.get("/health", summary="Quality engine health probe")
def health():
    return {
        "status": "ok",
        "engine": "AnalysisQualityEngine",
        "sub_validators": [
            "DataQualityValidator", "ModelQualityValidator", "CrossValidationEngine",
            "UncertaintyEstimator", "EnsembleVerifier", "ScientificConsistencyChecker",
            "ExplainabilityChecker"
        ],
        "checked_at": datetime.utcnow().isoformat() + "Z",
    }
