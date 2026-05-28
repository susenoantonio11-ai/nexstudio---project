"""
Advanced AI Models — FastAPI router

Generic dispatcher endpoint:
  POST /api/ai/advanced/{model_id}/run  → executes the model with payload
  GET  /api/ai/advanced/list             → lists all 100 models
  GET  /api/ai/advanced/categories       → lists 10 category labels
  GET  /api/ai/advanced/{model_id}       → model metadata
"""
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.ai_models.advanced import run_model, list_models, get_model
from app.ai_models.advanced.registry import CATEGORY_LABELS

router = APIRouter(prefix="/api/ai/advanced", tags=["advanced_ai"])


class RunRequest(BaseModel):
    payload: Optional[Dict[str, Any]] = None


@router.get("/list")
def list_all(category: Optional[str] = None) -> Dict[str, Any]:
    models = list_models(category=category)
    return {"status": "success", "count": len(models), "models": models}


@router.get("/categories")
def categories() -> Dict[str, Any]:
    return {"status": "success", "categories": CATEGORY_LABELS}


@router.get("/{model_id}")
def model_meta(model_id: str) -> Dict[str, Any]:
    cls = get_model(model_id)
    if cls is None:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")
    return {
        "status": "success",
        "id": cls.model_id, "name": cls.name, "domain": cls.domain, "category": cls.category,
        "description": cls.description, "why_used": cls.why_used, "why_not_others": cls.why_not_others,
        "formulas": cls.formulas, "limitations": cls.limitations, "citations": cls.citations,
        "dependencies": cls.dependencies, "fallback_available": cls.fallback_available,
        "realtime_capable": cls.realtime_capable, "integration_targets": cls.integration_targets,
    }


@router.post("/{model_id}/run")
def run(model_id: str, req: Optional[RunRequest] = None) -> Dict[str, Any]:
    payload = (req.payload if req else None) or {}
    result = run_model(model_id, payload)
    if result.get("status") == "error":
        raise HTTPException(status_code=404, detail=result.get("message"))
    return result
