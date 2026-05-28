"""
AI Model Registry API — /api/ai/registry/*
Auto-discovery, health monitoring, and Method Monitor metadata for every registered model.
"""
from __future__ import annotations
from fastapi import APIRouter, HTTPException
from typing import Any, Dict
import sys, pathlib
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ml_engine.registry import registry
from ml_engine.registry import bootstrap  # triggers registration on import

router = APIRouter(prefix="/api/ai/registry", tags=["AIModelRegistry"])


@router.get("/list", summary="List every registered AI model")
def list_models(domain: str | None = None):
    if domain:
        return {"domain": domain, "models": registry.list_by_domain(domain)}
    return {"total": len(registry.list_all()), "models": registry.list_all()}


@router.get("/health", summary="Health check across all registered models")
def health_check():
    return registry.health_check_all()


@router.get("/explain/{model_id}", summary="Method Monitor metadata for a model")
def explain_model(model_id: str):
    entry = registry.get(model_id)
    if not entry:
        raise HTTPException(404, f"Model not found: {model_id}")
    return entry.to_dict()
