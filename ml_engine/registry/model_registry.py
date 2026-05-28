"""
Nexlytics AI Model Registry
============================
Single source of truth for every AI model in the platform.

Every new model MUST register itself here. The registry then powers:
  - GET /api/ai/registry/list       (frontend auto-discovery)
  - GET /api/ai/registry/health     (system health monitoring)
  - GET /api/ai/registry/explain/X  (Method Monitor metadata)
  - Auto API routing (each model can declare its endpoint path)
  - Auto Method Monitor integration

Use the @register_model decorator on new model classes, or call
`register_class(...)` for procedural registration.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional
from datetime import datetime
import time


@dataclass
class ModelEntry:
    """Metadata + handle for one registered AI model."""
    id: str                                  # "risk_score_engine"
    name: str                                # "RiskScoreEngine"
    domain: str                              # "disaster" | "quality" | "geospatial" | ...
    category: str                            # "risk_scoring" | "classifier" | "validator" | ...
    description: str
    formula: Optional[str] = None
    citations: List[str] = field(default_factory=list)
    api_endpoint: Optional[str] = None       # "/api/disaster/risk/assess"
    method_monitor: Dict[str, Any] = field(default_factory=dict)
    handler: Optional[Callable] = None       # Optional callable for direct invocation
    dependencies: List[str] = field(default_factory=list)
    fallback_available: bool = True
    version: str = "1.0.0"
    registered_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    last_health_check: Optional[str] = None
    last_status: str = "registered"          # registered | healthy | degraded | unavailable
    response_latency_ms: Optional[float] = None
    integration_targets: List[str] = field(default_factory=list)  # ["EWC", "DataWorkspace", ...]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("handler", None)  # callable not serializable
        return d


class ModelRegistry:
    """Registry singleton. Tracks every AI model in Nexlytics."""

    def __init__(self):
        self._models: Dict[str, ModelEntry] = {}

    def register(self, entry: ModelEntry) -> ModelEntry:
        if entry.id in self._models:
            existing = self._models[entry.id]
            # update non-handler metadata while preserving registered_at
            entry.registered_at = existing.registered_at
        self._models[entry.id] = entry
        return entry

    def get(self, model_id: str) -> Optional[ModelEntry]:
        return self._models.get(model_id)

    def list_all(self) -> List[Dict[str, Any]]:
        return [m.to_dict() for m in self._models.values()]

    def list_by_domain(self, domain: str) -> List[Dict[str, Any]]:
        return [m.to_dict() for m in self._models.values() if m.domain == domain]

    def health_check_all(self) -> Dict[str, Any]:
        results = []
        healthy_count = 0
        total = 0
        for entry in self._models.values():
            total += 1
            check = self._health_check_one(entry)
            entry.last_health_check = check["timestamp"]
            entry.last_status = check["status"]
            entry.response_latency_ms = check.get("latency_ms")
            if check["status"] == "healthy":
                healthy_count += 1
            results.append(check)
        return {
            "total_models": total,
            "healthy": healthy_count,
            "unhealthy": total - healthy_count,
            "checked_at": datetime.utcnow().isoformat() + "Z",
            "results": results,
        }

    def _health_check_one(self, entry: ModelEntry) -> Dict[str, Any]:
        start = time.time()
        status = "healthy"
        message = "OK"
        # Optional handler ping
        if entry.handler and callable(entry.handler):
            try:
                # Don't actually call the handler with random args; just verify it's callable.
                _ = callable(entry.handler)
            except Exception as e:
                status = "degraded"
                message = f"Handler check failed: {e}"
        # Check declared dependencies
        for dep in entry.dependencies:
            try:
                __import__(dep)
            except ImportError:
                status = "degraded"
                message = f"Optional dependency '{dep}' unavailable; fallback active"
                break
        latency_ms = round((time.time() - start) * 1000, 2)
        return {
            "id": entry.id,
            "name": entry.name,
            "status": status,
            "message": message,
            "latency_ms": latency_ms,
            "fallback_available": entry.fallback_available,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }


# Global singleton
registry = ModelRegistry()


def register_class(
    *, id: str, name: str, domain: str, category: str, description: str,
    formula: Optional[str] = None, citations: Optional[List[str]] = None,
    api_endpoint: Optional[str] = None, method_monitor: Optional[Dict[str, Any]] = None,
    handler: Optional[Callable] = None, dependencies: Optional[List[str]] = None,
    fallback_available: bool = True, version: str = "1.0.0",
    integration_targets: Optional[List[str]] = None,
) -> ModelEntry:
    """Procedurally register a model entry."""
    return registry.register(ModelEntry(
        id=id, name=name, domain=domain, category=category, description=description,
        formula=formula or None, citations=citations or [], api_endpoint=api_endpoint,
        method_monitor=method_monitor or {}, handler=handler,
        dependencies=dependencies or [], fallback_available=fallback_available,
        version=version, integration_targets=integration_targets or [],
    ))


def register_model(**meta):
    """Decorator form. Apply to a class to auto-register it."""
    def decorator(cls):
        entry = register_class(handler=cls, **meta)
        cls._registry_id = entry.id
        return cls
    return decorator
