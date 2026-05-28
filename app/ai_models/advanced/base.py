"""
AdvancedAIModel — base class for the 100 next-generation models.

Provides:
  * Standard envelope output
  * Method Monitor metadata helpers
  * Confidence + uncertainty estimation utilities
  * Fallback execution that always returns a valid envelope
  * Auto-timing
"""
from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


def _safe_mean(arr: List[float]) -> float:
    if not arr: return 0.0
    return sum(arr) / len(arr)


def _safe_std(arr: List[float]) -> float:
    if len(arr) < 2: return 0.0
    m = _safe_mean(arr)
    return math.sqrt(sum((x - m) ** 2 for x in arr) / (len(arr) - 1))


class AdvancedAIModel:
    """Base for all advanced models. Subclasses must implement `run(payload)`
    and set class-level metadata. The base supplies the envelope wrapper."""

    name: str = "AdvancedAIModel"
    model_id: str = "advanced_base"
    category: str = "advanced"
    domain: str = "ai_intelligence"
    description: str = "Base advanced AI model."
    why_used: str = "—"
    why_not_others: str = "—"
    formulas: List[str] = []
    limitations: List[str] = []
    citations: List[str] = []
    dependencies: List[str] = []
    fallback_available: bool = True
    realtime_capable: bool = False
    integration_targets: List[str] = ["MethodMonitor", "PrimaryMonitor"]

    # ------------------------------------------------------------------
    def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Override in subclass. Default returns ready envelope w/ no result."""
        return self._envelope({}, confidence=0.0, uncertainty=1.0,
                              note="Base run not implemented.")

    # ------------------------------------------------------------------
    def execute(self, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Wrapper that times + handles errors with fallback envelope."""
        t0 = time.perf_counter()
        try:
            res = self.run(payload or {})
            res["duration_ms"] = int((time.perf_counter() - t0) * 1000)
            return res
        except Exception as e:
            return self._envelope(
                {}, confidence=0.20, uncertainty=0.85,
                status="fallback", note=f"Exception: {e.__class__.__name__}: {e}",
                duration_ms=int((time.perf_counter() - t0) * 1000),
            )

    # ------------------------------------------------------------------
    def _envelope(
        self,
        result: Dict[str, Any],
        confidence: float = 0.75,
        uncertainty: float = 0.25,
        status: str = "success",
        note: Optional[str] = None,
        duration_ms: Optional[int] = None,
    ) -> Dict[str, Any]:
        env = {
            "status": status,
            "model_id": self.model_id,
            "model_name": self.name,
            "category": self.category,
            "domain": self.domain,
            "result": result,
            "confidence": round(min(1.0, max(0.0, confidence)), 4),
            "uncertainty": round(min(1.0, max(0.0, uncertainty)), 4),
            "method_monitor": {
                "method": self.name,
                "why_used": self.why_used,
                "why_not_other_methods": self.why_not_others,
                "description": self.description,
                "formulas": self.formulas,
                "limitations": self.limitations,
                "citations": self.citations,
                "dependencies": self.dependencies,
                "fallback_available": self.fallback_available,
                "realtime_capable": self.realtime_capable,
            },
            "integration_targets": self.integration_targets,
        }
        if note: env["note"] = note
        if duration_ms is not None: env["duration_ms"] = duration_ms
        return env


def run_model(model_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Generic dispatcher. Used by the API router."""
    from .registry import get_model
    cls = get_model(model_id)
    if cls is None:
        return {
            "status": "error",
            "model_id": model_id,
            "message": f"Model not found: {model_id}",
        }
    return cls().execute(payload)


# ---------------------------------------------------------------------------
# Lightweight numeric utilities reused by models
# ---------------------------------------------------------------------------
def confidence_from_signal(signal: float, baseline: float = 0.5,
                           amplitude: float = 0.3) -> float:
    """Map a unitless signal in [0, 1] to a confidence in [0.4, 0.95]."""
    s = max(0.0, min(1.0, signal))
    return round(0.4 + 0.55 * (baseline + amplitude * (s - 0.5)), 4)


def uncertainty_from_inputs(n: int = 0, completeness: float = 1.0,
                            baseline: float = 0.20) -> float:
    """Higher uncertainty when sample size small or completeness low."""
    n_factor = 1.0 / (1.0 + (n / 50.0)) if n > 0 else 1.0
    incomplete = max(0.0, 1.0 - completeness)
    return round(min(1.0, baseline + 0.4 * n_factor + 0.3 * incomplete), 4)
