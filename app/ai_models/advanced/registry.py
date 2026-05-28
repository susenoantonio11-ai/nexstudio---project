"""
Advanced AI Model Registry — central catalog of all 100 models.
Auto-registers them in the ml_engine ModelRegistry with full metadata.
"""
from typing import Any, Dict, List, Type
from .base import AdvancedAIModel

# Phase-1 — original 10 categories (100 models)
from .scientific_reasoning import MODELS as M_A
from .multimodal import MODELS as M_B
from .forecasting import MODELS as M_C
from .disaster_intelligence import MODELS as M_D
from .vision import MODELS as M_E
from .nlp_knowledge import MODELS as M_F
from .analytical import MODELS as M_G
from .quality_trust import MODELS as M_H
from .realtime import MODELS as M_I
from .future_research import MODELS as M_J

# Phase-2 — next-phase 10 categories (100 more models)
from .agentic import MODELS as M_K
from .digital_twin import MODELS as M_L
from .causal import MODELS as M_M
from .geo_foundation import MODELS as M_N
from .simulation import MODELS as M_O
from .uncertainty_trust import MODELS as M_P
from .llm_research import MODELS as M_Q
from .self_learning import MODELS as M_R
from .national_intelligence import MODELS as M_S
from .explainable import MODELS as M_T


CATEGORY_LABELS = {
    # Phase-1
    "scientific_reasoning":  "A · Scientific Reasoning",
    "multimodal":            "B · Multimodal AI",
    "forecasting":           "C · Forecasting",
    "disaster_intelligence": "D · Disaster Intelligence",
    "vision":                "E · Computer Vision",
    "nlp_knowledge":         "F · NLP & Knowledge",
    "analytical":            "G · Analytical Intelligence",
    "quality_trust":         "H · Quality & Trust",
    "realtime":              "I · Realtime Intelligence",
    "future_research":       "J · Future Research",
    # Phase-2
    "agentic":               "K · Agentic AI Systems",
    "digital_twin":          "L · Digital Twin AI",
    "causal":                "M · Causal AI",
    "geo_foundation":        "N · Geo-AI Foundation Models",
    "simulation":            "O · Simulation AI",
    "uncertainty_trust":     "P · Uncertainty & Trust",
    "llm_research":          "Q · LLM Research AI",
    "self_learning":         "R · Self-Learning AI",
    "national_intelligence": "S · National Intelligence",
    "explainable":           "T · Human-Centered Explainable",
}

ALL_MODEL_CLASSES: List[Type[AdvancedAIModel]] = (
    list(M_A) + list(M_B) + list(M_C) + list(M_D) + list(M_E)
    + list(M_F) + list(M_G) + list(M_H) + list(M_I) + list(M_J)
    + list(M_K) + list(M_L) + list(M_M) + list(M_N) + list(M_O)
    + list(M_P) + list(M_Q) + list(M_R) + list(M_S) + list(M_T)
)

MODEL_CATALOG: Dict[str, Type[AdvancedAIModel]] = {
    cls.model_id: cls for cls in ALL_MODEL_CLASSES
}


def get_model(model_id: str):
    return MODEL_CATALOG.get(model_id)


def list_models(category: str = None) -> List[Dict[str, Any]]:
    out = []
    for cls in ALL_MODEL_CLASSES:
        if category and cls.category != category:
            continue
        out.append({
            "id": cls.model_id, "name": cls.name, "category": cls.category, "domain": cls.domain,
            "description": cls.description, "citations": cls.citations, "dependencies": cls.dependencies,
            "fallback_available": cls.fallback_available, "realtime_capable": cls.realtime_capable,
            "integration_targets": cls.integration_targets,
        })
    return out


def register_all_advanced(register_class):
    """Register every advanced model into the existing ml_engine ModelRegistry."""
    n = 0
    for cls in ALL_MODEL_CLASSES:
        try:
            register_class(
                id=cls.model_id, name=cls.name, domain=cls.domain, category=cls.category,
                description=cls.description, formula=";  ".join(cls.formulas) if cls.formulas else None,
                citations=cls.citations, dependencies=cls.dependencies,
                api_endpoint=f"/api/ai/advanced/{cls.model_id}/run",
                fallback_available=cls.fallback_available,
                method_monitor={"method": cls.name, "why_used": cls.why_used,
                                "limitations": cls.limitations},
                integration_targets=cls.integration_targets,
            )
            n += 1
        except Exception as e:
            print(f"[advanced.register] failed {cls.model_id}: {e}")
    return n
