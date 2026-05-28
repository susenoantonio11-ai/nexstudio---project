"""
Advanced AI Model Ecosystem — 100 next-generation models grouped into 10
categories. All models inherit from AdvancedAIModel base class, share the
Method Monitor + confidence + uncertainty + fallback contract, and
auto-register in the existing ml_engine ModelRegistry.

Categories:
  A. Scientific Reasoning      (scientific_reasoning.py)
  B. Multimodal AI             (multimodal.py)
  C. Forecasting               (forecasting.py)
  D. Disaster Intelligence     (disaster_intelligence.py)
  E. Computer Vision           (vision.py)
  F. NLP & Knowledge           (nlp_knowledge.py)
  G. Analytical Intelligence   (analytical.py)
  H. Quality & Trust           (quality_trust.py)
  I. Realtime Intelligence     (realtime.py)
  J. Future Research           (future_research.py)
"""
from .base import AdvancedAIModel, run_model
from .registry import register_all_advanced, get_model, list_models, MODEL_CATALOG

__all__ = ["AdvancedAIModel", "run_model", "register_all_advanced",
           "get_model", "list_models", "MODEL_CATALOG"]
