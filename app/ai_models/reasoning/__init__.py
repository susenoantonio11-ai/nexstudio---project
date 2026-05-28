"""
Reasoning sub-package — AI reasoning before model selection.

Contains:
  DynamicModelSelectionEngine — algorithm-agnostic, dataset-aware ranking.
"""
from .dynamic_model_selector import DynamicModelSelectionEngine

__all__ = ["DynamicModelSelectionEngine"]
