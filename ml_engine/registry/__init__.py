"""Nexlytics AI Model Registry — auto-registration & discovery."""
from .model_registry import (
    ModelRegistry, ModelEntry, registry, register_model, register_class
)

__all__ = ["ModelRegistry", "ModelEntry", "registry", "register_model", "register_class"]
