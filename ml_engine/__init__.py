"""
NEXLYTICS ML Engine
====================
Modular AI engine following CRISP-DM methodology.

Pipeline:
    Dataset -> Profiler -> Target Detector -> Feature Selector
    -> Model Selector -> ML Runner -> Method Monitor -> Insight

Each component returns structured output that includes WHY decisions
were made, supporting the Explainable AI principle.
"""

from .profilers.data_profiler import DataProfiler
from .detectors.target_detector import TargetDetector
from .selectors.model_selector import ModelSelector
from .monitors.method_monitor import MethodMonitor

__version__ = "1.0.0"

__all__ = [
    "DataProfiler",
    "TargetDetector",
    "ModelSelector",
    "MethodMonitor",
]
