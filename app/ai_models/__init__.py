"""
Nexlytics ComplexDataScienceAIEngine
====================================
Top-level package that hosts the eight new AI models requested for complex
dataset and image analysis. Models live in two sub-packages:

  complex_data/     ComplexDatasetAnalyzer
                    LargeDatasetProcessingEngine
                    MultimodalDataScienceEngine

  computer_vision/  ImageAnalysisAIModel
                    ComputerVisionPipeline
                    ImageDatasetBuilder
                    VisualFeatureExtractor
                    ImageExplainabilityEngine

Design contract (every model):
  * input         : explicit, validated dictionaries (no implicit globals).
  * output        : standardized "analysis envelope" — success/error,
                    model_name, method_monitor block, timing, citations.
  * dependencies  : never hard-fail. PyTorch / TensorFlow / OpenCV are all
                    optional; the engine uses scikit-learn + numpy + Pillow
                    + scikit-image as a guaranteed fallback path so the
                    backend keeps running even on a minimal Python env.
  * citations     : every method names its primary source paper, in line
                    with the FAIR data principles (Wilkinson et al., 2016)
                    and the W3C PROV-DM provenance model (W3C, 2013).
"""
from .complex_data.complex_dataset_analyzer import ComplexDatasetAnalyzer
from .complex_data.large_dataset_processing_engine import LargeDatasetProcessingEngine
from .complex_data.multimodal_engine import MultimodalDataScienceEngine
from .computer_vision.image_analysis_model import ImageAnalysisAIModel
from .computer_vision.computer_vision_pipeline import ComputerVisionPipeline
from .computer_vision.image_dataset_builder import ImageDatasetBuilder
from .computer_vision.visual_feature_extractor import VisualFeatureExtractor
from .computer_vision.image_explainability_engine import ImageExplainabilityEngine
from .multisource.multisource_flood_fusion import MultisourceFloodFusion
from .reasoning.dynamic_model_selector import DynamicModelSelectionEngine
from .research import (
    FloodPanelBuilder, HybridLSTMXGBoost, HybridSHAPExplainer,
    FloodResearchOrchestrator, INDONESIA_PROVINCES,
)

__all__ = [
    "ComplexDatasetAnalyzer",
    "LargeDatasetProcessingEngine",
    "MultimodalDataScienceEngine",
    "ImageAnalysisAIModel",
    "ComputerVisionPipeline",
    "ImageDatasetBuilder",
    "VisualFeatureExtractor",
    "ImageExplainabilityEngine",
    "MultisourceFloodFusion",
    "DynamicModelSelectionEngine",
    "FloodPanelBuilder",
    "HybridLSTMXGBoost",
    "HybridSHAPExplainer",
    "FloodResearchOrchestrator",
    "INDONESIA_PROVINCES",
]
