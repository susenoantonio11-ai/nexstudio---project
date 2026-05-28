"""
ComputerVisionPipeline
======================
Orchestrates the full image-analysis flow for the frontend Image Analysis
page. Each step is logged in the response so the Method Monitor drawer can
display the actual operations performed:

  upload → preprocess → extract_features → classify/segment/detect →
  evaluate → explain → emit_widget_payload

Pipeline never crashes on missing DL libs; it simply skips the deep step
and continues with handcrafted features.
"""
from __future__ import annotations

import io
import time
from typing import Any, Dict, Optional

import numpy as np

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

from .image_analysis_model import ImageAnalysisAIModel
from .visual_feature_extractor import VisualFeatureExtractor
from .image_explainability_engine import ImageExplainabilityEngine


def _preprocess(image_bytes: bytes, target_size=(256, 256)) -> Dict[str, Any]:
    if not HAS_PIL:
        return {"status": "error", "message": "Pillow not available"}
    try:
        im = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        original = im.size
        im2 = im.resize(target_size)
        return {
            "status": "success",
            "original_size": list(original),
            "resized_size": list(target_size),
            "mode": im.mode,
            "method": "Pillow.Image.resize (bilinear, default)",
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


class ComputerVisionPipeline:
    """End-to-end CV pipeline orchestrator."""

    name = "ComputerVisionPipeline"
    domain = "computer_vision"

    def __init__(self) -> None:
        self.analyzer = ImageAnalysisAIModel()
        self.extractor = VisualFeatureExtractor()
        self.explainer = ImageExplainabilityEngine()

    def run(self, image_bytes: bytes, payload: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.perf_counter()
        steps = []
        # 1. Preprocess
        pre = _preprocess(image_bytes)
        steps.append({"step": "preprocess", "method": pre.get("method", "—"), "status": pre["status"]})

        # 2. Feature extraction
        feats = self.extractor.extract_one(image_bytes)
        steps.append({"step": "feature_extraction", "method": "VisualFeatureExtractor (color hist + Sobel + LBP + Hu)", "status": feats.get("status", "error"), "feature_length": feats.get("feature_length", 0)})

        # 3. Analysis (classification / segmentation / detection / all)
        analysis = self.analyzer.analyze(image_bytes, payload)
        steps.append({"step": "analysis", "method": analysis.get("method_monitor", {}).get("method", "—"), "status": analysis["status"]})

        # 4. Explainability
        explain = self.explainer.explain(image_bytes, payload, prediction=analysis.get("classification"))
        steps.append({"step": "explain", "method": explain.get("method", "—"), "status": explain.get("status", "ok")})

        # 5. Widget payload
        widget = self._widget_payload(analysis, feats, explain)

        duration_ms = int((time.perf_counter() - t0) * 1000)
        return {
            "status": "success",
            "model_name": self.name,
            "image_meta": pre,
            "analysis": analysis,
            "features": {"feature_length": feats.get("feature_length", 0), "vector": feats.get("feature_vector", [])},
            "explanation": explain,
            "widget_payload": widget,
            "pipeline_steps": steps,
            "duration_ms": duration_ms,
            "method_monitor": {
                "method": "Linear pipeline: preprocess → feature_extraction → analysis → explain",
                "why_used": "Reproducible, debuggable, every step is logged & citable.",
                "limitations": ["No model training inside the pipeline; analyzer is rule-based/k-NN."],
            },
        }

    def _widget_payload(self, analysis: Dict[str, Any], feats: Dict[str, Any], explain: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "type": "image_classification_widget",
            "predicted_label": analysis.get("classification", {}).get("predicted_label"),
            "confidence": analysis.get("classification", {}).get("confidence"),
            "top_k": analysis.get("classification", {}).get("top_k", []),
            "image_quality": analysis.get("image_quality"),
            "feature_length": feats.get("feature_length", 0),
            "saliency_summary": explain.get("region_importance"),
        }
