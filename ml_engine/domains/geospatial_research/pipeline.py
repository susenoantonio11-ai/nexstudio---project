"""
FloodResearchPipeline - end-to-end orchestrator untuk flood research.
=====================================================================
Menggabungkan: RasterReader → Preprocessor → SpectralIndices →
Classifier (supervised atau threshold) → Evaluator → Output.

Mengikuti CRISP-DM lengkap dengan Method Monitor logging.
"""
from __future__ import annotations
from typing import Dict, Any, List, Optional
import time
import numpy as np

from .raster_reader import RasterReader
from .raster_preprocessor import RasterPreprocessor
from .spectral_indices import SpectralIndexCalculator
from .threshold_classifier import ThresholdFloodClassifier
from .flood_classifier import FloodClassifier
from .change_detection import ChangeDetector
from .flood_evaluator import FloodEvaluator


class FloodResearchPipeline:
    """End-to-end pipeline flood research."""

    def __init__(self):
        self.reader = RasterReader()
        self.preprocessor = RasterPreprocessor()
        self.indices_calc = SpectralIndexCalculator()
        self.threshold_clf = ThresholdFloodClassifier()
        self.change_detector = ChangeDetector()
        self.evaluator = FloodEvaluator()
        self.steps: List[Dict[str, Any]] = []

    def run_unsupervised(
        self,
        bands: Dict[str, np.ndarray],
        method: str = "auto",
    ) -> Dict[str, Any]:
        """
        Pipeline tanpa label - threshold-based flood detection.
        method: 'auto' / 'mndwi' / 'ndwi' / 'sar_vv' / 'combined'
        """
        t0 = time.time()
        result: Dict[str, Any] = {
            "pipeline": "unsupervised_threshold",
            "started_at": time.time(),
            "method_steps": [],
        }

        # Step 1: Calculate indices
        indices = self.indices_calc.calculate_all_indices(bands)
        result["indices"] = {
            k: {"stats": v["stats"], "formula": v["formula"], "shape": v["shape"]}
            for k, v in indices["results"].items()
        }
        result["method_steps"].append({
            "step": "spectral_indices_calculation",
            "method": f"Calculated {indices['n_indices']} indices",
            "indices": indices["indices_calculated"],
        })

        # Step 2: Pilih method
        if method == "auto":
            method = self._auto_select_method(indices["results"], bands)

        # Step 3: Klasifikasi
        if method == "mndwi" and "mndwi" in indices["results"]:
            classification = self.threshold_clf.classify_with_mndwi(
                indices["results"]["mndwi"]["array"]
            )
        elif method == "ndwi" and "ndwi" in indices["results"]:
            classification = self.threshold_clf.classify_with_ndwi(
                indices["results"]["ndwi"]["array"]
            )
        elif method == "sar_vv" and "vv" in bands:
            classification = self.threshold_clf.classify_with_sar_vv(
                bands["vv"]
            )
        elif method == "combined":
            classification = self.threshold_clf.classify_combined(
                mndwi=indices["results"].get("mndwi", {}).get("array"),
                ndwi=indices["results"].get("ndwi", {}).get("array"),
                vv_db=bands.get("vv"),
                slope=indices["results"].get("slope", {}).get("array"),
            )
        else:
            return {"status": "error", "error": f"Method '{method}' tidak applicable dengan bands yang tersedia"}

        result["classification"] = {
            "method": classification["method"],
            "n_flooded_pixels": classification.get("n_flooded_pixels"),
            "flooded_percentage": classification.get("flooded_percentage"),
            "method_monitor": classification.get("method_monitor"),
            "flood_mask_shape": list(classification["flood_mask"].shape),
        }
        result["flood_mask"] = classification["flood_mask"]
        result["method_steps"].append({
            "step": "flood_classification",
            "method": classification["method"],
        })

        result["status"] = "success"
        result["duration_seconds"] = round(time.time() - t0, 2)
        return result

    def run_supervised(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        algorithm: str = "random_forest",
        raster_shape: Optional[tuple] = None,
    ) -> Dict[str, Any]:
        """
        Pipeline dengan label: train classifier + evaluate + (optional) reshape ke raster.
        """
        t0 = time.time()

        # Train/test split
        try:
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y if len(set(y)) > 1 else None,
            )
        except Exception as e:
            return {"status": "error", "error": str(e)}

        clf = FloodClassifier(algorithm=algorithm)
        fit_result = clf.fit(X_train, y_train, feature_names=feature_names)
        if fit_result["status"] != "success":
            return fit_result

        # Predict + evaluate
        pred_result = clf.predict(X_test)
        if pred_result["status"] != "success":
            return pred_result

        # Get probabilities for ROC-AUC
        y_proba = pred_result.get("probabilities")
        evaluation = self.evaluator.evaluate(y_test, pred_result["predictions"], y_proba)

        # Feature importance
        fi = clf.feature_importance()

        result = {
            "pipeline": "supervised_classification",
            "status": "success",
            "training": fit_result,
            "predictions_test": {
                "n_predicted": pred_result["n_predicted"],
                "predicted_distribution": pred_result["predicted_distribution"],
            },
            "evaluation": evaluation,
            "feature_importance": fi,
            "duration_seconds": round(time.time() - t0, 2),
        }

        # Predict full raster jika diminta
        if raster_shape is not None:
            full_pred = clf.predict_to_raster(X, raster_shape)
            if full_pred["status"] == "success":
                result["full_raster_prediction"] = {
                    "shape": list(full_pred["raster_2d"].shape),
                    "n_pixels": int(full_pred["raster_2d"].size),
                }
                result["flood_mask"] = full_pred["raster_2d"]
                if "probability_raster_2d" in full_pred:
                    result["probability_mask"] = full_pred["probability_raster_2d"]

        return result

    def _auto_select_method(self, indices: Dict[str, Any], bands: Dict[str, np.ndarray]) -> str:
        """Pilih method otomatis berdasarkan data yang tersedia."""
        if "mndwi" in indices:
            return "mndwi"
        if "vv" in bands:
            return "sar_vv"
        if "ndwi" in indices:
            return "ndwi"
        return "combined"
