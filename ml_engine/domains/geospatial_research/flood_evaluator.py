"""
FloodEvaluator - evaluation metrics khusus flood mapping.
==========================================================
Selain metrik klasifikasi standar, tambahkan IoU, Kappa untuk flood masks.
"""
from __future__ import annotations
from typing import Dict, Any, Optional
import numpy as np


class FloodEvaluator:
    """Evaluasi prediksi flood mask vs ground truth."""

    def evaluate(
        self,
        y_true,
        y_pred,
        y_proba=None,
    ) -> Dict[str, Any]:
        """Compute comprehensive metrics."""
        try:
            from sklearn.metrics import (
                accuracy_score, precision_score, recall_score, f1_score,
                cohen_kappa_score, confusion_matrix, classification_report,
                roc_auc_score, jaccard_score,
            )
        except ImportError:
            return {"status": "error", "error": "scikit-learn not installed"}

        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        unique = np.unique(np.concatenate([y_true, y_pred]))
        is_binary = len(unique) == 2

        avg = "binary" if is_binary else "weighted"

        result = {
            "n_samples": int(len(y_true)),
            "n_classes": int(len(unique)),
            "is_binary": is_binary,
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "precision": round(float(precision_score(y_true, y_pred, average=avg, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true, y_pred, average=avg, zero_division=0)), 4),
            "f1": round(float(f1_score(y_true, y_pred, average=avg, zero_division=0)), 4),
            "f1_macro": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
            "kappa": round(float(cohen_kappa_score(y_true, y_pred)), 4),
            "iou": round(float(jaccard_score(y_true, y_pred, average=avg, zero_division=0)), 4),
            "iou_macro": round(float(jaccard_score(y_true, y_pred, average="macro", zero_division=0)), 4),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
            "classification_report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
        }

        if y_proba is not None and is_binary:
            try:
                proba_pos = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
                result["roc_auc"] = round(float(roc_auc_score(y_true, proba_pos)), 4)
            except Exception:
                pass

        # Flood-specific interpretation
        result["interpretation"] = self._interpret(result)
        result["method_monitor"] = {
            "evaluation_metrics_chosen": (
                "F1, IoU, dan Kappa adalah TRINITY untuk flood mapping. "
                "F1 menyeimbangkan precision-recall. "
                "IoU (Intersection over Union / Jaccard) = standar segmentation: |intersection| / |union|. "
                "Kappa = agreement diluar chance — robust terhadap class imbalance."
            ),
            "why_not_accuracy_only": (
                "Flood mapping biasanya IMBALANCED (10-30% pixel saja yang flood). "
                "Accuracy 80% bisa dicapai dengan model yang selalu prediksi non-flood — useless. "
                "F1 dan IoU memberikan gambaran lebih jujur."
            ),
        }
        return result

    def _interpret(self, m: Dict[str, Any]) -> Dict[str, Any]:
        f1 = m.get("f1", 0)
        iou = m.get("iou", 0)
        kappa = m.get("kappa", 0)

        f1_grade = "EXCELLENT" if f1 > 0.85 else "GOOD" if f1 > 0.7 else "FAIR" if f1 > 0.5 else "POOR"
        iou_grade = "EXCELLENT" if iou > 0.75 else "GOOD" if iou > 0.5 else "FAIR" if iou > 0.3 else "POOR"
        kappa_grade = (
            "ALMOST PERFECT" if kappa > 0.8 else
            "SUBSTANTIAL" if kappa > 0.6 else
            "MODERATE" if kappa > 0.4 else
            "FAIR" if kappa > 0.2 else
            "SLIGHT"
        )

        return {
            "f1_grade": f1_grade,
            "iou_grade": iou_grade,
            "kappa_grade": kappa_grade,
            "deployment_ready": f1 > 0.7 and iou > 0.5 and kappa > 0.6,
            "summary": (
                f"Model {f1_grade} dengan F1={f1:.3f} (precision-recall balance), "
                f"IoU={iou:.3f} ({iou_grade} segmentation overlap), "
                f"Kappa={kappa:.3f} ({kappa_grade} agreement). "
                f"{'Model siap deploy.' if (f1 > 0.7 and iou > 0.5) else 'Perlu peningkatan sebelum deploy.'}"
            ),
        }
