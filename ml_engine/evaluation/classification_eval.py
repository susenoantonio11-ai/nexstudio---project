"""
Classification Evaluator
========================
Computes a comprehensive set of classification metrics.
"""
from __future__ import annotations
from typing import Dict, Any
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
    log_loss,
)


class ClassificationEvaluator:
    """Compute comprehensive classification metrics."""

    def evaluate(
        self,
        y_true,
        y_pred,
        y_proba=None,
        labels=None,
    ) -> Dict[str, Any]:
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        unique = np.unique(np.concatenate([y_true, y_pred]))
        n_classes = len(unique)
        is_binary = n_classes == 2

        avg = "binary" if is_binary else "weighted"

        result = {
            "n_classes": int(n_classes),
            "is_binary": is_binary,
            "n_samples": int(len(y_true)),
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
            "precision": round(float(precision_score(y_true, y_pred, average=avg, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true, y_pred, average=avg, zero_division=0)), 4),
            "f1": round(float(f1_score(y_true, y_pred, average=avg, zero_division=0)), 4),
            "f1_macro": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
            "f1_weighted": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
            "mcc": round(float(matthews_corrcoef(y_true, y_pred)), 4),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
            "classification_report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
        }

        # Probability-based metrics (only if y_proba provided)
        if y_proba is not None:
            try:
                if is_binary:
                    proba_pos = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
                    result["roc_auc"] = round(float(roc_auc_score(y_true, proba_pos)), 4)
                    result["pr_auc"] = round(float(average_precision_score(y_true, proba_pos)), 4)
                    result["log_loss"] = round(float(log_loss(y_true, proba_pos)), 4)
                else:
                    result["roc_auc_ovr"] = round(float(roc_auc_score(y_true, y_proba, multi_class="ovr")), 4)
                    result["log_loss"] = round(float(log_loss(y_true, y_proba)), 4)
            except Exception as e:
                result["proba_metrics_error"] = str(e)

        return result
