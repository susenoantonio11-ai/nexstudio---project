"""
Threshold Tuner
===============
For binary classification: optimizes the decision threshold beyond the
default 0.5. Critical for imbalanced data where default threshold
produces poor recall on the minority class.

Optimization criteria:
- 'f1': maximize F1 (balance precision-recall)
- 'youden': maximize TPR - FPR (best ROC point)
- 'precision': hit precision target while maximizing recall
- 'recall': hit recall target while maximizing precision
- 'business': custom cost matrix
"""
from __future__ import annotations
from typing import Dict, Any, Optional, Callable
import numpy as np
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    f1_score,
    precision_score,
    recall_score,
)


class ThresholdTuner:
    """Find optimal decision threshold for binary classifier."""

    def tune(
        self,
        y_true,
        y_proba,
        criterion: str = "f1",
        target_value: float = 0.8,
    ) -> Dict[str, Any]:
        """
        Args:
            y_true: ground truth binary labels (0/1)
            y_proba: predicted probabilities for positive class
            criterion: 'f1', 'youden', 'precision', 'recall'
            target_value: for 'precision'/'recall', the minimum value to maintain

        Returns:
            Optimal threshold + metric values + reasoning.
        """
        y_true = np.asarray(y_true)
        y_proba = np.asarray(y_proba)

        if criterion == "f1":
            return self._tune_f1(y_true, y_proba)
        if criterion == "youden":
            return self._tune_youden(y_true, y_proba)
        if criterion == "precision":
            return self._tune_constraint(y_true, y_proba, "precision", target_value)
        if criterion == "recall":
            return self._tune_constraint(y_true, y_proba, "recall", target_value)
        return self._tune_f1(y_true, y_proba)

    def _tune_f1(self, y_true, y_proba) -> Dict[str, Any]:
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
        # F1 per threshold
        f1s = []
        for p, r in zip(precisions[:-1], recalls[:-1]):
            if (p + r) > 0:
                f1s.append(2 * p * r / (p + r))
            else:
                f1s.append(0)
        f1s = np.array(f1s)
        best_idx = int(np.argmax(f1s))
        best_thresh = float(thresholds[best_idx])

        return {
            "criterion": "f1",
            "optimal_threshold": round(best_thresh, 4),
            "default_threshold": 0.5,
            "f1_at_optimal": round(float(f1s[best_idx]), 4),
            "precision_at_optimal": round(float(precisions[best_idx]), 4),
            "recall_at_optimal": round(float(recalls[best_idx]), 4),
            "improvement_over_default": self._compute_improvement_over_default(
                y_true, y_proba, best_thresh
            ),
            "reasoning": (
                f"Optimal threshold {best_thresh:.3f} maximizes F1 = {f1s[best_idx]:.3f} "
                f"(precision={precisions[best_idx]:.3f}, recall={recalls[best_idx]:.3f}). "
                f"Default 0.5 may produce different precision-recall tradeoff. "
                f"For imbalanced data, threshold tuning often substantially improves "
                f"minority class detection without retraining the model."
            ),
        }

    def _tune_youden(self, y_true, y_proba) -> Dict[str, Any]:
        fprs, tprs, thresholds = roc_curve(y_true, y_proba)
        youden = tprs - fprs
        best_idx = int(np.argmax(youden))
        best_thresh = float(thresholds[best_idx])

        return {
            "criterion": "youden_index",
            "optimal_threshold": round(best_thresh, 4),
            "default_threshold": 0.5,
            "youden_at_optimal": round(float(youden[best_idx]), 4),
            "tpr_at_optimal": round(float(tprs[best_idx]), 4),
            "fpr_at_optimal": round(float(fprs[best_idx]), 4),
            "reasoning": (
                f"Optimal threshold {best_thresh:.3f} maximizes Youden index "
                f"(TPR - FPR = {youden[best_idx]:.3f}). This is the point on ROC curve "
                f"farthest from the diagonal, balancing sensitivity and specificity."
            ),
        }

    def _tune_constraint(
        self, y_true, y_proba, fixed_metric: str, target: float
    ) -> Dict[str, Any]:
        thresholds = np.linspace(0.01, 0.99, 99)
        candidates = []
        for t in thresholds:
            preds = (y_proba >= t).astype(int)
            p = precision_score(y_true, preds, zero_division=0)
            r = recall_score(y_true, preds, zero_division=0)
            f1 = f1_score(y_true, preds, zero_division=0)
            candidates.append({"threshold": t, "precision": p, "recall": r, "f1": f1})

        valid = [c for c in candidates if c[fixed_metric] >= target]
        if not valid:
            return {
                "criterion": f"{fixed_metric} >= {target}",
                "optimal_threshold": None,
                "reasoning": (
                    f"No threshold achieves {fixed_metric} >= {target}. "
                    f"Either lower target or improve model first."
                ),
            }

        # Among valid, pick best of the OTHER metric
        other = "recall" if fixed_metric == "precision" else "precision"
        best = max(valid, key=lambda c: c[other])

        return {
            "criterion": f"max {other} subject to {fixed_metric} >= {target}",
            "optimal_threshold": round(float(best["threshold"]), 4),
            "precision": round(float(best["precision"]), 4),
            "recall": round(float(best["recall"]), 4),
            "f1": round(float(best["f1"]), 4),
            "reasoning": (
                f"Threshold {best['threshold']:.3f} achieves {fixed_metric}="
                f"{best[fixed_metric]:.3f} (target {target}) while maximizing "
                f"{other}={best[other]:.3f}."
            ),
        }

    def _compute_improvement_over_default(
        self, y_true, y_proba, opt_thresh: float
    ) -> Dict[str, Any]:
        default_preds = (y_proba >= 0.5).astype(int)
        opt_preds = (y_proba >= opt_thresh).astype(int)
        return {
            "default_f1": round(float(f1_score(y_true, default_preds, zero_division=0)), 4),
            "optimal_f1": round(float(f1_score(y_true, opt_preds, zero_division=0)), 4),
            "default_precision": round(float(precision_score(y_true, default_preds, zero_division=0)), 4),
            "optimal_precision": round(float(precision_score(y_true, opt_preds, zero_division=0)), 4),
            "default_recall": round(float(recall_score(y_true, default_preds, zero_division=0)), 4),
            "optimal_recall": round(float(recall_score(y_true, opt_preds, zero_division=0)), 4),
        }
