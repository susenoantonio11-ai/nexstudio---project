"""
Metric Selector
===============
Picks the correct primary scoring metric based on:
- Task type (regression / classification / forecasting / anomaly)
- Class balance (for classification)
- Number of classes (binary vs multiclass)

Also returns a list of secondary metrics for full evaluation.

CRITICAL: Accuracy is NEVER recommended as the primary metric for
imbalanced classification. F1, PR-AUC, MCC are preferred.
"""
from __future__ import annotations
from typing import Dict, Any


class MetricSelector:
    """Recommend metrics based on problem characteristics."""

    def select(
        self,
        task_type: str,
        n_classes: int = 2,
        is_imbalanced: bool = False,
        imbalance_severity: str = "balanced",
    ) -> Dict[str, Any]:
        """
        Returns:
            {
                "primary_metric": str (sklearn scoring name),
                "primary_metric_human": str (human-readable),
                "secondary_metrics": list[str],
                "avoid_metrics": list[str],
                "reasoning": str
            }
        """
        if task_type == "regression":
            return {
                "primary_metric": "r2",
                "primary_metric_human": "R² (coefficient of determination)",
                "primary_higher_is_better": True,
                "secondary_metrics": [
                    "neg_root_mean_squared_error",  # RMSE
                    "neg_mean_absolute_error",      # MAE
                    "neg_mean_absolute_percentage_error",  # MAPE
                ],
                "secondary_human": [
                    "RMSE (root mean squared error)",
                    "MAE (mean absolute error)",
                    "MAPE (mean absolute percentage error)",
                ],
                "avoid_metrics": [],
                "reasoning": (
                    "For regression: R² explains proportion of variance captured. "
                    "RMSE penalizes large errors more (good when big errors are costly). "
                    "MAE is robust to outliers. MAPE is scale-free, useful for comparing "
                    "across different targets. Reference: Hyndman & Athanasopoulos (2018)."
                ),
            }

        if task_type == "forecasting":
            return {
                "primary_metric": "neg_mean_absolute_percentage_error",
                "primary_metric_human": "MAPE (mean absolute percentage error)",
                "primary_higher_is_better": True,  # higher (less negative) is better in sklearn
                "secondary_metrics": [
                    "neg_root_mean_squared_error",
                    "neg_mean_absolute_error",
                ],
                "secondary_human": ["RMSE", "MAE"],
                "avoid_metrics": [],
                "reasoning": (
                    "Forecasting: MAPE is scale-free and intuitive (% error). "
                    "Combine with RMSE for absolute error magnitude. Reference: "
                    "Hyndman & Athanasopoulos (2018) - Forecasting: Principles and Practice."
                ),
            }

        if task_type == "anomaly_detection":
            return {
                "primary_metric": "average_precision",  # PR-AUC
                "primary_metric_human": "PR-AUC (precision-recall area under curve)",
                "primary_higher_is_better": True,
                "secondary_metrics": ["precision", "recall", "f1"],
                "secondary_human": ["Precision", "Recall", "F1"],
                "avoid_metrics": ["accuracy", "roc_auc"],
                "reasoning": (
                    "Anomaly detection: anomalies are RARE by definition (extreme imbalance). "
                    "PR-AUC focuses on the rare positive class and is robust to imbalance. "
                    "Accuracy and ROC-AUC are misleading at this imbalance level. "
                    "Reference: Chandola, Banerjee & Kumar (2009)."
                ),
            }

        # Classification
        is_binary = n_classes == 2

        if imbalance_severity == "severe":
            return {
                "primary_metric": "average_precision" if is_binary else "f1_macro",
                "primary_metric_human": "PR-AUC" if is_binary else "F1-macro",
                "primary_higher_is_better": True,
                "secondary_metrics": ["matthews_corrcoef", "balanced_accuracy", "f1_macro"],
                "secondary_human": ["MCC", "Balanced accuracy", "F1-macro"],
                "avoid_metrics": ["accuracy", "roc_auc"],
                "reasoning": (
                    f"Severe imbalance ({imbalance_severity}). Accuracy and ROC-AUC are "
                    f"OVER-OPTIMISTIC: a model predicting only majority class scores high. "
                    f"PR-AUC and MCC are robust to imbalance. Reference: He & Garcia (2009)."
                ),
            }

        if is_imbalanced:
            return {
                "primary_metric": "f1_macro",
                "primary_metric_human": "F1-macro (unweighted mean of per-class F1)",
                "primary_higher_is_better": True,
                "secondary_metrics": ["roc_auc" if is_binary else "f1_weighted", "average_precision", "balanced_accuracy"],
                "secondary_human": ["ROC-AUC", "PR-AUC", "Balanced accuracy"],
                "avoid_metrics": ["accuracy"],
                "reasoning": (
                    f"Imbalanced classification ({imbalance_severity}). F1-macro weights "
                    f"each class equally, preventing majority class from dominating the score. "
                    f"Avoid plain accuracy which is misleading."
                ),
            }

        # Balanced classification
        return {
            "primary_metric": "f1_weighted" if not is_binary else "f1",
            "primary_metric_human": "F1-weighted" if not is_binary else "F1",
            "primary_higher_is_better": True,
            "secondary_metrics": ["accuracy", "roc_auc" if is_binary else "f1_macro"],
            "secondary_human": ["Accuracy", "ROC-AUC" if is_binary else "F1-macro"],
            "avoid_metrics": [],
            "reasoning": (
                "Balanced classification. F1 captures both precision and recall. "
                "Accuracy is also reasonable when classes are balanced. "
                "ROC-AUC summarizes performance across all thresholds for binary problems."
            ),
        }
