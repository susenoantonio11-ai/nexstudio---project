"""
Class Imbalance Detector
========================
For classification problems: analyzes class distribution and recommends
imbalance-handling strategy (class_weight, SMOTE, ADASYN, threshold tuning).

Imbalance ratios:
- 1:1 to 1:1.5  -> balanced (no action needed)
- 1:1.5 to 1:5  -> mild imbalance (use class_weight='balanced')
- 1:5 to 1:50   -> moderate imbalance (SMOTE/ADASYN + class_weight)
- 1:50+         -> severe imbalance (use anomaly detection framing instead)

Reference:
    He, H., & Garcia, E. A. (2009). Learning from imbalanced data.
    IEEE Transactions on Knowledge and Data Engineering, 21(9), 1263-1284.
"""
from __future__ import annotations
from typing import Dict, Any, List
import pandas as pd
import numpy as np


class ImbalanceDetector:
    """Analyze class balance for classification target."""

    def detect(self, target: pd.Series) -> Dict[str, Any]:
        if target is None or target.empty:
            return {"is_imbalanced": False, "reason": "Empty target"}

        if pd.api.types.is_numeric_dtype(target) and target.nunique() > 50:
            return {
                "is_imbalanced": False,
                "reason": "Target appears continuous (nunique > 50). Not a classification problem.",
                "task_inferred": "regression",
            }

        # Compute class distribution
        counts = target.value_counts(dropna=False)
        n_classes = len(counts)
        total = int(counts.sum())
        proportions = (counts / total).to_dict()

        if n_classes < 2:
            return {
                "is_imbalanced": False,
                "reason": "Only 1 class found. Cannot train classifier.",
            }

        majority = counts.iloc[0]
        minority = counts.iloc[-1]
        imbalance_ratio = majority / minority if minority > 0 else float("inf")

        severity, recommendation = self._classify_severity(
            imbalance_ratio, n_classes, total
        )

        return {
            "is_imbalanced": severity != "balanced",
            "severity": severity,
            "n_classes": n_classes,
            "total_samples": total,
            "imbalance_ratio": round(float(imbalance_ratio), 2),
            "majority_class": str(counts.index[0]),
            "minority_class": str(counts.index[-1]),
            "majority_count": int(majority),
            "minority_count": int(minority),
            "class_distribution": [
                {
                    "class": str(cls),
                    "count": int(cnt),
                    "proportion": round(float(cnt / total), 4),
                }
                for cls, cnt in counts.items()
            ],
            "recommendation": recommendation,
            "metric_implications": self._metric_implications(severity, n_classes),
        }

    def _classify_severity(
        self, ratio: float, n_classes: int, total: int
    ) -> tuple:
        if ratio <= 1.5:
            return "balanced", {
                "action": "no_action",
                "reasoning": (
                    f"Class distribution is balanced (ratio {ratio:.2f}). "
                    f"Standard training should produce reliable results."
                ),
                "alternatives_considered": [],
            }

        if ratio <= 5:
            return "mild", {
                "action": "class_weight_balanced",
                "reasoning": (
                    f"Mild imbalance (ratio {ratio:.2f}). Setting class_weight='balanced' "
                    f"automatically inverse-weights samples by class frequency, sufficient "
                    f"to compensate without resampling."
                ),
                "alternatives_considered": [
                    {
                        "action": "smote",
                        "reason_rejected": "Overkill for mild imbalance; class_weight is simpler and avoids synthetic data risk",
                    },
                    {
                        "action": "no_action",
                        "reason_rejected": "Could lead to majority-class bias",
                    },
                ],
            }

        if ratio <= 50:
            if total < 1000:
                return "moderate", {
                    "action": "smote_plus_class_weight",
                    "reasoning": (
                        f"Moderate imbalance (ratio {ratio:.2f}) with relatively small dataset "
                        f"({total} samples). Use SMOTE to synthesize minority samples, combined "
                        f"with class_weight for the model. Be careful: SMOTE must be applied "
                        f"INSIDE the cross-validation fold, not before splitting (data leakage risk)."
                    ),
                    "alternatives_considered": [
                        {
                            "action": "class_weight_only",
                            "reason_rejected": "May not be enough at this ratio",
                        },
                        {
                            "action": "undersample_majority",
                            "reason_rejected": f"Loses too much data with only {total} samples",
                        },
                    ],
                }
            else:
                return "moderate", {
                    "action": "class_weight_with_threshold_tuning",
                    "reasoning": (
                        f"Moderate imbalance (ratio {ratio:.2f}) with sufficient data ({total} samples). "
                        f"Use class_weight='balanced' and tune classification threshold using "
                        f"precision-recall curve to optimize for the business metric."
                    ),
                    "alternatives_considered": [
                        {
                            "action": "smote",
                            "reason_rejected": "With abundant data, threshold tuning is more principled and avoids synthetic risk",
                        },
                    ],
                }

        # Severe (ratio > 50)
        return "severe", {
            "action": "reframe_as_anomaly_detection",
            "reasoning": (
                f"Severe imbalance (ratio {ratio:.2f}). Standard classification will struggle "
                f"to learn the minority pattern. Recommend reframing as anomaly detection "
                f"(Isolation Forest, One-Class SVM) where the minority class is the 'anomaly'. "
                f"Reference: He & Garcia (2009)."
            ),
            "alternatives_considered": [
                {
                    "action": "smote",
                    "reason_rejected": "At extreme ratio, synthesizing many minority samples produces unreliable patterns",
                },
                {
                    "action": "ensemble_balanced",
                    "reason_rejected": "BalancedRandomForest is alternative but anomaly detection paradigm fits better at this ratio",
                },
            ],
        }

    def _metric_implications(self, severity: str, n_classes: int) -> Dict[str, Any]:
        """Tell the user what metrics to use based on imbalance severity."""
        if severity == "balanced":
            return {
                "primary_metrics": ["accuracy", "f1_macro", "roc_auc"] if n_classes == 2 else ["accuracy", "f1_macro"],
                "warning": None,
            }

        if severity in ("mild", "moderate"):
            return {
                "primary_metrics": ["f1_macro", "pr_auc", "precision_minority", "recall_minority"],
                "avoid_metrics": ["accuracy"],
                "warning": (
                    "Accuracy is MISLEADING for imbalanced data. A model predicting majority "
                    "class always can score high accuracy while being useless. Use F1-macro and "
                    "PR-AUC instead."
                ),
            }

        # severe
        return {
            "primary_metrics": ["pr_auc", "recall_minority", "mcc", "balanced_accuracy"],
            "avoid_metrics": ["accuracy", "roc_auc"],
            "warning": (
                "ROC-AUC is OVER-OPTIMISTIC for severely imbalanced data because it weights "
                "TP and FP equally. Use Precision-Recall AUC and Matthews Correlation Coefficient (MCC) "
                "which are robust to imbalance."
            ),
        }
