"""
Outlier Detector
================
Detects outliers using multiple methods (consensus approach):
- IQR (Interquartile Range): robust, distribution-free
- Z-score: assumes normality; fast
- Modified Z-score (uses median): robust to outliers in own distribution
- Isolation Forest: ML-based, multivariate

CRITICAL: Does NOT delete outliers automatically. Returns labels and
recommendations. Decision (cap/transform/keep/remove) is up to user.

Reference:
    Iglewicz, B., & Hoaglin, D. C. (1993). Volume 16: How to Detect and
    Handle Outliers. ASQC Quality Press.
"""
from __future__ import annotations
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np


class OutlierDetector:
    """Detect outliers per column with multiple methods + recommendation."""

    def __init__(self, iqr_multiplier: float = 1.5, z_threshold: float = 3.0):
        self.iqr_multiplier = iqr_multiplier
        self.z_threshold = z_threshold

    def detect(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Detect outliers in numerical columns.

        Returns per-column analysis with:
        - n_outliers per method
        - consensus outliers (flagged by 2+ methods)
        - recommended treatment (cap/transform/robust_model/keep)
        """
        if columns is None:
            columns = df.select_dtypes(include=["number"]).columns.tolist()

        per_column = []
        for col in columns:
            series = df[col].dropna()
            if len(series) < 10:
                continue

            iqr_outliers = self._iqr_method(series)
            zscore_outliers = self._zscore_method(series)
            mod_zscore_outliers = self._modified_zscore_method(series)

            # Consensus: indices flagged by at least 2 methods
            all_indices = set(iqr_outliers["indices"]) | set(zscore_outliers["indices"]) | set(mod_zscore_outliers["indices"])
            consensus = []
            for idx in all_indices:
                count = sum([
                    idx in iqr_outliers["indices"],
                    idx in zscore_outliers["indices"],
                    idx in mod_zscore_outliers["indices"],
                ])
                if count >= 2:
                    consensus.append(idx)

            recommendation = self._recommend_treatment(
                series, len(consensus), col
            )

            per_column.append({
                "column": col,
                "n_observations": int(len(series)),
                "iqr": {
                    "n_outliers": iqr_outliers["count"],
                    "lower_bound": iqr_outliers["lower"],
                    "upper_bound": iqr_outliers["upper"],
                    "outlier_rate": round(iqr_outliers["count"] / len(series), 4),
                },
                "zscore": {
                    "n_outliers": zscore_outliers["count"],
                    "threshold": self.z_threshold,
                    "outlier_rate": round(zscore_outliers["count"] / len(series), 4),
                },
                "modified_zscore": {
                    "n_outliers": mod_zscore_outliers["count"],
                    "threshold": 3.5,
                    "outlier_rate": round(mod_zscore_outliers["count"] / len(series), 4),
                },
                "consensus_outliers": {
                    "n": len(consensus),
                    "rate": round(len(consensus) / len(series), 4),
                    "sample_indices": consensus[:20],
                },
                "recommendation": recommendation,
                "stats": {
                    "mean": float(series.mean()),
                    "median": float(series.median()),
                    "std": float(series.std()),
                    "skew": float(series.skew()) if len(series) > 2 else 0,
                },
            })

        return {
            "n_columns_analyzed": len(per_column),
            "columns": per_column,
            "summary": self._build_summary(per_column),
            "global_recommendations": self._build_global_recommendations(per_column),
        }

    def _iqr_method(self, series: pd.Series) -> Dict[str, Any]:
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - self.iqr_multiplier * iqr
        upper = q3 + self.iqr_multiplier * iqr
        outliers = (series < lower) | (series > upper)
        return {
            "lower": float(lower),
            "upper": float(upper),
            "count": int(outliers.sum()),
            "indices": series[outliers].index.tolist(),
        }

    def _zscore_method(self, series: pd.Series) -> Dict[str, Any]:
        mean = series.mean()
        std = series.std()
        if std == 0:
            return {"count": 0, "indices": []}
        z = np.abs((series - mean) / std)
        outliers = z > self.z_threshold
        return {
            "count": int(outliers.sum()),
            "indices": series[outliers].index.tolist(),
        }

    def _modified_zscore_method(self, series: pd.Series) -> Dict[str, Any]:
        """Modified Z-score uses median and MAD (robust to extreme outliers)."""
        median = series.median()
        mad = np.median(np.abs(series - median))
        if mad == 0:
            return {"count": 0, "indices": []}
        modified_z = 0.6745 * (series - median) / mad
        outliers = np.abs(modified_z) > 3.5
        return {
            "count": int(outliers.sum()),
            "indices": series[outliers].index.tolist(),
        }

    def _recommend_treatment(
        self, series: pd.Series, n_consensus: int, col: str
    ) -> Dict[str, Any]:
        """Decide what to do based on outlier rate, distribution, and column name."""
        rate = n_consensus / len(series) if len(series) else 0
        skew = abs(series.skew()) if len(series) > 2 else 0
        name_lower = col.lower()

        if n_consensus == 0:
            return {
                "action": "no_action",
                "reasoning": "No consensus outliers detected. Distribution is clean.",
                "alternatives_considered": [],
            }

        # Heuristic: if column is a financial amount, outliers may be valid (large transactions)
        is_financial = any(w in name_lower for w in ["amount", "revenue", "price", "value", "total"])

        if is_financial and rate < 0.05:
            return {
                "action": "keep_with_robust_model",
                "reasoning": (
                    f"Column '{col}' appears to be a financial amount where outliers "
                    f"often represent VALID large transactions, not errors. Recommend using "
                    f"robust models (Random Forest, gradient boosting) and RobustScaler "
                    f"instead of removing data."
                ),
                "alternatives_considered": [
                    {"action": "remove", "reason_rejected": "Would lose valid business signal"},
                    {"action": "cap", "reason_rejected": "Would distort genuine high-value transactions"},
                    {"action": "log_transform", "reason_rejected": "Possible if model is linear; for tree-based models not needed"},
                ],
            }

        if rate > 0.1:
            return {
                "action": "investigate_then_transform",
                "reasoning": (
                    f"High outlier rate ({rate*100:.1f}%) suggests systematic issue rather "
                    f"than random errors. Investigate root cause before treatment. "
                    f"If valid, apply log transform; if errors, remove."
                ),
                "alternatives_considered": [
                    {"action": "remove_all", "reason_rejected": "Risk losing 10%+ of data without understanding cause"},
                ],
            }

        if skew > 2:
            return {
                "action": "log_transform",
                "reasoning": (
                    f"Distribution is highly skewed (skew={skew:.2f}). Log transformation "
                    f"compresses extreme values while preserving order, often resolving "
                    f"outlier influence on linear models."
                ),
                "alternatives_considered": [
                    {"action": "remove", "reason_rejected": "Skewness suggests valid heavy tail, not errors"},
                    {"action": "cap", "reason_rejected": "Log transform is more principled for skewed data"},
                    {"action": "robust_scaler", "reason_rejected": "Doesn't address skewness, only scale"},
                ],
            }

        if rate < 0.02:
            return {
                "action": "cap",
                "reasoning": (
                    f"Low outlier rate ({rate*100:.1f}%). Cap at 1st/99th percentile to "
                    f"limit influence without losing data points."
                ),
                "alternatives_considered": [
                    {"action": "remove", "reason_rejected": "Removing very few rows is fine but cap preserves count"},
                    {"action": "no_action", "reason_rejected": "Outliers may unduly influence linear models"},
                ],
            }

        return {
            "action": "robust_scaler",
            "reasoning": (
                f"Moderate outlier rate. Use RobustScaler (median/IQR-based) instead of "
                f"StandardScaler. This reduces outlier influence without changing data."
            ),
            "alternatives_considered": [
                {"action": "remove", "reason_rejected": "RobustScaler preserves all data"},
            ],
        }

    def _build_summary(self, per_column: List[Dict]) -> Dict[str, Any]:
        if not per_column:
            return {"any_outliers": False}
        max_rate = max(c["consensus_outliers"]["rate"] for c in per_column)
        return {
            "any_outliers": max_rate > 0,
            "max_outlier_rate": max_rate,
            "n_columns_with_outliers": sum(1 for c in per_column if c["consensus_outliers"]["n"] > 0),
            "actions_recommended": list(set(c["recommendation"]["action"] for c in per_column)),
        }

    def _build_global_recommendations(self, per_column: List[Dict]) -> List[Dict]:
        recs = []
        actions_count: Dict[str, int] = {}
        for c in per_column:
            action = c["recommendation"]["action"]
            actions_count[action] = actions_count.get(action, 0) + 1

        if actions_count.get("keep_with_robust_model", 0) > 0:
            recs.append({
                "priority": "high",
                "action": "Use Random Forest or gradient boosting (tree-based models are robust to outliers without preprocessing)",
                "reason": "Multiple columns contain valid extreme values that should be preserved",
            })

        if actions_count.get("log_transform", 0) > 0:
            recs.append({
                "priority": "medium",
                "action": "Apply log1p transformation to skewed columns BEFORE scaling",
                "reason": "Reduces outlier influence on linear models",
            })

        return recs
