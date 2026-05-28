"""
Correlation Analyzer
====================
Analyzes:
1. Feature-target correlation (signal strength)
2. Feature-feature correlation (multicollinearity)
3. Recommendations for feature pruning

High inter-feature correlation (>0.9) hurts:
- Linear model coefficients (unstable)
- Model interpretability
- Computation time

Reference:
    Hastie, Tibshirani & Friedman (2009) - Elements of Statistical Learning
"""
from __future__ import annotations
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np


class CorrelationAnalyzer:
    """Analyze correlations to guide feature selection."""

    def analyze(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        multicollinearity_threshold: float = 0.9,
    ) -> Dict[str, Any]:
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        if len(numeric_cols) < 2:
            return {
                "n_numeric_features": len(numeric_cols),
                "feature_target_correlations": [],
                "multicollinearity_pairs": [],
                "recommendations": [],
            }

        # Build correlation matrix
        corr_matrix = df[numeric_cols].corr().fillna(0)

        # Feature-target correlations
        target_corrs = []
        if target_column and target_column in corr_matrix.columns:
            for col in numeric_cols:
                if col == target_column:
                    continue
                corr_val = corr_matrix.loc[col, target_column]
                target_corrs.append({
                    "feature": col,
                    "correlation": round(float(corr_val), 4),
                    "abs_correlation": round(abs(float(corr_val)), 4),
                    "signal_strength": self._classify_signal(abs(corr_val)),
                })
            target_corrs.sort(key=lambda x: x["abs_correlation"], reverse=True)

        # Multicollinearity pairs
        multicol_pairs = []
        for i, col1 in enumerate(numeric_cols):
            if col1 == target_column:
                continue
            for col2 in numeric_cols[i + 1 :]:
                if col2 == target_column:
                    continue
                corr_val = corr_matrix.loc[col1, col2]
                if abs(corr_val) >= multicollinearity_threshold:
                    multicol_pairs.append({
                        "feature_a": col1,
                        "feature_b": col2,
                        "correlation": round(float(corr_val), 4),
                        "recommendation": (
                            f"Highly correlated pair (|r|={abs(corr_val):.3f} >= {multicollinearity_threshold}). "
                            f"Consider dropping one of them OR using a regularized model "
                            f"(Ridge/Lasso). Linear model coefficients will be unstable otherwise."
                        ),
                    })

        recommendations = self._build_recommendations(target_corrs, multicol_pairs)

        return {
            "n_numeric_features": len(numeric_cols),
            "feature_target_correlations": target_corrs,
            "n_multicollinearity_pairs": len(multicol_pairs),
            "multicollinearity_pairs": multicol_pairs,
            "weak_signal_features": [
                t["feature"] for t in target_corrs if t["signal_strength"] == "weak"
            ],
            "recommendations": recommendations,
        }

    def _classify_signal(self, abs_corr: float) -> str:
        if abs_corr >= 0.5:
            return "strong"
        if abs_corr >= 0.2:
            return "moderate"
        if abs_corr >= 0.05:
            return "weak"
        return "noise"

    def _build_recommendations(
        self,
        target_corrs: List[Dict],
        multicol_pairs: List[Dict],
    ) -> List[Dict[str, Any]]:
        recs = []

        # Weak signal features
        weak_features = [
            t["feature"] for t in target_corrs if t["signal_strength"] in ("weak", "noise")
        ]
        if weak_features:
            recs.append({
                "priority": "low",
                "action": "Consider feature selection / RFE",
                "reason": (
                    f"Features with weak target correlation: {', '.join(weak_features[:5])}"
                    + (f" (and {len(weak_features) - 5} more)" if len(weak_features) > 5 else "")
                    + ". They may add noise without predictive signal. Recursive feature "
                    + "elimination or model-based importance can identify if they help."
                ),
            })

        # Multicollinearity
        if multicol_pairs:
            recs.append({
                "priority": "medium",
                "action": (
                    "Address multicollinearity: either drop one feature from each correlated pair, "
                    "use Ridge/Lasso for linear models, OR rely on tree-based models (which are "
                    "less affected by multicollinearity)"
                ),
                "reason": (
                    f"{len(multicol_pairs)} highly correlated feature pair(s) detected. "
                    f"Linear model coefficients will be unstable; tree-based models are robust."
                ),
            })

        # Strong target correlation - good news
        strong = [t for t in target_corrs if t["signal_strength"] == "strong"]
        if strong:
            recs.append({
                "priority": "info",
                "action": (
                    f"Top features by signal: {', '.join(t['feature'] for t in strong[:5])}. "
                    f"These are likely the strongest predictors."
                ),
                "reason": "Strong correlation suggests genuine predictive signal.",
            })

        return recs
