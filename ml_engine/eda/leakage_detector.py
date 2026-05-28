"""
Data Leakage Detector
=====================
Identifies columns that risk causing data leakage - the #1 cause of
"too good to be true" model performance that fails in production.

Types of leakage detected:
1. Target leakage: features that include information from the future or
   are direct functions of the target (e.g., 'profit' when target is 'revenue')
2. Train-test contamination: features computed using all data (e.g., target encoding
   without proper CV)
3. Time leakage: using future timestamps to predict past
4. Identifier leakage: row identifiers that incidentally correlate with target

Reference:
    Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O. (2012). Leakage
    in data mining: Formulation, detection, and avoidance. ACM TKDD, 6(4).
"""
from __future__ import annotations
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np


class LeakageDetector:
    """Detect potentially leaky features."""

    # Suspicious feature name patterns relative to common targets
    LEAK_PATTERN_PAIRS = [
        # (target_keyword, leaky_keywords)
        ("revenue", ["profit", "margin", "loss", "net_revenue"]),
        ("sales", ["profit", "margin", "commission"]),
        ("price", ["discounted_price", "final_price", "total"]),
        ("churn", ["last_login", "days_since_active"]),
        ("default", ["recovery", "writeoff", "collection"]),
        ("fraud", ["fraud_alert", "investigation", "blocked"]),
        ("target", ["target_pred", "target_score"]),
    ]

    def detect(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        datetime_column: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Returns:
            {
                "high_risk_columns": [...],
                "medium_risk_columns": [...],
                "warnings": [...],
                "recommendations": [...]
            }
        """
        high_risk: List[Dict[str, Any]] = []
        medium_risk: List[Dict[str, Any]] = []
        warnings: List[str] = []
        recommendations: List[Dict[str, Any]] = []

        if target_column and target_column in df.columns:
            # 1. Check for highly correlated features (suspicious)
            high_corr = self._check_target_correlation(df, target_column)
            high_risk.extend(high_corr)

            # 2. Check for feature names that suggest leakage
            name_based = self._check_name_patterns(df, target_column)
            high_risk.extend(name_based)

            # 3. Check for perfect/near-perfect rank correlation (likely leaky)
            rank_corr = self._check_rank_correlation(df, target_column)
            medium_risk.extend(rank_corr)

        # 4. Time leakage check
        if datetime_column and datetime_column in df.columns:
            time_leak = self._check_time_leakage(df, datetime_column)
            warnings.extend(time_leak)

        # 5. Identifier columns
        id_columns = self._check_identifier_columns(df)
        if id_columns:
            warnings.append(
                f"Identifier-like columns detected: {', '.join(id_columns)}. "
                f"These should be EXCLUDED from features as they may cause "
                f"row-level overfitting."
            )
            for col in id_columns:
                recommendations.append({
                    "priority": "high",
                    "column": col,
                    "action": f"DROP column '{col}' before training",
                    "reason": "Identifier columns add no predictive signal but can cause overfitting",
                })

        # Generate recommendations for risky features
        for entry in high_risk:
            recommendations.append({
                "priority": "high",
                "column": entry["column"],
                "action": f"DROP or carefully validate '{entry['column']}'",
                "reason": entry["reason"],
            })

        for entry in medium_risk:
            recommendations.append({
                "priority": "medium",
                "column": entry["column"],
                "action": f"REVIEW '{entry['column']}' for potential leakage",
                "reason": entry["reason"],
            })

        return {
            "n_high_risk": len(high_risk),
            "n_medium_risk": len(medium_risk),
            "high_risk_columns": high_risk,
            "medium_risk_columns": medium_risk,
            "warnings": warnings,
            "recommendations": recommendations,
            "best_practices": self._best_practices(),
        }

    def _check_target_correlation(
        self, df: pd.DataFrame, target_col: str
    ) -> List[Dict[str, Any]]:
        """Find features with suspiciously high correlation to target."""
        risky = []
        if not pd.api.types.is_numeric_dtype(df[target_col]):
            return risky

        target = df[target_col]
        for col in df.select_dtypes(include=["number"]).columns:
            if col == target_col:
                continue
            try:
                corr = df[col].corr(target)
                if not np.isnan(corr) and abs(corr) > 0.95:
                    risky.append({
                        "column": col,
                        "correlation": round(float(corr), 4),
                        "reason": (
                            f"Correlation with target '{target_col}' is {abs(corr):.3f} "
                            f"(>0.95). This is suspiciously high and often indicates leakage. "
                            f"Verify '{col}' is not a direct function of the target."
                        ),
                    })
            except Exception:
                continue
        return risky

    def _check_name_patterns(
        self, df: pd.DataFrame, target_col: str
    ) -> List[Dict[str, Any]]:
        """Check feature names against known leakage patterns."""
        risky = []
        target_lower = target_col.lower()
        for pattern_target, leaky_keywords in self.LEAK_PATTERN_PAIRS:
            if pattern_target in target_lower:
                for col in df.columns:
                    col_lower = col.lower()
                    for kw in leaky_keywords:
                        if kw in col_lower and col != target_col:
                            risky.append({
                                "column": col,
                                "reason": (
                                    f"Column name '{col}' contains '{kw}' which is "
                                    f"commonly a downstream function of target '{target_col}'. "
                                    f"This often causes target leakage."
                                ),
                            })
                            break
        return risky

    def _check_rank_correlation(
        self, df: pd.DataFrame, target_col: str
    ) -> List[Dict[str, Any]]:
        """Spearman rank correlation - catches monotonic leakage even if not linear."""
        risky = []
        if not pd.api.types.is_numeric_dtype(df[target_col]):
            return risky

        for col in df.select_dtypes(include=["number"]).columns:
            if col == target_col:
                continue
            try:
                rank_corr = df[col].corr(df[target_col], method="spearman")
                if not np.isnan(rank_corr) and 0.85 < abs(rank_corr) <= 0.95:
                    risky.append({
                        "column": col,
                        "rank_correlation": round(float(rank_corr), 4),
                        "reason": (
                            f"Spearman rank correlation with target = {abs(rank_corr):.3f}. "
                            f"High monotonic relationship. Validate if causal or coincidental."
                        ),
                    })
            except Exception:
                continue
        return risky

    def _check_time_leakage(
        self, df: pd.DataFrame, datetime_col: str
    ) -> List[str]:
        """Warn about time-related leakage patterns."""
        warnings = []
        try:
            parsed = pd.to_datetime(df[datetime_col], errors="coerce")
            if parsed.notna().sum() > 0:
                warnings.append(
                    f"Datetime column '{datetime_col}' detected. CRITICAL: when splitting "
                    f"train/test, use TimeSeriesSplit or date-based split (NOT random) "
                    f"to prevent the model from seeing 'future' data during training."
                )
        except Exception:
            pass
        return warnings

    def _check_identifier_columns(self, df: pd.DataFrame) -> List[str]:
        """Find columns that look like row identifiers."""
        ids = []
        for col in df.columns:
            name_lower = col.lower()
            if any(kw in name_lower for kw in ["id", "uuid", "key", "code"]):
                # High cardinality + ID-like name = likely identifier
                if df[col].nunique() / len(df) > 0.95 if len(df) > 0 else False:
                    ids.append(col)
        return ids

    def _best_practices(self) -> List[str]:
        return [
            "ALWAYS split train/test BEFORE any preprocessing that uses statistics (scaling, imputation, encoding)",
            "Use sklearn Pipeline to bundle preprocessing + model so transformations only see training data",
            "For time series: use TimeSeriesSplit, never random KFold",
            "Beware target encoding without proper CV - it leaks target into features",
            "Re-test the model on a HOLD-OUT set that was never seen during any tuning",
            "If accuracy seems too good (>99%), suspect leakage and audit each feature",
        ]
