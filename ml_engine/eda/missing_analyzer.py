"""
Missing Value Analyzer
======================
Detects missing patterns and recommends imputation strategy.

Mechanism types:
- MCAR (Missing Completely At Random): missingness independent of observed/unobserved
- MAR (Missing At Random): missingness depends on observed variables
- MNAR (Missing Not At Random): missingness depends on the missing values themselves

Heuristic detection:
- If missing rate uniform across other columns -> likely MCAR
- If missing rate correlates with another column -> likely MAR
- If missingness has business meaning (e.g., "no_response") -> likely MNAR

Reference:
    Little, R. J., & Rubin, D. B. (2019). Statistical Analysis with Missing Data.
"""
from __future__ import annotations
from typing import Dict, Any, List
import pandas as pd
import numpy as np


class MissingAnalyzer:
    """Analyzes missing patterns and recommends imputation strategy per column."""

    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        if df.empty:
            return {"columns": [], "summary": {}, "recommendations": []}

        n_rows = len(df)
        per_column = []

        for col in df.columns:
            missing_count = int(df[col].isna().sum())
            missing_rate = missing_count / n_rows if n_rows else 0

            if missing_count == 0:
                continue

            mechanism = self._detect_mechanism(df, col)
            strategy = self._recommend_strategy(df, col, missing_rate, mechanism)

            per_column.append({
                "column": col,
                "n_missing": missing_count,
                "missing_rate": round(missing_rate, 4),
                "missing_pct": round(missing_rate * 100, 2),
                "dtype": str(df[col].dtype),
                "mechanism": mechanism,
                "recommended_strategy": strategy["method"],
                "strategy_reasoning": strategy["reasoning"],
                "alternatives": strategy["alternatives"],
                "use_indicator": strategy["use_indicator"],
            })

        per_column.sort(key=lambda x: x["missing_rate"], reverse=True)

        return {
            "n_rows": n_rows,
            "n_columns_with_missing": len(per_column),
            "total_missing_cells": int(df.isna().sum().sum()),
            "columns": per_column,
            "summary": self._build_summary(per_column),
            "recommendations": self._build_recommendations(per_column),
        }

    def _detect_mechanism(self, df: pd.DataFrame, target_col: str) -> str:
        """Heuristic detection of missing mechanism."""
        is_missing = df[target_col].isna()
        if is_missing.sum() < 5:
            return "insufficient_data"

        # Check if missingness correlates with another column
        # If correlation > 0.3 with any other col's missing pattern -> MAR
        max_corr = 0
        for other_col in df.columns:
            if other_col == target_col:
                continue
            other_missing = df[other_col].isna()
            if other_missing.sum() < 5:
                continue
            # Phi correlation between binary missing indicators
            try:
                corr = is_missing.astype(int).corr(other_missing.astype(int))
                if not np.isnan(corr):
                    max_corr = max(max_corr, abs(corr))
            except Exception:
                continue

        if max_corr > 0.3:
            return "MAR_likely"

        # If missingness is concentrated (clusters) -> MNAR likely
        # Simple check: are missing rows contiguous?
        if is_missing.diff().fillna(False).sum() < is_missing.sum() / 4:
            return "MNAR_likely"

        return "MCAR_likely"

    def _recommend_strategy(
        self, df: pd.DataFrame, col: str, missing_rate: float, mechanism: str
    ) -> Dict[str, Any]:
        """Recommend imputation method based on dtype, rate, and mechanism."""
        is_numeric = pd.api.types.is_numeric_dtype(df[col])
        is_categorical = (
            pd.api.types.is_object_dtype(df[col]) or
            pd.api.types.is_categorical_dtype(df[col])
        )

        # If too many missing, drop is best
        if missing_rate > 0.5:
            return {
                "method": "drop_column",
                "reasoning": (
                    f"Column has >{int(missing_rate*100)}% missing. Imputation would introduce "
                    f"too much synthetic data, harming model reliability."
                ),
                "alternatives": [
                    {"method": "knn_imputer", "reason_rejected": "Risk of creating noise from few real data points"},
                ],
                "use_indicator": False,
            }

        # MNAR: missingness has meaning, use indicator + imputation
        if mechanism == "MNAR_likely":
            if is_numeric:
                return {
                    "method": "median_with_indicator",
                    "reasoning": (
                        "MNAR pattern detected: missingness may carry signal. "
                        "Use median imputation AND add binary missing indicator feature so model can learn."
                    ),
                    "alternatives": [
                        {"method": "mean", "reason_rejected": "Sensitive to outliers"},
                        {"method": "drop_rows", "reason_rejected": "Loses informative signal"},
                    ],
                    "use_indicator": True,
                }
            else:
                return {
                    "method": "constant_with_indicator",
                    "reasoning": (
                        "MNAR pattern detected for categorical: missingness carries signal. "
                        "Impute with sentinel value 'missing' and add indicator."
                    ),
                    "alternatives": [
                        {"method": "mode", "reason_rejected": "Ignores potentially meaningful missingness"},
                    ],
                    "use_indicator": True,
                }

        # MAR: depends on other columns, KNN imputation works well
        if mechanism == "MAR_likely" and missing_rate > 0.05:
            if is_numeric:
                return {
                    "method": "knn_imputer",
                    "reasoning": (
                        "MAR pattern detected (missingness correlates with another column). "
                        "KNN imputer leverages relationships between columns to estimate missing values. "
                        "Reference: Little & Rubin (2019)."
                    ),
                    "alternatives": [
                        {"method": "iterative_imputer", "reason_rejected": "More expensive; KNN sufficient for moderate missingness"},
                        {"method": "median", "reason_rejected": "Ignores cross-column information"},
                    ],
                    "use_indicator": False,
                }

        # MCAR or low missing rate: simple imputation is fine
        if is_numeric:
            # Check skewness - if skewed, prefer median
            non_null = df[col].dropna()
            try:
                skew = float(non_null.skew())
            except Exception:
                skew = 0
            if abs(skew) > 1:
                return {
                    "method": "median",
                    "reasoning": (
                        f"Numerical column with low missing rate ({missing_rate*100:.1f}%) "
                        f"and skewed distribution (skew={skew:.2f}). Median is robust to skewness."
                    ),
                    "alternatives": [
                        {"method": "mean", "reason_rejected": "Mean is biased for skewed distributions"},
                        {"method": "knn_imputer", "reason_rejected": "Overkill for low missing rate"},
                    ],
                    "use_indicator": False,
                }
            else:
                return {
                    "method": "mean",
                    "reasoning": (
                        f"Numerical column with low missing rate ({missing_rate*100:.1f}%) "
                        f"and approximately normal distribution. Mean imputation is appropriate."
                    ),
                    "alternatives": [
                        {"method": "median", "reason_rejected": "Slight loss of information for symmetric data"},
                    ],
                    "use_indicator": False,
                }

        # Categorical
        return {
            "method": "mode",
            "reasoning": (
                f"Categorical column with {missing_rate*100:.1f}% missing. "
                f"Mode imputation (most frequent value) is the standard approach for low-rate missing."
            ),
            "alternatives": [
                {"method": "constant_missing", "reason_rejected": "Adds new category that may bias model"},
            ],
            "use_indicator": False,
        }

    def _build_summary(self, per_column: List[Dict]) -> Dict[str, Any]:
        if not per_column:
            return {"any_missing": False, "max_missing_rate": 0}
        return {
            "any_missing": True,
            "max_missing_rate": max(c["missing_rate"] for c in per_column),
            "n_columns_critical": sum(1 for c in per_column if c["missing_rate"] > 0.5),
            "n_columns_warning": sum(1 for c in per_column if 0.1 < c["missing_rate"] <= 0.5),
            "mechanism_distribution": self._count_mechanisms(per_column),
        }

    def _count_mechanisms(self, per_column: List[Dict]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for c in per_column:
            counts[c["mechanism"]] = counts.get(c["mechanism"], 0) + 1
        return counts

    def _build_recommendations(self, per_column: List[Dict]) -> List[Dict[str, Any]]:
        recs = []
        for c in per_column:
            if c["recommended_strategy"] == "drop_column":
                recs.append({
                    "priority": "high",
                    "column": c["column"],
                    "action": f"DROP column '{c['column']}' before modeling",
                    "reason": c["strategy_reasoning"],
                })
            else:
                recs.append({
                    "priority": "medium" if c["missing_rate"] > 0.1 else "low",
                    "column": c["column"],
                    "action": (
                        f"Impute '{c['column']}' using strategy: {c['recommended_strategy']}"
                        + (" + add missing indicator" if c["use_indicator"] else "")
                    ),
                    "reason": c["strategy_reasoning"],
                })
        return recs
