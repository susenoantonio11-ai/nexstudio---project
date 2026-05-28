"""
Feature Engineer
================
Generates new features from existing ones, BEFORE the train/test split is fine
for deterministic features (e.g., year/month from date) but UNSAFE for
features that use statistics (target-encoding, rolling averages on whole data).

This implementation only produces leak-safe features:
- Datetime decomposition (year, month, weekday, etc) - deterministic
- Log transformation (log1p) - deterministic
- Binning - simple cuts (NOT quantile-based which uses data stats)
"""
from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np


class FeatureEngineer:
    """Add deterministic engineered features (no leakage)."""

    def transform(
        self,
        df: pd.DataFrame,
        datetime_columns: Optional[List[str]] = None,
        log_columns: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """
        Returns:
            (transformed_df, list_of_added_features_with_reasoning)
        """
        df = df.copy()
        added: List[Dict[str, Any]] = []

        # 1. Datetime decomposition
        if datetime_columns is None:
            datetime_columns = self._auto_detect_datetime_cols(df)

        for col in datetime_columns:
            if col not in df.columns:
                continue
            try:
                parsed = pd.to_datetime(df[col], errors="coerce")
                if parsed.notna().sum() == 0:
                    continue

                df[f"{col}_year"] = parsed.dt.year
                df[f"{col}_month"] = parsed.dt.month
                df[f"{col}_day"] = parsed.dt.day
                df[f"{col}_dayofweek"] = parsed.dt.dayofweek
                df[f"{col}_quarter"] = parsed.dt.quarter
                df[f"{col}_is_weekend"] = parsed.dt.dayofweek.isin([5, 6]).astype(int)

                added.append({
                    "source_column": col,
                    "features_added": [
                        f"{col}_year", f"{col}_month", f"{col}_day",
                        f"{col}_dayofweek", f"{col}_quarter", f"{col}_is_weekend",
                    ],
                    "reasoning": (
                        f"Datetime '{col}' decomposed into temporal components. "
                        f"This is deterministic (no leakage) and lets tree-based models "
                        f"capture seasonality, day-of-week effects, and trends."
                    ),
                })

                # Drop original datetime (sklearn cannot handle datetime dtype)
                df = df.drop(columns=[col])
            except Exception:
                continue

        # 2. Log transformation for highly-skewed positive numerics
        if log_columns:
            for col in log_columns:
                if col not in df.columns:
                    continue
                if not pd.api.types.is_numeric_dtype(df[col]):
                    continue
                if (df[col].dropna() < 0).any():
                    continue  # log on negative values is undefined
                df[f"{col}_log"] = np.log1p(df[col])
                added.append({
                    "source_column": col,
                    "features_added": [f"{col}_log"],
                    "reasoning": (
                        f"log1p({col}) added to handle right-skew. log1p = log(1+x), "
                        f"safe for zero values. Compresses extreme values, often "
                        f"improves linear model performance significantly."
                    ),
                })

        return df, added

    def _auto_detect_datetime_cols(self, df: pd.DataFrame) -> List[str]:
        cols = []
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                cols.append(col)
            elif df[col].dtype == "object":
                # Try parsing first 50 values
                try:
                    sample = df[col].dropna().head(50)
                    parsed = pd.to_datetime(sample, errors="coerce")
                    if parsed.notna().sum() / max(1, len(sample)) > 0.8:
                        cols.append(col)
                except Exception:
                    continue
        return cols
