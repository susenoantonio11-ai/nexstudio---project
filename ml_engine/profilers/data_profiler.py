"""
Data Profiler - CRISP-DM Step 2: Data Understanding
====================================================
Performs automatic profiling on uploaded datasets:
- Column type detection (numerical, categorical, datetime, text, boolean)
- Missing value analysis
- Unique value counting
- Statistical summaries
- Encoding detection
- Format inference

Reference: Wang & Strong (1996) - Data Quality Framework
"""
from __future__ import annotations
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import re


class DataProfiler:
    """
    Profiles a pandas DataFrame and produces structured metadata
    used downstream by the Target Detector and Model Selector.
    """

    DATETIME_PATTERNS = [
        r"^\d{4}-\d{2}-\d{2}",
        r"^\d{2}/\d{2}/\d{4}",
        r"^\d{2}-\d{2}-\d{4}",
        r"^\d{4}/\d{2}/\d{2}",
    ]

    def __init__(self, sample_size: int = 1000):
        self.sample_size = sample_size

    def profile(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate complete profile of dataframe.

        Returns:
            {
                "n_rows": int,
                "n_columns": int,
                "memory_usage_bytes": int,
                "columns": [
                    {
                        "name": str,
                        "inferred_type": str,
                        "pandas_dtype": str,
                        "n_missing": int,
                        "n_unique": int,
                        "completeness_pct": float,
                        "stats": {...}  # type-specific
                    }
                ],
                "summary": {...}
            }
        """
        if df is None or df.empty:
            raise ValueError("DataFrame cannot be empty")

        n_rows, n_columns = df.shape
        columns_meta = []

        for position, col_name in enumerate(df.columns):
            col_meta = self._profile_column(df[col_name], col_name, position, n_rows)
            columns_meta.append(col_meta)

        return {
            "n_rows": int(n_rows),
            "n_columns": int(n_columns),
            "memory_usage_bytes": int(df.memory_usage(deep=True).sum()),
            "columns": columns_meta,
            "summary": self._build_summary(columns_meta),
        }

    def _profile_column(
        self, series: pd.Series, col_name: str, position: int, total_rows: int
    ) -> Dict[str, Any]:
        n_missing = int(series.isna().sum())
        n_unique = int(series.nunique(dropna=True))
        completeness_pct = round((1 - n_missing / total_rows) * 100, 2) if total_rows else 0.0

        inferred_type = self._infer_type(series)
        pandas_dtype = str(series.dtype)

        stats: Dict[str, Any] = {
            "n_missing": n_missing,
            "n_unique": n_unique,
            "completeness_pct": completeness_pct,
        }

        if inferred_type == "numerical":
            non_null = series.dropna()
            if len(non_null) > 0:
                stats.update({
                    "mean": self._safe_float(non_null.mean()),
                    "std": self._safe_float(non_null.std()),
                    "min": self._safe_float(non_null.min()),
                    "max": self._safe_float(non_null.max()),
                    "median": self._safe_float(non_null.median()),
                    "q25": self._safe_float(non_null.quantile(0.25)),
                    "q75": self._safe_float(non_null.quantile(0.75)),
                })
        elif inferred_type in ("categorical", "boolean", "text"):
            top = series.value_counts(dropna=True).head(5)
            stats["top_values"] = [
                {"value": str(v), "count": int(c)} for v, c in top.items()
            ]

        return {
            "name": col_name,
            "position": position,
            "inferred_type": inferred_type,
            "pandas_dtype": pandas_dtype,
            "n_missing": n_missing,
            "n_unique": n_unique,
            "completeness_pct": completeness_pct,
            "stats": stats,
        }

    def _infer_type(self, series: pd.Series) -> str:
        """
        Heuristic type inference - more robust than pandas dtype alone.

        Decision rules:
        1. If pandas dtype is datetime -> datetime
        2. If can parse as datetime (>80% rows) -> datetime
        3. If numeric dtype -> numerical
        4. If 2 unique non-null values -> boolean
        5. If unique ratio low (<5% of total) and dtype object -> categorical
        6. If avg string length > 50 -> text
        7. Otherwise -> categorical
        """
        if pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"

        non_null = series.dropna()
        if len(non_null) == 0:
            return "unknown"

        # Try datetime parsing
        if non_null.dtype == "object":
            sample = non_null.head(20).astype(str)
            if self._looks_like_datetime(sample):
                return "datetime"

        if pd.api.types.is_numeric_dtype(series):
            return "numerical"

        if pd.api.types.is_bool_dtype(series):
            return "boolean"

        n_unique = non_null.nunique()
        n_total = len(non_null)

        if n_unique == 2:
            return "boolean"

        if n_total > 0:
            unique_ratio = n_unique / n_total
            if unique_ratio < 0.05 and n_unique <= 50:
                return "categorical"

        # Check if text (long strings)
        if series.dtype == "object":
            try:
                avg_len = non_null.astype(str).str.len().mean()
                if avg_len > 50:
                    return "text"
            except Exception:
                pass

        return "categorical"

    def _looks_like_datetime(self, sample: pd.Series) -> bool:
        if len(sample) == 0:
            return False
        match_count = 0
        for val in sample:
            for pattern in self.DATETIME_PATTERNS:
                if re.match(pattern, str(val)):
                    match_count += 1
                    break
        return match_count / len(sample) > 0.8

    def _safe_float(self, value: Any) -> Optional[float]:
        try:
            f = float(value)
            if np.isnan(f) or np.isinf(f):
                return None
            return round(f, 4)
        except (TypeError, ValueError):
            return None

    def _build_summary(self, columns_meta: List[Dict]) -> Dict[str, Any]:
        type_counts: Dict[str, int] = {}
        total_missing = 0
        for col in columns_meta:
            t = col["inferred_type"]
            type_counts[t] = type_counts.get(t, 0) + 1
            total_missing += col["n_missing"]
        return {
            "type_distribution": type_counts,
            "total_missing": total_missing,
            "n_numerical": type_counts.get("numerical", 0),
            "n_categorical": type_counts.get("categorical", 0),
            "n_datetime": type_counts.get("datetime", 0),
            "n_text": type_counts.get("text", 0),
            "n_boolean": type_counts.get("boolean", 0),
        }
