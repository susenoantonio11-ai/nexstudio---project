"""
Data Quality Checker
====================
Implements Wang & Strong (1996) data quality framework with 6 dimensions:
- Completeness (% non-missing)
- Consistency (format consistency across rows)
- Accuracy (validity of values)
- Validity (type matches schema)
- Uniqueness (duplicate detection)
- Timeliness (data freshness, if datetime present)

Output:
    - Overall quality score (0-100)
    - Per-dimension scores
    - List of issues with severity
    - Cleaning recommendations

Reference:
    Wang, R. Y., & Strong, D. M. (1996). Beyond Accuracy: What Data Quality
    Means to Data Consumers. Journal of MIS, 12(4), 5-33.
"""
from __future__ import annotations
from typing import Dict, Any, List
import pandas as pd
import numpy as np
from datetime import datetime


class DataQualityChecker:
    """Compute comprehensive data quality report."""

    def check(self, df: pd.DataFrame) -> Dict[str, Any]:
        if df is None or df.empty:
            return self._empty_report("Empty DataFrame")

        completeness = self._completeness(df)
        consistency = self._consistency(df)
        validity = self._validity(df)
        uniqueness = self._uniqueness(df)
        timeliness = self._timeliness(df)
        accuracy = self._accuracy(df)  # heuristic

        scores = {
            "completeness": completeness["score"],
            "consistency": consistency["score"],
            "accuracy": accuracy["score"],
            "validity": validity["score"],
            "uniqueness": uniqueness["score"],
            "timeliness": timeliness["score"],
        }

        # Weighted overall (completeness and validity weighted higher)
        weights = {
            "completeness": 0.25,
            "validity": 0.20,
            "consistency": 0.15,
            "accuracy": 0.15,
            "uniqueness": 0.15,
            "timeliness": 0.10,
        }
        overall = sum(scores[k] * weights[k] for k in scores)

        issues: List[Dict[str, Any]] = []
        recommendations: List[Dict[str, Any]] = []

        for dim_name, dim_data in [
            ("completeness", completeness),
            ("consistency", consistency),
            ("accuracy", accuracy),
            ("validity", validity),
            ("uniqueness", uniqueness),
            ("timeliness", timeliness),
        ]:
            issues.extend(dim_data.get("issues", []))
            recommendations.extend(dim_data.get("recommendations", []))

        return {
            "overall_score": round(overall, 2),
            "dimension_scores": {k: round(v, 2) for k, v in scores.items()},
            "n_issues": len(issues),
            "issues": issues,
            "cleaning_recommendations": recommendations,
            "n_rows": int(len(df)),
            "n_columns": int(len(df.columns)),
            "timestamp": datetime.utcnow().isoformat(),
        }

    def _completeness(self, df: pd.DataFrame) -> Dict[str, Any]:
        total = len(df) * len(df.columns)
        n_missing = int(df.isna().sum().sum())
        score = 100 * (1 - n_missing / total) if total > 0 else 0

        issues = []
        recs = []

        for col in df.columns:
            missing_pct = df[col].isna().mean() * 100
            if missing_pct > 50:
                issues.append({
                    "column": col,
                    "issue_type": "high_missing",
                    "severity": "critical",
                    "description": f"Column has {missing_pct:.1f}% missing values",
                    "value": missing_pct,
                })
                recs.append({
                    "step": "drop_or_imputation",
                    "column": col,
                    "action": (
                        f"Consider dropping column '{col}' or using advanced imputation "
                        f"(KNN/Iterative). >50% missing weakens any imputation."
                    ),
                    "reason": f"Column completeness only {100 - missing_pct:.1f}%",
                })
            elif missing_pct > 10:
                issues.append({
                    "column": col,
                    "issue_type": "moderate_missing",
                    "severity": "warning",
                    "description": f"Column has {missing_pct:.1f}% missing values",
                    "value": missing_pct,
                })
                recs.append({
                    "step": "imputation",
                    "column": col,
                    "action": (
                        f"Apply imputation on '{col}'. For numerical use median; "
                        f"for categorical use mode. Consider missing indicator."
                    ),
                    "reason": f"Moderate missing rate {missing_pct:.1f}%",
                })

        return {"score": score, "issues": issues, "recommendations": recs}

    def _consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check format consistency within string columns."""
        issues = []
        score = 100
        for col in df.select_dtypes(include=["object"]).columns:
            non_null = df[col].dropna().astype(str)
            if len(non_null) == 0:
                continue

            # Check leading/trailing whitespace
            with_space = non_null[non_null != non_null.str.strip()]
            if len(with_space) / len(non_null) > 0.05:
                issues.append({
                    "column": col,
                    "issue_type": "whitespace",
                    "severity": "warning",
                    "description": f"{len(with_space)} values have leading/trailing whitespace",
                })
                score -= 5

            # Check mixed case for low-cardinality
            if non_null.nunique() < 50:
                lower_unique = non_null.str.lower().nunique()
                if lower_unique < non_null.nunique():
                    issues.append({
                        "column": col,
                        "issue_type": "case_inconsistency",
                        "severity": "info",
                        "description": (
                            f"Same values appear in different cases "
                            f"({non_null.nunique()} actual vs {lower_unique} normalized)"
                        ),
                    })
                    score -= 3

        return {"score": max(0, score), "issues": issues, "recommendations": []}

    def _accuracy(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Heuristic accuracy: check for impossible values."""
        issues = []
        score = 100
        for col in df.select_dtypes(include=["number"]).columns:
            series = df[col].dropna()
            if len(series) == 0:
                continue

            # Negative values where not expected (e.g., quantity, price, age)
            name_lower = col.lower()
            if any(w in name_lower for w in ["quantity", "price", "amount", "age", "count"]):
                neg_count = int((series < 0).sum())
                if neg_count > 0:
                    issues.append({
                        "column": col,
                        "issue_type": "impossible_negative",
                        "severity": "warning",
                        "description": f"{neg_count} negative values in '{col}' (suggested non-negative)",
                    })
                    score -= 5

        return {"score": max(0, score), "issues": issues, "recommendations": []}

    def _validity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check that types match expected schema."""
        score = 100
        issues = []
        for col in df.columns:
            if df[col].dtype == "object":
                # Check if all values look numeric (should be number column)
                non_null = df[col].dropna().astype(str)
                if len(non_null) > 0:
                    numeric_count = pd.to_numeric(non_null, errors="coerce").notna().sum()
                    if numeric_count / len(non_null) > 0.95 and numeric_count < len(non_null):
                        issues.append({
                            "column": col,
                            "issue_type": "type_mismatch",
                            "severity": "warning",
                            "description": (
                                f"Column '{col}' is object but {numeric_count}/{len(non_null)} "
                                f"values are numeric. Possible type coercion issue."
                            ),
                        })
                        score -= 3

        return {"score": max(0, score), "issues": issues, "recommendations": []}

    def _uniqueness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect duplicate rows."""
        n_duplicates = int(df.duplicated().sum())
        rate = n_duplicates / len(df) if len(df) else 0
        score = 100 * (1 - rate)

        issues = []
        recs = []
        if rate > 0.01:
            issues.append({
                "column": "<all>",
                "issue_type": "duplicates",
                "severity": "warning" if rate < 0.05 else "critical",
                "description": f"{n_duplicates} duplicate rows ({rate*100:.1f}% of dataset)",
            })
            recs.append({
                "step": "deduplicate",
                "column": "<all>",
                "action": "Apply df.drop_duplicates() before training to avoid biased models",
                "reason": f"{rate*100:.1f}% duplicate rate",
            })

        return {"score": score, "issues": issues, "recommendations": recs}

    def _timeliness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check data freshness if datetime column present."""
        date_cols = []
        for col in df.columns:
            try:
                parsed = pd.to_datetime(df[col], errors="coerce")
                if parsed.notna().sum() / len(df) > 0.5:
                    date_cols.append((col, parsed))
            except Exception:
                continue

        if not date_cols:
            return {"score": 100, "issues": [], "recommendations": []}

        # Use the column with most parseable dates
        col, parsed = max(date_cols, key=lambda x: x[1].notna().sum())
        max_date = parsed.max()
        if pd.isna(max_date):
            return {"score": 100, "issues": [], "recommendations": []}

        days_old = (datetime.utcnow() - max_date.to_pydatetime()).days
        score = max(0, 100 - days_old / 30)  # lose 1 point per month old

        issues = []
        if days_old > 365:
            issues.append({
                "column": col,
                "issue_type": "stale_data",
                "severity": "warning",
                "description": f"Latest data is {days_old} days old (>1 year). Model may not reflect current patterns.",
            })

        return {"score": score, "issues": issues, "recommendations": []}

    def _empty_report(self, msg: str) -> Dict[str, Any]:
        return {
            "overall_score": 0,
            "dimension_scores": {},
            "n_issues": 1,
            "issues": [{"column": "<all>", "issue_type": "empty", "severity": "critical", "description": msg}],
            "cleaning_recommendations": [],
            "n_rows": 0,
            "n_columns": 0,
        }
