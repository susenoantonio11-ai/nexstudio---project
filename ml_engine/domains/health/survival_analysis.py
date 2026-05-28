"""
Survival Analysis - Kaplan-Meier estimator without lifelines dependency.
"""
from __future__ import annotations
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np


class SurvivalAnalyzer:
    """Kaplan-Meier survival curve estimator + log-rank test."""

    def kaplan_meier(
        self,
        df: pd.DataFrame,
        time_column: str,
        event_column: str,
        group_column: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Compute KM survival curve.

        Args:
            time_column: time-to-event column (numeric, e.g. days)
            event_column: 1 = event observed, 0 = censored
            group_column: optional grouping (e.g., treatment arm)
        """
        df = df.copy()
        df = df.dropna(subset=[time_column, event_column])
        df[time_column] = pd.to_numeric(df[time_column], errors="coerce")
        df[event_column] = pd.to_numeric(df[event_column], errors="coerce")
        df = df.dropna(subset=[time_column, event_column])

        if group_column and group_column in df.columns:
            curves = []
            for group_val, group_df in df.groupby(group_column):
                curve = self._km_curve(group_df, time_column, event_column)
                curve["group"] = str(group_val)
                curves.append(curve)
            # Log-rank test if 2 groups
            log_rank = None
            if len(curves) == 2:
                log_rank = self._log_rank_test(
                    df[df[group_column] == list(df[group_column].unique())[0]],
                    df[df[group_column] == list(df[group_column].unique())[1]],
                    time_column, event_column,
                )
        else:
            curves = [self._km_curve(df, time_column, event_column)]
            curves[0]["group"] = "All"
            log_rank = None

        return {
            "method": "Kaplan-Meier",
            "n_subjects": int(len(df)),
            "n_events": int(df[event_column].sum()),
            "censoring_rate": round(float(1 - df[event_column].mean()), 4),
            "curves": curves,
            "log_rank_test": log_rank,
            "method_explanation": (
                "Kaplan-Meier estimator computes survival probability over time. "
                "Censored observations (subjects who left study before event) are correctly handled. "
                "Drops in curve represent events; horizontal segments = censoring. "
                "Reference: Kaplan & Meier (1958)."
            ),
            "method_monitor": {
                "selected_method": "Kaplan-Meier non-parametric estimator",
                "why_chosen": (
                    "KM makes no assumption about underlying survival distribution, "
                    "handles right-censoring naturally, and is the standard in clinical research."
                ),
                "why_not_alternatives": [
                    {"alternative": "Cox Proportional Hazards",
                     "reason_rejected": "Requires covariates; KM is for descriptive survival without features"},
                    {"alternative": "Parametric (Weibull, exponential)",
                     "reason_rejected": "KM avoids distributional assumptions"},
                ],
                "limitations": [
                    "Assumes censoring is non-informative",
                    "Cannot adjust for covariates (use Cox regression for that)",
                ],
            },
        }

    def _km_curve(self, df: pd.DataFrame, time_col: str, event_col: str) -> Dict[str, Any]:
        """Compute Kaplan-Meier curve from raw data."""
        sorted_df = df.sort_values(time_col).reset_index(drop=True)
        n = len(sorted_df)

        # Group by unique time
        unique_times = sorted_df[time_col].unique()
        survival = 1.0
        points = [{"time": 0.0, "survival": 1.0, "n_at_risk": n, "n_events": 0}]

        n_at_risk = n
        for t in sorted(unique_times):
            at_t = sorted_df[sorted_df[time_col] == t]
            n_events = int(at_t[event_col].sum())
            n_censored = int(len(at_t) - n_events)
            if n_at_risk > 0 and n_events > 0:
                survival *= (1 - n_events / n_at_risk)
            points.append({
                "time": float(t),
                "survival": round(float(survival), 4),
                "n_at_risk": int(n_at_risk),
                "n_events": n_events,
                "n_censored": n_censored,
            })
            n_at_risk -= len(at_t)

        # Median survival time (where survival crosses 0.5)
        median_survival = None
        for p in points:
            if p["survival"] <= 0.5:
                median_survival = p["time"]
                break

        return {
            "n_subjects": n,
            "n_events": int(sorted_df[event_col].sum()),
            "median_survival_time": median_survival,
            "max_followup_time": float(sorted_df[time_col].max()),
            "points": points,
        }

    def _log_rank_test(
        self,
        group1: pd.DataFrame,
        group2: pd.DataFrame,
        time_col: str,
        event_col: str,
    ) -> Dict[str, Any]:
        """Simplified log-rank test for two groups."""
        try:
            from scipy.stats import chi2
        except ImportError:
            return {"available": False, "reason": "scipy not installed"}

        all_times = sorted(set(group1[time_col].tolist() + group2[time_col].tolist()))
        observed1 = 0
        expected1 = 0
        variance = 0

        for t in all_times:
            n1 = int((group1[time_col] >= t).sum())
            n2 = int((group2[time_col] >= t).sum())
            d1 = int(((group1[time_col] == t) & (group1[event_col] == 1)).sum())
            d2 = int(((group2[time_col] == t) & (group2[event_col] == 1)).sum())
            n = n1 + n2
            d = d1 + d2
            if n > 1:
                observed1 += d1
                expected1 += d * n1 / n
                variance += (d * n1 * n2 * (n - d)) / (n * n * (n - 1)) if n > 1 else 0

        if variance <= 0:
            return {"available": False, "reason": "Insufficient variance"}

        z2 = (observed1 - expected1) ** 2 / variance
        p_value = 1 - chi2.cdf(z2, df=1)

        return {
            "available": True,
            "test_statistic": round(float(z2), 4),
            "p_value": round(float(p_value), 4),
            "significant_at_0.05": p_value < 0.05,
            "interpretation": (
                f"{'SIGNIFICANT' if p_value < 0.05 else 'Not significant'} difference in survival "
                f"between groups (p = {p_value:.4f}). "
                f"{'Reject null hypothesis of equal survival.' if p_value < 0.05 else 'Cannot reject null.'}"
            ),
        }
