"""
Data Drift Detector
===================
Detects changes in feature distributions between training (reference)
and production (current) data.

Methods:
- PSI (Population Stability Index): industry-standard for binned features
- KS test (Kolmogorov-Smirnov): non-parametric distribution comparison
- Mean/std shift: simple sanity check

PSI interpretation:
- < 0.1   : no significant change
- 0.1-0.25: moderate change, monitor
- > 0.25  : significant change, investigate

Reference:
    Tsymbal, A. (2004). The problem of concept drift: definitions and related work.
"""
from __future__ import annotations
from typing import Dict, Any, List
import numpy as np
import pandas as pd


class DataDriftDetector:
    """Detect distribution drift in features."""

    def detect(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
        n_bins: int = 10,
    ) -> Dict[str, Any]:
        """
        Args:
            reference: training data (or reference window)
            current: new production data

        Returns drift score per feature + overall verdict.
        """
        common_cols = [c for c in reference.columns if c in current.columns]
        per_column = []

        for col in common_cols:
            if pd.api.types.is_numeric_dtype(reference[col]):
                psi = self._psi_numeric(reference[col].dropna(), current[col].dropna(), n_bins)
                ks_stat, ks_pvalue = self._ks_test(reference[col].dropna(), current[col].dropna())
                mean_ref = float(reference[col].mean())
                mean_cur = float(current[col].mean())
                mean_shift = (mean_cur - mean_ref) / mean_ref if mean_ref != 0 else 0
                per_column.append({
                    "column": col,
                    "type": "numeric",
                    "psi": round(psi, 4),
                    "ks_statistic": round(ks_stat, 4),
                    "ks_pvalue": round(ks_pvalue, 4),
                    "mean_shift_pct": round(float(mean_shift) * 100, 2),
                    "drift_verdict": self._psi_verdict(psi),
                })
            else:
                # Categorical drift via PSI on category proportions
                psi = self._psi_categorical(reference[col].dropna(), current[col].dropna())
                per_column.append({
                    "column": col,
                    "type": "categorical",
                    "psi": round(psi, 4),
                    "drift_verdict": self._psi_verdict(psi),
                })

        n_drifted = sum(1 for c in per_column if c["drift_verdict"] != "stable")
        max_psi = max((c.get("psi", 0) for c in per_column), default=0)

        return {
            "n_features_analyzed": len(per_column),
            "n_features_drifted": n_drifted,
            "max_psi": round(float(max_psi), 4),
            "overall_verdict": self._overall_verdict(n_drifted, len(per_column), max_psi),
            "per_column": sorted(per_column, key=lambda x: x.get("psi", 0), reverse=True),
            "recommendations": self._build_recommendations(n_drifted, max_psi),
        }

    def _psi_numeric(self, ref, cur, n_bins) -> float:
        """PSI for numeric features using equal-frequency bins."""
        try:
            quantiles = np.linspace(0, 1, n_bins + 1)
            bin_edges = np.unique(np.quantile(ref, quantiles))
            if len(bin_edges) < 3:
                return 0.0
            bin_edges[0] = -np.inf
            bin_edges[-1] = np.inf

            ref_counts, _ = np.histogram(ref, bins=bin_edges)
            cur_counts, _ = np.histogram(cur, bins=bin_edges)

            ref_pct = ref_counts / max(ref_counts.sum(), 1)
            cur_pct = cur_counts / max(cur_counts.sum(), 1)

            # Avoid divide-by-zero
            ref_pct = np.where(ref_pct == 0, 0.0001, ref_pct)
            cur_pct = np.where(cur_pct == 0, 0.0001, cur_pct)

            psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
            return abs(psi)
        except Exception:
            return 0.0

    def _psi_categorical(self, ref, cur) -> float:
        ref_props = ref.value_counts(normalize=True)
        cur_props = cur.value_counts(normalize=True)
        all_categories = set(ref_props.index) | set(cur_props.index)

        psi = 0.0
        for cat in all_categories:
            r = max(ref_props.get(cat, 0), 0.0001)
            c = max(cur_props.get(cat, 0), 0.0001)
            psi += (c - r) * np.log(c / r)
        return float(abs(psi))

    def _ks_test(self, ref, cur) -> tuple:
        try:
            from scipy.stats import ks_2samp
            stat, pvalue = ks_2samp(ref, cur)
            return float(stat), float(pvalue)
        except ImportError:
            return 0.0, 1.0

    def _psi_verdict(self, psi: float) -> str:
        if psi < 0.1:
            return "stable"
        if psi < 0.25:
            return "moderate_drift"
        return "significant_drift"

    def _overall_verdict(self, n_drifted: int, n_total: int, max_psi: float) -> str:
        if max_psi >= 0.25 or (n_total > 0 and n_drifted / n_total > 0.3):
            return "RETRAIN_RECOMMENDED"
        if max_psi >= 0.1 or n_drifted > 0:
            return "MONITOR_CLOSELY"
        return "STABLE"

    def _build_recommendations(self, n_drifted: int, max_psi: float) -> List[str]:
        recs = []
        if max_psi >= 0.25:
            recs.append(
                "Significant drift detected (PSI >= 0.25). Retrain model on recent data. "
                "Investigate root cause: market shift, seasonality, data pipeline change?"
            )
        elif max_psi >= 0.1:
            recs.append(
                "Moderate drift (PSI 0.1-0.25). Increase monitoring frequency. "
                "Plan retraining if drift persists."
            )
        else:
            recs.append("No significant drift. Continue monitoring.")
        return recs
