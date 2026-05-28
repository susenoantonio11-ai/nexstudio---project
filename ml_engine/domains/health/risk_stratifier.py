"""
Risk Stratifier
===============
Stratifies patients into risk tiers based on predicted probability:
- Critical (top 5%)
- High (top 5-25%)
- Medium (top 25-50%)
- Low (bottom 50%)

Each tier maps to recommended intervention intensity.
"""
from __future__ import annotations
from typing import Dict, Any, List
import pandas as pd
import numpy as np


class RiskStratifier:
    """Stratify patients by risk score percentile."""

    DEFAULT_TIERS = [
        {"name": "low", "lower_pct": 0.0, "upper_pct": 0.5,
         "color": "#22C55E",
         "intervention": "Standard care - routine follow-up"},
        {"name": "medium", "lower_pct": 0.5, "upper_pct": 0.75,
         "color": "#F59E0B",
         "intervention": "Enhanced monitoring + scheduled follow-up within 7 days"},
        {"name": "high", "lower_pct": 0.75, "upper_pct": 0.95,
         "color": "#EF4444",
         "intervention": "Active intervention: nurse follow-up call, medication review, home visit consideration"},
        {"name": "critical", "lower_pct": 0.95, "upper_pct": 1.0,
         "color": "#DC2626",
         "intervention": "Immediate intervention: care coordinator assignment, daily monitoring, clinical review"},
    ]

    def stratify(
        self,
        risk_scores: List[float],
        patient_ids: List[Any] = None,
        custom_tiers: List[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Args:
            risk_scores: predicted risk probabilities or scores
            patient_ids: optional patient IDs (defaults to row index)
            custom_tiers: override default tier definitions
        """
        if not risk_scores:
            return {"error": "No risk scores provided"}

        tiers = custom_tiers or self.DEFAULT_TIERS
        n = len(risk_scores)
        scores_arr = np.array(risk_scores)
        if patient_ids is None:
            patient_ids = list(range(n))

        # Compute thresholds at percentiles
        tier_with_thresholds = []
        for t in tiers:
            tier_with_thresholds.append({
                **t,
                "lower_threshold": float(np.quantile(scores_arr, t["lower_pct"])),
                "upper_threshold": float(np.quantile(scores_arr, t["upper_pct"])),
            })

        # Assign each patient to tier (inclusive on upper edge for last tier)
        assignments = []
        last_idx = len(tier_with_thresholds) - 1
        for pid, score in zip(patient_ids, risk_scores):
            assigned_tier = None
            for ti, t in enumerate(tier_with_thresholds):
                upper = t["upper_threshold"]
                lower = t["lower_threshold"]
                if ti == last_idx:
                    if lower <= score <= upper:
                        assigned_tier = t["name"]
                        break
                elif lower <= score < upper:
                    assigned_tier = t["name"]
                    break
            if assigned_tier is None:
                assigned_tier = tier_with_thresholds[-1]["name"]
            assignments.append({
                "patient_id": str(pid),
                "risk_score": round(float(score), 4),
                "tier": assigned_tier,
            })

        # Tier counts
        tier_counts: Dict[str, int] = {}
        for a in assignments:
            tier_counts[a["tier"]] = tier_counts.get(a["tier"], 0) + 1

        # Tier summary with proportions
        tier_summary = []
        for t in tier_with_thresholds:
            count = tier_counts.get(t["name"], 0)
            tier_summary.append({
                "tier": t["name"],
                "color": t["color"],
                "intervention": t["intervention"],
                "lower_threshold": round(t["lower_threshold"], 4),
                "upper_threshold": round(t["upper_threshold"], 4),
                "n_patients": count,
                "proportion": round(count / n, 4) if n else 0,
            })

        return {
            "n_patients": n,
            "tier_assignments": assignments,
            "tier_summary": tier_summary,
            "tier_counts": tier_counts,
            "method_explanation": (
                "Risk stratification by percentile of predicted probability. "
                "Tiers are PERCENTILE-BASED (relative ranking), not absolute thresholds. "
                "Useful when predicted probabilities are calibrated to overall population. "
                "Each tier has matched intervention intensity for resource allocation."
            ),
            "method_monitor": {
                "selected_method": "Percentile-based stratification (4 tiers)",
                "why_chosen": (
                    "Distributes resources based on RELATIVE risk within population. "
                    "Avoids problem of absolute thresholds becoming meaningless when population characteristics change. "
                    "Top 5% always gets most attention, regardless of absolute risk values."
                ),
                "why_not_alternatives": [
                    {"alternative": "Absolute threshold (e.g., score > 0.7 = high)",
                     "reason_rejected": "Threshold may not align with resource capacity; can leave high-risk patients without intervention"},
                    {"alternative": "K-means clustering of scores",
                     "reason_rejected": "Cluster boundaries non-intuitive for clinicians"},
                ],
                "limitations": [
                    "Percentile is relative: if entire population is high-risk, 'low' tier still has meaningful risk",
                    "Tier boundaries should be validated against intervention capacity",
                ],
            },
        }
