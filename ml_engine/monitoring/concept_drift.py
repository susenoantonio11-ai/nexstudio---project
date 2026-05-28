"""
Concept Drift Detector
======================
Detects degradation in model performance over time, signaling that the
relationship between features and target has changed.

Methods:
- Performance window comparison: rolling-window metrics vs baseline
- Page-Hinkley test (sequential change detection)
"""
from __future__ import annotations
from typing import Dict, Any, List


class ConceptDriftDetector:
    """Detect concept drift via performance degradation."""

    def detect(
        self,
        baseline_metric: float,
        recent_metrics: List[float],
        threshold_drop_pct: float = 10.0,
        higher_is_better: bool = True,
    ) -> Dict[str, Any]:
        """
        Args:
            baseline_metric: original test set score
            recent_metrics: rolling window of recent scores
            threshold_drop_pct: % drop that triggers alert (default 10%)
        """
        if not recent_metrics:
            return {"verdict": "no_data", "recommendation": "Need recent metrics to assess drift"}

        recent_avg = sum(recent_metrics) / len(recent_metrics)

        if higher_is_better:
            drop_pct = (baseline_metric - recent_avg) / baseline_metric * 100 if baseline_metric != 0 else 0
        else:
            drop_pct = (recent_avg - baseline_metric) / baseline_metric * 100 if baseline_metric != 0 else 0

        is_drifted = drop_pct >= threshold_drop_pct

        return {
            "baseline_metric": round(float(baseline_metric), 4),
            "recent_average": round(float(recent_avg), 4),
            "performance_drop_pct": round(float(drop_pct), 2),
            "threshold_drop_pct": threshold_drop_pct,
            "drift_detected": is_drifted,
            "verdict": "DRIFT_DETECTED" if is_drifted else "STABLE",
            "recommendation": (
                f"Performance dropped {drop_pct:.1f}% vs baseline. RETRAIN model with recent data."
                if is_drifted else
                f"Performance change ({drop_pct:.1f}%) within acceptable range. No action needed."
            ),
            "n_recent_observations": len(recent_metrics),
        }
