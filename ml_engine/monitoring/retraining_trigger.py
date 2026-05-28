"""
Retraining Trigger
==================
Combines data drift + concept drift signals to decide if retraining is needed.

Decision rules:
- Significant data drift OR concept drift -> retrain
- Moderate drift + extended period -> retrain
- Stable -> no action
"""
from __future__ import annotations
from typing import Dict, Any


class RetrainingTrigger:
    """Decide whether to retrain based on monitoring signals."""

    def evaluate(
        self,
        data_drift_result: Dict[str, Any],
        concept_drift_result: Dict[str, Any],
        days_since_last_retrain: int = 0,
        max_days_between_retrains: int = 90,
    ) -> Dict[str, Any]:
        """
        Returns retraining decision with reasoning.
        """
        reasons = []
        priority = "none"

        if data_drift_result.get("overall_verdict") == "RETRAIN_RECOMMENDED":
            reasons.append(
                f"Data drift detected (max PSI = {data_drift_result.get('max_psi', 0):.3f})"
            )
            priority = "high"

        if concept_drift_result.get("drift_detected"):
            reasons.append(
                f"Concept drift: performance dropped "
                f"{concept_drift_result.get('performance_drop_pct', 0):.1f}%"
            )
            priority = "high"

        if days_since_last_retrain > max_days_between_retrains:
            reasons.append(
                f"Model hasn't been retrained in {days_since_last_retrain} days "
                f"(policy max: {max_days_between_retrains})"
            )
            if priority == "none":
                priority = "medium"

        # Moderate signals
        if data_drift_result.get("overall_verdict") == "MONITOR_CLOSELY":
            if priority == "none":
                priority = "low"
            reasons.append("Moderate data drift, increase monitoring frequency")

        decision = priority != "none"

        return {
            "should_retrain": decision,
            "priority": priority,
            "reasons": reasons,
            "data_drift_verdict": data_drift_result.get("overall_verdict"),
            "concept_drift_verdict": concept_drift_result.get("verdict"),
            "days_since_last_retrain": days_since_last_retrain,
            "recommendation": self._build_recommendation(priority, reasons),
        }

    def _build_recommendation(self, priority: str, reasons: list) -> str:
        if priority == "high":
            return (
                "RETRAIN IMMEDIATELY. " +
                " ".join(reasons) +
                ". Validate retrained model on a fresh holdout before redeploying."
            )
        if priority == "medium":
            return (
                "Schedule retraining within 1 week. " +
                " ".join(reasons)
            )
        if priority == "low":
            return (
                "Monitor closely. " +
                " ".join(reasons) +
                ". Consider retraining if drift persists for 2+ weeks."
            )
        return "Model is stable. Continue routine monitoring."
