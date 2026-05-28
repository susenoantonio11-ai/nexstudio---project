"""
Overfitting Detector
====================
Compares CV training scores vs CV test scores AND test set scores.
Large gap = overfitting; need regularization, more data, or simpler model.

Heuristics:
- gap < 0.05  → low risk
- gap 0.05-0.15 → moderate risk (possibly acceptable)
- gap > 0.15  → high risk (model memorized training data)

Reference:
    Hastie, Tibshirani & Friedman (2009) - Elements of Statistical Learning
"""
from __future__ import annotations
from typing import Dict, Any


class OverfittingDetector:
    """Diagnose overfitting from CV vs test scores."""

    def diagnose(
        self,
        cv_train_score: float,
        cv_test_score: float,
        held_out_test_score: float,
        scoring_metric: str = "f1_macro",
    ) -> Dict[str, Any]:
        """
        Args:
            cv_train_score: mean CV training score
            cv_test_score: mean CV validation score
            held_out_test_score: score on the held-out test set
            scoring_metric: name (just for context)
        """
        # Negative sklearn metrics need sign flip for human interpretation
        higher_is_better = "neg_" not in scoring_metric

        cv_gap = cv_train_score - cv_test_score if higher_is_better else cv_test_score - cv_train_score
        cv_test_to_holdout_drift = abs(cv_test_score - held_out_test_score)

        cv_gap_severity = self._severity(cv_gap)
        drift_severity = self._severity(cv_test_to_holdout_drift)

        diagnosis: Dict[str, Any] = {
            "cv_train_score": round(cv_train_score, 4),
            "cv_test_score": round(cv_test_score, 4),
            "held_out_test_score": round(held_out_test_score, 4),
            "cv_train_test_gap": round(cv_gap, 4),
            "cv_test_to_holdout_drift": round(cv_test_to_holdout_drift, 4),
            "scoring_metric": scoring_metric,
            "overfitting_severity": cv_gap_severity,
            "generalization_drift_severity": drift_severity,
        }

        recommendations = []
        warnings = []

        if cv_gap_severity == "high":
            warnings.append(
                f"HIGH overfitting risk: CV train-test gap = {cv_gap:.3f}. "
                f"Model memorized training data."
            )
            recommendations.extend([
                "Reduce model complexity (max_depth, n_estimators)",
                "Increase regularization (Ridge/Lasso, dropout, alpha)",
                "Collect more training data",
                "Apply feature selection to remove noise",
            ])
        elif cv_gap_severity == "moderate":
            warnings.append(
                f"Moderate overfitting: CV train-test gap = {cv_gap:.3f}. "
                f"Generalization may be acceptable but improvement possible."
            )
            recommendations.append("Consider mild regularization or early stopping")

        if drift_severity == "high":
            warnings.append(
                f"CV→holdout drift = {cv_test_to_holdout_drift:.3f}. "
                f"CV may not be representative of true holdout performance. "
                f"Suspect: data distribution shift, leakage in CV setup, or unlucky split."
            )
            recommendations.append(
                "Audit CV setup: ensure preprocessing is inside Pipeline, not pre-fit"
            )

        if not warnings:
            warnings.append("No significant overfitting or drift detected.")

        diagnosis["warnings"] = warnings
        diagnosis["recommendations"] = recommendations
        diagnosis["overall_assessment"] = self._build_assessment(
            cv_gap_severity, drift_severity, held_out_test_score, scoring_metric
        )

        return diagnosis

    def _severity(self, gap: float) -> str:
        if gap > 0.15:
            return "high"
        if gap > 0.05:
            return "moderate"
        return "low"

    def _build_assessment(
        self,
        cv_gap_sev: str,
        drift_sev: str,
        holdout_score: float,
        scoring: str,
    ) -> str:
        if cv_gap_sev == "low" and drift_sev == "low":
            return (
                f"GENERALIZES WELL. Holdout {scoring} = {holdout_score:.3f}. "
                f"Model performance on unseen data is consistent with cross-validation. "
                f"Safe to deploy with monitoring."
            )
        if cv_gap_sev == "high":
            return (
                f"OVERFITTING. Holdout {scoring} = {holdout_score:.3f} but CV gap is large. "
                f"Reduce complexity or add regularization before deployment."
            )
        if drift_sev == "high":
            return (
                f"DRIFT DETECTED between CV and holdout. Holdout {scoring} = {holdout_score:.3f}. "
                f"Audit CV setup before trusting either estimate."
            )
        return (
            f"ACCEPTABLE. Holdout {scoring} = {holdout_score:.3f}. "
            f"Some room for improvement via tuning."
        )
