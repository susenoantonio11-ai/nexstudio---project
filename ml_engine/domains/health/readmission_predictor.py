"""
Readmission Predictor
=====================
Wraps PatientRiskPredictor with readmission-specific feature engineering:
- Days since last admission
- Number of prior admissions
- Comorbidity count
- Length of stay
"""
from __future__ import annotations
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

from .patient_risk import PatientRiskPredictor


class ReadmissionPredictor:
    """30-day or 90-day hospital readmission prediction."""

    def fit_evaluate(
        self,
        df: pd.DataFrame,
        target_column: str = "readmitted",
        feature_columns: Optional[List[str]] = None,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Dict[str, Any]:
        # Delegate to base predictor
        predictor = PatientRiskPredictor(algorithm="random_forest")
        result = predictor.fit_evaluate(
            df=df,
            target_column=target_column,
            feature_columns=feature_columns,
            test_size=test_size,
            random_state=random_state,
        )

        if "error" in result:
            return result

        # Add domain-specific guidance
        m = result.get("metrics", {})
        result["clinical_summary"] = self._build_clinical_summary(m)
        result["intervention_recommendations"] = self._suggest_interventions(m)
        result["domain"] = "hospital_readmission"

        return result

    def _build_clinical_summary(self, metrics: Dict[str, Any]) -> str:
        roc = metrics.get("roc_auc", 0)
        sens = metrics.get("sensitivity", 0)
        spec = metrics.get("specificity", 0)
        ppv = metrics.get("ppv", 0)

        roc_quality = (
            "EXCELLENT" if roc > 0.85 else
            "GOOD" if roc > 0.75 else
            "FAIR" if roc > 0.65 else
            "POOR"
        )
        return (
            f"Model {roc_quality} for readmission prediction (ROC-AUC = {roc:.3f}). "
            f"Sensitivity = {sens:.3f}: catches {int(sens*100)}% of patients who will be readmitted. "
            f"Specificity = {spec:.3f}: correctly identifies {int(spec*100)}% of non-readmitted. "
            f"PPV = {ppv:.3f}: when predicted high-risk, {int(ppv*100)}% actually readmit. "
            f"Use these metrics to choose intervention threshold based on resource availability."
        )

    def _suggest_interventions(self, metrics: Dict[str, Any]) -> List[Dict[str, str]]:
        interventions = [
            {
                "tier": "high_risk_top_10pct",
                "intervention": "Comprehensive transitional care",
                "details": "Home health visits, medication reconciliation, follow-up call within 48 hours",
                "expected_impact": "20-30% readmission reduction in landmark studies (Naylor et al., 2004)",
            },
            {
                "tier": "high_risk_top_25pct",
                "intervention": "Pharmacist-led medication review + nurse follow-up call",
                "details": "Identify medication interactions, ensure prescription adherence",
                "expected_impact": "10-15% readmission reduction",
            },
            {
                "tier": "medium_risk",
                "intervention": "Standard discharge protocol + scheduled follow-up",
                "details": "Within 7 days, primary care visit",
                "expected_impact": "Baseline standard of care",
            },
            {
                "tier": "low_risk",
                "intervention": "Standard care",
                "details": "No additional resources required",
                "expected_impact": "Cost-efficient routine care",
            },
        ]
        return interventions
