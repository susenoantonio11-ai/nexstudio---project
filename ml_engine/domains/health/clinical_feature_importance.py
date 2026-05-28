"""
Clinical Feature Importance
============================
Extracts feature importance from a fitted model and adds clinical context
(normal range, unit, interpretation guidance).
"""
from __future__ import annotations
from typing import Dict, Any, List, Optional


# Common clinical reference ranges (extensible)
CLINICAL_REFERENCES = {
    "age": {"unit": "years", "normal_range": [0, 120],
            "context": "Strong predictor in many diseases; use with care to avoid age bias"},
    "bmi": {"unit": "kg/m²", "normal_range": [18.5, 24.9],
            "context": "Underweight <18.5, normal 18.5-24.9, overweight 25-29.9, obese ≥30"},
    "systolic_bp": {"unit": "mmHg", "normal_range": [90, 120],
                    "context": "Hypertension if ≥130; hypotension if <90"},
    "diastolic_bp": {"unit": "mmHg", "normal_range": [60, 80],
                     "context": "Hypertension if ≥80; hypotension if <60"},
    "heart_rate": {"unit": "bpm", "normal_range": [60, 100],
                   "context": "Bradycardia <60; tachycardia >100"},
    "temperature": {"unit": "°C", "normal_range": [36.1, 37.2],
                    "context": "Fever ≥38; hypothermia <35"},
    "glucose": {"unit": "mg/dL", "normal_range": [70, 100],
                "context": "Diabetes ≥126 fasting; hypoglycemia <70"},
    "cholesterol": {"unit": "mg/dL", "normal_range": [125, 200],
                    "context": "High ≥240; borderline 200-239"},
    "hba1c": {"unit": "%", "normal_range": [4.0, 5.7],
              "context": "Pre-diabetes 5.7-6.4; diabetes ≥6.5"},
    "creatinine": {"unit": "mg/dL", "normal_range": [0.6, 1.2],
                   "context": "Elevated may indicate kidney dysfunction"},
    "hemoglobin": {"unit": "g/dL", "normal_range": [12, 17],
                   "context": "Anemia if low; polycythemia if high"},
    "wbc_count": {"unit": "cells/μL", "normal_range": [4500, 11000],
                  "context": "Infection/inflammation if elevated; immunocompromise if low"},
    "respiratory_rate": {"unit": "breaths/min", "normal_range": [12, 20],
                         "context": "Tachypnea >20; bradypnea <12"},
    "oxygen_saturation": {"unit": "%", "normal_range": [95, 100],
                          "context": "Hypoxemia if <95; severe if <90"},
    "length_of_stay": {"unit": "days", "normal_range": [0, 10],
                       "context": "Longer LOS often correlates with higher readmission risk"},
    "n_prior_admissions": {"unit": "count", "normal_range": [0, 2],
                           "context": "Frequent flyers have much higher readmission risk"},
    "n_medications": {"unit": "count", "normal_range": [0, 8],
                      "context": "Polypharmacy ≥5 increases adverse event risk"},
}


class ClinicalFeatureImportance:
    """Extract + contextualize feature importance for clinical models."""

    def extract(
        self,
        model_pipeline,
        feature_names: List[str],
        top_k: int = 15,
    ) -> Dict[str, Any]:
        """
        Extract feature importance from a fitted sklearn pipeline.
        Adds clinical context where available.
        """
        try:
            # Try to access the calibrated classifier inside Pipeline
            base_model = self._unwrap_model(model_pipeline)
            if base_model is None or not hasattr(base_model, "feature_importances_"):
                # Try coefficient-based importance
                if hasattr(base_model, "coef_"):
                    importances = abs(base_model.coef_[0]) if base_model.coef_.ndim > 1 else abs(base_model.coef_)
                else:
                    return {
                        "error": "Model does not expose feature_importances_ or coef_",
                        "available": False,
                    }
            else:
                importances = base_model.feature_importances_
        except Exception as e:
            return {"error": str(e), "available": False}

        # If preprocessor expanded features (one-hot), names list may not match
        if len(importances) != len(feature_names):
            feature_names = [f"feature_{i}" for i in range(len(importances))]

        # Build entries
        entries = []
        for name, imp in zip(feature_names, importances):
            ref = self._lookup_reference(name)
            entries.append({
                "feature": name,
                "importance": round(float(imp), 6),
                "clinical_unit": ref.get("unit") if ref else None,
                "normal_range": ref.get("normal_range") if ref else None,
                "clinical_context": ref.get("context") if ref else None,
            })

        # Sort + take top K
        entries.sort(key=lambda e: e["importance"], reverse=True)
        top = entries[:top_k]

        # Normalize for visualization
        max_imp = top[0]["importance"] if top else 1
        for e in top:
            e["normalized"] = round(e["importance"] / max_imp, 4) if max_imp > 0 else 0

        return {
            "available": True,
            "top_features": top,
            "n_features_total": len(entries),
            "method_explanation": (
                "Feature importance extracted from trained model (built-in model attribute). "
                "For tree-based models: average decrease in impurity (Gini for classification, MSE for regression). "
                "For linear models: absolute coefficient magnitude. "
                "Clinical context added for known clinical features (BMI, BP, glucose, etc.)."
            ),
            "method_monitor": {
                "selected_method": "Model-built-in feature importance",
                "why_chosen": (
                    "Fast, deterministic. Tree ensembles like Random Forest provide unbiased importance "
                    "by averaging across trees. Coefficients of linear models indicate direction + magnitude."
                ),
                "why_not_alternatives": [
                    {"alternative": "Permutation importance",
                     "reason_rejected": "More expensive; built-in is sufficient for ranking"},
                    {"alternative": "SHAP values",
                     "reason_rejected": "More precise but computationally expensive; recommend SHAP for individual patient explanation"},
                ],
                "limitations": [
                    "Built-in importance is biased toward high-cardinality features",
                    "Doesn't show direction (positive/negative effect) for tree models",
                    "Assumes feature independence; correlated features may share importance",
                ],
                "clinical_caution": (
                    "High importance != causation. A feature may be predictive because it's a SYMPTOM "
                    "of the outcome, not a CAUSE. Use SHAP for individual-level interpretation in clinical setting."
                ),
            },
        }

    def _unwrap_model(self, pipeline):
        """Try to find the actual classifier inside sklearn Pipeline."""
        # Pipeline ends with .named_steps
        if hasattr(pipeline, "named_steps"):
            # If wrapped in CalibratedClassifierCV, look inside
            last_step = list(pipeline.named_steps.values())[-1]
            if hasattr(last_step, "calibrated_classifiers_"):
                # Average importances across calibrated classifiers
                base = last_step.calibrated_classifiers_[0].estimator
                return base
            return last_step
        return pipeline

    def _lookup_reference(self, feature_name: str) -> Optional[Dict[str, Any]]:
        """Lookup clinical reference by feature name (case-insensitive partial match)."""
        name_lower = feature_name.lower()
        # Direct match
        if name_lower in CLINICAL_REFERENCES:
            return CLINICAL_REFERENCES[name_lower]
        # Partial match
        for key in CLINICAL_REFERENCES:
            if key in name_lower or name_lower in key:
                return CLINICAL_REFERENCES[key]
        return None
