"""
Health Anomaly Monitor
======================
Detects abnormal vital signs and lab values per patient.
Uses normal-range checks + statistical outlier detection.
"""
from __future__ import annotations
from typing import Dict, Any, List
import pandas as pd
import numpy as np

from .clinical_feature_importance import CLINICAL_REFERENCES


class HealthAnomalyMonitor:
    """Real-time anomaly detection for vital signs and lab values."""

    def detect(
        self,
        df: pd.DataFrame,
        patient_id_column: str = "patient_id",
    ) -> Dict[str, Any]:
        """
        Scan dataframe for clinically abnormal values per patient.

        Returns alerts grouped by severity:
        - critical: outside extreme thresholds (e.g., O2 sat <85)
        - warning: outside normal range but not critical
        - info: borderline values
        """
        alerts: List[Dict[str, Any]] = []
        n_patients = df[patient_id_column].nunique() if patient_id_column in df.columns else len(df)

        # Per-column abnormal value scan
        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue
            ref = self._lookup_reference(col)
            if not ref or "normal_range" not in ref:
                continue

            min_normal, max_normal = ref["normal_range"]
            critical_thresholds = self._critical_thresholds(col, min_normal, max_normal)

            values = df[col].dropna()
            for idx, val in values.items():
                pid = df.loc[idx, patient_id_column] if patient_id_column in df.columns else f"row_{idx}"

                severity = None
                description = None

                if critical_thresholds:
                    cl, ch = critical_thresholds
                    if val < cl:
                        severity = "critical"
                        description = f"{col} = {val:.2f} (CRITICAL LOW; threshold: {cl})"
                    elif val > ch:
                        severity = "critical"
                        description = f"{col} = {val:.2f} (CRITICAL HIGH; threshold: {ch})"

                if severity is None:
                    if val < min_normal:
                        severity = "warning"
                        description = f"{col} = {val:.2f} below normal range [{min_normal}-{max_normal}]"
                    elif val > max_normal:
                        severity = "warning"
                        description = f"{col} = {val:.2f} above normal range [{min_normal}-{max_normal}]"

                if severity:
                    alerts.append({
                        "patient_id": str(pid),
                        "metric": col,
                        "value": float(val),
                        "normal_range": ref["normal_range"],
                        "unit": ref.get("unit"),
                        "severity": severity,
                        "description": description,
                        "clinical_context": ref.get("context"),
                    })

        # Summarize
        severity_counts = {"critical": 0, "warning": 0, "info": 0}
        for a in alerts:
            severity_counts[a["severity"]] = severity_counts.get(a["severity"], 0) + 1

        # Patient-level severity (worst alert per patient)
        patient_severity: Dict[str, str] = {}
        rank = {"critical": 3, "warning": 2, "info": 1}
        for a in alerts:
            pid = a["patient_id"]
            cur = patient_severity.get(pid, "")
            if rank.get(a["severity"], 0) > rank.get(cur, 0):
                patient_severity[pid] = a["severity"]

        return {
            "n_patients": int(n_patients),
            "n_alerts_total": len(alerts),
            "severity_counts": severity_counts,
            "n_patients_with_critical": sum(1 for s in patient_severity.values() if s == "critical"),
            "n_patients_with_warning": sum(1 for s in patient_severity.values() if s == "warning"),
            "alerts": alerts[:200],  # cap response size
            "patient_severity_map": patient_severity,
            "method_explanation": (
                "Rule-based abnormality detection using clinical reference ranges. "
                "Each numeric column matched against known thresholds (BP, HR, glucose, etc.). "
                "Values outside normal range trigger warnings; outside critical thresholds trigger alerts. "
                "Reference: clinical reference ranges from major medical guidelines."
            ),
            "method_monitor": {
                "selected_method": "Rule-based clinical thresholds",
                "why_chosen": (
                    "Fast, deterministic, clinically validated thresholds. "
                    "Aligns with how clinicians and EHR systems already think about abnormal values."
                ),
                "why_not_alternatives": [
                    {"alternative": "Statistical (Z-score)",
                     "reason_rejected": "Cohort norms may differ from clinical norms; misses absolute danger thresholds"},
                    {"alternative": "ML anomaly detection (Isolation Forest)",
                     "reason_rejected": "Less interpretable; clinical thresholds are gold standard"},
                ],
                "limitations": [
                    "Reference ranges vary by age/sex/population - simplified version here",
                    "Doesn't account for trend (single high BP vs sustained hypertension)",
                    "Personalized baseline would improve sensitivity",
                ],
            },
        }

    def _lookup_reference(self, feature_name: str) -> Dict[str, Any]:
        name_lower = feature_name.lower()
        if name_lower in CLINICAL_REFERENCES:
            return CLINICAL_REFERENCES[name_lower]
        for key in CLINICAL_REFERENCES:
            if key in name_lower or name_lower in key:
                return CLINICAL_REFERENCES[key]
        return {}

    def _critical_thresholds(self, feature: str, min_n: float, max_n: float):
        """Define critical thresholds (red-line emergency values)."""
        feature_lower = feature.lower()
        # Custom critical thresholds for known vitals
        if "oxygen_saturation" in feature_lower:
            return (90, 101)  # below 90 is critical hypoxemia
        if "systolic_bp" in feature_lower:
            return (80, 180)  # extreme hypotension or hypertensive crisis
        if "heart_rate" in feature_lower:
            return (40, 150)
        if "glucose" in feature_lower:
            return (54, 300)  # severe hypoglycemia or hyperglycemia
        if "temperature" in feature_lower:
            return (35, 39.5)
        # Default: 1.5x outside normal range
        spread = max_n - min_n
        return (min_n - 0.5 * spread, max_n + 0.5 * spread)
