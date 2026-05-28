"""
HEALTH DATA SCIENCE DOMAIN
==========================
Clinical / health analytics modules.

Components:
- PatientRiskPredictor: classification model for risk score
- SurvivalAnalyzer: Kaplan-Meier survival curves
- ReadmissionPredictor: hospital readmission risk
- ClinicalFeatureImportance: feature importance with clinical context
- HealthAnomalyMonitor: vital sign anomalies / abnormal lab values
- RiskStratifier: low/medium/high/critical patient stratification

CRISP-DM compliance:
- Data Understanding: medical feature stats, missing labs
- Preparation: medical-specific imputation, normal-range awareness
- Modeling: medical-appropriate algorithms (CoxPH, RF, calibrated probabilities)
- Evaluation: ROC-AUC, calibration, sensitivity/specificity, NPV/PPV
"""
from .patient_risk import PatientRiskPredictor
from .survival_analysis import SurvivalAnalyzer
from .readmission_predictor import ReadmissionPredictor
from .clinical_feature_importance import ClinicalFeatureImportance
from .health_anomaly_monitor import HealthAnomalyMonitor
from .risk_stratifier import RiskStratifier

__all__ = [
    "PatientRiskPredictor",
    "SurvivalAnalyzer",
    "ReadmissionPredictor",
    "ClinicalFeatureImportance",
    "HealthAnomalyMonitor",
    "RiskStratifier",
]
