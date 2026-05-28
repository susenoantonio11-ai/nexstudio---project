"""
Evaluation Layer
================
- MetricSelector: picks correct metrics by task + balance
- ClassificationEvaluator: F1, ROC-AUC, PR-AUC, MCC, balanced acc, confusion matrix
- RegressionEvaluator: RMSE, MAE, R², MAPE, Huber loss
- ForecastingEvaluator: MAPE, RMSE, MAE, sMAPE
- AnomalyEvaluator: precision/recall/F1 on anomalies (when labels available)
- OverfittingDetector: train-test gap analysis with severity
"""
from .metric_selector import MetricSelector
from .classification_eval import ClassificationEvaluator
from .regression_eval import RegressionEvaluator
from .overfitting_detector import OverfittingDetector

__all__ = [
    "MetricSelector",
    "ClassificationEvaluator",
    "RegressionEvaluator",
    "OverfittingDetector",
]
