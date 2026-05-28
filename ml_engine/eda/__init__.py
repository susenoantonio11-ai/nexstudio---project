"""
Exploratory Data Analysis (EDA) - CRISP-DM Step 2
==================================================
Components that run BEFORE modeling to understand the dataset
and identify potential issues that could harm model accuracy:

- DataQualityChecker: overall quality scoring (Wang & Strong, 1996)
- MissingAnalyzer: MCAR/MAR/MNAR pattern detection + imputation strategy
- OutlierDetector: IQR, Z-score, Modified Z-score, Isolation Forest
- ImbalanceDetector: class balance analysis for classification
- LeakageDetector: identifies columns that may cause data leakage
- CorrelationAnalyzer: feature-target correlation, multicollinearity

Each component returns structured output suitable for the Method Monitor.
"""
from .data_quality_checker import DataQualityChecker
from .missing_analyzer import MissingAnalyzer
from .outlier_detector import OutlierDetector
from .imbalance_detector import ImbalanceDetector
from .leakage_detector import LeakageDetector
from .correlation_analyzer import CorrelationAnalyzer

__all__ = [
    "DataQualityChecker",
    "MissingAnalyzer",
    "OutlierDetector",
    "ImbalanceDetector",
    "LeakageDetector",
    "CorrelationAnalyzer",
]
