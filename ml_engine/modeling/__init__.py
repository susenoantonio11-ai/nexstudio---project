"""
Modeling Layer
==============
- BaselineModel: DummyClassifier/Regressor as floor (any model must beat this)
- ModelComparator: trains multiple candidates with CV, picks best
- HyperparameterTuner: GridSearch / RandomizedSearch with leak-safe pipeline
- ThresholdTuner: optimize classification threshold for business metric

Order of operations (enforced by AccuracyPipeline orchestrator):
1. Train baseline → get floor metric
2. Compare candidates with CV → pick top 1-3
3. Tune hyperparameters of best on training data only
4. Final evaluation on held-out test set
"""
from .baseline_model import BaselineModel
from .model_comparator import ModelComparator
from .tuner import HyperparameterTuner
from .threshold_tuner import ThresholdTuner

__all__ = [
    "BaselineModel",
    "ModelComparator",
    "HyperparameterTuner",
    "ThresholdTuner",
]
