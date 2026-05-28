"""
Data Splitting Strategies (Leak-Safe)
======================================
- TrainTestSplitter: stratified for classification, time-aware for time series
- CVStrategy: KFold, StratifiedKFold, TimeSeriesSplit selector

ALL preprocessing must happen INSIDE the cross-validation loop to prevent
leakage. The orchestrator ensures this via sklearn Pipeline.
"""
from .train_test_splitter import TrainTestSplitter
from .cv_strategy import CVStrategy

__all__ = ["TrainTestSplitter", "CVStrategy"]
