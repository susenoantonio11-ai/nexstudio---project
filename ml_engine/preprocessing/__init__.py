"""
Preprocessing Pipeline (Leak-Safe)
====================================
CRITICAL: All preprocessing here uses sklearn's ColumnTransformer + Pipeline,
which automatically prevents data leakage by:
- Fitting transformers on TRAINING data only
- Applying same fitted transformations to test/CV folds
- Allowing nested CV (StratifiedKFold inside hyperparameter tuning)

NEVER fit a scaler/imputer/encoder on the entire dataset before split.
The PipelineBuilder enforces this pattern.
"""
from .pipeline_builder import PreprocessingPipelineBuilder
from .imputer import build_imputer_step
from .scaler import build_scaler_step
from .encoder import build_encoder_step
from .feature_engineer import FeatureEngineer

__all__ = [
    "PreprocessingPipelineBuilder",
    "build_imputer_step",
    "build_scaler_step",
    "build_encoder_step",
    "FeatureEngineer",
]
