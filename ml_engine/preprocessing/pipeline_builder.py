"""
Preprocessing Pipeline Builder
==============================
Builds an sklearn ColumnTransformer + Pipeline that applies the right
preprocessing per column type, while remaining LEAK-SAFE.

Key safety: this pipeline is FIT only on training data inside the modeling
layer. The same fitted instance is then APPLIED to test/CV folds, ensuring
no test data informs the transformation parameters.
"""
from __future__ import annotations
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from .imputer import build_imputer_step
from .scaler import build_scaler_step
from .encoder import build_encoder_step


class PreprocessingPipelineBuilder:
    """
    Build a leak-safe preprocessing ColumnTransformer.

    Returns:
        A scikit-learn Pipeline that can be fit/transformed inside CV.
    """

    def build(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        imputation_strategy: Dict[str, str] = None,
        scaler_strategy: str = "standard",
        encoder_strategy: str = "onehot",
        drop_columns: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Args:
            df: Input dataframe (only column types are inspected, NOT data values)
            target_column: Name of Y to exclude from preprocessing
            imputation_strategy: per-column override (e.g., {"col1": "knn_imputer"})
            scaler_strategy: 'standard' / 'robust' / 'minmax' / 'none'
            encoder_strategy: 'onehot' / 'ordinal'
            drop_columns: columns to exclude entirely

        Returns:
            {
                "pipeline": sklearn ColumnTransformer,
                "numeric_cols": [...],
                "categorical_cols": [...],
                "dropped_cols": [...],
                "config": {...}
            }
        """
        drop_columns = drop_columns or []
        if target_column:
            drop_columns = list(set(drop_columns + [target_column]))

        feature_cols = [c for c in df.columns if c not in drop_columns]
        numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
        categorical_cols = [
            c for c in feature_cols
            if (pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_categorical_dtype(df[c]))
            and not pd.api.types.is_datetime64_any_dtype(df[c])
        ]
        # Datetime cols should be feature-engineered before reaching here
        datetime_cols = [c for c in feature_cols if pd.api.types.is_datetime64_any_dtype(df[c])]
        if datetime_cols:
            # Auto-drop raw datetime
            drop_columns.extend(datetime_cols)
            for d in datetime_cols:
                if d in numeric_cols:
                    numeric_cols.remove(d)
                if d in categorical_cols:
                    categorical_cols.remove(d)

        imputation_strategy = imputation_strategy or {}

        # Numeric pipeline
        numeric_steps = []
        if numeric_cols:
            # Use 'median' as safe default; per-col override via imputation_strategy
            default_num_strategy = "median"
            numeric_steps = [
                ("imputer", build_imputer_step(default_num_strategy)),
            ]
            scaler = build_scaler_step(scaler_strategy)
            if scaler != "passthrough":
                numeric_steps.append(("scaler", scaler))
            numeric_pipeline = Pipeline(numeric_steps)
        else:
            numeric_pipeline = "drop"

        # Categorical pipeline
        if categorical_cols:
            categorical_pipeline = Pipeline([
                ("imputer", build_imputer_step("constant_missing")),
                ("encoder", build_encoder_step(encoder_strategy)),
            ])
        else:
            categorical_pipeline = "drop"

        # Build ColumnTransformer
        transformers = []
        if numeric_cols:
            transformers.append(("num", numeric_pipeline, numeric_cols))
        if categorical_cols:
            transformers.append(("cat", categorical_pipeline, categorical_cols))

        if not transformers:
            raise ValueError("No usable feature columns found after exclusions")

        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
            verbose_feature_names_out=False,
        )

        return {
            "pipeline": preprocessor,
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
            "dropped_cols": drop_columns,
            "config": {
                "scaler_strategy": scaler_strategy,
                "encoder_strategy": encoder_strategy,
                "default_numeric_imputer": "median",
                "default_categorical_imputer": "constant_missing",
            },
            "leak_safety_note": (
                "This preprocessor must be fit on TRAINING data only. "
                "Use it inside sklearn Pipeline along with the model so cross_val_score "
                "and GridSearchCV automatically enforce leak-safe boundaries."
            ),
        }
