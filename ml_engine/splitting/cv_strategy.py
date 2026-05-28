"""
Cross-Validation Strategy Selector
==================================
Picks the right CV strategy:
- StratifiedKFold for classification (maintains class balance per fold)
- KFold for regression
- TimeSeriesSplit for time series (always trains on past, tests on future)
- RepeatedStratifiedKFold for small datasets (more reliable estimates)
"""
from __future__ import annotations
from typing import Dict, Any, Optional
import pandas as pd
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    RepeatedStratifiedKFold,
    TimeSeriesSplit,
)


class CVStrategy:
    """Select cross-validation splitter based on task and data."""

    def select(
        self,
        y: pd.Series,
        task_type: str = "auto",
        n_splits: int = 5,
        is_time_series: bool = False,
        random_state: int = 42,
    ) -> Dict[str, Any]:
        """
        Returns:
            {
                "splitter": sklearn CV splitter,
                "strategy": str,
                "n_splits": int,
                "reasoning": str
            }
        """
        n_samples = len(y)

        # Time series → TimeSeriesSplit
        if is_time_series:
            return {
                "splitter": TimeSeriesSplit(n_splits=n_splits),
                "strategy": "time_series_split",
                "n_splits": n_splits,
                "reasoning": (
                    f"TimeSeriesSplit with {n_splits} folds. Each fold uses earlier data "
                    f"as train and later data as validation, preventing the model from "
                    f"seeing future during training."
                ),
            }

        is_classification = task_type == "classification" or (
            task_type == "auto" and (
                pd.api.types.is_object_dtype(y)
                or pd.api.types.is_bool_dtype(y)
                or (pd.api.types.is_numeric_dtype(y) and y.nunique() <= 20)
            )
        )

        if is_classification:
            min_class_size = y.value_counts().min()
            if min_class_size < n_splits:
                # Reduce n_splits to match smallest class
                effective_splits = max(2, int(min_class_size))
                splitter = StratifiedKFold(
                    n_splits=effective_splits, shuffle=True, random_state=random_state
                )
                return {
                    "splitter": splitter,
                    "strategy": "stratified_kfold_reduced",
                    "n_splits": effective_splits,
                    "reasoning": (
                        f"Classification detected. Smallest class has only {min_class_size} samples. "
                        f"Reducing n_splits to {effective_splits} to ensure each fold has at least "
                        f"one sample per class."
                    ),
                }

            # For small datasets, use repeated CV for stability
            if n_samples < 200:
                splitter = RepeatedStratifiedKFold(
                    n_splits=n_splits, n_repeats=3, random_state=random_state
                )
                return {
                    "splitter": splitter,
                    "strategy": "repeated_stratified_kfold",
                    "n_splits": n_splits * 3,  # effective evaluations
                    "reasoning": (
                        f"Classification task with small dataset (n={n_samples}). "
                        f"RepeatedStratifiedKFold ({n_splits} splits × 3 repeats) provides more "
                        f"stable performance estimates by averaging across random splits."
                    ),
                }

            return {
                "splitter": StratifiedKFold(
                    n_splits=n_splits, shuffle=True, random_state=random_state
                ),
                "strategy": "stratified_kfold",
                "n_splits": n_splits,
                "reasoning": (
                    f"StratifiedKFold with {n_splits} folds. Maintains class balance in each fold, "
                    f"essential for reliable F1 and PR-AUC estimates."
                ),
            }

        # Regression
        return {
            "splitter": KFold(n_splits=n_splits, shuffle=True, random_state=random_state),
            "strategy": "kfold",
            "n_splits": n_splits,
            "reasoning": (
                f"Regression task. KFold with {n_splits} folds and shuffling provides "
                f"unbiased performance estimate."
            ),
        }
