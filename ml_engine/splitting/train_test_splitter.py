"""
Train/Test Splitter
===================
Smart splitter that picks the right strategy based on task type:
- Classification: stratified split (maintains class proportions)
- Regression: random split (binning is overkill for MVP)
- Time series: chronological split (no future data in train)
"""
from __future__ import annotations
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class TrainTestSplitter:
    """Build train/test split with proper strategy."""

    def split(
        self,
        df: pd.DataFrame,
        target_column: str,
        test_size: float = 0.2,
        random_state: int = 42,
        task_type: str = "auto",
        datetime_column: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Returns:
            {
                "X_train", "X_test", "y_train", "y_test": pandas objects,
                "strategy": str (random / stratified / chronological),
                "reasoning": str (why this strategy was chosen),
                "size_summary": dict
            }
        """
        if target_column not in df.columns:
            raise ValueError(f"Target '{target_column}' not in dataframe")

        # Decide strategy
        strategy, reasoning = self._decide_strategy(
            df, target_column, task_type, datetime_column
        )

        X = df.drop(columns=[target_column])
        y = df[target_column]

        if strategy == "chronological":
            # Sort by datetime first, then split last test_size as test
            df_sorted = df.sort_values(datetime_column).reset_index(drop=True)
            split_idx = int(len(df_sorted) * (1 - test_size))
            train_df = df_sorted.iloc[:split_idx]
            test_df = df_sorted.iloc[split_idx:]
            X_train = train_df.drop(columns=[target_column])
            X_test = test_df.drop(columns=[target_column])
            y_train = train_df[target_column]
            y_test = test_df[target_column]
        elif strategy == "stratified":
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=random_state,
                stratify=y,
            )
        else:
            # Random
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=random_state,
            )

        return {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "strategy": strategy,
            "reasoning": reasoning,
            "size_summary": {
                "n_train": int(len(X_train)),
                "n_test": int(len(X_test)),
                "test_ratio": round(len(X_test) / len(df), 3) if len(df) else 0,
            },
            "leak_safety_note": (
                "Train/test split is performed BEFORE any preprocessing. "
                "Preprocessor will be fit on X_train only and applied to X_test as transform."
            ),
        }

    def _decide_strategy(
        self,
        df: pd.DataFrame,
        target_column: str,
        task_type: str,
        datetime_column: Optional[str],
    ) -> Tuple[str, str]:
        # Time series check first
        if datetime_column and datetime_column in df.columns:
            return "chronological", (
                f"Datetime column '{datetime_column}' detected. Using chronological split "
                f"to prevent the model from seeing 'future' data in training. "
                f"Random split would cause time leakage."
            )

        # Classification check
        target = df[target_column]
        if task_type == "classification" or (
            task_type == "auto" and (
                pd.api.types.is_object_dtype(target)
                or pd.api.types.is_bool_dtype(target)
                or (pd.api.types.is_numeric_dtype(target) and target.nunique() <= 20)
            )
        ):
            # Verify each class has at least 2 samples for stratification
            counts = target.value_counts(dropna=False)
            if counts.min() >= 2:
                return "stratified", (
                    f"Classification task detected with {len(counts)} classes. "
                    f"Stratified split preserves class proportions in both train and test, "
                    f"essential for reliable evaluation when classes are imbalanced."
                )
            else:
                return "random", (
                    "Classification task but some classes have <2 samples. "
                    "Falling back to random split (stratification would fail)."
                )

        # Regression
        return "random", (
            "Regression or generic task. Random split is appropriate when target is "
            "continuous and there is no temporal ordering."
        )
