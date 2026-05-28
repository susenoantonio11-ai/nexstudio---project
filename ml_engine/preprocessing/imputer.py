"""
Imputation strategy builder.
Returns scikit-learn transformers that respect the leak-safe principle.
"""
from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def build_imputer_step(strategy: str, fill_value=None):
    """
    Returns a sklearn imputer based on strategy name.

    Strategies:
    - 'mean'                  : numerical, normal-ish distributions
    - 'median'                : numerical, skewed/has-outliers
    - 'most_frequent' / 'mode': categorical/numerical (fills with mode)
    - 'constant'              : numerical (default 0) or categorical (default 'missing')
    - 'knn_imputer'           : numerical with MAR pattern
    - 'iterative_imputer'     : numerical with complex MAR
    - 'median_with_indicator' : adds binary missing indicator + median impute
    """
    if strategy == "mean":
        return SimpleImputer(strategy="mean")
    if strategy == "median":
        return SimpleImputer(strategy="median")
    if strategy in ("most_frequent", "mode"):
        return SimpleImputer(strategy="most_frequent")
    if strategy == "constant_missing":
        return SimpleImputer(strategy="constant", fill_value=fill_value or "missing")
    if strategy == "constant":
        return SimpleImputer(strategy="constant", fill_value=fill_value if fill_value is not None else 0)

    if strategy == "knn_imputer":
        try:
            from sklearn.impute import KNNImputer
            return KNNImputer(n_neighbors=5, weights="uniform")
        except ImportError:
            return SimpleImputer(strategy="median")

    if strategy == "iterative_imputer":
        try:
            from sklearn.experimental import enable_iterative_imputer  # noqa
            from sklearn.impute import IterativeImputer
            return IterativeImputer(max_iter=10, random_state=42)
        except ImportError:
            return SimpleImputer(strategy="median")

    if strategy == "median_with_indicator":
        return SimpleImputer(strategy="median", add_indicator=True)

    if strategy == "constant_with_indicator":
        return SimpleImputer(
            strategy="constant",
            fill_value=fill_value or "missing",
            add_indicator=True,
        )

    # Default fallback
    return SimpleImputer(strategy="median")
