"""
Baseline Model
==============
Establishes a floor performance using DummyClassifier/Regressor.
Any real model must significantly beat this to justify deployment.

Strategies:
- Classification: 'most_frequent' (predicts majority class)
- Regression: 'mean' or 'median'

If your tuned model can't beat the baseline by >5%, the model is not learning
useful patterns - data quality or feature engineering issue, not model issue.
"""
from __future__ import annotations
from typing import Dict, Any
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.model_selection import cross_val_score


class BaselineModel:
    """Train a dummy baseline and return CV score."""

    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        task_type: str,
        cv_splitter,
        scoring: str,
    ) -> Dict[str, Any]:
        if task_type == "classification":
            model = DummyClassifier(strategy="most_frequent", random_state=42)
            strategy_name = "most_frequent (predicts majority class)"
            description = (
                "DummyClassifier with strategy='most_frequent' always predicts the most "
                "common class. This represents the absolute minimum any classifier should "
                "achieve. If your sophisticated model only matches this, it means the "
                "features have no predictive signal."
            )
        else:
            model = DummyRegressor(strategy="mean")
            strategy_name = "mean (predicts training mean)"
            description = (
                "DummyRegressor with strategy='mean' always predicts the training set mean. "
                "Any regression model must achieve significantly lower RMSE/MAE than this "
                "to justify use."
            )

        try:
            scores = cross_val_score(model, X, y, cv=cv_splitter, scoring=scoring, n_jobs=-1)
            mean_score = float(scores.mean())
            std_score = float(scores.std())
        except Exception as e:
            return {
                "baseline_strategy": strategy_name,
                "error": str(e),
                "description": description,
            }

        return {
            "baseline_strategy": strategy_name,
            "task_type": task_type,
            "scoring_metric": scoring,
            "mean_cv_score": round(mean_score, 4),
            "std_cv_score": round(std_score, 4),
            "individual_fold_scores": [round(s, 4) for s in scores.tolist()],
            "description": description,
            "interpretation": self._interpret(scoring, mean_score),
        }

    def _interpret(self, scoring: str, score: float) -> str:
        if scoring in ("accuracy", "f1", "f1_macro", "f1_weighted", "roc_auc", "average_precision", "balanced_accuracy"):
            return (
                f"Baseline {scoring} = {score:.3f}. Your trained model should target "
                f">{score + 0.05:.3f} to demonstrate meaningful learning."
            )
        if scoring == "r2":
            return (
                f"Baseline R² = {score:.3f} (mean predictor has R² = 0 by definition; "
                f"any deviation reflects CV variance). Your model should achieve R² > 0.3 "
                f"for moderate, R² > 0.7 for strong performance."
            )
        if "neg_" in scoring:
            return (
                f"Baseline {scoring} = {score:.3f}. Your model should achieve "
                f"a HIGHER (less negative) score to demonstrate value."
            )
        return f"Baseline {scoring} = {score:.3f}."
