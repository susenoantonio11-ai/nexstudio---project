"""
Model Comparator
================
Trains multiple candidate models with the SAME preprocessing pipeline,
evaluates with cross-validation, and ranks them.

The preprocessing is INSIDE the sklearn Pipeline so each CV fold gets
its own fitted transformer (no leakage).

Returns:
- Per-model: mean_cv_score, std_cv_score, fit_time
- Recommendation: top model with reasoning
- Comparison table for Method Monitor
"""
from __future__ import annotations
from typing import Dict, Any, List
import time
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate

# Models
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    GradientBoostingRegressor,
    GradientBoostingClassifier,
)
from sklearn.svm import SVC


class ModelComparator:
    """Train + compare multiple models."""

    def get_default_candidates(
        self, task_type: str, is_imbalanced: bool = False
    ) -> List[Dict[str, Any]]:
        """Return list of candidate models for the task type."""
        if task_type == "regression":
            return [
                {
                    "name": "linear_regression",
                    "model": LinearRegression(),
                    "rationale": "Simple linear baseline. Highly interpretable. Tests if relationship is linear.",
                },
                {
                    "name": "ridge_regression",
                    "model": Ridge(alpha=1.0, random_state=42),
                    "rationale": "Linear with L2 regularization. More stable than plain linear when features are correlated.",
                },
                {
                    "name": "random_forest_regressor",
                    "model": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
                    "rationale": "Ensemble of trees. Captures non-linearities. Usually strong default for tabular data.",
                },
                {
                    "name": "gradient_boosting_regressor",
                    "model": GradientBoostingRegressor(n_estimators=100, random_state=42),
                    "rationale": "Sequential boosting. Often top performer on tabular data, slower than RF.",
                },
            ]

        # Classification
        class_weight = "balanced" if is_imbalanced else None
        return [
            {
                "name": "logistic_regression",
                "model": LogisticRegression(
                    max_iter=1000, random_state=42, class_weight=class_weight
                ),
                "rationale": (
                    "Linear classifier with probability output. Highly interpretable. "
                    + ("Uses class_weight='balanced' due to imbalance." if is_imbalanced else "")
                ),
            },
            {
                "name": "random_forest_classifier",
                "model": RandomForestClassifier(
                    n_estimators=100, random_state=42, n_jobs=-1, class_weight=class_weight
                ),
                "rationale": "Ensemble of trees. Robust to outliers and missing values, captures non-linearities.",
            },
            {
                "name": "gradient_boosting_classifier",
                "model": GradientBoostingClassifier(n_estimators=100, random_state=42),
                "rationale": "Sequential boosting. Often top performer; slower than RF.",
            },
        ]

    def compare(
        self,
        candidates: List[Dict[str, Any]],
        preprocessor,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        cv_splitter,
        scoring: str,
        task_type: str = "classification",
    ) -> Dict[str, Any]:
        """
        Compare candidates with CV.

        Args:
            candidates: list of {name, model, rationale}
            preprocessor: ColumnTransformer (will be wrapped in Pipeline with each model)
            X_train, y_train: training data only (test set is held out)
            cv_splitter: KFold/StratifiedKFold/etc
            scoring: sklearn scoring name (e.g., 'f1_macro', 'r2', 'neg_root_mean_squared_error')

        Returns:
            Comparison results with reasoning.
        """
        results = []

        for cand in candidates:
            pipe = Pipeline([
                ("preprocessor", preprocessor),
                ("model", cand["model"]),
            ])
            t0 = time.time()
            try:
                cv_results = cross_validate(
                    pipe, X_train, y_train,
                    cv=cv_splitter,
                    scoring=scoring,
                    n_jobs=-1,
                    return_train_score=True,
                )
                fit_time = time.time() - t0
                test_scores = cv_results["test_score"]
                train_scores = cv_results["train_score"]

                # Detect overfitting (large train-test gap)
                gap = float(train_scores.mean() - test_scores.mean())

                results.append({
                    "name": cand["name"],
                    "rationale": cand["rationale"],
                    "mean_test_score": round(float(test_scores.mean()), 4),
                    "std_test_score": round(float(test_scores.std()), 4),
                    "mean_train_score": round(float(train_scores.mean()), 4),
                    "train_test_gap": round(gap, 4),
                    "overfitting_risk": self._classify_overfitting(gap, scoring),
                    "fit_time_seconds": round(fit_time, 2),
                    "fold_scores": [round(s, 4) for s in test_scores.tolist()],
                    "status": "success",
                })
            except Exception as e:
                results.append({
                    "name": cand["name"],
                    "rationale": cand["rationale"],
                    "status": "failed",
                    "error": str(e),
                })

        # Rank by mean test score (descending - higher is better in sklearn convention)
        successful = [r for r in results if r["status"] == "success"]
        successful.sort(key=lambda r: r["mean_test_score"], reverse=True)

        if not successful:
            return {
                "results": results,
                "best_model": None,
                "reasoning": "All candidates failed during CV.",
            }

        best = successful[0]
        ranked = successful

        # Penalize models with high overfitting risk in selection
        # Adjusted score = mean_test - 0.5 * gap (favor generalization)
        for r in ranked:
            r["adjusted_score"] = round(
                r["mean_test_score"] - 0.5 * max(0, r["train_test_gap"]), 4
            )
        ranked_by_adjusted = sorted(
            ranked, key=lambda r: r["adjusted_score"], reverse=True
        )
        best_by_adjusted = ranked_by_adjusted[0]

        reasoning = self._build_reasoning(best_by_adjusted, ranked, scoring)

        return {
            "scoring_metric": scoring,
            "n_candidates": len(candidates),
            "n_successful": len(successful),
            "results": ranked_by_adjusted,
            "best_model": best_by_adjusted["name"],
            "best_model_score": best_by_adjusted["mean_test_score"],
            "best_model_std": best_by_adjusted["std_test_score"],
            "reasoning": reasoning,
            "alternatives_considered": [
                {
                    "model": r["name"],
                    "score": r["mean_test_score"],
                    "reason_rejected": (
                        f"Lower CV score ({r['mean_test_score']:.4f} vs "
                        f"{best_by_adjusted['mean_test_score']:.4f}) "
                        + (
                            "and higher overfitting risk." if r["overfitting_risk"] != "low"
                            else "."
                        )
                    ),
                }
                for r in ranked_by_adjusted[1:]
            ],
        }

    def _classify_overfitting(self, gap: float, scoring: str) -> str:
        """Train-test gap interpretation depends on scoring direction."""
        if "neg_" in scoring:
            # Negative scoring: gap meaning inverted, take absolute
            gap = abs(gap)
        if gap > 0.15:
            return "high"
        if gap > 0.07:
            return "moderate"
        return "low"

    def _build_reasoning(
        self, best: Dict[str, Any], ranked: List[Dict[str, Any]], scoring: str
    ) -> str:
        parts = []
        parts.append(
            f"Best model selected: '{best['name']}' with CV {scoring} = "
            f"{best['mean_test_score']:.4f} ± {best['std_test_score']:.4f} "
            f"({len(best.get('fold_scores', []))} folds)."
        )

        if best["overfitting_risk"] != "low":
            parts.append(
                f"NOTE: train-test gap of {best['train_test_gap']:.3f} indicates "
                f"{best['overfitting_risk']} overfitting risk. Consider regularization "
                f"or more training data."
            )

        parts.append(best["rationale"])

        if len(ranked) > 1:
            second = ranked[1]
            parts.append(
                f"Second best: '{second['name']}' with score {second['mean_test_score']:.4f}. "
                f"Selected model wins by {(best['mean_test_score'] - second['mean_test_score']):.4f}."
            )

        return " ".join(parts)
