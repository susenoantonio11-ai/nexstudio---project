"""
Hyperparameter Tuner
====================
Wraps GridSearchCV / RandomizedSearchCV with sklearn Pipeline so
preprocessing is fit per-fold (leak-safe).

Default param grids are conservative defaults that work on most data.
"""
from __future__ import annotations
from typing import Dict, Any, Optional
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


class HyperparameterTuner:
    """Tune a model's hyperparameters with leak-safe CV."""

    DEFAULT_GRIDS = {
        "random_forest_regressor": {
            "model__n_estimators": [50, 100, 200],
            "model__max_depth": [None, 10, 20],
            "model__min_samples_split": [2, 5],
        },
        "random_forest_classifier": {
            "model__n_estimators": [50, 100, 200],
            "model__max_depth": [None, 10, 20],
            "model__min_samples_split": [2, 5],
        },
        "gradient_boosting_regressor": {
            "model__n_estimators": [50, 100, 200],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [3, 5],
        },
        "gradient_boosting_classifier": {
            "model__n_estimators": [50, 100, 200],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [3, 5],
        },
        "logistic_regression": {
            "model__C": [0.01, 0.1, 1.0, 10.0],
            "model__solver": ["lbfgs", "liblinear"],
        },
        "ridge_regression": {
            "model__alpha": [0.1, 1.0, 10.0, 100.0],
        },
        "linear_regression": {},  # no real hyperparameters
    }

    def tune(
        self,
        pipeline: Pipeline,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        cv_splitter,
        scoring: str,
        model_name: str,
        param_grid: Optional[Dict[str, Any]] = None,
        search_strategy: str = "grid",
        n_iter: int = 20,
        random_state: int = 42,
    ) -> Dict[str, Any]:
        """
        Args:
            pipeline: Pipeline([("preprocessor", ...), ("model", model)])
            X_train, y_train: training data only
            cv_splitter: CV splitter
            scoring: metric to optimize
            model_name: lookup key for default grid
            param_grid: optional override
            search_strategy: 'grid' or 'random'
        """
        grid = param_grid if param_grid is not None else self.DEFAULT_GRIDS.get(model_name, {})

        if not grid:
            return {
                "tuning_performed": False,
                "reason": (
                    f"No hyperparameters to tune for '{model_name}' "
                    f"(or empty grid provided). Returning the unchanged pipeline."
                ),
                "best_pipeline": pipeline,
                "best_params": {},
                "best_score": None,
            }

        if search_strategy == "random":
            search = RandomizedSearchCV(
                pipeline,
                param_distributions=grid,
                n_iter=n_iter,
                cv=cv_splitter,
                scoring=scoring,
                n_jobs=-1,
                random_state=random_state,
                refit=True,
                return_train_score=True,
            )
        else:
            search = GridSearchCV(
                pipeline,
                param_grid=grid,
                cv=cv_splitter,
                scoring=scoring,
                n_jobs=-1,
                refit=True,
                return_train_score=True,
            )

        search.fit(X_train, y_train)

        return {
            "tuning_performed": True,
            "search_strategy": search_strategy,
            "n_combinations_evaluated": len(search.cv_results_["params"]),
            "best_pipeline": search.best_estimator_,
            "best_params": search.best_params_,
            "best_score": float(search.best_score_),
            "scoring": scoring,
            "reasoning": (
                f"Tuned {len(grid)} hyperparameter(s) using {search_strategy.upper()} search "
                f"with {len(search.cv_results_['params'])} configurations. "
                f"Best params: {search.best_params_}. "
                f"Best CV {scoring} = {search.best_score_:.4f}. "
                f"Tuning is leak-safe because preprocessing is inside the Pipeline and "
                f"refit per fold."
            ),
        }
