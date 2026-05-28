"""
Regression Runner - actually trains regression models.
Uses scikit-learn under the hood.
"""
from __future__ import annotations
from typing import Dict, Any, List
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline


class RegressionRunner:
    """Trains and evaluates regression models with full metrics."""

    def __init__(self, algorithm: str = "random_forest_regressor", hyperparameters: Dict = None):
        self.algorithm = algorithm
        self.hparams = hyperparameters or {}
        self.model = None
        self.feature_names = []
        self.preprocessing_info = {}

    def run(
        self,
        df: pd.DataFrame,
        target_column: str,
        feature_columns: List[str] = None,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Dict[str, Any]:
        """Train regression model and return full evaluation."""
        # Prepare features
        if feature_columns is None:
            feature_columns = [c for c in df.columns if c != target_column]

        X = df[feature_columns].copy()
        y = df[target_column].copy()

        # Drop rows with missing target
        valid = ~y.isna()
        X, y = X[valid], y[valid]

        # Handle non-numerical features (one-hot)
        X_encoded = pd.get_dummies(X, drop_first=True, dummy_na=False)

        # Fill remaining numeric NaN with median
        for col in X_encoded.columns:
            if X_encoded[col].dtype in ("float64", "int64"):
                X_encoded[col] = X_encoded[col].fillna(X_encoded[col].median())

        self.feature_names = list(X_encoded.columns)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y, test_size=test_size, random_state=random_state
        )

        # Build model
        if self.algorithm == "random_forest_regressor":
            model = RandomForestRegressor(**self.hparams)
        elif self.algorithm == "linear_regression":
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("linreg", LinearRegression(**self.hparams)),
            ])
        else:
            raise ValueError(f"Unknown regression algorithm: {self.algorithm}")

        model.fit(X_train, y_train)
        self.model = model

        # Evaluate
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        metrics = {
            "rmse_train": float(np.sqrt(mean_squared_error(y_train, y_pred_train))),
            "rmse_test": float(np.sqrt(mean_squared_error(y_test, y_pred_test))),
            "mae_train": float(mean_absolute_error(y_train, y_pred_train)),
            "mae_test": float(mean_absolute_error(y_test, y_pred_test)),
            "r2_train": float(r2_score(y_train, y_pred_train)),
            "r2_test": float(r2_score(y_test, y_pred_test)),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
        }

        # Feature importance (if available)
        feature_importance = None
        if self.algorithm == "random_forest_regressor":
            importances = model.feature_importances_
            feature_importance = sorted(
                [
                    {"feature": f, "importance": float(imp)}
                    for f, imp in zip(self.feature_names, importances)
                ],
                key=lambda x: x["importance"],
                reverse=True,
            )

        # Confidence (rough heuristic from R²)
        confidence = max(0.0, min(1.0, metrics["r2_test"]))

        return {
            "algorithm": self.algorithm,
            "task": "regression",
            "metrics": metrics,
            "confidence_score": round(confidence, 3),
            "feature_importance": feature_importance[:10] if feature_importance else None,
            "feature_count": len(self.feature_names),
            "training_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
        }
