"""
Classification Runner - trains classification models with sklearn.
"""
from __future__ import annotations
from typing import Dict, Any, List
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.pipeline import Pipeline


class ClassificationRunner:
    """Trains and evaluates classification models."""

    def __init__(self, algorithm: str = "random_forest_classifier", hyperparameters: Dict = None):
        self.algorithm = algorithm
        self.hparams = hyperparameters or {}
        self.model = None
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        self.classes_ = []

    def run(
        self,
        df: pd.DataFrame,
        target_column: str,
        feature_columns: List[str] = None,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Dict[str, Any]:
        if feature_columns is None:
            feature_columns = [c for c in df.columns if c != target_column]

        X = df[feature_columns].copy()
        y = df[target_column].copy()

        valid = ~y.isna()
        X, y = X[valid], y[valid]

        # Encode target
        y_encoded = self.label_encoder.fit_transform(y.astype(str))
        self.classes_ = list(self.label_encoder.classes_)

        # Encode features
        X_encoded = pd.get_dummies(X, drop_first=True, dummy_na=False)
        for col in X_encoded.columns:
            if X_encoded[col].dtype in ("float64", "int64"):
                X_encoded[col] = X_encoded[col].fillna(X_encoded[col].median())

        self.feature_names = list(X_encoded.columns)

        # Stratified split for class balance
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y_encoded, test_size=test_size,
            random_state=random_state, stratify=y_encoded
        )

        if self.algorithm == "random_forest_classifier":
            model = RandomForestClassifier(**self.hparams)
        elif self.algorithm == "logistic_regression":
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("logreg", LogisticRegression(**self.hparams)),
            ])
        else:
            raise ValueError(f"Unknown classification algorithm: {self.algorithm}")

        model.fit(X_train, y_train)
        self.model = model

        y_pred = model.predict(X_test)

        avg_strategy = "binary" if len(self.classes_) == 2 else "weighted"

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, average=avg_strategy, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, average=avg_strategy, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, average=avg_strategy, zero_division=0)),
            "n_classes": len(self.classes_),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
        }

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred).tolist()

        # Feature importance
        feature_importance = None
        if self.algorithm == "random_forest_classifier":
            importances = model.feature_importances_
            feature_importance = sorted(
                [
                    {"feature": f, "importance": float(imp)}
                    for f, imp in zip(self.feature_names, importances)
                ],
                key=lambda x: x["importance"],
                reverse=True,
            )

        confidence = metrics["f1"]

        return {
            "algorithm": self.algorithm,
            "task": "classification",
            "metrics": metrics,
            "confidence_score": round(confidence, 3),
            "confusion_matrix": cm,
            "classes": [str(c) for c in self.classes_],
            "feature_importance": feature_importance[:10] if feature_importance else None,
            "training_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
        }
