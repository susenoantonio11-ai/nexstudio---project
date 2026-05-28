"""
Patient Risk Predictor
======================
Trains a calibrated classifier for patient risk (binary outcome:
high-risk vs low-risk, or readmission, or mortality).

Uses RandomForest with class_weight balanced + sigmoid calibration
(Platt scaling) so probability outputs are reliable for clinical decision making.
"""
from __future__ import annotations
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np


class PatientRiskPredictor:
    """Calibrated classification for clinical risk."""

    def __init__(self, algorithm: str = "random_forest"):
        self.algorithm = algorithm
        self.pipeline = None
        self.classes_ = []
        self.feature_names = []

    def fit_evaluate(
        self,
        df: pd.DataFrame,
        target_column: str,
        feature_columns: Optional[List[str]] = None,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Dict[str, Any]:
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.linear_model import LogisticRegression
            from sklearn.calibration import CalibratedClassifierCV
            from sklearn.preprocessing import StandardScaler
            from sklearn.impute import SimpleImputer
            from sklearn.compose import ColumnTransformer
            from sklearn.pipeline import Pipeline
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import (
                roc_auc_score, average_precision_score,
                f1_score, accuracy_score, precision_score, recall_score,
                confusion_matrix, classification_report,
                brier_score_loss,
            )
        except ImportError:
            return {"error": "scikit-learn not installed"}

        if feature_columns is None:
            feature_columns = [c for c in df.columns if c != target_column]

        X = df[feature_columns].copy()
        y = df[target_column].copy()
        valid = ~y.isna()
        X, y = X[valid], y[valid]

        # Split numeric vs categorical
        numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
        cat_cols = [c for c in X.columns if c not in numeric_cols]

        # Build leak-safe preprocessing
        from sklearn.preprocessing import OneHotEncoder
        transformers = []
        if numeric_cols:
            transformers.append(("num", Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("sc", StandardScaler()),
            ]), numeric_cols))
        if cat_cols:
            transformers.append(("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), cat_cols))
        if not transformers:
            return {"error": "No usable feature columns"}

        preprocessor = ColumnTransformer(transformers, remainder="drop")

        # Base classifier
        if self.algorithm == "logistic_regression":
            base = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state)
        else:
            base = RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1,
            )

        # Calibrate (Platt scaling) - critical for clinical probability output
        calibrated = CalibratedClassifierCV(base, method="sigmoid", cv=5)

        self.pipeline = Pipeline([
            ("preproc", preprocessor),
            ("clf", calibrated),
        ])

        # Stratified split
        from collections import Counter
        if Counter(y).most_common()[-1][1] >= 2:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y,
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state,
            )

        self.pipeline.fit(X_train, y_train)
        self.classes_ = sorted(y.unique())
        self.feature_names = numeric_cols + cat_cols

        y_pred = self.pipeline.predict(X_test)
        y_proba = self.pipeline.predict_proba(X_test)
        is_binary = len(self.classes_) == 2

        avg = "binary" if is_binary else "weighted"

        metrics = {
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "precision": round(float(precision_score(y_test, y_pred, average=avg, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, y_pred, average=avg, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, y_pred, average=avg, zero_division=0)), 4),
            "f1_macro": round(float(f1_score(y_test, y_pred, average="macro", zero_division=0)), 4),
        }
        if is_binary:
            try:
                metrics["roc_auc"] = round(float(roc_auc_score(y_test, y_proba[:, 1])), 4)
                metrics["pr_auc"] = round(float(average_precision_score(y_test, y_proba[:, 1])), 4)
                metrics["brier_score"] = round(float(brier_score_loss(y_test, y_proba[:, 1])), 4)
            except Exception:
                pass

            # Sensitivity / Specificity / NPV / PPV (clinical metrics)
            cm = confusion_matrix(y_test, y_pred)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                metrics["sensitivity"] = round(float(tp / (tp + fn)) if (tp + fn) else 0, 4)
                metrics["specificity"] = round(float(tn / (tn + fp)) if (tn + fp) else 0, 4)
                metrics["ppv"] = round(float(tp / (tp + fp)) if (tp + fp) else 0, 4)
                metrics["npv"] = round(float(tn / (tn + fn)) if (tn + fn) else 0, 4)

        cm_full = confusion_matrix(y_test, y_pred, labels=self.classes_).tolist()
        report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        # Calibration curve (binned)
        calibration = self._calibration_curve(y_test, y_proba, is_binary)

        return {
            "algorithm": self.algorithm + " + sigmoid calibration",
            "n_classes": len(self.classes_),
            "classes": [str(c) for c in self.classes_],
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "feature_count": len(self.feature_names),
            "metrics": metrics,
            "confusion_matrix": cm_full,
            "classification_report": report_dict,
            "calibration": calibration,
            "method_explanation": (
                f"Trained {self.algorithm} with class_weight='balanced' for imbalanced clinical data, "
                f"then calibrated probabilities using sigmoid (Platt scaling) with 5-fold CV. "
                f"Calibration is critical: a 70% predicted risk should translate to actual 70% rate "
                f"in similar patients, otherwise clinicians cannot trust thresholds."
            ),
            "method_monitor": {
                "selected_method": f"{self.algorithm} + Platt-scaled calibration",
                "why_chosen": (
                    "Random Forest handles missing values, mixed types, and non-linear medical relationships. "
                    "Sigmoid calibration ensures probability outputs are reliable for clinical decision thresholds. "
                    "Class-weight balanced compensates for typical imbalance (most patients are low-risk)."
                ),
                "why_not_alternatives": [
                    {"alternative": "Logistic Regression (uncalibrated)",
                     "reason_rejected": "Linear assumption may miss complex clinical interactions"},
                    {"alternative": "XGBoost",
                     "reason_rejected": "Better performance possible but less interpretable for clinical use"},
                    {"alternative": "Deep Neural Network",
                     "reason_rejected": "Hard to explain to clinicians; overfit risk on medical datasets"},
                    {"alternative": "Uncalibrated RF",
                     "reason_rejected": "Probability outputs may be biased; clinical thresholds need calibration"},
                ],
                "limitations": [
                    "Trained only on past data; concept drift requires retraining",
                    "Imputation may obscure clinically meaningful missingness",
                    "Performance varies by demographic; check fairness across subgroups",
                ],
                "clinical_guidance": (
                    "Use predicted probability with clinically validated threshold (often 0.5 or based on ROC). "
                    "Consider sensitivity vs specificity trade-off based on cost of false negatives "
                    "(missed high-risk patient) vs false positives (unnecessary intervention)."
                ),
            },
        }

    def _calibration_curve(self, y_true, y_proba, is_binary: bool) -> Dict[str, Any]:
        """Compute calibration bins for plotting."""
        if not is_binary:
            return {"available": False, "reason": "Only binary classification supported"}
        try:
            from sklearn.calibration import calibration_curve
            prob_pred, prob_true = calibration_curve(y_true, y_proba[:, 1], n_bins=10, strategy="quantile")
            return {
                "available": True,
                "predicted_proba_bin": [round(float(p), 4) for p in prob_pred],
                "actual_rate_bin": [round(float(p), 4) for p in prob_true],
                "interpretation": (
                    "Perfect calibration: predicted = actual along diagonal. "
                    "Above diagonal = under-confident; below = over-confident."
                ),
            }
        except Exception as e:
            return {"available": False, "error": str(e)}
