"""
FloodClassifier - supervised flood classification.
====================================================
Pakai feature matrix (pixel × bands+indices) + label untuk training model.
Mendukung Random Forest, XGBoost, SVM, Logistic Regression.
"""
from __future__ import annotations
from typing import Dict, Any, List, Optional
import numpy as np


class FloodClassifier:
    """Supervised classifier untuk pixel-based flood classification."""

    SUPPORTED_ALGORITHMS = [
        "random_forest", "logistic_regression", "svm", "xgboost", "gradient_boosting",
    ]

    def __init__(self, algorithm: str = "random_forest"):
        self.algorithm = algorithm
        self.model = None
        self.feature_names: List[str] = []
        self.classes_: List[str] = []
        self.is_imbalanced = False

    def fit(
        self,
        X,
        y,
        feature_names: Optional[List[str]] = None,
        balance_classes: bool = True,
    ) -> Dict[str, Any]:
        """
        Train classifier.

        Args:
            X: feature matrix shape (n_pixels, n_features)
            y: labels shape (n_pixels,) — biner (0=non_flood, 1=flood) atau kategori
            feature_names: nama kolom features (untuk feature importance)
            balance_classes: jika True, gunakan class_weight='balanced'
        """
        try:
            import numpy as np
            X = np.asarray(X, dtype=float)
            y = np.asarray(y)

            # Handle imbalance
            from collections import Counter
            class_counts = Counter(y.tolist())
            counts = sorted(class_counts.values())
            self.is_imbalanced = len(counts) > 1 and counts[-1] / max(1, counts[0]) > 3

            class_weight = "balanced" if (balance_classes and self.is_imbalanced) else None

            self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]
            self.classes_ = sorted(set(y.tolist()))

            if self.algorithm == "random_forest":
                from sklearn.ensemble import RandomForestClassifier
                self.model = RandomForestClassifier(
                    n_estimators=100, max_depth=None, n_jobs=-1,
                    random_state=42, class_weight=class_weight,
                )
            elif self.algorithm == "gradient_boosting":
                from sklearn.ensemble import GradientBoostingClassifier
                self.model = GradientBoostingClassifier(
                    n_estimators=100, learning_rate=0.1, random_state=42,
                )
            elif self.algorithm == "logistic_regression":
                from sklearn.linear_model import LogisticRegression
                self.model = LogisticRegression(
                    max_iter=1000, random_state=42, class_weight=class_weight,
                )
            elif self.algorithm == "svm":
                from sklearn.svm import SVC
                self.model = SVC(
                    kernel="rbf", probability=True, random_state=42,
                    class_weight=class_weight,
                )
            elif self.algorithm == "xgboost":
                try:
                    from xgboost import XGBClassifier
                    self.model = XGBClassifier(
                        n_estimators=100, max_depth=6, random_state=42,
                        use_label_encoder=False, eval_metric="logloss",
                    )
                except ImportError:
                    # Fallback ke gradient boosting
                    from sklearn.ensemble import GradientBoostingClassifier
                    self.model = GradientBoostingClassifier(
                        n_estimators=100, random_state=42,
                    )
                    self.algorithm = "gradient_boosting"
            else:
                return {"status": "error", "error": f"Unknown algorithm: {self.algorithm}"}

            self.model.fit(X, y)

            return {
                "status": "success",
                "algorithm": self.algorithm,
                "n_features": int(X.shape[1]),
                "n_samples": int(X.shape[0]),
                "classes": [str(c) for c in self.classes_],
                "is_imbalanced": self.is_imbalanced,
                "class_weight_used": class_weight,
                "method_monitor": {
                    "selected_method": self.algorithm,
                    "why_chosen": self._reason_for_choice(self.algorithm, X.shape[0], self.is_imbalanced),
                    "why_not_alternatives": self._alternatives_reasons(self.algorithm),
                },
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def predict(self, X) -> Dict[str, Any]:
        """Prediksi label per-pixel."""
        if self.model is None:
            return {"status": "error", "error": "Model belum di-fit"}
        try:
            import numpy as np
            X = np.asarray(X, dtype=float)
            preds = self.model.predict(X)
            try:
                probas = self.model.predict_proba(X)
            except Exception:
                probas = None
            return {
                "status": "success",
                "predictions": preds,
                "probabilities": probas,
                "n_predicted": int(len(preds)),
                "predicted_distribution": {
                    str(c): int(np.sum(preds == c)) for c in self.classes_
                },
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def predict_to_raster(
        self,
        X,
        raster_shape: tuple,
    ) -> Dict[str, Any]:
        """Prediksi + reshape ke 2D raster format."""
        result = self.predict(X)
        if result["status"] != "success":
            return result
        try:
            import numpy as np
            preds = result["predictions"]
            raster = preds.reshape(raster_shape)
            result["raster_2d"] = raster
            if result.get("probabilities") is not None:
                # Probability untuk class flood (asumsi class 1 = flood)
                if 1 in self.classes_:
                    flood_idx = list(self.classes_).index(1)
                    proba_flood = result["probabilities"][:, flood_idx]
                    result["probability_raster_2d"] = proba_flood.reshape(raster_shape)
            return result
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def feature_importance(self) -> Dict[str, Any]:
        """Get feature importance dari trained model."""
        if self.model is None:
            return {"available": False, "reason": "Model belum di-fit"}

        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
            sorted_idx = importances.argsort()[::-1]
            return {
                "available": True,
                "method": "Tree-based (mean decrease impurity)",
                "ranked_features": [
                    {
                        "feature": self.feature_names[i],
                        "importance": float(importances[i]),
                        "rank": int(rank + 1),
                    }
                    for rank, i in enumerate(sorted_idx)
                ],
            }
        if hasattr(self.model, "coef_"):
            coefs = abs(self.model.coef_[0]) if self.model.coef_.ndim > 1 else abs(self.model.coef_)
            sorted_idx = coefs.argsort()[::-1]
            return {
                "available": True,
                "method": "Linear coefficient magnitude",
                "ranked_features": [
                    {
                        "feature": self.feature_names[i],
                        "importance": float(coefs[i]),
                        "rank": int(rank + 1),
                    }
                    for rank, i in enumerate(sorted_idx)
                ],
            }

        return {"available": False, "reason": "Model tidak expose feature importance"}

    def _reason_for_choice(self, algo: str, n_samples: int, is_imbalanced: bool) -> str:
        if algo == "random_forest":
            return (
                f"Random Forest dipilih untuk pixel-based classification: handle non-linear, "
                f"robust terhadap outlier raster, mendukung feature importance. "
                f"{'Class_weight=balanced karena imbalance ratio.' if is_imbalanced else ''}"
            )
        if algo == "xgboost":
            return "XGBoost untuk akurasi maksimal pada tabular pixel data — sequential boosting."
        if algo == "logistic_regression":
            return "Logistic Regression sebagai BASELINE interpretable. Cocok untuk binary flood/non-flood."
        if algo == "svm":
            return "SVM dengan RBF kernel untuk decision boundary kompleks. Lambat untuk jutaan pixel."
        if algo == "gradient_boosting":
            return "Gradient Boosting alternatif XGBoost. Akurasi tinggi pada tabular features."
        return f"{algo} dipilih"

    def _alternatives_reasons(self, chosen: str) -> List[Dict[str, str]]:
        all_alts = []
        if chosen != "random_forest":
            all_alts.append({"alternative": "random_forest", "reason_rejected": "Mungkin kurang akurat dibanding gradient boosting"})
        if chosen != "logistic_regression":
            all_alts.append({"alternative": "logistic_regression", "reason_rejected": "Hanya menangkap pola linear"})
        if chosen != "svm":
            all_alts.append({"alternative": "svm", "reason_rejected": "Lambat untuk jutaan pixel"})
        if chosen != "xgboost":
            all_alts.append({"alternative": "xgboost", "reason_rejected": "Membutuhkan dependency tambahan"})
        return all_alts
