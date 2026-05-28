"""
GEOSPATIAL XGBOOST MODEL (with graceful degradation)
====================================================

Boosted decision trees untuk prediksi probabilistik bencana berbasis fitur
geospatial dan meteorologi.

Sitasi:
    Chen & Guestrin (2016). XGBoost: A Scalable Tree Boosting System. KDD 2016.
    Friedman (2001). Greedy function approximation: a gradient boosting machine.
        The Annals of Statistics 29(5).

Jika xgboost belum terinstal, fallback ke scikit-learn GradientBoostingClassifier.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

try:
    import xgboost as xgb  # type: ignore
    HAVE_XGB = True
except Exception:
    HAVE_XGB = False

try:
    from sklearn.ensemble import GradientBoostingClassifier  # type: ignore
    from sklearn.ensemble import GradientBoostingRegressor  # type: ignore
    HAVE_SKLEARN = True
except Exception:
    HAVE_SKLEARN = False


@dataclass
class XGBoostFitReport:
    backend: str
    n_train: int
    feature_importance: Dict[str, float]
    notes: str = ""


class GeospatialXGBoostModel:
    """
    XGBoost untuk klasifikasi event/non-event atau regresi intensitas.

    Args:
        task: 'classification' atau 'regression'
        params: dict parameter XGBoost (n_estimators, max_depth, lr, dst.)
    """

    def __init__(
        self,
        task: str = "classification",
        params: Optional[Dict[str, Any]] = None,
        feature_names: Optional[List[str]] = None,
    ) -> None:
        if task not in ("classification", "regression"):
            raise ValueError("task harus 'classification' atau 'regression'")
        self.task = task
        self.params = params or {
            "n_estimators": 200, "max_depth": 5, "learning_rate": 0.05,
            "subsample": 0.85, "colsample_bytree": 0.85,
        }
        self.feature_names = feature_names or []
        self._model = None
        self._backend = "uninitialized"

    def fit(
        self,
        X: Sequence[Sequence[float]],
        y: Sequence[float],
    ) -> XGBoostFitReport:
        if HAVE_XGB:
            return self._fit_xgb(X, y)
        if HAVE_SKLEARN:
            return self._fit_sklearn(X, y)
        return self._fit_fallback(X, y)

    def _fit_xgb(self, X, y) -> XGBoostFitReport:
        if self.task == "classification":
            self._model = xgb.XGBClassifier(
                eval_metric="logloss",
                use_label_encoder=False,
                **self.params,
            )
        else:
            self._model = xgb.XGBRegressor(**self.params)
        self._model.fit(X, y)
        importances = list(self._model.feature_importances_)
        names = self.feature_names or [f"f{i}" for i in range(len(importances))]
        fi = {n: float(v) for n, v in zip(names, importances)}
        self._backend = "xgboost"
        return XGBoostFitReport(
            backend="xgboost",
            n_train=len(y),
            feature_importance=fi,
            notes="Chen & Guestrin (2016) XGBoost.",
        )

    def _fit_sklearn(self, X, y) -> XGBoostFitReport:
        if self.task == "classification":
            self._model = GradientBoostingClassifier(
                n_estimators=int(self.params.get("n_estimators", 200)),
                max_depth=int(self.params.get("max_depth", 3)),
                learning_rate=float(self.params.get("learning_rate", 0.05)),
            )
        else:
            self._model = GradientBoostingRegressor(
                n_estimators=int(self.params.get("n_estimators", 200)),
                max_depth=int(self.params.get("max_depth", 3)),
                learning_rate=float(self.params.get("learning_rate", 0.05)),
            )
        self._model.fit(X, y)
        importances = list(self._model.feature_importances_)
        names = self.feature_names or [f"f{i}" for i in range(len(importances))]
        fi = {n: float(v) for n, v in zip(names, importances)}
        self._backend = "sklearn_gbm"
        return XGBoostFitReport(
            backend="sklearn_gradient_boosting",
            n_train=len(y),
            feature_importance=fi,
            notes="Friedman (2001) gradient boosting via scikit-learn.",
        )

    def _fit_fallback(self, X, y) -> XGBoostFitReport:
        # Linear logistic regression sederhana via gradient descent
        n = len(y)
        d = len(X[0]) if X else 0
        w = [0.0] * d
        b = 0.0
        lr = 0.05
        for _ in range(100):
            grad_w = [0.0] * d
            grad_b = 0.0
            for i in range(n):
                z = b + sum(w[j] * X[i][j] for j in range(d))
                p = 1.0 / (1.0 + 2.718281828 ** (-z))
                err = p - float(y[i])
                for j in range(d):
                    grad_w[j] += err * X[i][j]
                grad_b += err
            for j in range(d):
                w[j] -= lr * grad_w[j] / max(1, n)
            b -= lr * grad_b / max(1, n)
        self._model = {"w": w, "b": b}
        self._backend = "logistic_fallback"
        names = self.feature_names or [f"f{i}" for i in range(d)]
        max_abs = max((abs(x) for x in w), default=1.0) or 1.0
        fi = {n: float(abs(v)) / max_abs for n, v in zip(names, w)}
        return XGBoostFitReport(
            backend="logistic_fallback",
            n_train=n,
            feature_importance=fi,
            notes="xgboost & sklearn tidak tersedia. Fallback logistic regression.",
        )

    def predict_proba(self, X: Sequence[Sequence[float]]) -> List[float]:
        if self._model is None:
            raise RuntimeError("Model belum di-fit")
        if self._backend == "xgboost":
            preds = self._model.predict_proba(X)
            # asumsi binary classification, ambil kolom positif
            return [float(p[1]) for p in preds]
        if self._backend == "sklearn_gbm":
            if self.task == "classification":
                preds = self._model.predict_proba(X)
                return [float(p[1]) for p in preds]
            return [float(v) for v in self._model.predict(X)]
        if self._backend == "logistic_fallback":
            w = self._model["w"]
            b = self._model["b"]
            out = []
            for row in X:
                z = b + sum(w[j] * row[j] for j in range(len(w)))
                out.append(1.0 / (1.0 + 2.718281828 ** (-z)))
            return out
        return [0.0] * len(X)

    def predict(self, X: Sequence[Sequence[float]]) -> List[float]:
        proba = self.predict_proba(X)
        if self.task == "classification":
            return [1.0 if p >= 0.5 else 0.0 for p in proba]
        return proba
