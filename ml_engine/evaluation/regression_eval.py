"""
Regression Evaluator
====================
Computes RMSE, MAE, R², MAPE for regression models.
"""
from __future__ import annotations
from typing import Dict, Any
import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
    explained_variance_score,
    median_absolute_error,
)


class RegressionEvaluator:
    """Compute comprehensive regression metrics."""

    def evaluate(self, y_true, y_pred) -> Dict[str, Any]:
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()

        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))
        ev = float(explained_variance_score(y_true, y_pred))
        med_ae = float(median_absolute_error(y_true, y_pred))

        # MAPE (handle zeros in y_true)
        try:
            mape = float(mean_absolute_percentage_error(y_true, y_pred))
        except Exception:
            mape = None

        # Residual analysis
        residuals = y_true - y_pred
        return {
            "n_samples": int(len(y_true)),
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "median_absolute_error": round(med_ae, 4),
            "r2": round(r2, 4),
            "explained_variance": round(ev, 4),
            "mape": round(mape, 4) if mape is not None else None,
            "residual_mean": round(float(residuals.mean()), 4),
            "residual_std": round(float(residuals.std()), 4),
            "residual_min": round(float(residuals.min()), 4),
            "residual_max": round(float(residuals.max()), 4),
            "interpretation": self._interpret(r2, rmse, mae, mape),
        }

    def _interpret(self, r2, rmse, mae, mape) -> str:
        if r2 >= 0.7:
            quality = "strong"
        elif r2 >= 0.4:
            quality = "moderate"
        elif r2 >= 0.1:
            quality = "weak"
        else:
            quality = "very weak / no signal"

        msg = f"Model shows {quality} predictive power (R² = {r2:.3f})."
        if mape is not None:
            msg += f" On average, predictions deviate by {mape*100:.1f}% from actual values."
        msg += f" RMSE = {rmse:.2f}, MAE = {mae:.2f}."
        return msg
