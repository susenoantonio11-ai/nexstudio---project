"""
Forecasting Runner - simple but real time-series forecasting.
Uses statsmodels ARIMA as baseline (no Prophet dependency required).

Reference: Hyndman & Athanasopoulos (2018) - Forecasting: Principles and Practice
"""
from __future__ import annotations
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np


class ForecastingRunner:
    """Forecasts future values using ARIMA."""

    def __init__(self, algorithm: str = "arima", hyperparameters: Dict = None):
        self.algorithm = algorithm
        self.hparams = hyperparameters or {"order": [1, 1, 1]}

    def run(
        self,
        df: pd.DataFrame,
        target_column: str,
        date_column: Optional[str] = None,
        forecast_periods: int = 30,
    ) -> Dict[str, Any]:
        # Prepare time series
        if date_column and date_column in df.columns:
            try:
                df = df.copy()
                df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
                df = df.dropna(subset=[date_column, target_column])
                df = df.sort_values(date_column)
                ts = df.set_index(date_column)[target_column]
            except Exception as e:
                return self._error(f"Failed to parse datetime: {e}")
        else:
            # No date column - assume sequential
            ts = df[target_column].dropna().reset_index(drop=True)

        if len(ts) < 30:
            return self._error("Need at least 30 observations for forecasting")

        try:
            from statsmodels.tsa.arima.model import ARIMA
            order = tuple(self.hparams.get("order", [1, 1, 1]))
            model = ARIMA(ts.values, order=order)
            fit = model.fit()

            forecast = fit.forecast(steps=forecast_periods)
            forecast_values = forecast.tolist() if hasattr(forecast, 'tolist') else list(forecast)

            # In-sample fit metrics
            fitted = fit.fittedvalues
            residuals = ts.values[len(ts.values)-len(fitted):] - fitted
            rmse = float(np.sqrt(np.mean(residuals**2)))
            mae = float(np.mean(np.abs(residuals)))

            return {
                "algorithm": self.algorithm,
                "task": "forecasting",
                "metrics": {
                    "rmse": rmse,
                    "mae": mae,
                    "n_observations": int(len(ts)),
                    "forecast_periods": int(forecast_periods),
                    "aic": float(fit.aic),
                    "bic": float(fit.bic),
                },
                "confidence_score": 0.7,
                "forecast": [
                    {"period": i + 1, "value": self._safe(v)}
                    for i, v in enumerate(forecast_values)
                ],
                "historical_last": self._safe(ts.values[-1]),
                "historical_mean": self._safe(np.mean(ts.values)),
            }
        except ImportError:
            return self._error("statsmodels not installed. Run: pip install statsmodels")
        except Exception as e:
            return self._error(f"Forecasting failed: {e}")

    def _safe(self, v):
        try:
            f = float(v)
            if np.isnan(f) or np.isinf(f):
                return None
            return f
        except (TypeError, ValueError):
            return None

    def _error(self, msg: str) -> Dict:
        return {
            "algorithm": self.algorithm,
            "task": "forecasting",
            "error": msg,
            "metrics": {},
            "confidence_score": 0.0,
        }
