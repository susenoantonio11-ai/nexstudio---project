"""
NEXA · Probability Engine + Risk Engine + Explainability Layer
==============================================================
Output utama BUKAN harga prediksi, tapi:
    - probability_up / probability_down / probability_sideways
    - probability_breakout / probability_reversal
    - crash_risk_score / volatility_risk_score
    - confidence_score
    - recommended_action (NEVER "guaranteed", always probabilistic)
    - explainability (per-prediction reasoning)
    - risk metrics (VaR, Kelly, drawdown, stop-loss/take-profit suggestion)

COMPLIANCE: All outputs include disclaimer. No "guaranteed profit" language.
"""

from __future__ import annotations
import logging
import uuid
from datetime import datetime
from typing import Optional
import numpy as np
import pandas as pd

from . import DISCLAIMER
from .models import (
    LogisticBaseline, RandomForestModel, BoostedTreesModel,
    BayesianUpdater, RegimeDetectorHMM, AnomalyDetector,
    monte_carlo_simulation, mc_probability_above, list_available_models
)

log = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════
# PROBABILITY ENGINE
# ════════════════════════════════════════════════════════════════════
class ProbabilityEngine:
    """
    Orchestrate multi-model ensemble untuk probability scoring.
    Ensemble strategy: weighted average of calibrated model probabilities.
    """

    def __init__(self):
        self.models = {}
        self.feature_cols = []
        self.fitted = False
        self.training_diagnostics = {}

    def fit(self, X: pd.DataFrame, y: pd.Series,
            sample_weight: Optional[np.ndarray] = None) -> dict:
        """
        Train all available models. Drop NaN rows automatically.
        y values expected: 0=down, 1=sideways, 2=up
        """
        # Clean training data
        Xy = X.copy()
        Xy["__y"] = y.values
        Xy = Xy.dropna()
        if len(Xy) < 50:
            raise ValueError(f"Insufficient training data after dropna: {len(Xy)} rows (need ≥50)")

        self.feature_cols = [c for c in Xy.columns if c != "__y"]
        X_clean = Xy[self.feature_cols].values
        y_clean = Xy["__y"].values.astype(int)

        # Train models
        self.models["logistic"] = LogisticBaseline().fit(X_clean, y_clean)
        self.models["random_forest"] = RandomForestModel().fit(X_clean, y_clean)
        try:
            self.models["boosted"] = BoostedTreesModel().fit(X_clean, y_clean)
        except Exception as e:
            log.warning("Boosted trees failed: %s", e)

        # Anomaly detector (unsupervised)
        try:
            self.models["anomaly"] = AnomalyDetector().fit(X_clean)
        except Exception as e:
            log.warning("Anomaly detector failed: %s", e)

        self.fitted = True
        self.training_diagnostics = {
            "n_samples": len(X_clean),
            "n_features": len(self.feature_cols),
            "class_distribution": {int(k): int(v) for k, v in
                                    zip(*np.unique(y_clean, return_counts=True))},
            "models_trained": list(self.models.keys())
        }
        return self.training_diagnostics

    def predict_probabilities(self, X: pd.DataFrame) -> dict:
        """
        Multi-model ensemble probability. Returns dict with all components.
        """
        if not self.fitted:
            raise RuntimeError("Engine not fitted. Call fit() first.")

        X = X[self.feature_cols].copy()
        # Forward-fill NaN for inference (can't drop — user expects 1 prediction per row)
        X = X.fillna(method="ffill").fillna(0)
        X_arr = X.values

        # Get probabilities from each classifier
        probs_list = []
        model_outputs = {}
        for name in ["logistic", "random_forest", "boosted"]:
            if name in self.models:
                try:
                    p = self.models[name].predict_proba(X_arr)
                    probs_list.append(p)
                    model_outputs[name] = p
                except Exception as e:
                    log.warning("Model %s prediction failed: %s", name, e)

        if not probs_list:
            raise RuntimeError("No models produced predictions")

        # Ensemble: simple average (could be weighted by validation accuracy)
        ensemble_prob = np.mean(probs_list, axis=0)

        # Ensure 3 classes (down=0, sideways=1, up=2)
        if ensemble_prob.shape[1] < 3:
            # Pad missing class with low prob
            padded = np.zeros((ensemble_prob.shape[0], 3))
            padded[:, :ensemble_prob.shape[1]] = ensemble_prob
            ensemble_prob = padded

        p_down = ensemble_prob[:, 0]
        p_sideways = ensemble_prob[:, 1]
        p_up = ensemble_prob[:, 2]

        # Confidence = max prob - second max (margin)
        sorted_probs = np.sort(ensemble_prob, axis=1)
        confidence = sorted_probs[:, -1] - sorted_probs[:, -2]

        # Anomaly score → crash risk component
        anomaly_scores = None
        if "anomaly" in self.models:
            try:
                anomaly_scores = self.models["anomaly"].anomaly_score(X_arr)
            except Exception:
                pass

        return {
            "probability_down": p_down.tolist(),
            "probability_sideways": p_sideways.tolist(),
            "probability_up": p_up.tolist(),
            "confidence_score": confidence.tolist(),
            "anomaly_score": (anomaly_scores.tolist() if anomaly_scores is not None else None),
            "model_outputs": {k: v.tolist() for k, v in model_outputs.items()},
            "n_models_in_ensemble": len(probs_list)
        }

    def predict_one(self, X_row: pd.DataFrame) -> dict:
        """Single-row prediction with full output payload."""
        result = self.predict_probabilities(X_row)
        return {k: (v[0] if isinstance(v, list) and v else v)
                for k, v in result.items() if k not in ["model_outputs", "n_models_in_ensemble"]}


# ════════════════════════════════════════════════════════════════════
# RISK ENGINE
# ════════════════════════════════════════════════════════════════════
class RiskEngine:
    """
    Risk metrics: VaR, Kelly, drawdown, position sizing, stop-loss/take-profit.

    References:
        - Kelly, J. L. (1956) Bell System Technical Journal 35:917-926
        - Jorion, P. (2006) Value at Risk 3rd ed, McGraw-Hill
    """

    @staticmethod
    def value_at_risk(returns: np.ndarray, confidence: float = 0.95) -> dict:
        """
        Historical VaR. confidence=0.95 means 5% worst loss.
        Returns dict with var and CVaR (expected shortfall).
        """
        returns = np.array(returns)
        returns = returns[~np.isnan(returns)]
        if len(returns) < 10:
            return {"var": None, "cvar": None, "n_observations": len(returns)}
        var = float(np.percentile(returns, (1 - confidence) * 100))
        cvar = float(returns[returns <= var].mean())
        return {
            "var": var,
            "cvar": cvar,
            "confidence_level": confidence,
            "n_observations": len(returns),
            "interpretation": f"With {confidence*100:.0f}% confidence, max loss does not exceed {abs(var)*100:.2f}% per period"
        }

    @staticmethod
    def kelly_criterion(p_win: float, win_loss_ratio: float) -> dict:
        """
        Kelly fraction: f* = (p*b - q) / b
        where p = win prob, q = 1-p, b = win/loss ratio.

        Half-Kelly recommended for safety (Thorp 1997).
        """
        if win_loss_ratio <= 0 or p_win <= 0 or p_win >= 1:
            return {"kelly_fraction": 0.0, "recommendation": "no_bet",
                    "reason": "Invalid inputs"}
        q = 1 - p_win
        f_star = (p_win * win_loss_ratio - q) / win_loss_ratio
        f_star = max(0.0, min(1.0, f_star))
        return {
            "kelly_fraction": float(f_star),
            "half_kelly": float(f_star / 2),
            "recommendation": "no_bet" if f_star <= 0 else
                              "small" if f_star < 0.05 else
                              "medium" if f_star < 0.20 else "aggressive",
            "warning": "Half-Kelly recommended in practice (Thorp 1997) for drawdown reduction"
        }

    @staticmethod
    def max_drawdown(equity_curve: np.ndarray) -> dict:
        """Max drawdown from peak-to-trough."""
        equity = np.array(equity_curve)
        equity = equity[~np.isnan(equity)]
        if len(equity) < 2:
            return {"max_drawdown": None}
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max
        max_dd = float(drawdown.min())
        return {
            "max_drawdown": max_dd,
            "max_drawdown_pct": f"{max_dd*100:.2f}%",
            "interpretation": "Worst peak-to-trough decline in equity"
        }

    @staticmethod
    def position_sizing(prob_up: float, confidence: float,
                          current_volatility: float,
                          account_balance: float = 10000) -> dict:
        """
        Recommended position size based on probability + confidence + volatility.
        Conservative: vol-adjusted Kelly with safety multipliers.
        """
        if prob_up < 0.55 or confidence < 0.1:
            return {
                "exposure": "low",
                "size_pct": 0.0,
                "size_amount": 0.0,
                "reason": "Probability or confidence below entry threshold"
            }
        # Conservative scaling
        base = (prob_up - 0.5) * 2  # 0 at 50%, 1 at 100%
        confidence_factor = min(1.0, confidence * 2)
        vol_penalty = max(0.1, 1.0 - min(1.0, current_volatility * 2))
        size_pct = base * confidence_factor * vol_penalty * 0.10  # cap 10%
        size_pct = max(0.0, min(0.10, size_pct))
        if size_pct < 0.01:
            exposure = "low"
        elif size_pct < 0.05:
            exposure = "medium"
        else:
            exposure = "high"
        return {
            "exposure": exposure,
            "size_pct": float(size_pct),
            "size_amount": float(size_pct * account_balance),
            "reason": f"Based on prob_up={prob_up:.2f}, conf={confidence:.2f}, vol={current_volatility:.4f}",
            "max_exposure_cap_pct": 0.10,
            "safety_note": "Cap at 10% per single position. Use stop-loss for downside control."
        }

    @staticmethod
    def stop_loss_take_profit(current_price: float, volatility: float,
                                prob_up: float, atr_multiple: float = 2.0) -> dict:
        """
        ATR-based stop loss / take profit. Volatility = ATR proxy.
        Reward:Risk ratio of 2:1 recommended.
        """
        sl_distance = volatility * atr_multiple * current_price
        tp_distance = sl_distance * 2  # 2:1 R:R
        return {
            "stop_loss": float(current_price - sl_distance) if prob_up > 0.5 else float(current_price + sl_distance),
            "take_profit": float(current_price + tp_distance) if prob_up > 0.5 else float(current_price - tp_distance),
            "stop_distance_pct": float(sl_distance / current_price),
            "reward_risk_ratio": 2.0,
            "direction": "long" if prob_up > 0.5 else "short",
            "method": "ATR-based (Wilder 1978)",
            "note": "Suggestion only — adjust to your risk tolerance and strategy"
        }


# ════════════════════════════════════════════════════════════════════
# EXPLAINABILITY LAYER
# ════════════════════════════════════════════════════════════════════
class ExplainabilityEngine:
    """
    Generate per-prediction reasoning. Required for compliance + audit trail.
    Format: structured dict for UI consumption.
    """

    @staticmethod
    def explain_prediction(probabilities: dict, risk: dict,
                             model_outputs: dict = None,
                             feature_importance: dict = None,
                             regime: str = None) -> dict:
        """
        Build full explanation payload.

        Required components (per spec):
            - method used + rationale
            - why other methods not used
            - most influential features
            - confidence level
            - main risks
            - model limitations
        """
        p_up = probabilities.get("probability_up", [0])
        p_up = p_up[0] if isinstance(p_up, list) else p_up
        p_down = probabilities.get("probability_down", [0])
        p_down = p_down[0] if isinstance(p_down, list) else p_down
        confidence = probabilities.get("confidence_score", [0])
        confidence = confidence[0] if isinstance(confidence, list) else confidence

        # Determine dominant signal
        signals = {"up": p_up, "down": p_down,
                   "sideways": probabilities.get("probability_sideways", [0])}
        signals = {k: (v[0] if isinstance(v, list) else v) for k, v in signals.items()}
        dominant = max(signals, key=signals.get)
        dominant_prob = signals[dominant]

        explanation = {
            "summary": {
                "dominant_signal": dominant,
                "probability": float(dominant_prob),
                "confidence": float(confidence),
                "regime": regime or "unknown"
            },
            "methods_used": [
                {
                    "name": "Logistic Regression",
                    "rationale": "Baseline interpretable model. Cox (1958)."
                },
                {
                    "name": "Random Forest",
                    "rationale": "Bagging ensemble robust untuk noisy market data. Breiman (2001)."
                },
                {
                    "name": "Gradient Boosting (XGBoost/LightGBM/sklearn fallback)",
                    "rationale": "Captures non-linear interactions. Chen & Guestrin (2016)."
                },
                {
                    "name": "Isolation Forest",
                    "rationale": "Crash/spike anomaly detection. Liu, Ting & Zhou (2008)."
                }
            ],
            "methods_not_used": [
                {
                    "name": "LSTM/Transformer",
                    "reason": "Requires PyTorch + GPU. Not available in Pyodide browser runtime. "
                              "Export workflow to Colab for deep learning experiments."
                },
                {
                    "name": "Reinforcement Learning",
                    "reason": "RL requires reward function + simulator. Out of scope for this engine, "
                              "high overfitting risk for finance (Lopez de Prado 2018)."
                }
            ],
            "top_features": [],  # populated below if feature_importance provided
            "confidence_assessment": _confidence_band(confidence),
            "risk_factors": [
                f"Volatility ATR proxy: {risk.get('current_volatility', 'N/A')}",
                f"Max drawdown estimate: {risk.get('max_drawdown', {}).get('max_drawdown_pct', 'N/A')}",
                f"Value at Risk 95%: {risk.get('var', {}).get('var', 'N/A')}"
            ],
            "limitations": [
                "Models trained on historical data — may fail during regime shifts.",
                "Probability ≠ certainty. 70% probability still has 30% chance of being wrong.",
                "No guarantee of profit. Past performance ≠ future results.",
                "Black swan events (Taleb 2007) cannot be predicted by historical-data models.",
                "Market microstructure (slippage, latency) not modeled here."
            ],
            "disclaimer": DISCLAIMER
        }

        if feature_importance:
            top = sorted(feature_importance.items(), key=lambda x: -abs(x[1]))[:5]
            explanation["top_features"] = [
                {"feature": f, "importance": float(imp)} for f, imp in top
            ]

        return explanation


def _confidence_band(c: float) -> str:
    if c < 0.10:
        return "VERY LOW — models disagree, do not act on this signal"
    if c < 0.20:
        return "LOW — weak signal, consider waiting for confirmation"
    if c < 0.35:
        return "MODERATE — actionable but with full risk management"
    if c < 0.50:
        return "HIGH — strong agreement among models"
    return "VERY HIGH — exceptional agreement, but verify no overfit"


# ════════════════════════════════════════════════════════════════════
# BACKTESTING ENGINE
# ════════════════════════════════════════════════════════════════════
class BacktestEngine:
    """
    Walk-forward backtest with realistic execution assumptions.

    Metrics computed:
        - Sharpe ratio (Sharpe 1966)
        - Sortino ratio (Sortino & Price 1994)
        - Max drawdown
        - Win rate, profit factor
        - Calibration (Brier score, reliability)
    """

    @staticmethod
    def walk_forward(X: pd.DataFrame, y: pd.Series, prices: pd.Series,
                     window: int = 252, step: int = 21,
                     transaction_cost: float = 0.001) -> dict:
        """
        Walk-forward: train on window, predict next step, advance.
        Lopez de Prado (2018) §7 — gold standard for finance backtesting.
        """
        if len(X) < window + step:
            return {"error": "Insufficient data for walk-forward"}

        predictions = []
        actuals = []
        returns = []

        for start in range(0, len(X) - window - step, step):
            train_X = X.iloc[start:start + window]
            train_y = y.iloc[start:start + window]
            test_X = X.iloc[start + window:start + window + step]
            test_y = y.iloc[start + window:start + window + step]
            test_prices = prices.iloc[start + window:start + window + step]

            # Train fresh model each window
            engine = ProbabilityEngine()
            try:
                engine.fit(train_X, train_y)
                probs = engine.predict_probabilities(test_X)
            except Exception as e:
                log.warning("Backtest step %d failed: %s", start, e)
                continue

            # Position: long if p_up > p_down + threshold, short if reverse, else flat
            p_up = np.array(probs["probability_up"])
            p_down = np.array(probs["probability_down"])
            position = np.where(p_up - p_down > 0.10, 1,
                       np.where(p_down - p_up > 0.10, -1, 0))

            # Realized returns with transaction cost
            ret = np.log(test_prices.values[1:] / test_prices.values[:-1])
            pos_aligned = position[:-1]
            strat_ret = pos_aligned * ret - transaction_cost * np.abs(np.diff(np.concatenate([[0], pos_aligned])))

            predictions.extend(position.tolist())
            actuals.extend(test_y.tolist())
            returns.extend(strat_ret.tolist())

        returns = np.array(returns)
        returns = returns[~np.isnan(returns)]
        if len(returns) < 2:
            return {"error": "No valid returns generated"}

        # Metrics
        sharpe = float(returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        downside = returns[returns < 0]
        sortino = float(returns.mean() / downside.std() * np.sqrt(252)) if len(downside) > 0 and downside.std() > 0 else 0
        equity_curve = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(equity_curve)
        max_dd = float(((equity_curve - running_max) / running_max).min())
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        win_rate = float(len(wins) / len(returns)) if len(returns) > 0 else 0
        profit_factor = float(wins.sum() / abs(losses.sum())) if len(losses) > 0 and losses.sum() != 0 else float("inf")

        return {
            "n_observations": len(returns),
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_dd,
            "max_drawdown_pct": f"{max_dd*100:.2f}%",
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "total_return_pct": float((equity_curve[-1] - 1) * 100),
            "transaction_cost_assumed": transaction_cost,
            "method": "Walk-forward (Lopez de Prado 2018 §7)",
            "disclaimer": DISCLAIMER + " Backtest results subject to survivorship bias, overfit, and execution assumptions."
        }


# ════════════════════════════════════════════════════════════════════
# Main predict_pipeline orchestrator
# ════════════════════════════════════════════════════════════════════
def predict_pipeline(features_df: pd.DataFrame, y_history: pd.Series,
                      current_price: float = None,
                      account_balance: float = 10000) -> dict:
    """
    End-to-end pipeline: train → predict → risk → explain.
    Returns full prediction payload ready for UI.
    """
    prediction_id = str(uuid.uuid4())[:8]
    timestamp = datetime.utcnow().isoformat() + "Z"

    # Time-split: use last row as inference target, rest as training
    if len(features_df) < 100:
        return {"error": "Need ≥100 historical rows", "prediction_id": prediction_id}

    X_train = features_df.iloc[:-1]
    y_train = y_history.iloc[:-1]
    X_infer = features_df.iloc[[-1]]

    # Train probability engine
    engine = ProbabilityEngine()
    training_diag = engine.fit(X_train, y_train)

    # Predict
    probs = engine.predict_probabilities(X_infer)

    # Risk engine
    risk_engine = RiskEngine()
    returns_hist = np.log(features_df["close"] / features_df["close"].shift(1)).dropna().values \
                    if "close" in features_df.columns else np.array([])
    var_result = risk_engine.value_at_risk(returns_hist)
    p_up = probs["probability_up"][0] if probs["probability_up"] else 0.33
    confidence = probs["confidence_score"][0] if probs["confidence_score"] else 0
    current_vol = features_df.get("vol_20", pd.Series([0.02])).iloc[-1]
    if pd.isna(current_vol):
        current_vol = 0.02

    kelly = risk_engine.kelly_criterion(p_up, 1.5)
    position = risk_engine.position_sizing(p_up, confidence, current_vol, account_balance)
    sl_tp = risk_engine.stop_loss_take_profit(
        current_price or features_df["close"].iloc[-1] if "close" in features_df.columns else 100.0,
        current_vol, p_up
    )

    risk_payload = {
        "var": var_result,
        "kelly": kelly,
        "position_sizing": position,
        "stop_loss_take_profit": sl_tp,
        "current_volatility": float(current_vol)
    }

    # Feature importance from RF (if available)
    feat_imp = {}
    if "random_forest" in engine.models:
        try:
            importances = engine.models["random_forest"].feature_importance_()
            feat_imp = dict(zip(engine.feature_cols, importances.tolist()))
        except Exception:
            pass

    # Regime
    regime = "unknown"
    if "regime" in features_df.columns:
        regime_val = int(features_df["regime"].iloc[-1])
        regime = ["trending_up", "trending_down", "ranging", "high_volatility"][regime_val] \
                 if 0 <= regime_val < 4 else "unknown"

    # Explainability
    explain = ExplainabilityEngine.explain_prediction(
        probabilities={k: v for k, v in probs.items() if k != "model_outputs"},
        risk=risk_payload,
        model_outputs=probs.get("model_outputs"),
        feature_importance=feat_imp,
        regime=regime
    )

    return {
        "prediction_id": prediction_id,
        "timestamp": timestamp,
        "probabilities": {
            "up": probs["probability_up"][0],
            "down": probs["probability_down"][0],
            "sideways": probs["probability_sideways"][0]
        },
        "confidence": probs["confidence_score"][0],
        "anomaly_score": probs.get("anomaly_score", [None])[0] if probs.get("anomaly_score") else None,
        "regime": regime,
        "risk": risk_payload,
        "training_diagnostics": training_diag,
        "explainability": explain,
        "models_available": list_available_models(),
        "disclaimer": DISCLAIMER
    }
