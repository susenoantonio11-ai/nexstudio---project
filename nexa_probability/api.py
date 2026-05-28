"""
NEXA · FastAPI Backend
======================
Endpoints:
    POST /api/ai/predict/market
    POST /api/ai/backtest
    GET  /api/ai/model/status
    GET  /api/ai/model/explain/{prediction_id}
    POST /api/ai/regime-detection
    POST /api/ai/risk-score

Run:
    uvicorn backend.nexa_probability.api:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations
import logging
import uuid
from typing import Optional, List, Dict, Any
from datetime import datetime

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from . import DISCLAIMER, __version__
from .preprocessing import run_full_pipeline, time_series_split
from .features import build_features, make_target
from .engine import (
    ProbabilityEngine, RiskEngine, ExplainabilityEngine, BacktestEngine,
    predict_pipeline
)
from .models import (
    list_available_models, RegimeDetectorHMM, monte_carlo_simulation,
    mc_probability_above
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s")
log = logging.getLogger("nexa.api")

app = FastAPI(
    title="NEXA Probability Engine",
    description="Probabilistic market intelligence for NXLYTICS. NOT financial advice.",
    version=__version__
)

# CORS — allow NXLYTICS frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # in production, restrict to actual NXLYTICS domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory prediction cache for /explain endpoint
PREDICTION_CACHE: Dict[str, dict] = {}


# ──────────────────────────────────────────────────────────────────────
# Pydantic schemas
# ──────────────────────────────────────────────────────────────────────
class OHLCVRow(BaseModel):
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float


class PredictRequest(BaseModel):
    ohlcv: List[OHLCVRow] = Field(..., description="Historical OHLCV bars (chronological)")
    current_price: Optional[float] = Field(None, description="Current spot price (defaults to last close)")
    account_balance: Optional[float] = Field(10000, description="USD account balance for position sizing")
    horizon: Optional[int] = Field(1, description="Forecast horizon in bars")
    up_threshold: Optional[float] = Field(0.002, description="Up move threshold for target labeling")
    down_threshold: Optional[float] = Field(-0.002, description="Down move threshold")
    sentiment_score: Optional[float] = Field(None, description="External sentiment score [-1, 1]")


class BacktestRequest(BaseModel):
    ohlcv: List[OHLCVRow]
    window: Optional[int] = 252
    step: Optional[int] = 21
    transaction_cost: Optional[float] = 0.001


class RegimeRequest(BaseModel):
    ohlcv: List[OHLCVRow]
    n_states: Optional[int] = 3


class RiskScoreRequest(BaseModel):
    probability_up: float
    confidence: float
    current_price: float
    volatility: float
    account_balance: Optional[float] = 10000


class MonteCarloRequest(BaseModel):
    initial_price: float
    mu: float  # expected return
    sigma: float  # volatility
    horizon: int
    n_paths: Optional[int] = 1000
    target_price: Optional[float] = None


# ──────────────────────────────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────────────────────────────
def ohlcv_to_df(ohlcv: List[OHLCVRow]) -> pd.DataFrame:
    df = pd.DataFrame([row.dict() for row in ohlcv])
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    return df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "service": "NEXA Probability Engine",
        "version": __version__,
        "disclaimer": DISCLAIMER,
        "endpoints": [
            "POST /api/ai/predict/market",
            "POST /api/ai/backtest",
            "GET  /api/ai/model/status",
            "GET  /api/ai/model/explain/{prediction_id}",
            "POST /api/ai/regime-detection",
            "POST /api/ai/risk-score",
            "POST /api/ai/monte-carlo"
        ]
    }


@app.get("/api/ai/model/status")
def model_status():
    """Diagnostic: which models are available in current environment."""
    return {
        "service": "NEXA Probability Engine",
        "version": __version__,
        "models_available": list_available_models(),
        "disclaimer": DISCLAIMER
    }


@app.post("/api/ai/predict/market")
def predict_market(req: PredictRequest):
    """
    Main prediction endpoint. Returns probabilities + risk + explainability.
    """
    df = ohlcv_to_df(req.ohlcv)
    if len(df) < 100:
        raise HTTPException(status_code=400, detail="Need at least 100 OHLCV rows")

    # Pipeline
    pipeline_out = run_full_pipeline(df)
    df_clean = pipeline_out["df"]

    # Sentiment series (optional)
    sentiment = None
    if req.sentiment_score is not None:
        sentiment = pd.Series([req.sentiment_score] * len(df_clean), index=df_clean.index)

    # Feature engineering
    features_df = build_features(df_clean, sentiment=sentiment)

    # Target
    y = make_target(features_df, horizon=req.horizon,
                     up_threshold=req.up_threshold,
                     down_threshold=req.down_threshold)

    # Strip rows with NaN in features
    feature_cols = [c for c in features_df.columns
                     if c not in ["timestamp"] and features_df[c].dtype != "object"]
    X = features_df[feature_cols].copy()

    # Drop training rows where target is NaN (last `horizon` rows)
    valid_mask = ~y.isna()
    X_valid = X[valid_mask]
    y_valid = y[valid_mask].astype(int)

    if len(X_valid) < 100:
        raise HTTPException(status_code=400, detail=f"After feature engineering: only {len(X_valid)} valid rows")

    # Add latest (no target) row back for inference
    X_for_pipeline = pd.concat([X_valid, X.iloc[[-1]]], ignore_index=True)
    y_for_pipeline = pd.concat([y_valid, pd.Series([1])], ignore_index=True)  # dummy target

    current_price = req.current_price or float(df_clean["close"].iloc[-1])

    try:
        result = predict_pipeline(
            features_df=X_for_pipeline,
            y_history=y_for_pipeline,
            current_price=current_price,
            account_balance=req.account_balance
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    # Cache for /explain endpoint
    PREDICTION_CACHE[result["prediction_id"]] = result
    return result


@app.get("/api/ai/model/explain/{prediction_id}")
def explain_prediction(prediction_id: str):
    """Retrieve cached prediction explainability."""
    if prediction_id not in PREDICTION_CACHE:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return PREDICTION_CACHE[prediction_id]


@app.post("/api/ai/backtest")
def backtest(req: BacktestRequest):
    """Walk-forward backtest on provided OHLCV."""
    df = ohlcv_to_df(req.ohlcv)
    if len(df) < req.window + req.step + 50:
        raise HTTPException(status_code=400,
                            detail=f"Need at least {req.window + req.step + 50} rows for backtest")
    pipeline_out = run_full_pipeline(df)
    df_clean = pipeline_out["df"]
    features_df = build_features(df_clean)
    y = make_target(features_df).fillna(1).astype(int)
    feature_cols = [c for c in features_df.columns if features_df[c].dtype != "object"]
    X = features_df[feature_cols].fillna(0)
    prices = df_clean["close"]

    result = BacktestEngine.walk_forward(
        X=X, y=y, prices=prices,
        window=req.window, step=req.step,
        transaction_cost=req.transaction_cost
    )
    return result


@app.post("/api/ai/regime-detection")
def regime_detection(req: RegimeRequest):
    """Detect market regime via HMM (or heuristic fallback)."""
    df = ohlcv_to_df(req.ohlcv)
    returns = np.log(df["close"] / df["close"].shift(1)).dropna().values

    detector = RegimeDetectorHMM(n_states=req.n_states)
    if detector.model is not None:
        detector.fit(returns)
        states = detector.predict_states(returns)
        return {
            "method": "Gaussian HMM (Hamilton 1989)",
            "n_states": req.n_states,
            "current_regime": int(states[-1]) if len(states) > 0 else None,
            "regime_history": states.tolist() if states is not None else [],
            "disclaimer": DISCLAIMER
        }
    else:
        # Heuristic fallback
        from .features import regime_label_heuristic, build_features
        feats = build_features(df)
        regime = feats["regime"].iloc[-1] if not feats["regime"].empty else None
        return {
            "method": "Heuristic (MA crossover + volatility)",
            "n_states": 4,
            "current_regime": int(regime) if regime is not None and not pd.isna(regime) else None,
            "regime_labels": ["trending_up", "trending_down", "ranging", "high_volatility"],
            "note": "hmmlearn not installed — using heuristic. Install hmmlearn for true HMM.",
            "disclaimer": DISCLAIMER
        }


@app.post("/api/ai/risk-score")
def risk_score(req: RiskScoreRequest):
    """Compute risk metrics for a hypothetical position."""
    re = RiskEngine()
    kelly = re.kelly_criterion(req.probability_up, 1.5)
    position = re.position_sizing(req.probability_up, req.confidence,
                                    req.volatility, req.account_balance)
    sl_tp = re.stop_loss_take_profit(req.current_price, req.volatility, req.probability_up)
    return {
        "kelly": kelly,
        "position_sizing": position,
        "stop_loss_take_profit": sl_tp,
        "disclaimer": DISCLAIMER
    }


@app.post("/api/ai/monte-carlo")
def run_monte_carlo(req: MonteCarloRequest):
    """Monte Carlo price simulation."""
    paths = monte_carlo_simulation(
        initial_price=req.initial_price,
        mu=req.mu, sigma=req.sigma,
        horizon=req.horizon, n_paths=req.n_paths
    )
    final = paths[:, -1]
    payload = {
        "initial_price": req.initial_price,
        "horizon": req.horizon,
        "n_paths": req.n_paths,
        "final_price_stats": {
            "mean": float(np.mean(final)),
            "median": float(np.median(final)),
            "std": float(np.std(final)),
            "p5": float(np.percentile(final, 5)),
            "p95": float(np.percentile(final, 95))
        },
        "method": "Geometric Brownian Motion (Hull 2018 §14)",
        "disclaimer": DISCLAIMER
    }
    if req.target_price is not None:
        payload["probability_above_target"] = mc_probability_above(paths, req.target_price)
    return payload
