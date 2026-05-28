"""
NEXA · Synthetic Sample Data Generator
======================================
Generate realistic OHLCV time-series for demo + testing.
Mimics BTC/USD daily bars with regime shifts.

Usage:
    from nexa_probability.sample_data import generate_btc_demo
    df = generate_btc_demo(n_bars=500)
    df.to_csv("btc_demo.csv", index=False)
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_btc_demo(n_bars: int = 500, seed: int = 42,
                       initial_price: float = 30000.0) -> pd.DataFrame:
    """
    Synthetic BTC/USD daily bars with realistic features:
        - GBM trend
        - 3 regime shifts (bull → bear → bull)
        - Volatility clustering
        - Volume correlated with price moves
    """
    rng = np.random.default_rng(seed)
    start_date = datetime.utcnow() - timedelta(days=n_bars)

    # Regime schedule: 3 regimes
    regime_boundaries = [n_bars // 3, 2 * n_bars // 3]
    mu_regimes = [0.0015, -0.0020, 0.0010]   # daily drift
    sigma_regimes = [0.025, 0.040, 0.030]    # daily vol

    closes = [initial_price]
    sigmas = []
    for i in range(1, n_bars):
        if i < regime_boundaries[0]:
            mu, sigma = mu_regimes[0], sigma_regimes[0]
        elif i < regime_boundaries[1]:
            mu, sigma = mu_regimes[1], sigma_regimes[1]
        else:
            mu, sigma = mu_regimes[2], sigma_regimes[2]
        # Add volatility clustering (vol-of-vol)
        sigma_t = sigma * (1 + 0.3 * rng.standard_normal())
        sigma_t = max(0.005, sigma_t)
        sigmas.append(sigma_t)
        # GBM step
        z = rng.standard_normal()
        next_close = closes[-1] * np.exp((mu - 0.5 * sigma_t**2) + sigma_t * z)
        closes.append(next_close)

    closes = np.array(closes)

    # OHLC from close + intraday noise
    intraday_range = 0.015  # 1.5% typical intraday
    highs = closes * (1 + rng.uniform(0.001, intraday_range, n_bars))
    lows = closes * (1 - rng.uniform(0.001, intraday_range, n_bars))
    opens = np.concatenate([[initial_price], closes[:-1]])
    # Ensure OHLC consistency
    highs = np.maximum.reduce([highs, opens, closes])
    lows = np.minimum.reduce([lows, opens, closes])

    # Volume correlated with abs return + base level
    log_ret = np.concatenate([[0], np.diff(np.log(closes))])
    volume = 1000 + 5000 * np.abs(log_ret) * 100 + rng.uniform(500, 2000, n_bars)

    timestamps = [start_date + timedelta(days=i) for i in range(n_bars)]

    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volume.round(2)
    })
    return df


def generate_polymarket_demo(n_bars: int = 100, seed: int = 42) -> pd.DataFrame:
    """
    Synthetic Polymarket-style prediction market data.
    Each row = market state over time as probability evolves.
    """
    rng = np.random.default_rng(seed)
    start = datetime.utcnow() - timedelta(hours=n_bars)
    # Probability random walk in [0.05, 0.95]
    probs = [0.50]
    for _ in range(n_bars - 1):
        step = rng.normal(0, 0.02)
        new_p = np.clip(probs[-1] + step, 0.05, 0.95)
        probs.append(new_p)
    probs = np.array(probs)
    volumes = rng.uniform(500, 5000, n_bars)
    timestamps = [start + timedelta(hours=i) for i in range(n_bars)]

    return pd.DataFrame({
        "timestamp": timestamps,
        "open": probs,
        "high": np.clip(probs + rng.uniform(0, 0.02, n_bars), 0.05, 0.95),
        "low": np.clip(probs - rng.uniform(0, 0.02, n_bars), 0.05, 0.95),
        "close": probs,
        "volume": volumes,
        "market_type": "polymarket_binary",
        "outcome": "Will event X happen by date Y"
    })


if __name__ == "__main__":
    df_btc = generate_btc_demo(500)
    df_btc.to_csv("nexa_btc_demo.csv", index=False)
    print(f"Generated BTC demo: {len(df_btc)} bars, saved to nexa_btc_demo.csv")
    print(df_btc.head())
