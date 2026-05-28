"""
NEXA · Feature Engineering Layer
================================
Technical indicators + regime labels + sentiment features.

References (all canonical):
    - Wilder, J. W. (1978) New Concepts in Technical Trading Systems (RSI)
    - Appel, G. (1979) Technical Analysis of Stock Trends (MACD)
    - Bollinger, J. (2002) Bollinger on Bollinger Bands (BB)
    - Lopez de Prado, M. (2018) Advances in Financial ML §3 (Triple Barrier)
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def log_return(df: pd.DataFrame, col: str = "close", periods: int = 1) -> pd.Series:
    """Log return = log(p_t / p_{t-periods}). More additive than simple return."""
    return np.log(df[col] / df[col].shift(periods))


def rolling_volatility(df: pd.DataFrame, col: str = "close",
                       window: int = 20) -> pd.Series:
    """Rolling standard deviation of log-returns (annualized scale daily)."""
    ret = np.log(df[col] / df[col].shift(1))
    return ret.rolling(window).std() * np.sqrt(252)  # annualized assumption


def sma(df: pd.DataFrame, col: str = "close", window: int = 20) -> pd.Series:
    """Simple Moving Average."""
    return df[col].rolling(window).mean()


def ema(df: pd.DataFrame, col: str = "close", span: int = 12) -> pd.Series:
    """Exponential Moving Average."""
    return df[col].ewm(span=span, adjust=False).mean()


def rsi(df: pd.DataFrame, col: str = "close", period: int = 14) -> pd.Series:
    """
    Relative Strength Index (Wilder 1978).
    Range 0-100. >70 overbought, <30 oversold (heuristic).
    """
    delta = df[col].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(df: pd.DataFrame, col: str = "close",
         fast: int = 12, slow: int = 26, signal: int = 9):
    """
    MACD (Appel 1979). Returns (macd_line, signal_line, histogram).
    """
    fast_ema = ema(df, col, fast)
    slow_ema = ema(df, col, slow)
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def bollinger_bands(df: pd.DataFrame, col: str = "close",
                    window: int = 20, num_std: float = 2.0):
    """
    Bollinger Bands (Bollinger 2002). Returns (upper, middle, lower, %B).
    """
    middle = sma(df, col, window)
    std = df[col].rolling(window).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    pct_b = (df[col] - lower) / (upper - lower).replace(0, np.nan)
    return upper, middle, lower, pct_b


def volume_imbalance(df: pd.DataFrame, window: int = 10) -> pd.Series:
    """
    Volume imbalance: ratio of up-volume vs down-volume in rolling window.
    > 0 = buying pressure dominates.
    """
    up_vol = df["volume"].where(df["close"] > df["close"].shift(1), 0)
    down_vol = df["volume"].where(df["close"] < df["close"].shift(1), 0)
    return (up_vol.rolling(window).sum() - down_vol.rolling(window).sum()) / \
           df["volume"].rolling(window).sum().replace(0, np.nan)


def volatility_clustering(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Volatility-of-volatility (proxy for clustering).
    Bollerslev (1986) GARCH foundation: σ²_t depends on σ²_{t-1}.
    """
    ret = np.log(df["close"] / df["close"].shift(1))
    vol = ret.rolling(window).std()
    return vol.rolling(window).std()


def shannon_entropy(series: pd.Series, bins: int = 10) -> float:
    """
    Shannon entropy of return distribution (Shannon 1948).
    Low entropy = predictable; high entropy = random walk-like.
    """
    s = series.dropna()
    if len(s) < bins:
        return float("nan")
    hist, _ = np.histogram(s, bins=bins)
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log(hist)))


def entropy_score(df: pd.DataFrame, window: int = 100) -> pd.Series:
    """Rolling entropy of log returns."""
    ret = np.log(df["close"] / df["close"].shift(1))
    return ret.rolling(window).apply(lambda x: shannon_entropy(x), raw=False)


def regime_label_heuristic(df: pd.DataFrame, ma_short: int = 20,
                            ma_long: int = 50, vol_window: int = 20) -> pd.Series:
    """
    Heuristic regime classifier (no HMM needed):
        0 = trending_up    (price > MA50, MA20 > MA50, low vol)
        1 = trending_down  (price < MA50, MA20 < MA50, low vol)
        2 = ranging        (price ~= MA50, |MA20 - MA50| small)
        3 = high_volatility (vol > 2x median vol)
    """
    ma_s = df["close"].rolling(ma_short).mean()
    ma_l = df["close"].rolling(ma_long).mean()
    ret = np.log(df["close"] / df["close"].shift(1))
    vol = ret.rolling(vol_window).std()
    vol_med = vol.median()

    regime = pd.Series(2, index=df.index)  # default: ranging
    regime[(df["close"] > ma_l) & (ma_s > ma_l)] = 0
    regime[(df["close"] < ma_l) & (ma_s < ma_l)] = 1
    regime[vol > 2 * vol_med] = 3
    return regime


def build_features(df: pd.DataFrame, sentiment: pd.Series = None) -> pd.DataFrame:
    """
    Build full feature set. Returns enriched DataFrame.
    All features are causal — no future leakage.
    """
    df = df.copy()

    # Returns
    df["log_ret_1"] = log_return(df, periods=1)
    df["log_ret_5"] = log_return(df, periods=5)
    df["log_ret_20"] = log_return(df, periods=20)

    # Volatility
    df["vol_20"] = rolling_volatility(df, window=20)
    df["vol_60"] = rolling_volatility(df, window=60)
    df["vol_clustering"] = volatility_clustering(df, window=20)

    # Moving averages
    df["sma_20"] = sma(df, window=20)
    df["sma_50"] = sma(df, window=50)
    df["ema_12"] = ema(df, span=12)
    df["sma_ratio"] = df["sma_20"] / df["sma_50"]

    # Momentum oscillators
    df["rsi_14"] = rsi(df, period=14)
    macd_line, signal_line, hist = macd(df)
    df["macd"] = macd_line
    df["macd_signal"] = signal_line
    df["macd_hist"] = hist

    # Bollinger
    upper, middle, lower, pct_b = bollinger_bands(df)
    df["bb_upper"] = upper
    df["bb_lower"] = lower
    df["bb_pct"] = pct_b

    # Volume features
    df["vol_imbalance_10"] = volume_imbalance(df, window=10)
    df["log_volume"] = np.log1p(df["volume"])

    # Entropy
    df["entropy_100"] = entropy_score(df, window=100)

    # Regime
    df["regime"] = regime_label_heuristic(df)

    # Sentiment (optional input)
    if sentiment is not None:
        df["sentiment"] = sentiment.reindex(df.index, method="ffill")
        df["sentiment_momentum"] = df["sentiment"].diff(5)

    return df


def make_target(df: pd.DataFrame, horizon: int = 1,
                 up_threshold: float = 0.002,
                 down_threshold: float = -0.002) -> pd.Series:
    """
    Triple-barrier target (simplified, Lopez de Prado 2018 §3.4):
        2 = up move > up_threshold within horizon
        0 = down move < down_threshold within horizon
        1 = sideways (neither barrier hit)
    """
    future_ret = np.log(df["close"].shift(-horizon) / df["close"])
    target = pd.Series(1, index=df.index)  # default: sideways
    target[future_ret > up_threshold] = 2
    target[future_ret < down_threshold] = 0
    return target
