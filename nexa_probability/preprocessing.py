"""
NEXA · Data Processing Layer
============================
Pipeline preprocessing untuk financial time-series.
Aligned dengan CRISP-DM Phase 3 + ISO 8000-8 quality dimensions.

Critical for financial ML: prevent data leakage (Lopez de Prado 2018).
"""

from __future__ import annotations
import logging
from typing import Optional, Tuple
import pandas as pd
import numpy as np

log = logging.getLogger(__name__)


def validate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Verifikasi struktur OHLCV minimal. Raise ValueError kalau invalid."""
    required = {"open", "high", "low", "close", "volume"}
    cols_lower = {c.lower() for c in df.columns}
    missing = required - cols_lower
    if missing:
        raise ValueError(f"Missing OHLCV columns: {missing}")
    # Rename to lowercase for consistency
    df = df.rename(columns={c: c.lower() for c in df.columns})
    # Sanity: high >= low, prices positive, volume non-negative
    bad_hl = (df["high"] < df["low"]).sum()
    if bad_hl > 0:
        log.warning("Found %d rows where high < low — likely data error", bad_hl)
    return df


def handle_missing_ohlcv(df: pd.DataFrame, max_gap: int = 3) -> pd.DataFrame:
    """
    Handle missing rows di time-series OHLCV.
    Strategy: forward-fill close + recompute volume to 0 (no trade gap).
    LIMIT: forward-fill ≤ max_gap consecutive rows (gap besar = suspicious data).

    Reference: Lopez de Prado (2018) §4 — handling holidays vs missing.
    """
    df = df.copy()
    if df.empty:
        return df
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)

    # Track missing
    missing_before = df[["open", "high", "low", "close"]].isnull().sum().sum()
    if missing_before == 0:
        log.info("No missing OHLCV values — skipping imputation")
        return df

    # Forward-fill close-based (preserves no-trade semantics)
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].ffill(limit=max_gap)
    # Volume → 0 when missing (no trades)
    df["volume"] = df["volume"].fillna(0)

    missing_after = df[["open", "high", "low", "close"]].isnull().sum().sum()
    log.info("Imputed %d missing OHLCV values (forward-fill, max_gap=%d). Remaining: %d",
             missing_before - missing_after, max_gap, missing_after)
    return df


def detect_outliers_zscore(df: pd.DataFrame, column: str = "close",
                            threshold: float = 4.0) -> pd.Series:
    """
    Detect outlier candles via Z-score on log-returns (more robust for prices).
    threshold=4 means 4-sigma events (very rare ~0.006% in normal).
    """
    if column not in df.columns:
        return pd.Series(False, index=df.index)
    log_ret = np.log(df[column] / df[column].shift(1))
    z = (log_ret - log_ret.mean()) / log_ret.std()
    return z.abs() > threshold


def normalize_timestamps(df: pd.DataFrame, ts_col: str = "timestamp",
                         target_freq: Optional[str] = None) -> pd.DataFrame:
    """
    Normalize timestamps to ISO 8601 UTC + optional resampling.
    target_freq: pandas freq string like '1H', '1D', '15min'.
    """
    df = df.copy()
    if ts_col not in df.columns:
        log.warning("No timestamp column '%s' found", ts_col)
        return df

    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
    df = df.dropna(subset=[ts_col]).sort_values(ts_col).reset_index(drop=True)

    if target_freq:
        df = df.set_index(ts_col)
        agg = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum"
        }
        agg = {k: v for k, v in agg.items() if k in df.columns}
        df = df.resample(target_freq).agg(agg).dropna().reset_index()
    return df


def detect_leakage(df: pd.DataFrame, target_col: str) -> dict:
    """
    Detect potential future-leakage features.
    Flags any column with correlation > 0.95 to future target (shifted -1).

    Reference: Lopez de Prado (2018) §7 — backtesting pitfalls.
    """
    if target_col not in df.columns:
        return {"checked": False, "reason": f"target '{target_col}' missing"}
    future_target = df[target_col].shift(-1)
    suspects = []
    num_df = df.select_dtypes(include=[np.number])
    for col in num_df.columns:
        if col == target_col:
            continue
        try:
            corr = num_df[col].corr(future_target)
            if abs(corr) > 0.95:
                suspects.append({"column": col, "future_corr": float(corr)})
        except Exception:
            continue
    return {
        "checked": True,
        "suspects": suspects,
        "warning": "Features with |corr| > 0.95 to future target — possible leakage!"
                   if suspects else "No leakage detected"
    }


def time_series_split(df: pd.DataFrame, train_ratio: float = 0.7,
                       val_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Walk-forward time-aware split (Lopez de Prado 2018 §7.3).
    No shuffling. Train comes first chronologically.
    """
    n = len(df)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = df.iloc[:n_train].copy()
    val = df.iloc[n_train:n_train + n_val].copy()
    test = df.iloc[n_train + n_val:].copy()
    return train, val, test


def run_full_pipeline(df: pd.DataFrame, target_freq: Optional[str] = None) -> dict:
    """
    Run full cleaning pipeline + return audit report.
    Returns: { df, report }
    """
    report = {"pipeline": "nexa_probability.preprocessing.run_full_pipeline",
              "steps": []}

    # Step 1: validate
    df = validate_ohlcv(df)
    report["steps"].append({"step": "validate_ohlcv", "rows": len(df)})

    # Step 2: normalize timestamps
    df = normalize_timestamps(df, target_freq=target_freq)
    report["steps"].append({"step": "normalize_timestamps",
                             "rows": len(df), "freq": target_freq})

    # Step 3: handle missing
    df = handle_missing_ohlcv(df)
    report["steps"].append({"step": "handle_missing_ohlcv", "rows": len(df)})

    # Step 4: flag outliers (don't remove — extreme moves matter in trading)
    outlier_mask = detect_outliers_zscore(df, "close")
    df["is_outlier"] = outlier_mask.astype(int)
    report["steps"].append({"step": "detect_outliers_zscore",
                             "outliers_flagged": int(outlier_mask.sum())})

    return {"df": df, "report": report}
