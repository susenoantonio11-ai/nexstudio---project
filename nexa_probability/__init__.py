"""
NEXA Probability Engine — financial/crypto/prediction-market intelligence module
for NXLYTICS platform.

COMPLIANCE NOTICE
=================
This engine produces PROBABILISTIC predictions, not financial advice.
- No guarantee of profit
- No claim of certainty
- No future leakage in training
- All outputs are explainable and auditable
- User retains full responsibility for any trading decisions

Architecture (per spec):
    1. Data Input Layer        — OHLCV, orderbook, on-chain, sentiment, macro
    2. Data Processing Layer   — cleaning, missing handling, leakage detection
    3. Feature Engineering     — technical indicators, regime labels
    4. Model Layer             — baseline → ensemble → deep → bayesian → HMM
    5. Probability Engine      — multi-outcome probability scoring
    6. Risk Engine             — VaR, Kelly, drawdown, position sizing
    7. Explainability Layer    — per-prediction reasoning + feature impact
    8. Backtesting             — walk-forward, calibration, Sharpe/Sortino
    9. API Backend             — FastAPI endpoints
    10. Frontend Integration   — NXLYTICS widgets

Standards:
    - ISO/IEC 23053:2022 — AI lifecycle framework
    - CRISP-DM (Chapman et al. 2000) — methodology
    - CFA Institute Code of Ethics — fair representation

References (canonical):
    - Lopez de Prado, M. (2018) Advances in Financial Machine Learning, Wiley
    - Lopez de Prado, M. (2020) Machine Learning for Asset Managers, Cambridge
    - Hamilton, J. D. (1989) Econometrica 57(2):357-384 (HMM regime)
    - Black, F. & Scholes, M. (1973) JPE 81(3):637-654 (volatility)
    - Bollerslev, T. (1986) J Econometrics 31(3):307-327 (GARCH)
"""

__version__ = "0.1.0"
__author__ = "NXLYTICS Team"

DISCLAIMER = (
    "This is NOT financial advice. Predictions are probabilistic and may be "
    "wrong. Past performance does not guarantee future results. Trading "
    "involves substantial risk of loss. Consult a licensed financial advisor "
    "before making investment decisions."
)
