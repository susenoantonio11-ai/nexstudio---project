# NEXA Probability Engine

Probabilistic market intelligence module for NXLYTICS platform. Designed for financial prediction, trading intelligence, stocks, Bitcoin, crypto, and prediction markets like Polymarket.

## Compliance Notice

**This is NOT financial advice.** Predictions are probabilistic and may be wrong. Past performance does not guarantee future results. Trading involves substantial risk of loss. Consult a licensed financial advisor before making investment decisions.

The engine intentionally avoids:
- Language of "guaranteed profit" or "pasti menang"
- Future leakage in training (Lopez de Prado 2018 §7)
- Overfit-prone deep architectures without proper validation
- Storing API keys in frontend

## Architecture

```
backend/nexa_probability/
├── __init__.py              Module entry + DISCLAIMER constant
├── preprocessing.py         Pipeline: validate → normalize → clean → leakage detection
├── features.py              Technical indicators (RSI, MACD, BB, volume imbalance, entropy, regime)
├── models.py                Multi-model layer with graceful fallback
├── engine.py                ProbabilityEngine, RiskEngine, ExplainabilityEngine, BacktestEngine
├── api.py                   FastAPI endpoints
├── sample_data.py           Synthetic OHLCV generator (BTC + Polymarket)
├── requirements.txt
└── README.md (this file)
```

### Model Hierarchy (with fallback)

| Model | Status | Library | Fallback |
|---|---|---|---|
| Logistic Regression | Always | sklearn | — |
| Random Forest | Always | sklearn | — |
| XGBoost | Optional | xgboost | LightGBM → sklearn GradientBoosting |
| LightGBM | Optional | lightgbm | sklearn GradientBoosting |
| LSTM / Transformer | Optional | PyTorch | (skipped — not feasible in Pyodide) |
| Bayesian Updating | Always | numpy + scipy | normal approximation |
| Hidden Markov Model | Optional | hmmlearn | Heuristic regime (MA + vol) |
| Monte Carlo | Always | numpy | — |
| Anomaly Detection | Always | sklearn (IsolationForest) | — |

If optional libraries are missing, the engine continues working with available models. Status visible at `GET /api/ai/model/status`.

## Installation

```bash
# From repo root:
cd backend
pip install -r nexa_probability/requirements.txt

# Optional advanced models:
pip install xgboost lightgbm hmmlearn
```

## Running the Backend

```bash
# From repo root:
uvicorn backend.nexa_probability.api:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`. Swagger UI at `http://localhost:8000/docs`.

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/` | Service info + endpoint list |
| GET | `/api/ai/model/status` | List available models in current environment |
| POST | `/api/ai/predict/market` | Main prediction: probabilities + risk + explainability |
| POST | `/api/ai/backtest` | Walk-forward backtest with Sharpe/Sortino/win-rate |
| POST | `/api/ai/regime-detection` | Detect market regime (HMM or heuristic) |
| POST | `/api/ai/risk-score` | Position sizing + Kelly + stop-loss/take-profit |
| POST | `/api/ai/monte-carlo` | GBM Monte Carlo price simulation |
| GET | `/api/ai/model/explain/{prediction_id}` | Retrieve cached prediction with full explainability |

### Example: Predict

```bash
curl -X POST http://localhost:8000/api/ai/predict/market \
  -H "Content-Type: application/json" \
  -d '{
    "ohlcv": [
      {"timestamp": "2024-01-01T00:00:00Z", "open": 42000, "high": 42500, "low": 41800, "close": 42300, "volume": 1200},
      {"timestamp": "2024-01-02T00:00:00Z", "open": 42300, "high": 43000, "low": 42100, "close": 42800, "volume": 1500}
    ],
    "account_balance": 10000,
    "horizon": 1
  }'
```

For demo data, generate with:

```python
from backend.nexa_probability.sample_data import generate_btc_demo
df = generate_btc_demo(500)
df.to_csv("btc_demo.csv", index=False)
```

## Frontend Integration

Sidebar menu **NEXA Probability** in NXLYTICS contains:
- **Overview & Predict** (`np-overview`) — main interactive dashboard

The frontend connects to the FastAPI backend at the URL configured in the page (default `http://localhost:8000`). For production, configure via Settings → API Integration.

### Widgets Implemented

| Widget | Purpose |
|---|---|
| Probability Card | UP / DOWN / SIDEWAYS percentage |
| Market Regime Badge | bull / bear / ranging / high-vol |
| Risk Meter | VaR + Kelly + position sizing |
| Confidence Score | Model agreement margin |
| Feature Impact Chart | Top 5 features bar chart |
| Backtest Result Panel | Sharpe, Sortino, drawdown, win rate |
| Explainability Monitor | Per-prediction method reasoning |
| Compliance Disclaimer | Always visible at top + bottom |

## Standards & References

### Methodology

- **Lopez de Prado, M. (2018)**. *Advances in Financial Machine Learning*. Wiley. — Walk-forward backtesting, leakage detection, triple-barrier labeling.
- **Lopez de Prado, M. (2020)**. *Machine Learning for Asset Managers*. Cambridge.
- **Chapman et al. (2000)**. *CRISP-DM 1.0 Step-by-Step Guide*. SPSS Inc. — Phase methodology.

### Models

- **Cox, D. R. (1958)**. JRSS B 20(2):215-242. Logistic Regression.
- **Breiman, L. (2001)**. Machine Learning 45(1):5-32. Random Forest.
- **Chen, T. & Guestrin, C. (2016)**. KDD 785-794. XGBoost.
- **Hamilton, J. D. (1989)**. Econometrica 57(2):357-384. HMM regime switching.
- **Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008)**. ICDM 413-422. Isolation Forest.

### Risk

- **Kelly, J. L. (1956)**. Bell System Technical Journal 35:917-926. Kelly Criterion.
- **Thorp, E. O. (1997)**. *The Kelly Criterion in Blackjack, Sports Betting, and the Stock Market*. — Half-Kelly recommendation.
- **Jorion, P. (2006)**. *Value at Risk* 3rd ed, McGraw-Hill.
- **Sharpe, W. F. (1966)**. Journal of Business 39(1):119-138. Sharpe ratio.
- **Sortino, F. A. & Price, L. N. (1994)**. JOI 3(3):59-64. Sortino ratio.

### Technical Indicators

- **Wilder, J. W. (1978)**. *New Concepts in Technical Trading Systems*. RSI.
- **Appel, G. (1979)**. *Technical Analysis of Stock Trends*. MACD.
- **Bollinger, J. (2002)**. *Bollinger on Bollinger Bands*.
- **Bollerslev, T. (1986)**. J Econometrics 31(3):307-327. GARCH.

### Stochastic Models

- **Black, F. & Scholes, M. (1973)**. JPE 81(3):637-654. Geometric Brownian Motion.
- **Hull, J. C. (2018)**. *Options, Futures, and Other Derivatives* 10th ed. Monte Carlo simulation.

## Critical Safety Practices

1. **No future leakage.** `detect_leakage()` in preprocessing flags any feature with `|corr| > 0.95` to future target.
2. **Walk-forward validation only.** Random train/test split is forbidden for time-series.
3. **Half-Kelly default.** Position sizing uses Half-Kelly + 10% cap regardless of model confidence.
4. **Compliance disclaimer everywhere.** Every API response includes `disclaimer` field.
5. **API keys in backend only.** Frontend reads from environment, never from frontend code.
6. **Probabilistic language only.** UI never displays "guaranteed", "pasti", or definitive predictions.

## Limitations Acknowledged

- Models are trained on historical data; **black swan events** (Taleb 2007) cannot be predicted.
- Market microstructure (slippage, latency, partial fills) is **not modeled**.
- Sentiment integration is **placeholder** — production needs news + social sentiment ingestion.
- LSTM/Transformer require GPU + PyTorch — **not feasible** in browser Pyodide; use Colab/local for training.
- Backtest results are subject to survivorship bias and overfit. Treat as upper-bound estimates.

## Roadmap

- [ ] Add real exchange data ingestion (Binance / Coinbase API)
- [ ] Polymarket / Kalshi probability ingestion
- [ ] On-chain crypto metrics (Glassnode, IntoTheBlock)
- [ ] News sentiment NLP pipeline (FinBERT)
- [ ] Reinforcement learning fine-tuning for position sizing
- [ ] SHAP-based deep explainability for tree models
- [ ] Calibration plot in backtest output
- [ ] Multi-asset portfolio optimization (Markowitz 1952)

## Contact & Legal

Built for NXLYTICS research platform as a thesis-grade probabilistic engine. Not licensed for retail trading platforms or financial advisory services without proper regulatory compliance (e.g., OJK in Indonesia, SEC in US, FCA in UK). User assumes all risk for any usage.
