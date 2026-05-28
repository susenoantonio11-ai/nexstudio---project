"""
DynamicModelSelectionEngine
===========================
Algorithm-agnostic model selection engine. Replaces hardcoded "Random Forest
chosen" / "XGBoost selected" defaults with reasoning-first analysis.

PHILOSOPHY
  Nexlytics must NOT be biased toward any particular family of algorithms.
  The engine reads dataset characteristics, scores every candidate algorithm
  in the relevant problem domain, and returns a ranked list with explicit
  reasoning, computational cost, and risk analysis.

INPUT
  dataset_characteristics dict, e.g.:
    {
      "n_rows": 8000,
      "n_features": 24,
      "n_numeric": 18,
      "n_categorical": 6,
      "missing_pct": 4.2,
      "imbalance_ratio": 4.8,
      "has_temporal": False,
      "has_spatial": True,
      "has_text": False,
      "has_image": False,
      "high_dimensionality": False,
      "nonlinear_relationship": True,
      "sparse_features": False,
      "target_type": "binary",
      "domain_hint": "geospatial"
    }

OUTPUT
  Standard reasoning envelope:
    {
      "problem_type": "classification",
      "dataset_characteristics": { ... echoes input + derivations },
      "reasoning_timeline": [ { step, finding, evidence } ... ],
      "candidate_evaluation": [ all candidates with scores ],
      "recommended_models": [ top-N with score >= 0.60 ],
      "non_prioritized_models": [ scored < 0.40 with reason ],
      "confidence_score": 0..1,
      "method_monitor": { method, why_used, formulas, limitations, citations }
    }

CITATIONS
  * Wolpert, D. H. (1996). The Lack of A Priori Distinctions Between
    Learning Algorithms. Neural Computation 8(7) — "No Free Lunch" theorem
    motivates algorithm-agnostic reasoning.
  * Olson, R. S. et al. (2017) Data-driven advice for applying machine
    learning to bioinformatics problems. Pacific Symposium on Biocomputing.
  * Probst, P., Boulesteix, A.-L., Bischl, B. (2019) Tunability and
    importance of hyperparameters of machine learning algorithms. JMLR 20.
"""
from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional


# ===========================================================================
# Algorithm catalog — every entry has scoring rules expressed as functions
# of dataset characteristics. A score in [0, 1] indicates suitability.
# ===========================================================================

def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _smallish(n_rows: int, threshold_low: int = 500, threshold_high: int = 5000) -> float:
    """Returns 1.0 when n_rows >= threshold_high, 0.0 below threshold_low,
    linear in between."""
    if n_rows <= threshold_low: return 0.0
    if n_rows >= threshold_high: return 1.0
    return (n_rows - threshold_low) / (threshold_high - threshold_low)


def _data_hungry(n_rows: int) -> float:
    """1.0 when dataset is large enough for deep models (>= 50k rows)."""
    if n_rows < 5000: return 0.0
    if n_rows >= 50000: return 1.0
    return (n_rows - 5000) / 45000.0


# ---------------------------------------------------------------------------
# Tabular classification candidates
# ---------------------------------------------------------------------------
TABULAR_CLASSIFICATION = [
    {
        "name": "Logistic Regression", "family": "linear",
        "computational_cost": "low", "interpretability": "high",
        "score_fn": lambda c: 0.10 + 0.60 * (1 - c.get("nonlinear_relationship", False))
                              + 0.20 * (1 - min(1.0, c.get("missing_pct", 0) / 30))
                              + 0.10 * _clip(c.get("n_features", 10) / 50),
        "pros": ["Calibrated probabilities", "Maximum interpretability via coefficients", "Fast training and inference"],
        "cons": ["Cannot capture non-linear interactions", "Sensitive to multi-collinearity"],
        "citation": "Cox (1958) JRSS Series B 20(2):215–242.",
        "deps": ["scikit-learn"],
    },
    {
        "name": "Random Forest", "family": "tree_ensemble",
        "computational_cost": "medium", "interpretability": "medium",
        "score_fn": lambda c: 0.20
                              + 0.30 * (1 if c.get("nonlinear_relationship", False) else 0)
                              + 0.15 * (1 - min(1.0, c.get("missing_pct", 0) / 40))
                              + 0.15 * _clip(c.get("n_categorical", 0) / max(c.get("n_features", 1), 1))
                              + 0.10 * _smallish(c.get("n_rows", 0))
                              + 0.10 * (1 if c.get("class_imbalance", False) else 0.5),
        "pros": ["Robust to outliers", "Handles mixed numeric/categorical naturally", "Built-in feature importance"],
        "cons": ["Larger models can be memory-heavy", "Less accurate than boosting on average for clean tabular"],
        "citation": "Breiman (2001) Machine Learning 45:5–32.",
        "deps": ["scikit-learn"],
    },
    {
        "name": "XGBoost", "family": "gradient_boosting",
        "computational_cost": "medium", "interpretability": "medium",
        "score_fn": lambda c: 0.20
                              + 0.35 * (1 if c.get("nonlinear_relationship", False) else 0)
                              + 0.15 * _smallish(c.get("n_rows", 0), 200, 2000)
                              + 0.15 * (1 - min(1.0, c.get("missing_pct", 0) / 40))
                              + 0.10 * (1 if c.get("class_imbalance", False) else 0.5)
                              + 0.05 * (1 - c.get("sparse_features", False)),
        "pros": ["State-of-the-art on tabular benchmarks", "Native handling of missing values", "Class-weight + scale_pos_weight for imbalance"],
        "cons": ["More hyperparameters to tune", "Less explainable without SHAP"],
        "citation": "Chen & Guestrin (2016) KDD '16: 785–794.",
        "deps": ["xgboost"],
    },
    {
        "name": "LightGBM", "family": "gradient_boosting",
        "computational_cost": "medium-low", "interpretability": "medium",
        "score_fn": lambda c: 0.18
                              + 0.30 * (1 if c.get("nonlinear_relationship", False) else 0)
                              + 0.20 * _data_hungry(c.get("n_rows", 0))
                              + 0.15 * _clip(c.get("n_features", 10) / 100)
                              + 0.10 * (1 if c.get("class_imbalance", False) else 0.5)
                              + 0.07,
        "pros": ["Histogram-based — fast on large datasets", "Memory-efficient", "Excellent on high-cardinality categoricals"],
        "cons": ["Can overfit on small datasets", "Sensitive to data leakage in default settings"],
        "citation": "Ke et al. (2017) NeurIPS 30: 3146–3154.",
        "deps": ["lightgbm"],
    },
    {
        "name": "CatBoost", "family": "gradient_boosting",
        "computational_cost": "medium", "interpretability": "medium",
        "score_fn": lambda c: 0.18
                              + 0.30 * (1 if c.get("nonlinear_relationship", False) else 0)
                              + 0.30 * _clip(c.get("n_categorical", 0) / max(c.get("n_features", 1), 1))
                              + 0.15 * _smallish(c.get("n_rows", 0), 500, 10000)
                              + 0.07,
        "pros": ["Best-in-class for high-cardinality categorical features", "Ordered boosting reduces target leakage", "Fewer hyperparameters than XGB/LGBM"],
        "cons": ["Slower training than LightGBM", "Less ecosystem tooling"],
        "citation": "Prokhorenkova et al. (2018) NeurIPS 31.",
        "deps": ["catboost"],
    },
    {
        "name": "Support Vector Machine", "family": "kernel",
        "computational_cost": "high", "interpretability": "low",
        "score_fn": lambda c: 0.15
                              + 0.20 * (1 if c.get("nonlinear_relationship", False) else 0)
                              + 0.20 * _clip(1 - c.get("n_rows", 0) / 10000)
                              + 0.15 * _clip(c.get("n_features", 10) / 30)
                              + 0.10 * (1 - min(1.0, c.get("missing_pct", 0) / 20)),
        "pros": ["Strong on small clean datasets with informative features", "Effective in high-dimensional spaces"],
        "cons": ["O(n²)–O(n³) training cost — does not scale", "Black-box without kernel introspection"],
        "citation": "Cortes & Vapnik (1995) Machine Learning 20(3):273–297.",
        "deps": ["scikit-learn"],
    },
    {
        "name": "K-Nearest Neighbors", "family": "instance_based",
        "computational_cost": "low_train_high_inference", "interpretability": "medium",
        "score_fn": lambda c: 0.12
                              + 0.20 * _smallish(c.get("n_rows", 0), 100, 2000)
                              + 0.20 * (1 - _clip(c.get("n_features", 10) / 30))
                              + 0.15 * (1 - min(1.0, c.get("missing_pct", 0) / 15)),
        "pros": ["No training phase", "Naturally handles non-linear decision boundaries"],
        "cons": ["Curse of dimensionality", "Slow at inference for large datasets", "Sensitive to feature scaling"],
        "citation": "Fix & Hodges (1951) USAF School of Aviation Medicine, Tech. Rep. 4.",
        "deps": ["scikit-learn"],
    },
    {
        "name": "Naive Bayes", "family": "probabilistic",
        "computational_cost": "low", "interpretability": "high",
        "score_fn": lambda c: 0.10
                              + 0.30 * (1 if c.get("sparse_features", False) else 0)
                              + 0.25 * (1 if c.get("has_text", False) else 0)
                              + 0.15 * _smallish(c.get("n_rows", 0), 100, 1000)
                              + 0.10,
        "pros": ["Extremely fast", "Strong baseline for text classification"],
        "cons": ["Assumes feature independence — usually violated", "Underperforms on dense numeric tabular"],
        "citation": "Hand & Yu (2001) International Statistical Review 69(3):385–398.",
        "deps": ["scikit-learn"],
    },
    {
        "name": "Multi-Layer Perceptron", "family": "neural_network",
        "computational_cost": "medium-high", "interpretability": "low",
        "score_fn": lambda c: 0.10
                              + 0.25 * (1 if c.get("nonlinear_relationship", False) else 0)
                              + 0.30 * _data_hungry(c.get("n_rows", 0))
                              + 0.10 * (1 - min(1.0, c.get("missing_pct", 0) / 20))
                              - 0.20 * (1 if c.get("n_rows", 1e9) < 1000 else 0),
        "pros": ["Universal approximator", "Scales with data"],
        "cons": ["Data-hungry", "Requires careful regularization", "Black-box"],
        "citation": "Rumelhart, Hinton & Williams (1986) Nature 323:533–536.",
        "deps": ["scikit-learn"],
    },
    {
        "name": "Stacking Ensemble", "family": "meta_learner",
        "computational_cost": "high", "interpretability": "low",
        "score_fn": lambda c: 0.10
                              + 0.30 * _data_hungry(c.get("n_rows", 0))
                              + 0.20 * (1 if c.get("nonlinear_relationship", False) else 0)
                              + 0.15,
        "pros": ["Often best performance via meta-learning", "Reduces variance vs single model"],
        "cons": ["Highest compute cost", "Harder to deploy", "Risk of meta-overfitting"],
        "citation": "Wolpert (1992) Neural Networks 5(2):241–259.",
        "deps": ["scikit-learn"],
    },
]


# ---------------------------------------------------------------------------
# Time series forecasting candidates
# ---------------------------------------------------------------------------
TIME_SERIES_FORECASTING = [
    {
        "name": "ARIMA / SARIMA", "family": "classical_stat",
        "computational_cost": "low", "interpretability": "high",
        "score_fn": lambda c: 0.20
                              + 0.30 * (1 if c.get("has_temporal", False) else 0)
                              + 0.20 * _clip(1 - c.get("n_rows", 1e9) / 10000)
                              + 0.10 * (1 - c.get("nonlinear_relationship", False)),
        "pros": ["Strong on stationary linear series", "Statistical confidence intervals", "Well understood"],
        "cons": ["Assumes linearity + stationarity", "Manual differencing required"],
        "citation": "Box, Jenkins & Reinsel (1976) Time Series Analysis: Forecasting and Control.",
        "deps": ["statsmodels"],
    },
    {
        "name": "Prophet", "family": "decomposition",
        "computational_cost": "low-medium", "interpretability": "high",
        "score_fn": lambda c: 0.20
                              + 0.20 * (1 if c.get("has_temporal", False) else 0)
                              + 0.20 * (1 if c.get("seasonal", False) else 0)
                              + 0.20 * (1 if c.get("has_holidays", False) else 0),
        "pros": ["Built-in seasonality + holidays", "Resilient to missing data", "Few hyperparameters"],
        "cons": ["Less flexible than tree boosting", "Single-series — multivariate is awkward"],
        "citation": "Taylor & Letham (2018) The American Statistician 72(1):37–45.",
        "deps": ["prophet"],
    },
    {
        "name": "Exponential Smoothing (ETS)", "family": "classical_stat",
        "computational_cost": "low", "interpretability": "medium",
        "score_fn": lambda c: 0.15
                              + 0.25 * (1 if c.get("has_temporal", False) else 0)
                              + 0.20 * (1 if c.get("seasonal", False) else 0)
                              + 0.10,
        "pros": ["Robust default for short series", "Captures level + trend + seasonality"],
        "cons": ["Univariate; no exogenous covariates"],
        "citation": "Hyndman, Koehler, Snyder & Grose (2002) IJF 18(3):439–454.",
        "deps": ["statsmodels"],
    },
    {
        "name": "LSTM", "family": "rnn",
        "computational_cost": "high", "interpretability": "low",
        "score_fn": lambda c: 0.10
                              + 0.30 * (1 if c.get("has_temporal", False) else 0)
                              + 0.30 * _data_hungry(c.get("n_rows", 0))
                              + 0.15 * (1 if c.get("nonlinear_relationship", False) else 0),
        "pros": ["Captures long-range temporal dependence", "Handles multivariate sequences"],
        "cons": ["Data-hungry", "Requires GPU for reasonable training time", "Black-box without SHAP / LIME"],
        "citation": "Hochreiter & Schmidhuber (1997) Neural Computation 9(8):1735–1780.",
        "deps": ["torch"],
    },
    {
        "name": "Temporal Fusion Transformer (TFT)", "family": "transformer",
        "computational_cost": "very_high", "interpretability": "medium",
        "score_fn": lambda c: 0.05
                              + 0.30 * (1 if c.get("has_temporal", False) else 0)
                              + 0.40 * _data_hungry(c.get("n_rows", 0))
                              + 0.15 * (1 if c.get("multivariate", False) else 0),
        "pros": ["State-of-the-art on long horizons", "Built-in attention-based interpretability"],
        "cons": ["Massive compute requirement", "Steep training data requirement"],
        "citation": "Lim et al. (2021) International Journal of Forecasting 37(4):1748–1764.",
        "deps": ["pytorch-forecasting"],
    },
    {
        "name": "XGBoost on lag features", "family": "gradient_boosting",
        "computational_cost": "medium", "interpretability": "medium",
        "score_fn": lambda c: 0.20
                              + 0.20 * (1 if c.get("has_temporal", False) else 0)
                              + 0.20 * (1 if c.get("multivariate", False) else 0)
                              + 0.15 * (1 if c.get("nonlinear_relationship", False) else 0)
                              + 0.10,
        "pros": ["Treats forecasting as supervised regression", "Handles many exogenous covariates", "Production-friendly"],
        "cons": ["Manual lag feature engineering", "Cannot extrapolate outside training range"],
        "citation": "Bojer & Meldgaard (2021) IJF 37(2):587–603 — M5 forecasting competition winner used XGBoost.",
        "deps": ["xgboost"],
    },
    {
        "name": "Hybrid LSTM + XGBoost (soft voting)", "family": "ensemble",
        "computational_cost": "high", "interpretability": "medium",
        "score_fn": lambda c: 0.10
                              + 0.25 * (1 if c.get("has_temporal", False) else 0)
                              + 0.20 * _data_hungry(c.get("n_rows", 0))
                              + 0.15 * (1 if c.get("nonlinear_relationship", False) else 0)
                              + 0.10 * (1 if c.get("has_static_features", False) else 0)
                              + 0.10,
        "pros": ["LSTM captures sequence + XGBoost captures static", "Diversity-based ensemble gain"],
        "cons": ["Two models to maintain", "Static fusion weight unless tuned via stacking"],
        "citation": "Wolpert (1992) Neural Networks 5(2); Nature Sci. Reports (2024) — storm prediction.",
        "deps": ["torch", "xgboost"],
    },
]


# ---------------------------------------------------------------------------
# NLP candidates
# ---------------------------------------------------------------------------
NLP_CLASSIFICATION = [
    {
        "name": "TF-IDF + Logistic Regression", "family": "linear",
        "computational_cost": "low", "interpretability": "high",
        "score_fn": lambda c: 0.20
                              + 0.20 * (1 if c.get("has_text", False) else 0)
                              + 0.20 * _smallish(c.get("n_rows", 0), 100, 5000)
                              + 0.20,
        "pros": ["Fast baseline", "Coefficients give per-token explanation"],
        "cons": ["Bag-of-words ignores order"],
        "citation": "Salton & Buckley (1988) Information Processing & Management 24(5):513–523.",
        "deps": ["scikit-learn"],
    },
    {
        "name": "FastText", "family": "shallow_nn",
        "computational_cost": "low-medium", "interpretability": "low",
        "score_fn": lambda c: 0.15
                              + 0.25 * (1 if c.get("has_text", False) else 0)
                              + 0.20 * _smallish(c.get("n_rows", 0), 1000, 20000),
        "pros": ["Sub-word units handle OOV", "Trains quickly on CPU"],
        "cons": ["No contextual semantics"],
        "citation": "Joulin, Grave, Bojanowski & Mikolov (2017) EACL.",
        "deps": ["fasttext"],
    },
    {
        "name": "DistilBERT", "family": "transformer",
        "computational_cost": "medium-high", "interpretability": "medium",
        "score_fn": lambda c: 0.12
                              + 0.30 * (1 if c.get("has_text", False) else 0)
                              + 0.25 * _data_hungry(c.get("n_rows", 0))
                              + 0.10,
        "pros": ["97% of BERT's quality at 60% of size", "Pretrained — works with little task data"],
        "cons": ["Needs GPU for fine-tuning at scale"],
        "citation": "Sanh et al. (2019) NeurIPS Workshop EMC²-NIPS.",
        "deps": ["transformers", "torch"],
    },
    {
        "name": "BERT (base / multilingual)", "family": "transformer",
        "computational_cost": "high", "interpretability": "medium",
        "score_fn": lambda c: 0.10
                              + 0.30 * (1 if c.get("has_text", False) else 0)
                              + 0.30 * _data_hungry(c.get("n_rows", 0)),
        "pros": ["Strong contextual embeddings", "Fine-tunes well across tasks"],
        "cons": ["Slower than DistilBERT", "Requires GPU"],
        "citation": "Devlin et al. (2019) NAACL-HLT.",
        "deps": ["transformers", "torch"],
    },
    {
        "name": "RoBERTa", "family": "transformer",
        "computational_cost": "high", "interpretability": "medium",
        "score_fn": lambda c: 0.10
                              + 0.30 * (1 if c.get("has_text", False) else 0)
                              + 0.35 * _data_hungry(c.get("n_rows", 0)),
        "pros": ["Improved pretraining objective", "Strong on sentence-level tasks"],
        "cons": ["Expensive", "Diminishing returns vs DistilBERT on small tasks"],
        "citation": "Liu et al. (2019) arXiv:1907.11692.",
        "deps": ["transformers", "torch"],
    },
    {
        "name": "Sentence-Transformers Embedding + classifier", "family": "embedding_pipeline",
        "computational_cost": "medium", "interpretability": "medium",
        "score_fn": lambda c: 0.15
                              + 0.30 * (1 if c.get("has_text", False) else 0)
                              + 0.20 * _smallish(c.get("n_rows", 0), 500, 10000),
        "pros": ["Pretrained embeddings + cheap downstream classifier", "Production-friendly"],
        "cons": ["Embedding quality depends on pretraining domain"],
        "citation": "Reimers & Gurevych (2019) EMNLP.",
        "deps": ["sentence-transformers"],
    },
]


# ---------------------------------------------------------------------------
# Computer vision candidates
# ---------------------------------------------------------------------------
COMPUTER_VISION = [
    {
        "name": "Handcrafted features (HOG/LBP/Color hist) + SVM", "family": "classical_cv",
        "computational_cost": "low", "interpretability": "medium",
        "score_fn": lambda c: 0.20
                              + 0.10 * (1 if c.get("has_image", False) else 0)
                              + 0.20 * _smallish(c.get("n_rows", 0), 50, 1000),
        "pros": ["Works without GPU", "Deterministic + citable per descriptor"],
        "cons": ["Misses high-level semantics"],
        "citation": "Dalal & Triggs (2005) HOG; Ojala et al. (2002) LBP.",
        "deps": ["scikit-image"],
    },
    {
        "name": "MobileNet (transfer learning)", "family": "cnn",
        "computational_cost": "medium", "interpretability": "low",
        "score_fn": lambda c: 0.15
                              + 0.30 * (1 if c.get("has_image", False) else 0)
                              + 0.20 * _smallish(c.get("n_rows", 0), 1000, 10000),
        "pros": ["Mobile-friendly", "Pretrained on ImageNet"],
        "cons": ["Lower accuracy than larger CNNs"],
        "citation": "Howard et al. (2017) arXiv:1704.04861.",
        "deps": ["torch", "torchvision"],
    },
    {
        "name": "ResNet-50 (transfer learning)", "family": "cnn",
        "computational_cost": "medium-high", "interpretability": "medium",
        "score_fn": lambda c: 0.15
                              + 0.30 * (1 if c.get("has_image", False) else 0)
                              + 0.30 * _data_hungry(c.get("n_rows", 0)),
        "pros": ["Robust pretrained backbone", "Works well with Grad-CAM"],
        "cons": ["Heavier than MobileNet"],
        "citation": "He, Zhang, Ren & Sun (2016) CVPR.",
        "deps": ["torch", "torchvision"],
    },
    {
        "name": "EfficientNet", "family": "cnn",
        "computational_cost": "medium", "interpretability": "low",
        "score_fn": lambda c: 0.15
                              + 0.30 * (1 if c.get("has_image", False) else 0)
                              + 0.30 * _data_hungry(c.get("n_rows", 0)),
        "pros": ["Best accuracy/parameter ratio", "Compound scaling"],
        "cons": ["Sensitive to image resolution"],
        "citation": "Tan & Le (2019) ICML.",
        "deps": ["torch", "torchvision"],
    },
    {
        "name": "Vision Transformer (ViT)", "family": "transformer",
        "computational_cost": "high", "interpretability": "medium",
        "score_fn": lambda c: 0.10
                              + 0.25 * (1 if c.get("has_image", False) else 0)
                              + 0.40 * _data_hungry(c.get("n_rows", 0)),
        "pros": ["Strong on large datasets", "Attention-based interpretability"],
        "cons": ["Data-hungry without pretraining"],
        "citation": "Dosovitskiy et al. (2021) ICLR.",
        "deps": ["torch", "transformers"],
    },
    {
        "name": "U-Net (segmentation)", "family": "encoder_decoder_cnn",
        "computational_cost": "medium-high", "interpretability": "medium",
        "score_fn": lambda c: 0.10
                              + 0.25 * (1 if c.get("has_image", False) else 0)
                              + 0.30 * (1 if c.get("task") == "segmentation" else 0)
                              + 0.20 * _data_hungry(c.get("n_rows", 0)),
        "pros": ["Designed for semantic segmentation", "Skip connections preserve detail"],
        "cons": ["Specific to segmentation"],
        "citation": "Ronneberger, Fischer & Brox (2015) MICCAI.",
        "deps": ["torch"],
    },
    {
        "name": "CLIP (zero-shot / few-shot)", "family": "vision_language",
        "computational_cost": "medium-high", "interpretability": "medium",
        "score_fn": lambda c: 0.10
                              + 0.20 * (1 if c.get("has_image", False) else 0)
                              + 0.20 * _smallish(c.get("n_rows", 0), 0, 500)
                              + 0.20,
        "pros": ["Zero-shot capable — almost no training data", "Works from text labels"],
        "cons": ["Less accurate than fine-tuned task-specific models"],
        "citation": "Radford et al. (2021) ICML.",
        "deps": ["open_clip"],
    },
]


# ---------------------------------------------------------------------------
# Geospatial candidates
# ---------------------------------------------------------------------------
GEOSPATIAL = [
    {
        "name": "Threshold classifier (NDWI / MNDWI)", "family": "spectral_index",
        "computational_cost": "very_low", "interpretability": "very_high",
        "score_fn": lambda c: 0.30
                              + 0.20 * (1 if c.get("has_spatial", False) else 0)
                              + 0.20 * (1 if c.get("task") in ("water", "flood") else 0),
        "pros": ["Maximum interpretability — single threshold rule", "Works with public satellite data"],
        "cons": ["Cannot capture multivariate interactions"],
        "citation": "McFeeters (1996); Xu (2006).",
        "deps": ["numpy"],
    },
    {
        "name": "Random Forest on raster pixels", "family": "tree_ensemble",
        "computational_cost": "medium", "interpretability": "medium",
        "score_fn": lambda c: 0.20
                              + 0.30 * (1 if c.get("has_spatial", False) else 0)
                              + 0.20 * (1 if c.get("nonlinear_relationship", False) else 0)
                              + 0.15,
        "pros": ["Handles multi-source raster stacks well", "Feature importance per band"],
        "cons": ["Ignores spatial context unless engineered"],
        "citation": "Belgiu & Drăguţ (2016) ISPRS J. Photogramm. Remote Sens. 114.",
        "deps": ["scikit-learn"],
    },
    {
        "name": "U-Net on multi-band raster", "family": "encoder_decoder_cnn",
        "computational_cost": "high", "interpretability": "medium",
        "score_fn": lambda c: 0.10
                              + 0.30 * (1 if c.get("has_spatial", False) else 0)
                              + 0.25 * (1 if c.get("task") == "segmentation" else 0)
                              + 0.25 * _data_hungry(c.get("n_rows", 0)),
        "pros": ["Captures spatial context", "State-of-the-art for flood/landcover segmentation"],
        "cons": ["Requires labeled tile data"],
        "citation": "Ronneberger et al. (2015) MICCAI.",
        "deps": ["torch"],
    },
    {
        "name": "Multisource Fusion (RF / XGB on stacked features)", "family": "ensemble",
        "computational_cost": "medium", "interpretability": "medium",
        "score_fn": lambda c: 0.20
                              + 0.30 * (1 if c.get("has_spatial", False) else 0)
                              + 0.25 * (1 if c.get("multimodal", False) else 0)
                              + 0.15,
        "pros": ["Combines optical + SAR + climate + topography", "Robust under cloud cover via SAR"],
        "cons": ["Pre-resampling to common grid required"],
        "citation": "Tehrany et al. (2014); Twele et al. (2016).",
        "deps": ["scikit-learn"],
    },
]


# ===========================================================================
# Engine
# ===========================================================================
class DynamicModelSelectionEngine:
    """Algorithm-agnostic, dataset-aware model recommender.

    Designed to make Nexlytics live up to the "Explainable AI Intelligence
    Platform" promise: every recommendation is paired with reasoning that
    a thesis reviewer can audit.
    """

    name = "DynamicModelSelectionEngine"
    domain = "reasoning"
    citations = [
        "Wolpert (1996) Neural Computation 8(7) — No Free Lunch theorem.",
        "Olson et al. (2017) Pacific Symposium on Biocomputing — data-driven advice.",
        "Probst, Boulesteix & Bischl (2019) JMLR 20 — hyperparameter tunability.",
    ]

    CANDIDATE_POOLS = {
        "tabular_classification": TABULAR_CLASSIFICATION,
        "tabular_regression": TABULAR_CLASSIFICATION,  # similar shape; refined per task below
        "time_series": TIME_SERIES_FORECASTING,
        "nlp": NLP_CLASSIFICATION,
        "image": COMPUTER_VISION,
        "geospatial": GEOSPATIAL,
    }

    # --------------------------------------------------------------
    def select(self, characteristics: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.perf_counter()
        c = self._normalize(characteristics)

        # Step 1 — problem type detection (reasoning, not lookup)
        problem_type, type_reasoning = self._detect_problem_type(c)

        # Step 2 — derived signals
        derived, signal_reasoning = self._derive_signals(c)
        c.update(derived)

        # Step 3 — score every candidate in the relevant pool
        pool = self.CANDIDATE_POOLS.get(problem_type, TABULAR_CLASSIFICATION)
        scored: List[Dict[str, Any]] = []
        for cand in pool:
            try:
                raw = float(cand["score_fn"](c))
            except Exception:
                raw = 0.0
            score = round(_clip(raw), 3)
            risks = self._risk_analysis(cand, c)
            scored.append({
                "name": cand["name"],
                "family": cand["family"],
                "suitability_score": score,
                "computational_cost": cand["computational_cost"],
                "interpretability": cand["interpretability"],
                "pros": cand["pros"],
                "cons": cand["cons"],
                "risks": risks,
                "citation": cand["citation"],
                "dependencies": cand["deps"],
                "reason": self._reason_for(cand, c, score),
            })
        scored.sort(key=lambda x: -x["suitability_score"])

        # Step 4 — recommended vs non-prioritized
        recommended = [s for s in scored if s["suitability_score"] >= 0.60]
        non_prioritized = [
            {**s, "reason_not_prioritized": self._why_not(s, c)}
            for s in scored if s["suitability_score"] < 0.40
        ]

        # Step 5 — confidence: gap between top-1 and top-3 mean
        top_score = scored[0]["suitability_score"] if scored else 0.0
        top3_mean = sum(s["suitability_score"] for s in scored[:3]) / max(1, min(3, len(scored)))
        confidence = round(_clip(0.5 + 0.4 * top_score + 0.1 * (top_score - top3_mean)), 3)

        # Step 6 — reasoning timeline
        timeline = []
        timeline.extend(type_reasoning)
        timeline.extend(signal_reasoning)
        timeline.append({
            "step": "Candidate Model Ranking",
            "finding": f"Scored {len(scored)} candidates in {problem_type} pool.",
            "evidence": [{"model": s["name"], "score": s["suitability_score"]} for s in scored[:5]],
        })
        timeline.append({
            "step": "Confidence Estimation",
            "finding": f"Confidence {confidence:.2f} based on top-1 score gap.",
            "evidence": {"top_score": top_score, "top3_mean": round(top3_mean, 3)},
        })

        return {
            "status": "success",
            "model_name": self.name,
            "problem_type": problem_type,
            "dataset_characteristics": c,
            "reasoning_timeline": timeline,
            "candidate_evaluation": scored,
            "recommended_models": recommended[:5],
            "non_prioritized_models": non_prioritized[:5],
            "confidence_score": confidence,
            "duration_ms": int((time.perf_counter() - t0) * 1000),
            "method_monitor": {
                "method": "Dataset-characteristics scoring across N candidates per problem domain",
                "why_used": "Algorithm-agnostic recommendation. Prevents bias toward any single family of models.",
                "formulas": [
                    "score_i = Σ w_k · f_k(characteristics)   per-candidate weighted rule",
                    "confidence = 0.5 + 0.4·top_score + 0.1·(top_score − mean(top3))",
                ],
                "limitations": [
                    "Heuristic scoring — not a learned ranker. Replace with learning-to-rank when telemetry accumulates.",
                    "Pool coverage is finite — extend per domain as new algorithms are introduced.",
                ],
                "citations": self.citations,
            },
        }

    # --------------------------------------------------------------
    # Reasoning helpers
    # --------------------------------------------------------------
    def _normalize(self, c: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(c or {})
        out.setdefault("n_rows", 0)
        out.setdefault("n_features", 0)
        out.setdefault("n_numeric", 0)
        out.setdefault("n_categorical", 0)
        out.setdefault("missing_pct", 0.0)
        out.setdefault("imbalance_ratio", 1.0)
        out.setdefault("has_temporal", False)
        out.setdefault("has_spatial", False)
        out.setdefault("has_text", False)
        out.setdefault("has_image", False)
        out.setdefault("nonlinear_relationship", False)
        out.setdefault("sparse_features", False)
        out.setdefault("class_imbalance", out.get("imbalance_ratio", 1.0) > 3.0)
        out.setdefault("seasonal", False)
        out.setdefault("multivariate", False)
        out.setdefault("multimodal", False)
        out.setdefault("has_static_features", False)
        out.setdefault("target_type", "binary")
        out.setdefault("domain_hint", "general")
        out.setdefault("task", "classification")
        return out

    def _detect_problem_type(self, c: Dict[str, Any]):
        timeline = []
        if c.get("has_image"):
            t = "image"; reason = "Image data detected (has_image=True)."
        elif c.get("has_text"):
            t = "nlp"; reason = "Text data detected (has_text=True)."
        elif c.get("has_temporal"):
            t = "time_series"; reason = "Temporal dependency detected (has_temporal=True)."
        elif c.get("has_spatial"):
            t = "geospatial"; reason = "Spatial structure detected (has_spatial=True)."
        elif c.get("target_type") in ("continuous",):
            t = "tabular_regression"; reason = "Continuous target → regression task."
        else:
            t = "tabular_classification"; reason = "Default tabular classification."
        timeline.append({"step": "Problem-Type Detection", "finding": reason})
        return t, timeline

    def _derive_signals(self, c: Dict[str, Any]):
        derived = {}
        timeline = []
        n_rows = int(c.get("n_rows", 0))
        n_features = int(c.get("n_features", 0))
        derived["high_dimensionality"] = n_features > 100 or (n_features > 20 and n_rows < 500)
        if derived["high_dimensionality"]:
            timeline.append({"step": "Statistical Characteristic Analysis",
                             "finding": "High dimensionality flag set — feature/sample ratio is elevated."})
        if c.get("imbalance_ratio", 1.0) > 3.0:
            derived["class_imbalance"] = True
            timeline.append({"step": "Feature Relationship Analysis",
                             "finding": f"Class imbalance ratio {c['imbalance_ratio']:.1f}:1 — boosting + class_weight recommended."})
        if c.get("missing_pct", 0) > 20:
            timeline.append({"step": "Statistical Characteristic Analysis",
                             "finding": f"High missingness ({c['missing_pct']:.1f}%). Tree-based or imputation pipelines needed."})
        if c.get("nonlinear_relationship"):
            timeline.append({"step": "Pattern Detection",
                             "finding": "Non-linear relationship detected — linear models likely underfit."})
        if c.get("has_temporal") and c.get("has_static_features"):
            timeline.append({"step": "Temporal / Spatial Analysis",
                             "finding": "Mixed temporal + static features → favors hybrid sequence + tree ensembles."})
        if c.get("has_spatial"):
            timeline.append({"step": "Temporal / Spatial Analysis",
                             "finding": "Spatial structure → consider context-aware models (CNN, U-Net)."})
        timeline.append({"step": "Complexity Analysis",
                         "finding": f"Dataset size {n_rows:,} rows × {n_features} features."})
        return derived, timeline

    def _reason_for(self, cand: Dict[str, Any], c: Dict[str, Any], score: float) -> str:
        if score >= 0.75:
            return f"Strong match: family {cand['family']} suits the detected dataset profile."
        if score >= 0.60:
            return f"Reasonable match: {cand['family']} fits parts of the dataset profile."
        if score >= 0.40:
            return f"Marginal: {cand['family']} could work but is not the strongest fit."
        return f"Not prioritized: dataset characteristics make {cand['family']} a poor match."

    def _why_not(self, scored: Dict[str, Any], c: Dict[str, Any]) -> str:
        msgs = []
        if "neural" in scored.get("family", "") or "transformer" in scored.get("family", ""):
            if c.get("n_rows", 0) < 5000:
                msgs.append("Dataset volume insufficient for deep architectures — high overfitting risk.")
        if scored.get("family") == "linear" and c.get("nonlinear_relationship"):
            msgs.append("Linear assumption violated — non-linear pattern detected.")
        if scored.get("family") == "kernel" and c.get("n_rows", 0) > 10000:
            msgs.append("SVM training scales O(n²)–O(n³) — unsuitable at this scale.")
        if scored.get("family") == "instance_based" and c.get("n_features", 0) > 30:
            msgs.append("Curse of dimensionality affects instance-based methods.")
        if not msgs:
            msgs.append("Suitability score below threshold for this dataset profile.")
        return " ".join(msgs)

    def _risk_analysis(self, cand: Dict[str, Any], c: Dict[str, Any]) -> List[str]:
        risks = []
        if c.get("class_imbalance") and "linear" in cand.get("family", ""):
            risks.append("Sensitive to class imbalance — pair with class_weight='balanced' or resampling.")
        if c.get("n_rows", 0) < 1000 and ("neural" in cand.get("family", "") or "transformer" in cand.get("family", "")):
            risks.append("Possible overfitting risk on small dataset.")
        if c.get("missing_pct", 0) > 10 and cand.get("family") == "instance_based":
            risks.append("Missing values inflate distance computations — impute first.")
        if cand.get("computational_cost") in ("high", "very_high"):
            risks.append("Inference latency / GPU dependency — verify deployment target supports it.")
        return risks
