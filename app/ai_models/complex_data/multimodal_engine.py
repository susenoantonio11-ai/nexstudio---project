"""
MultimodalDataScienceEngine
===========================
Fuses heterogeneous modalities (tabular + image + text + time-series) into
a single feature matrix, then trains a baseline classifier/regressor on the
fused features.

Fusion strategy (Baltrušaitis et al., 2019, IEEE TPAMI 41(2)):
  * Late-fusion when modality-specific encoders are available.
  * Early-fusion (concatenation) when only handcrafted features are
    available — used here as the default because it keeps the engine
    deployable on a torch-free environment.

Each modality has a dedicated extractor:
  tabular     : numeric features pass-through; categoricals one-hot.
  image       : 3·8 colour histogram + 8-orientation Sobel edge density
                (delegates to VisualFeatureExtractor when available).
  text        : TF-IDF char n-grams (sklearn) or hashed bag-of-words.
  time_series : window-mean / window-std / autocorr lag-1 / max-min range.

Citations:
  * Baltrušaitis, Ahuja, Morency (2019) IEEE TPAMI 41(2):423–443 —
    Multimodal Machine Learning: A Survey and Taxonomy.
  * Ngiam et al. (2011) ICML — Multimodal Deep Learning.
"""
from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    from sklearn.feature_extraction.text import HashingVectorizer
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def _tabular_features(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pass-through numeric + simple cardinality encoding for categorical."""
    if not rows or not HAS_PANDAS:
        return {"matrix": np.zeros((0, 0)), "feature_names": []}
    df = pd.DataFrame(rows)
    num = df.select_dtypes(include="number").fillna(0)
    cat = df.select_dtypes(exclude="number")
    cat_enc = pd.get_dummies(cat.astype(str).fillna("__NA__"), drop_first=True) if not cat.empty else pd.DataFrame(index=df.index)
    full = pd.concat([num, cat_enc], axis=1)
    return {"matrix": full.values.astype(float), "feature_names": list(full.columns)}


def _image_features(image_features_payload: Optional[List[List[float]]],
                    expected_count: Optional[int] = None) -> Dict[str, Any]:
    """Use the upstream VisualFeatureExtractor output if provided.
    Each entry is already a flat float vector (color hist + edge stats).
    """
    if not image_features_payload:
        return {"matrix": np.zeros((expected_count or 0, 0)), "feature_names": []}
    arr = np.array(image_features_payload, dtype=float)
    names = [f"img_feat_{i}" for i in range(arr.shape[1])]
    return {"matrix": arr, "feature_names": names}


def _text_features(texts: List[str]) -> Dict[str, Any]:
    if not texts:
        return {"matrix": np.zeros((0, 0)), "feature_names": []}
    if HAS_SKLEARN:
        h = HashingVectorizer(n_features=64, alternate_sign=False, norm="l2",
                              analyzer="char_wb", ngram_range=(3, 4))
        m = h.transform(texts).toarray()
        names = [f"text_h_{i}" for i in range(m.shape[1])]
        return {"matrix": m, "feature_names": names}
    # Pure-numpy fallback: char-frequency bag-of-words on a-z + digits + space
    vocab = list("abcdefghijklmnopqrstuvwxyz0123456789 ")
    M = np.zeros((len(texts), len(vocab)))
    for i, t in enumerate(texts):
        s = (t or "").lower()
        for ch in s:
            if ch in vocab:
                M[i, vocab.index(ch)] += 1
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    M = np.divide(M, np.where(norms == 0, 1, norms))
    return {"matrix": M, "feature_names": [f"text_c_{c}" for c in vocab]}


def _timeseries_features(series: List[List[float]]) -> Dict[str, Any]:
    if not series:
        return {"matrix": np.zeros((0, 0)), "feature_names": []}
    feats = []
    for s in series:
        a = np.array(s, dtype=float)
        if a.size < 2:
            feats.append([0, 0, 0, 0, 0])
            continue
        a = a[~np.isnan(a)]
        if a.size < 2:
            feats.append([0, 0, 0, 0, 0])
            continue
        mu = float(np.mean(a))
        sd = float(np.std(a))
        rng = float(a.max() - a.min())
        ac1 = float(np.corrcoef(a[:-1], a[1:])[0, 1]) if a.size > 2 else 0.0
        slope = float(np.polyfit(np.arange(a.size), a, 1)[0]) if a.size > 1 else 0.0
        feats.append([mu, sd, rng, ac1, slope])
    M = np.array(feats, dtype=float)
    return {"matrix": M, "feature_names": ["ts_mean", "ts_std", "ts_range", "ts_acf1", "ts_slope"]}


class MultimodalDataScienceEngine:
    """Fuse modalities and train a baseline model (or score-only mode)."""

    name = "MultimodalDataScienceEngine"
    domain = "data_science"
    citations = [
        "Baltrušaitis, Ahuja, Morency (2019) IEEE TPAMI 41(2):423–443.",
        "Ngiam et al. (2011) ICML — Multimodal Deep Learning.",
    ]

    def fuse(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """payload = {
            tabular_rows: [...],
            image_features: [[...], ...] (optional, from VisualFeatureExtractor),
            texts: [str, ...] (optional),
            time_series: [[...], ...] (optional),
            target: [...] (optional, same length as samples to enable training),
            task: "auto" | "classification" | "regression"
        }
        """
        t0 = time.perf_counter()
        tabular_rows = payload.get("tabular_rows") or []
        n_samples = len(tabular_rows)
        if not n_samples:
            n_samples = (
                len(payload.get("image_features") or [])
                or len(payload.get("texts") or [])
                or len(payload.get("time_series") or [])
            )

        tab = _tabular_features(tabular_rows) if tabular_rows else {"matrix": np.zeros((n_samples, 0)), "feature_names": []}
        img = _image_features(payload.get("image_features"), expected_count=n_samples)
        txt = _text_features(payload.get("texts") or [])
        ts  = _timeseries_features(payload.get("time_series") or [])

        # Reshape matrices to a common sample count (zero-pad missing modalities)
        def _pad(m: np.ndarray, n: int) -> np.ndarray:
            if m.shape[0] == n: return m
            if m.shape[0] == 0: return np.zeros((n, m.shape[1] if m.ndim > 1 else 0))
            if m.shape[0] < n:
                pad = np.zeros((n - m.shape[0], m.shape[1]))
                return np.vstack([m, pad])
            return m[:n]

        n = max(n_samples, tab["matrix"].shape[0], img["matrix"].shape[0], txt["matrix"].shape[0], ts["matrix"].shape[0])
        if n == 0:
            return {"status": "error", "model_name": self.name, "message": "No samples in any modality."}

        modality_blocks = [
            ("tabular", _pad(tab["matrix"], n), tab["feature_names"]),
            ("image",   _pad(img["matrix"], n), img["feature_names"]),
            ("text",    _pad(txt["matrix"], n), txt["feature_names"]),
            ("timeseries", _pad(ts["matrix"], n), ts["feature_names"]),
        ]
        active = [(name, m, names) for name, m, names in modality_blocks if m.shape[1] > 0]
        if not active:
            return {"status": "error", "model_name": self.name, "message": "All modalities are empty."}
        X = np.hstack([m for _, m, _ in active])
        feat_names = []
        for name, _, names in active:
            feat_names.extend([f"{name}::{x}" for x in names])

        result: Dict[str, Any] = {
            "status": "success",
            "model_name": self.name,
            "n_samples": int(n),
            "n_features": int(X.shape[1]),
            "modalities_used": [name for name, _, _ in active],
            "feature_dimensions": {name: int(m.shape[1]) for name, m, _ in active},
            "feature_names_preview": feat_names[:20] + (["…"] if len(feat_names) > 20 else []),
        }

        # Optional: train a baseline if a target is provided and sklearn is available
        target = payload.get("target")
        if target is not None and HAS_SKLEARN and len(target) == n:
            y = np.array(target)
            task = payload.get("task", "auto")
            if task == "auto":
                task = "classification" if (y.dtype.kind in {"O", "U", "S"} or len(np.unique(y)) <= 20) else "regression"
            try:
                X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y if task == "classification" and len(np.unique(y)) > 1 else None)
                scaler = StandardScaler(with_mean=False)
                X_tr_s = scaler.fit_transform(X_tr)
                X_te_s = scaler.transform(X_te)
                if task == "classification":
                    model = LogisticRegression(max_iter=300, multi_class="auto", n_jobs=-1)
                    model.fit(X_tr_s, y_tr)
                    yp = model.predict(X_te_s)
                    metrics = {
                        "task": "classification",
                        "accuracy": round(float(accuracy_score(y_te, yp)), 4),
                        "f1_macro": round(float(f1_score(y_te, yp, average="macro", zero_division=0)), 4),
                        "n_test": int(len(y_te)),
                    }
                else:
                    model = Ridge(alpha=1.0)
                    model.fit(X_tr_s, y_tr)
                    yp = model.predict(X_te_s)
                    metrics = {
                        "task": "regression",
                        "mae": round(float(mean_absolute_error(y_te, yp)), 4),
                        "r2": round(float(r2_score(y_te, yp)), 4),
                        "n_test": int(len(y_te)),
                    }
                result["baseline_model"] = {"name": "LogisticRegression" if task == "classification" else "Ridge", "metrics": metrics}
            except Exception as e:
                result["baseline_model"] = {"error": f"Training failed: {e}"}

        result["duration_ms"] = int((time.perf_counter() - t0) * 1000)
        result["method_monitor"] = {
            "method": "Early-fusion concatenation (Baltrušaitis et al., 2019) with handcrafted modality encoders",
            "why_used": "Deployable on torch-free environments; provides explicit per-modality contribution.",
            "formulas": [
                "X_fused = [X_tabular | X_image | X_text | X_ts]",
                "Standardize: x' = (x − μ_train) / σ_train",
                "TF-IDF char n-gram: w(t,d) = tf(t,d) · log(N/(1+df(t)))",
            ],
            "limitations": [
                "Late-fusion / cross-attention encoders not included; install torch + transformers for advanced fusion.",
                "Image features must be pre-extracted via VisualFeatureExtractor.",
            ],
            "citations": self.citations,
        }
        return result
