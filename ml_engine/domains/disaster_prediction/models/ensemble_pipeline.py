"""
HYBRID ENSEMBLE PIPELINE
========================

Menggabungkan TemporalLSTM (forecast time-series), GeospatialXGBoost
(klasifikasi spatial), dan BayesianRisk (integrasi evidence) menjadi satu
probabilitas event akhir.

Sitasi:
    Dietterich (2000). Ensemble Methods in Machine Learning. Springer.
    Caruana dkk (2004). Ensemble selection from libraries of models. ICML.
    Sagi & Rokach (2018). Ensemble learning: A survey. WIREs Data Mining.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence
import math

from .lstm_temporal import TemporalLSTMModel, TemporalForecast
from .xgboost_geospatial import GeospatialXGBoostModel
from .bayesian_risk import BayesianRiskModel


@dataclass
class EnsemblePrediction:
    final_probability: float
    components: Dict[str, float]
    weights: Dict[str, float]
    aggregation: str
    confidence: float
    explanation: str

    def to_dict(self) -> Dict:
        return {
            "final_probability": round(self.final_probability, 4),
            "components": {k: round(v, 4) for k, v in self.components.items()},
            "weights": {k: round(v, 4) for k, v in self.weights.items()},
            "aggregation": self.aggregation,
            "confidence": round(self.confidence, 4),
            "explanation": self.explanation,
        }


class HybridEnsemblePipeline:
    """
    Soft-voting ensemble dengan bobot yang bisa di-tune.

    Args:
        weights: dict {'lstm': w1, 'xgboost': w2, 'bayesian': w3} bobot soft-vote
        aggregation: 'mean' | 'weighted_mean' | 'logit_mean'
    """

    def __init__(
        self,
        lstm: Optional[TemporalLSTMModel] = None,
        xgboost: Optional[GeospatialXGBoostModel] = None,
        bayesian: Optional[BayesianRiskModel] = None,
        weights: Optional[Dict[str, float]] = None,
        aggregation: str = "weighted_mean",
    ) -> None:
        self.lstm = lstm
        self.xgboost = xgboost
        self.bayesian = bayesian
        self.weights = weights or {"lstm": 0.3, "xgboost": 0.45, "bayesian": 0.25}
        s = sum(self.weights.values())
        if s <= 0:
            raise ValueError("Total bobot harus > 0")
        self.weights = {k: v / s for k, v in self.weights.items()}
        if aggregation not in ("mean", "weighted_mean", "logit_mean"):
            raise ValueError("aggregation tidak dikenal")
        self.aggregation = aggregation

    def predict(
        self,
        time_series_input: Optional[Sequence[float]] = None,
        spatial_features: Optional[Sequence[Sequence[float]]] = None,
    ) -> EnsemblePrediction:
        components: Dict[str, float] = {}
        used_weights: Dict[str, float] = {}

        # LSTM forecast diubah ke "probability of exceedance threshold" sederhana
        if self.lstm is not None and time_series_input is not None:
            fc = self.lstm.predict(time_series_input)
            mean_forecast = sum(fc.forecast) / max(1, len(fc.forecast))
            recent = list(time_series_input)[-min(7, len(time_series_input)):]
            baseline = sum(recent) / max(1, len(recent))
            # naive: probability proportional to forecast vs baseline ratio
            ratio = mean_forecast / max(1e-6, baseline)
            p_lstm = max(0.0, min(1.0, (ratio - 1.0)))
            components["lstm"] = p_lstm
            used_weights["lstm"] = self.weights.get("lstm", 0.0)

        # XGBoost: probabilitas event langsung
        if self.xgboost is not None and spatial_features:
            try:
                p_xgb = float(self.xgboost.predict_proba(spatial_features)[-1])
            except Exception:
                p_xgb = 0.0
            components["xgboost"] = p_xgb
            used_weights["xgboost"] = self.weights.get("xgboost", 0.0)

        # Bayesian posterior mean
        if self.bayesian is not None:
            belief = self.bayesian.get_belief()
            components["bayesian"] = belief.posterior_mean
            used_weights["bayesian"] = self.weights.get("bayesian", 0.0)

        if not components:
            return EnsemblePrediction(
                final_probability=0.0,
                components={},
                weights={},
                aggregation=self.aggregation,
                confidence=0.0,
                explanation="Tidak ada komponen aktif. Output default 0.",
            )

        # Re-normalize weights pada komponen aktif
        s = sum(used_weights.values())
        if s <= 0:
            used_weights = {k: 1.0 / len(components) for k in components}
        else:
            used_weights = {k: v / s for k, v in used_weights.items()}

        if self.aggregation == "mean":
            p_final = sum(components.values()) / len(components)
        elif self.aggregation == "weighted_mean":
            p_final = sum(used_weights[k] * components[k] for k in components)
        else:  # logit_mean: rata-rata di ruang logit, lebih sensitif terhadap p ekstrem
            eps = 1e-6
            logits = []
            for k, p in components.items():
                p = max(eps, min(1 - eps, p))
                logits.append(used_weights[k] * math.log(p / (1 - p)))
            z = sum(logits)
            p_final = 1.0 / (1.0 + math.exp(-z))

        # Confidence: 1 - varians antar komponen (semakin sepakat semakin tinggi)
        mean = sum(components.values()) / len(components)
        var = sum((v - mean) ** 2 for v in components.values()) / len(components)
        confidence = max(0.0, 1.0 - 4.0 * var)

        explanation = (
            f"Ensemble {len(components)} komponen via {self.aggregation}: "
            + ", ".join(f"{k}={v:.3f}" for k, v in components.items())
            + f". Probabilitas akhir = {p_final:.3f}, kepercayaan "
            f"(berdasarkan kesepakatan antar model) = {confidence:.3f}. "
            f"Pendekatan ensemble mengikuti Dietterich (2000)."
        )

        return EnsemblePrediction(
            final_probability=max(0.0, min(1.0, p_final)),
            components=components,
            weights=used_weights,
            aggregation=self.aggregation,
            confidence=confidence,
            explanation=explanation,
        )
