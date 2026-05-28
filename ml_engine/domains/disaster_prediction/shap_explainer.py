"""
SHAP EXPLAINER (with graceful degradation)
==========================================

Pembungkus tipis untuk SHAP (SHapley Additive exPlanations, Lundberg & Lee 2017).
Jika library shap belum terinstal, modul fallback ke perturbation-based
attribution sederhana sehingga pipeline tetap berjalan untuk demo/research.

Sitasi:
    Lundberg & Lee (2017). A Unified Approach to Interpreting Model Predictions.
        NeurIPS 30.
    Strumbelj & Kononenko (2014). Explaining prediction models and individual
        predictions with feature contributions.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence
import math
import random

try:  # graceful degradation
    import shap  # type: ignore
    HAVE_SHAP = True
except Exception:
    HAVE_SHAP = False

try:
    import numpy as np  # type: ignore
    HAVE_NUMPY = True
except Exception:
    HAVE_NUMPY = False


@dataclass
class ExplanationReport:
    backend: str
    feature_names: List[str]
    base_value: float
    contributions: Dict[str, float]
    ranked_contributors: List[Dict]
    note: str = ""

    def to_dict(self) -> Dict:
        return {
            "backend": self.backend,
            "feature_names": self.feature_names,
            "base_value": round(self.base_value, 6),
            "contributions": {k: round(v, 6) for k, v in self.contributions.items()},
            "ranked_contributors": self.ranked_contributors,
            "note": self.note,
        }


class SHAPExplainer:
    """
    Wrapper terhadap SHAP.
    Jika shap tidak tersedia, gunakan permutation-importance lokal.

    Args:
        predict_fn: callable yang menerima list-of-records (list[dict] atau
            list[list]) dan mengembalikan probabilitas atau prediksi numerik.
        feature_names: daftar nama fitur (urutan harus konsisten dengan vector input).
        background_data: data latar untuk shap.Explainer; jika None akan
            digunakan satu sample default.
    """

    def __init__(
        self,
        predict_fn: Callable[[Any], Sequence[float]],
        feature_names: List[str],
        background_data: Optional[Sequence[Sequence[float]]] = None,
    ) -> None:
        self.predict_fn = predict_fn
        self.feature_names = list(feature_names)
        self.background_data = list(background_data) if background_data else None
        self._shap_explainer = None
        if HAVE_SHAP and HAVE_NUMPY:
            try:
                bg = self.background_data
                if bg is None:
                    bg = [[0.0] * len(self.feature_names)]
                self._shap_explainer = shap.Explainer(
                    self._np_predict, np.array(bg, dtype=float)
                )
            except Exception:
                self._shap_explainer = None

    def explain(self, instance: Sequence[float]) -> ExplanationReport:
        instance = [float(v) for v in instance]
        if len(instance) != len(self.feature_names):
            raise ValueError("Panjang instance harus sama dengan feature_names")

        if self._shap_explainer is not None and HAVE_NUMPY:
            return self._explain_with_shap(instance)
        return self._explain_with_permutation(instance)

    def _np_predict(self, X):
        # Bridge untuk numpy array shap
        records = [list(map(float, row)) for row in X]
        preds = self.predict_fn(records)
        if HAVE_NUMPY:
            return np.array(preds, dtype=float)
        return preds

    def _explain_with_shap(self, instance: List[float]) -> ExplanationReport:
        try:
            import numpy as _np
            arr = _np.array([instance], dtype=float)
            shap_values = self._shap_explainer(arr)
            # shap_values.values shape (1, n_features)
            vals = shap_values.values[0].tolist()
            base = float(getattr(shap_values, "base_values", _np.array([0.0]))[0])
        except Exception:
            return self._explain_with_permutation(instance)

        contributions = {n: float(v) for n, v in zip(self.feature_names, vals)}
        ranked = self._rank(contributions, instance)
        return ExplanationReport(
            backend="shap",
            feature_names=self.feature_names,
            base_value=base,
            contributions=contributions,
            ranked_contributors=ranked,
            note="Nilai SHAP mengikuti Lundberg & Lee (2017).",
        )

    def _explain_with_permutation(self, instance: List[float]) -> ExplanationReport:
        # Permutation-based attribution sederhana
        if self.background_data:
            bg = self.background_data[0]
        else:
            bg = [0.0] * len(self.feature_names)

        full_pred = float(self.predict_fn([instance])[0])
        base_pred = float(self.predict_fn([list(bg)])[0])

        contributions = {}
        for i, name in enumerate(self.feature_names):
            perturbed = list(instance)
            perturbed[i] = bg[i]
            p = float(self.predict_fn([perturbed])[0])
            contributions[name] = full_pred - p

        # Normalisasi agar jumlah kontribusi mendekati selisih
        total_attr = sum(contributions.values())
        target_total = full_pred - base_pred
        if abs(total_attr) > 1e-12:
            scale = target_total / total_attr
            contributions = {k: v * scale for k, v in contributions.items()}

        ranked = self._rank(contributions, instance)
        return ExplanationReport(
            backend="permutation_fallback",
            feature_names=self.feature_names,
            base_value=base_pred,
            contributions=contributions,
            ranked_contributors=ranked,
            note=(
                "Library shap tidak tersedia. Menggunakan permutation-based "
                "attribution mengikuti Strumbelj & Kononenko (2014)."
            ),
        )

    def _rank(self, contributions: Dict[str, float], instance: List[float]) -> List[Dict]:
        items = []
        for i, name in enumerate(self.feature_names):
            v = contributions.get(name, 0.0)
            items.append({
                "feature": name,
                "value": instance[i],
                "contribution": round(v, 6),
                "abs_contribution": round(abs(v), 6),
                "direction": "increases_risk" if v > 0 else ("decreases_risk" if v < 0 else "neutral"),
            })
        return sorted(items, key=lambda x: x["abs_contribution"], reverse=True)
