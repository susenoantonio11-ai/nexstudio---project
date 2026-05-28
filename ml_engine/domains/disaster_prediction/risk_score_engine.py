"""
RISK SCORE ENGINE
=================

Mengkombinasikan empat komponen risiko menjadi skor gabungan 0..1.

Skor Risiko = f(Hazard, Exposure, Vulnerability, Model Probability)

Formulasi mengikuti kerangka UNDRR (UN Office for Disaster Risk Reduction):
    Risk = Hazard x Exposure x Vulnerability / Capacity

Implementasi praktis menggunakan kombinasi tertimbang dengan opsi
agregasi geometric mean atau arithmetic mean.

Sitasi:
    UNISDR (2009). Terminology on Disaster Risk Reduction.
    Cardona dkk (2012). Determinants of Risk: Exposure and Vulnerability.
    IPCC AR6 WG2 (2022). Climate Change 2022: Impacts, Adaptation, Vulnerability.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Literal
import math


@dataclass
class RiskComponents:
    """
    Empat komponen risiko, masing-masing 0..1.

    hazard: intensitas/probabilitas fenomena alam (mis. magnitude gempa
        ternormalisasi, curah hujan ekstrem ternormalisasi).
    exposure: nilai/aset/populasi yang terpapar (mis. densitas penduduk,
        nilai infrastruktur dalam radius dampak).
    vulnerability: kerentanan struktural & sosial (mis. kualitas bangunan,
        akses evakuasi, indeks kemiskinan).
    model_probability: probabilitas event dari model ML (LSTM/XGBoost).
    """
    hazard: float = 0.0
    exposure: float = 0.0
    vulnerability: float = 0.0
    model_probability: float = 0.0

    def clamp(self) -> "RiskComponents":
        return RiskComponents(
            hazard=max(0.0, min(1.0, float(self.hazard))),
            exposure=max(0.0, min(1.0, float(self.exposure))),
            vulnerability=max(0.0, min(1.0, float(self.vulnerability))),
            model_probability=max(0.0, min(1.0, float(self.model_probability))),
        )

    def to_dict(self) -> Dict:
        return {k: round(v, 4) for k, v in asdict(self).items()}


@dataclass
class RiskAssessment:
    """Hasil agregasi risiko."""
    composite_risk: float
    components: RiskComponents
    weights: Dict[str, float]
    aggregation: str
    explanation: str
    contributors: List[Dict] = field(default_factory=list)
    timestamp: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "composite_risk": round(self.composite_risk, 4),
            "components": self.components.to_dict(),
            "weights": {k: round(v, 4) for k, v in self.weights.items()},
            "aggregation": self.aggregation,
            "explanation": self.explanation,
            "contributors": self.contributors,
            "timestamp": self.timestamp,
        }


class RiskScoreEngine:
    """
    Mesin skoring risiko gabungan.

    Mendukung dua mode agregasi:
    - 'arithmetic': weighted mean. Cocok jika komponen saling kompensasi.
    - 'geometric': weighted geometric mean. Cocok jika satu komponen nol
      harus menurunkan risiko drastis (sesuai prinsip UNISDR multiplicative).
    """

    DEFAULT_WEIGHTS = {
        "hazard": 0.30,
        "exposure": 0.25,
        "vulnerability": 0.20,
        "model_probability": 0.25,
    }

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        aggregation: Literal["arithmetic", "geometric"] = "arithmetic",
    ) -> None:
        self.weights = weights or dict(self.DEFAULT_WEIGHTS)
        # normalisasi bobot supaya jumlah = 1
        s = sum(self.weights.values())
        if s <= 0:
            raise ValueError("Total bobot harus > 0")
        self.weights = {k: v / s for k, v in self.weights.items()}
        if aggregation not in ("arithmetic", "geometric"):
            raise ValueError("aggregation harus 'arithmetic' atau 'geometric'")
        self.aggregation = aggregation

    def compute(self, components: RiskComponents) -> RiskAssessment:
        comp = components.clamp()
        cdict = asdict(comp)

        if self.aggregation == "arithmetic":
            score = sum(self.weights[k] * cdict[k] for k in self.weights)
        else:  # geometric weighted
            # tambah epsilon agar tidak nol murni
            eps = 1e-6
            log_sum = sum(
                self.weights[k] * math.log(cdict[k] + eps)
                for k in self.weights
            )
            score = math.exp(log_sum)

        score = max(0.0, min(1.0, score))

        contributors = sorted(
            [
                {
                    "component": k,
                    "value": round(cdict[k], 4),
                    "weight": round(self.weights[k], 4),
                    "contribution": round(self.weights[k] * cdict[k], 4),
                }
                for k in self.weights
            ],
            key=lambda x: x["contribution"],
            reverse=True,
        )

        explanation = self._build_explanation(
            score, comp, contributors
        )

        return RiskAssessment(
            composite_risk=score,
            components=comp,
            weights=dict(self.weights),
            aggregation=self.aggregation,
            explanation=explanation,
            contributors=contributors,
        )

    def _build_explanation(
        self,
        score: float,
        comp: RiskComponents,
        contributors: List[Dict],
    ) -> str:
        top = contributors[0]
        agg_text = (
            "rata-rata tertimbang (arithmetic)" if self.aggregation == "arithmetic"
            else "rata-rata geometrik tertimbang (geometric)"
        )
        return (
            f"Skor risiko komposit = {score:.3f}. Dihitung dengan "
            f"{agg_text} dari empat komponen UNISDR: hazard "
            f"({comp.hazard:.2f}), exposure ({comp.exposure:.2f}), "
            f"vulnerability ({comp.vulnerability:.2f}), dan probabilitas "
            f"model ({comp.model_probability:.2f}). Kontributor terbesar "
            f"adalah {top['component']} dengan kontribusi "
            f"{top['contribution']:.3f}. Kerangka mengikuti "
            f"UNISDR (2009) dan IPCC AR6 WG2 (2022)."
        )

    def update_weights(self, new_weights: Dict[str, float]) -> None:
        s = sum(new_weights.values())
        if s <= 0:
            raise ValueError("Total bobot harus > 0")
        self.weights = {k: v / s for k, v in new_weights.items()}

    def sensitivity_analysis(
        self, components: RiskComponents, delta: float = 0.10
    ) -> List[Dict]:
        """Hitung perubahan skor jika tiap komponen naik delta."""
        base = self.compute(components).composite_risk
        results = []
        cdict = asdict(components.clamp())
        for k in self.weights:
            perturbed = dict(cdict)
            perturbed[k] = max(0.0, min(1.0, perturbed[k] + delta))
            new_score = self.compute(RiskComponents(**perturbed)).composite_risk
            results.append({
                "component": k,
                "delta_input": delta,
                "delta_output": round(new_score - base, 4),
                "elasticity": round(
                    (new_score - base) / max(delta, 1e-9), 4
                ),
            })
        return sorted(results, key=lambda x: abs(x["elasticity"]), reverse=True)
