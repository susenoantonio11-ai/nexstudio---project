"""
WARNING LEVEL CLASSIFIER
========================

Mengubah skor risiko (0..1) menjadi level peringatan 5-tingkat.
Skala diadaptasi dari NOAA Storm Prediction Center (Convective Outlook),
USGS PAGER (Prompt Assessment of Global Earthquakes for Response),
dan BMKG Sistem Peringatan Dini Tsunami.

Sitasi:
    USGS (2010). PAGER: Rapid Assessment of an Earthquake's Impact.
    NOAA SPC. Convective Outlook Categories.
    BMKG (2012). Pedoman Operasional InaTEWS.

PENTING: Output sistem ini bersifat ADVISORY untuk RESEARCH.
Sistem ini TIDAK menggantikan peringatan resmi BMKG, BNPB, atau USGS.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


WARNING_DISCLAIMER = (
    "DISCLAIMER: Output ini adalah hasil analisis platform research "
    "Nexlytics dan TIDAK menggantikan peringatan resmi dari BMKG, BNPB, "
    "USGS, PVMBG, atau lembaga peringatan dini resmi lainnya. "
    "Untuk keputusan evakuasi atau tindakan darurat, selalu mengacu "
    "pada otoritas resmi. Platform ini dimaksudkan untuk validasi "
    "model, edukasi, dan studi banding metode prediksi."
)


class WarningLevel(str, Enum):
    """Lima tingkat peringatan, urutan menaik."""
    NORMAL = "NORMAL"
    ADVISORY = "ADVISORY"
    WATCH = "WATCH"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


# Threshold default skor risiko 0..1 untuk masing-masing level
DEFAULT_THRESHOLDS: Dict[WarningLevel, float] = {
    WarningLevel.NORMAL: 0.0,
    WarningLevel.ADVISORY: 0.30,
    WarningLevel.WATCH: 0.50,
    WarningLevel.WARNING: 0.70,
    WarningLevel.CRITICAL: 0.85,
}


# Aksi yang disarankan per level (advisory only, bukan instruksi resmi)
RECOMMENDED_ACTIONS: Dict[WarningLevel, List[str]] = {
    WarningLevel.NORMAL: [
        "Lanjutkan aktivitas normal.",
        "Pantau berkala status dari sumber resmi (BMKG, BNPB).",
    ],
    WarningLevel.ADVISORY: [
        "Tingkatkan kesadaran dan pantau update resmi.",
        "Periksa kesiapan kit darurat keluarga.",
    ],
    WarningLevel.WATCH: [
        "Bersiap untuk kemungkinan tindakan evakuasi.",
        "Pantau saluran resmi BMKG/BNPB secara intensif.",
        "Pastikan jalur evakuasi tersedia.",
    ],
    WarningLevel.WARNING: [
        "Ikuti instruksi otoritas resmi tanpa menunggu.",
        "Persiapkan evakuasi jika berada di zona rawan.",
        "Hubungi keluarga dan koordinator komunitas.",
    ],
    WarningLevel.CRITICAL: [
        "Ikuti perintah evakuasi resmi segera.",
        "Hindari area berbahaya yang ditetapkan otoritas.",
        "Aktivasi rencana kontinjensi komunitas.",
    ],
}


@dataclass
class WarningResult:
    """Hasil klasifikasi level peringatan."""
    level: WarningLevel
    risk_score: float
    confidence: float
    threshold_used: float
    distance_to_next: float
    recommended_actions: List[str]
    disclaimer: str = WARNING_DISCLAIMER
    explanation: str = ""

    def to_dict(self) -> Dict:
        return {
            "level": self.level.value,
            "risk_score": round(self.risk_score, 4),
            "confidence": round(self.confidence, 4),
            "threshold_used": round(self.threshold_used, 4),
            "distance_to_next": round(self.distance_to_next, 4),
            "recommended_actions": self.recommended_actions,
            "explanation": self.explanation,
            "disclaimer": self.disclaimer,
        }


class WarningLevelClassifier:
    """
    Mengkonversi skor risiko numerik menjadi level peringatan kategorikal.

    Default threshold mengikuti pendekatan PAGER alert level (Earle et al, 2009)
    yang menggunakan probabilitas dampak untuk memilih kategori warna.
    """

    def __init__(
        self,
        thresholds: Optional[Dict[WarningLevel, float]] = None,
    ) -> None:
        self.thresholds = thresholds or dict(DEFAULT_THRESHOLDS)
        self._sorted_levels = sorted(
            self.thresholds.items(), key=lambda kv: kv[1]
        )

    def classify(
        self,
        risk_score: float,
        confidence: float = 1.0,
    ) -> WarningResult:
        """
        Args:
            risk_score: skor risiko 0..1 (semakin besar semakin parah)
            confidence: keyakinan model 0..1

        Returns:
            WarningResult berisi level, aksi yang disarankan, disclaimer.
        """
        risk_score = max(0.0, min(1.0, float(risk_score)))
        confidence = max(0.0, min(1.0, float(confidence)))

        chosen = WarningLevel.NORMAL
        chosen_threshold = 0.0
        for level, thr in self._sorted_levels:
            if risk_score >= thr:
                chosen = level
                chosen_threshold = thr

        # jarak ke level berikutnya
        distance_to_next = 1.0 - risk_score
        for level, thr in self._sorted_levels:
            if thr > chosen_threshold:
                distance_to_next = thr - risk_score
                break

        explanation = self._build_explanation(
            risk_score, chosen, chosen_threshold, distance_to_next
        )

        return WarningResult(
            level=chosen,
            risk_score=risk_score,
            confidence=confidence,
            threshold_used=chosen_threshold,
            distance_to_next=max(0.0, distance_to_next),
            recommended_actions=list(RECOMMENDED_ACTIONS[chosen]),
            explanation=explanation,
        )

    def _build_explanation(
        self,
        risk_score: float,
        level: WarningLevel,
        threshold: float,
        distance: float,
    ) -> str:
        return (
            f"Skor risiko gabungan = {risk_score:.3f}. Threshold untuk "
            f"level {level.value} adalah {threshold:.2f}. Jarak ke "
            f"level berikutnya = {distance:.3f}. Klasifikasi mengikuti "
            f"pendekatan PAGER USGS (Earle dkk, 2009) yang memetakan "
            f"probabilitas dampak ke kategori peringatan diskrit."
        )

    def batch_classify(
        self,
        risk_scores: List[float],
        confidences: Optional[List[float]] = None,
    ) -> List[WarningResult]:
        if confidences is None:
            confidences = [1.0] * len(risk_scores)
        return [
            self.classify(rs, c)
            for rs, c in zip(risk_scores, confidences)
        ]

    def summary_stats(self, results: List[WarningResult]) -> Dict:
        counts = {level.value: 0 for level in WarningLevel}
        for r in results:
            counts[r.level.value] += 1
        n = max(1, len(results))
        return {
            "total": len(results),
            "by_level": counts,
            "by_level_pct": {k: round(v / n * 100, 2) for k, v in counts.items()},
            "max_risk": max((r.risk_score for r in results), default=0.0),
            "mean_risk": sum(r.risk_score for r in results) / n,
        }
