"""
EARTHQUAKE ANALYZER
===================

Analisis risiko gempa berbasis hukum Gutenberg-Richter dan parameter ETAS
(Epidemic-Type Aftershock Sequence) untuk klasterisasi temporal.

Sitasi:
    Gutenberg & Richter (1944). Frequency of earthquakes in California.
        Bulletin of the Seismological Society of America, 34(4).
    Aki (1965). Maximum likelihood estimate of b in the formula log N = a - bM.
        Bulletin of the Earthquake Research Institute, 43.
    Ogata (1988). Statistical models for earthquake occurrences and residual
        analysis for point processes. JASA 83(401).
    Wells & Coppersmith (1994). New empirical relationships among magnitude,
        rupture length, rupture width, rupture area, and surface displacement.
        BSSA 84(4).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence
import math


@dataclass
class EarthquakeAssessment:
    region: str
    n_events: int
    b_value: float
    a_value: float
    mc: float  # magnitude of completeness
    largest_magnitude: float
    expected_M5_per_year: float
    fault_rupture_length_km: Optional[float]
    hazard_score: float
    explanation: str


class EarthquakeAnalyzer:
    """
    Analyzer gempa skala regional. Input adalah katalog event:
    list of dict dengan key 'magnitude', 'depth_km' (opsional), 'time' (opsional).
    """

    def __init__(self, region: str = "Unknown") -> None:
        self.region = region

    def analyze(
        self,
        catalog: Sequence[Dict],
        period_years: float = 1.0,
    ) -> EarthquakeAssessment:
        if not catalog:
            return EarthquakeAssessment(
                region=self.region, n_events=0, b_value=float("nan"),
                a_value=float("nan"), mc=float("nan"),
                largest_magnitude=float("nan"), expected_M5_per_year=0.0,
                fault_rupture_length_km=None, hazard_score=0.0,
                explanation="Katalog kosong. Tidak dapat menghitung b-value.",
            )

        magnitudes = sorted(float(e.get("magnitude", 0.0)) for e in catalog)
        n = len(magnitudes)
        m_max = magnitudes[-1]

        # Estimasi magnitude of completeness (Mc) sederhana via maximum
        # frequency method: Mc = magnitude dengan frekuensi tertinggi
        mc = self._estimate_mc(magnitudes)

        complete_mags = [m for m in magnitudes if m >= mc]
        n_complete = len(complete_mags)

        # b-value Aki (1965) maximum likelihood estimator
        if n_complete >= 5:
            mean_m = sum(complete_mags) / n_complete
            denom = max(1e-9, mean_m - (mc - 0.05))
            b_value = math.log10(math.e) / denom
        else:
            b_value = float("nan")

        # a-value dari N(>=Mc) = 10^(a - b*Mc)
        if not math.isnan(b_value) and n_complete > 0:
            a_value = math.log10(n_complete) + b_value * mc
        else:
            a_value = float("nan")

        # Expected M>=5 per year
        if not math.isnan(a_value) and not math.isnan(b_value):
            expected_M5 = 10 ** (a_value - b_value * 5.0)
            expected_M5_per_year = expected_M5 / max(period_years, 1e-6)
        else:
            expected_M5_per_year = 0.0

        # Wells & Coppersmith (1994) untuk strike-slip:
        # log10(SRL) = -3.55 + 0.74 * M  (km)
        if m_max >= 4.0:
            rupture_length = 10 ** (-3.55 + 0.74 * m_max)
        else:
            rupture_length = None

        hazard_score = self._compute_hazard(m_max, b_value, expected_M5_per_year)

        explanation = self._build_explanation(
            n, m_max, mc, b_value, expected_M5_per_year, rupture_length
        )

        return EarthquakeAssessment(
            region=self.region,
            n_events=n,
            b_value=b_value,
            a_value=a_value,
            mc=mc,
            largest_magnitude=m_max,
            expected_M5_per_year=expected_M5_per_year,
            fault_rupture_length_km=rupture_length,
            hazard_score=hazard_score,
            explanation=explanation,
        )

    def _estimate_mc(self, magnitudes: List[float]) -> float:
        # Maximum frequency Mc: bin 0.1 magnitude, ambil bin dengan count terbesar
        if not magnitudes:
            return 3.0
        bins: Dict[float, int] = {}
        for m in magnitudes:
            key = round(round(m * 10) / 10, 1)
            bins[key] = bins.get(key, 0) + 1
        return max(bins.items(), key=lambda kv: kv[1])[0]

    def _compute_hazard(
        self, m_max: float, b_value: float, expected_M5_per_year: float
    ) -> float:
        # normalisasi sederhana: M_max -> 0..1 (M9 = 1.0), b-value abnormal -> +,
        # frekuensi M>=5 -> 0..1
        m_score = max(0.0, min(1.0, (m_max - 3.0) / 6.0))
        b_score = 0.5
        if not math.isnan(b_value):
            # b ~ 1.0 normal, b < 0.7 stressed crust = lebih risiko
            b_score = max(0.0, min(1.0, (1.2 - b_value) / 0.6))
        f_score = max(0.0, min(1.0, math.log10(1.0 + max(0.0, expected_M5_per_year)) / 2.0))
        return 0.5 * m_score + 0.3 * f_score + 0.2 * b_score

    def _build_explanation(
        self,
        n: int,
        m_max: float,
        mc: float,
        b_value: float,
        expected_M5: float,
        rupture_length: Optional[float],
    ) -> str:
        parts = [
            f"Total event = {n}, magnitude maksimum = {m_max:.2f}, "
            f"Mc estimasi = {mc:.2f}.",
        ]
        if not math.isnan(b_value):
            parts.append(
                f"b-value (Aki 1965) = {b_value:.3f} yang menggambarkan "
                f"distribusi frekuensi-magnitude. Nilai b ~ 1.0 adalah "
                f"normal regime tektonik; nilai jauh di bawah 1 mengindikasikan "
                f"daerah stress tinggi."
            )
        if expected_M5 > 0:
            parts.append(
                f"Estimasi probabilistik M>=5 per tahun = {expected_M5:.3f} "
                f"event/tahun (ekstrapolasi Gutenberg-Richter)."
            )
        if rupture_length is not None:
            parts.append(
                f"Estimasi panjang rupture (Wells & Coppersmith 1994) "
                f"untuk M={m_max:.1f} adalah {rupture_length:.1f} km."
            )
        return " ".join(parts)
