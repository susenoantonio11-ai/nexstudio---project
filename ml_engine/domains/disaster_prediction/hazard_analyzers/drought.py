"""
DROUGHT ANALYZER
================

Hitung indeks SPI (Standardized Precipitation Index) dan SPEI yang
disederhanakan untuk klasifikasi kekeringan.

Sitasi:
    McKee, Doesken, Kleist (1993). The relationship of drought frequency and
        duration to time scales. 8th Conference on Applied Climatology, AMS.
    Vicente-Serrano dkk (2010). A multiscalar drought index sensitive to global
        warming: the SPEI. Journal of Climate 23.
    Palmer (1965). Meteorological drought. Research Paper No. 45, US Weather
        Bureau.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Sequence
import math


@dataclass
class DroughtAssessment:
    timescale_months: int
    spi: float
    spei_proxy: float
    drought_class: str
    hazard_score: float
    explanation: str


class DroughtAnalyzer:
    """
    Hitung SPI dengan asumsi distribusi normal pada data presipitasi yang sudah
    distandardisasi. Untuk pipeline produksi gunakan fitting Gamma per stasiun.
    """

    def analyze(
        self,
        precipitation_history_mm: Sequence[float],
        temperature_c_history: Sequence[float] = (),
        timescale_months: int = 3,
    ) -> DroughtAssessment:
        n = len(precipitation_history_mm)
        if n < timescale_months + 1:
            return DroughtAssessment(
                timescale_months=timescale_months, spi=float("nan"),
                spei_proxy=float("nan"), drought_class="DATA_TIDAK_CUKUP",
                hazard_score=0.0,
                explanation="Riwayat presipitasi tidak cukup panjang untuk SPI.",
            )

        # Hitung agregat sliding window
        agg = []
        for i in range(timescale_months - 1, n):
            window = precipitation_history_mm[i - timescale_months + 1: i + 1]
            agg.append(sum(window))

        mean = sum(agg) / len(agg)
        var = sum((x - mean) ** 2 for x in agg) / max(1, len(agg) - 1)
        sd = math.sqrt(var) if var > 0 else 1e-9

        # SPI = (X - mean) / sd  (asumsi normal)
        spi = (agg[-1] - mean) / sd

        # SPEI proxy: koreksi defisit air dengan suhu tinggi
        spei = spi
        if temperature_c_history:
            t_recent = sum(temperature_c_history[-timescale_months:]) / timescale_months
            spei = spi - 0.05 * max(0.0, t_recent - 27.0)

        # Klasifikasi McKee 1993
        if spi >= 2.0:
            cls = "SANGAT_BASAH"
        elif spi >= 1.5:
            cls = "BASAH"
        elif spi >= 1.0:
            cls = "AGAK_BASAH"
        elif spi >= -1.0:
            cls = "NORMAL"
        elif spi >= -1.5:
            cls = "AGAK_KERING"
        elif spi >= -2.0:
            cls = "KERING"
        else:
            cls = "SANGAT_KERING"

        # Hazard hanya untuk sisi kering (spi negatif)
        hazard_score = max(0.0, min(1.0, -spi / 2.5)) if spi < 0 else 0.0

        explanation = (
            f"Time-scale {timescale_months} bulan. Agregat presipitasi terakhir "
            f"= {agg[-1]:.1f} mm vs rata-rata historis {mean:.1f} mm "
            f"(sd={sd:.1f}). SPI = {spi:.3f}, SPEI proxy = {spei:.3f}, "
            f"klasifikasi: {cls}. Aturan klasifikasi mengikuti McKee dkk (1993)."
        )

        return DroughtAssessment(
            timescale_months=timescale_months,
            spi=round(spi, 3),
            spei_proxy=round(spei, 3),
            drought_class=cls,
            hazard_score=hazard_score,
            explanation=explanation,
        )
