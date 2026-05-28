"""
CLIMATE RISK ANALYZER
=====================

Analisis tren iklim regional: anomali suhu, indeks ENSO (ONI proxy), dan
frekuensi event ekstrem.

Sitasi:
    Trenberth (1997). The Definition of El Nino. BAMS 78.
    IPCC AR6 WG1 (2021). Climate Change 2021: The Physical Science Basis.
    Hansen, Sato, Ruedy (2012). Perception of climate change. PNAS 109.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Sequence
import math


@dataclass
class ClimateRiskAssessment:
    n_years: int
    mean_temp_anomaly_c: float
    trend_per_decade_c: float
    enso_phase: str
    extremes_per_decade: float
    hazard_score: float
    explanation: str


class ClimateRiskAnalyzer:
    """
    Hitung anomali suhu vs baseline 1981-2010, regresi linier sederhana untuk
    trend, dan klasifikasi fase ENSO berdasarkan ONI proxy SST anomaly.
    """

    def analyze(
        self,
        temperature_history_c: Sequence[float],
        baseline_mean_c: float = 27.0,
        sst_nino34_anomaly_c: float = 0.0,
        years_per_record: float = 1.0,
    ) -> ClimateRiskAssessment:
        n = len(temperature_history_c)
        if n < 5:
            return ClimateRiskAssessment(
                n_years=n, mean_temp_anomaly_c=float("nan"),
                trend_per_decade_c=float("nan"), enso_phase="UNKNOWN",
                extremes_per_decade=0.0, hazard_score=0.0,
                explanation="Riwayat suhu terlalu pendek (minimal 5 record).",
            )

        anomalies = [float(t) - baseline_mean_c for t in temperature_history_c]
        mean_anom = sum(anomalies) / n

        # Regresi linier: y = a + b*x
        xs = [i * years_per_record for i in range(n)]
        x_mean = sum(xs) / n
        y_mean = mean_anom + baseline_mean_c
        ys = list(temperature_history_c)
        cov = sum((xs[i] - x_mean) * (ys[i] - y_mean) for i in range(n))
        var_x = sum((x - x_mean) ** 2 for x in xs)
        slope = cov / var_x if var_x > 0 else 0.0
        trend_decade = slope * 10.0

        # ENSO phase (ONI rule, NOAA): ambang +/- 0.5 C selama 3 bulan berturut
        if sst_nino34_anomaly_c >= 0.5:
            enso = "EL_NINO"
        elif sst_nino34_anomaly_c <= -0.5:
            enso = "LA_NINA"
        else:
            enso = "NETRAL"

        # Frekuensi extremes: anomali > 2*std
        sd = math.sqrt(sum((a - mean_anom) ** 2 for a in anomalies) / max(1, n - 1))
        extremes = sum(1 for a in anomalies if abs(a - mean_anom) > 2 * sd)
        extremes_decade = extremes / max(1.0, n * years_per_record / 10.0)

        # Hazard
        anom_score = max(0.0, min(1.0, mean_anom / 2.0))  # 2C anomaly -> 1.0
        trend_score = max(0.0, min(1.0, trend_decade / 0.5))  # 0.5C/decade -> 1.0
        enso_score = 0.7 if enso != "NETRAL" else 0.2
        hazard = 0.4 * anom_score + 0.4 * trend_score + 0.2 * enso_score

        explanation = (
            f"Riwayat {n} record. Anomali suhu rata-rata vs baseline "
            f"{baseline_mean_c:.1f} C = {mean_anom:.2f} C. Tren regresi linier "
            f"= {trend_decade:.3f} C/dekade. Fase ENSO (Trenberth 1997) = "
            f"{enso} berdasarkan ONI proxy {sst_nino34_anomaly_c:+.2f} C. "
            f"Frekuensi anomali > 2 sd = {extremes_decade:.2f} per dekade."
        )

        return ClimateRiskAssessment(
            n_years=n,
            mean_temp_anomaly_c=round(mean_anom, 3),
            trend_per_decade_c=round(trend_decade, 3),
            enso_phase=enso,
            extremes_per_decade=round(extremes_decade, 2),
            hazard_score=round(hazard, 3),
            explanation=explanation,
        )
