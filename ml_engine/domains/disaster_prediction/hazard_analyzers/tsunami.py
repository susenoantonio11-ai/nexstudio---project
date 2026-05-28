"""
TSUNAMI ANALYZER
================

Estimasi potensi tsunami dari parameter sumber gempa undersea.
Menggunakan kriteria Abe (1981) dan empirical Mw vs run-up.

Sitasi:
    Abe (1981). Magnitudes of large shallow earthquakes from 1904 to 1980.
        Physics of the Earth and Planetary Interiors 27.
    Murty & Loomis (1980). A new objective tsunami magnitude scale. Marine
        Geodesy.
    Iida (1963). Magnitude, energy and generation mechanisms of tsunamis.
    Pelinovsky & Mazova (1992). Exact analytical solutions of nonlinear problems
        of tsunami wave run-up.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
import math


@dataclass
class TsunamiAssessment:
    magnitude: float
    depth_km: float
    is_undersea: bool
    estimated_run_up_m: float
    arrival_time_minutes: Optional[float]
    tsunami_potential: str
    hazard_score: float
    explanation: str


class TsunamiAnalyzer:
    """
    Analyzer potensi tsunami dari sumber gempa.

    Aturan praktis BMKG-InaTEWS dan Tsunami Warning Centers:
    - M >= 6.5 dan kedalaman <= 60 km dan undersea -> potensi tsunami lokal
    - M >= 7.0 dan kedalaman <= 60 km dan undersea -> potensi tsunami signifikan
    - M >= 7.5 -> potensi tsunami regional
    """

    def analyze(
        self,
        magnitude: float,
        depth_km: float,
        is_undersea: bool = True,
        distance_to_coast_km: Optional[float] = None,
        water_depth_m: float = 1000.0,
    ) -> TsunamiAssessment:
        magnitude = float(magnitude)
        depth_km = float(depth_km)

        # Estimasi run-up empirik (Abe 1981, simplified):
        # log10(R) ~ 0.5 * Mw - 3.3
        if magnitude >= 6.0 and is_undersea:
            log_runup = 0.5 * magnitude - 3.3
            runup = max(0.0, 10 ** log_runup)
            # discount untuk gempa dalam (>60km) dan event darat
            if depth_km > 60:
                runup *= max(0.0, 1.0 - (depth_km - 60) / 100.0)
        else:
            runup = 0.0

        # Estimasi waktu tiba berdasarkan kecepatan gelombang dangkal
        # c = sqrt(g * h), g=9.81, h=water_depth (m)
        arrival_time = None
        if distance_to_coast_km is not None and water_depth_m > 0:
            c_ms = math.sqrt(9.81 * max(1.0, water_depth_m))  # m/s
            distance_m = distance_to_coast_km * 1000.0
            arrival_time = distance_m / c_ms / 60.0  # menit

        # Klasifikasi potensi
        if not is_undersea or magnitude < 6.5 or depth_km > 100:
            potential = "TIDAK_BERPOTENSI"
        elif magnitude < 7.0:
            potential = "BERPOTENSI_LOKAL"
        elif magnitude < 7.5:
            potential = "BERPOTENSI_SIGNIFIKAN"
        elif magnitude < 8.0:
            potential = "BERPOTENSI_REGIONAL"
        else:
            potential = "BERPOTENSI_TRANSOSEANIK"

        hazard_score = self._compute_hazard(magnitude, depth_km, is_undersea, runup)

        explanation = (
            f"Mw={magnitude:.1f}, kedalaman={depth_km:.1f} km, "
            f"{'undersea' if is_undersea else 'darat'}. "
            f"Estimasi run-up empiris (Abe 1981 yang disederhanakan) = "
            f"{runup:.2f} m. Potensi tsunami: {potential}. "
            f"Aturan ambang mengikuti pedoman InaTEWS BMKG (2012) dan "
            f"PTWC NOAA. Untuk peringatan operasional WAJIB mengacu pada "
            f"InaTEWS dan otoritas resmi."
        )

        return TsunamiAssessment(
            magnitude=magnitude,
            depth_km=depth_km,
            is_undersea=is_undersea,
            estimated_run_up_m=round(runup, 3),
            arrival_time_minutes=round(arrival_time, 2) if arrival_time else None,
            tsunami_potential=potential,
            hazard_score=hazard_score,
            explanation=explanation,
        )

    def _compute_hazard(
        self, m: float, d: float, undersea: bool, runup: float
    ) -> float:
        if not undersea:
            return 0.0
        if m < 6.0:
            return 0.0
        m_score = max(0.0, min(1.0, (m - 6.0) / 3.0))
        d_score = max(0.0, min(1.0, 1.0 - d / 200.0))
        r_score = max(0.0, min(1.0, runup / 20.0))
        return 0.5 * m_score + 0.2 * d_score + 0.3 * r_score
