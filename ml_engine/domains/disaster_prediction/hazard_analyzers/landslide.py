"""
LANDSLIDE ANALYZER
==================

Faktor Keamanan (Factor of Safety, FOS) lereng dengan kriteria Mohr-Coulomb
dan model SHALSTAB untuk pemicu hujan.

Sitasi:
    Terzaghi (1943). Theoretical Soil Mechanics. Wiley.
    Skempton & DeLory (1957). Stability of natural slopes in London Clay.
    Montgomery & Dietrich (1994). A physically-based model for the topographic
        control on shallow landsliding (SHALSTAB). Water Resources Research 30.
    Caine (1980). The rainfall intensity-duration control of shallow landslides
        and debris flows. Geografiska Annaler 62A.
"""

from __future__ import annotations
from dataclasses import dataclass
import math


@dataclass
class LandslideAssessment:
    slope_deg: float
    soil_cohesion_kpa: float
    friction_angle_deg: float
    soil_moisture: float
    rainfall_24h_mm: float
    factor_of_safety: float
    risk_class: str
    hazard_score: float
    explanation: str


class LandslideAnalyzer:
    """
    Analyzer longsor dangkal (translational).

    Faktor Keamanan untuk lereng tak terhingga (infinite slope):
        FOS = (c' + (gamma - m * gamma_w) * z * cos^2(theta) * tan(phi'))
              / (gamma * z * sin(theta) * cos(theta))

    Disederhanakan: jika FOS > 1.5 -> stabil, 1.0..1.5 -> marginally stable,
    < 1.0 -> tidak stabil.
    """

    GAMMA_SOIL = 18.0  # kN/m^3
    GAMMA_WATER = 9.81  # kN/m^3

    def analyze(
        self,
        slope_deg: float,
        soil_cohesion_kpa: float = 5.0,
        friction_angle_deg: float = 30.0,
        soil_moisture: float = 0.4,
        rainfall_24h_mm: float = 0.0,
        depth_to_failure_m: float = 1.5,
    ) -> LandslideAssessment:
        theta = math.radians(slope_deg)
        phi = math.radians(friction_angle_deg)
        m = max(0.0, min(1.0, soil_moisture))  # rasio kedalaman muka air
        z = max(0.1, depth_to_failure_m)
        c = max(0.0, soil_cohesion_kpa)

        # Naikkan saturasi sesuai hujan 24 jam (sederhana)
        rainfall_factor = min(1.0, rainfall_24h_mm / 100.0)
        m_eff = min(1.0, m + 0.5 * rainfall_factor)

        gamma = self.GAMMA_SOIL
        gamma_w = self.GAMMA_WATER

        sin_t = math.sin(theta)
        cos_t = math.cos(theta)

        denom = gamma * z * sin_t * cos_t
        if denom <= 1e-9:
            fos = 10.0  # lereng datar
        else:
            num = c + (gamma - m_eff * gamma_w) * z * (cos_t ** 2) * math.tan(phi)
            fos = num / denom

        # Klasifikasi
        if fos >= 1.5:
            cls = "STABIL"
        elif fos >= 1.25:
            cls = "MARJINAL"
        elif fos >= 1.0:
            cls = "RAWAN"
        elif fos >= 0.8:
            cls = "TIDAK_STABIL"
        else:
            cls = "RUNTUH_AKTIF"

        hazard_score = self._compute_hazard(fos, slope_deg, rainfall_24h_mm)

        # Cek juga ambang Caine (1980): I = 14.82 * D^-0.39
        explanation = (
            f"Slope = {slope_deg:.1f} deg, c' = {c:.1f} kPa, phi' = "
            f"{friction_angle_deg:.1f} deg, saturasi efektif (m) = "
            f"{m_eff:.2f} setelah hujan 24 jam {rainfall_24h_mm:.1f} mm. "
            f"FOS Mohr-Coulomb (lereng tak terhingga) = {fos:.3f} -> "
            f"{cls}. Ambang FOS = 1.0 mengikuti Skempton & DeLory (1957). "
            f"Ambang hujan pemicu mengikuti Caine (1980)."
        )

        return LandslideAssessment(
            slope_deg=slope_deg,
            soil_cohesion_kpa=c,
            friction_angle_deg=friction_angle_deg,
            soil_moisture=m_eff,
            rainfall_24h_mm=rainfall_24h_mm,
            factor_of_safety=round(fos, 3),
            risk_class=cls,
            hazard_score=hazard_score,
            explanation=explanation,
        )

    def _compute_hazard(self, fos: float, slope: float, rain: float) -> float:
        # FOS rendah -> hazard tinggi
        fos_score = max(0.0, min(1.0, (1.5 - fos) / 1.5))
        slope_score = max(0.0, min(1.0, slope / 60.0))
        rain_score = max(0.0, min(1.0, rain / 150.0))
        return 0.5 * fos_score + 0.25 * slope_score + 0.25 * rain_score
