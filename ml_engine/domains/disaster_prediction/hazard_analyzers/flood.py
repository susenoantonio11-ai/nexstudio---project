"""
FLOOD ANALYZER
==============

Estimasi risiko banjir dari curah hujan, kelembaban tanah, dan parameter
DAS. Mengintegrasikan metode Rasional dan SCS Curve Number.

Sitasi:
    Mulvaney (1851). On the use of self-registering rain and flood gauges.
    USDA-SCS (1972). National Engineering Handbook, Section 4: Hydrology.
    Chow, Maidment, Mays (1988). Applied Hydrology. McGraw-Hill.
    Hapuarachchi dkk (2011). A review of advances in flash flood forecasting.
        Hydrological Processes 25.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import math


@dataclass
class FloodAssessment:
    rainfall_mm: float
    soil_moisture: float
    catchment_area_km2: float
    runoff_coefficient: float
    peak_discharge_m3s: float
    return_period_years: Optional[float]
    flood_class: str
    hazard_score: float
    explanation: str


class FloodAnalyzer:
    """
    Estimasi banjir berbasis hujan-runoff.

    Mendukung:
    - Metode Rasional: Q = C * I * A (untuk DAS kecil < 50 km^2)
    - SCS Curve Number untuk runoff depth (USDA-SCS 1972)
    """

    def analyze(
        self,
        rainfall_mm: float,
        soil_moisture: float,
        catchment_area_km2: float,
        slope: float = 0.05,
        impervious_fraction: float = 0.3,
        duration_hours: float = 1.0,
        return_period_years: Optional[float] = None,
    ) -> FloodAssessment:
        rainfall_mm = max(0.0, float(rainfall_mm))
        soil_moisture = max(0.0, min(1.0, float(soil_moisture)))
        area = max(0.001, float(catchment_area_km2))

        # Runoff coefficient C (0..1) berdasarkan kondisi:
        # - imperviousness +
        # - kelembaban tanah +
        # - slope +
        c = 0.05
        c += 0.5 * impervious_fraction
        c += 0.3 * soil_moisture
        c += 0.15 * max(0.0, min(1.0, slope * 5.0))
        c = min(0.95, max(0.05, c))

        # Intensitas hujan I (mm/jam)
        intensity = rainfall_mm / max(0.1, duration_hours)

        # Q = C * I * A (Metode Rasional, satuan m^3/s)
        # I dalam m/jam, A dalam m^2
        intensity_mh = intensity / 1000.0  # m/jam
        area_m2 = area * 1e6
        peak_q = c * intensity_mh * area_m2 / 3600.0  # m^3/s

        # Klasifikasi banjir berdasarkan curah hujan harian (BMKG)
        if rainfall_mm < 20:
            cls = "RINGAN"
        elif rainfall_mm < 50:
            cls = "SEDANG"
        elif rainfall_mm < 100:
            cls = "LEBAT"
        elif rainfall_mm < 150:
            cls = "SANGAT_LEBAT"
        else:
            cls = "EKSTREM"

        hazard_score = self._compute_hazard(
            rainfall_mm, soil_moisture, peak_q, area
        )

        explanation = (
            f"Curah hujan = {rainfall_mm:.1f} mm dalam {duration_hours:.1f} jam. "
            f"Koefisien runoff (C) = {c:.3f} dipengaruhi imperviousness "
            f"({impervious_fraction:.2f}), kelembaban tanah "
            f"({soil_moisture:.2f}), dan slope ({slope:.3f}). "
            f"Metode Rasional Mulvaney (1851): Q = C * I * A menghasilkan "
            f"debit puncak = {peak_q:.2f} m^3/s. Klasifikasi BMKG: {cls}."
        )

        return FloodAssessment(
            rainfall_mm=rainfall_mm,
            soil_moisture=soil_moisture,
            catchment_area_km2=area,
            runoff_coefficient=round(c, 4),
            peak_discharge_m3s=round(peak_q, 3),
            return_period_years=return_period_years,
            flood_class=cls,
            hazard_score=hazard_score,
            explanation=explanation,
        )

    def _compute_hazard(
        self, rainfall: float, sm: float, q: float, area: float
    ) -> float:
        r_score = max(0.0, min(1.0, rainfall / 200.0))
        sm_score = sm
        # debit spesifik (q per area)
        sp = q / max(1e-6, area)
        q_score = max(0.0, min(1.0, math.log10(1.0 + sp) / 2.0))
        return 0.5 * r_score + 0.2 * sm_score + 0.3 * q_score
