"""
WILDFIRE ANALYZER
=================

Estimasi risiko kebakaran hutan dengan komponen Canadian Forest Fire Weather
Index System (FWI) yang disederhanakan dan Vapor Pressure Deficit (VPD).

Sitasi:
    Van Wagner (1987). Development and structure of the Canadian Forest Fire
        Weather Index System. Canadian Forest Service Forestry Technical
        Report 35.
    Seager dkk (2015). Climatology, variability, and trends in the U.S.
        vapor pressure deficit. Journal of Applied Meteorology and Climatology.
    Pyne dkk (1996). Introduction to Wildland Fire. Wiley.
"""

from __future__ import annotations
from dataclasses import dataclass
import math


@dataclass
class WildfireAssessment:
    temperature_c: float
    relative_humidity_pct: float
    wind_speed_kmh: float
    rainfall_24h_mm: float
    vpd_kpa: float
    ffmc_estimate: float
    fwi_estimate: float
    risk_class: str
    hazard_score: float
    explanation: str


class WildfireAnalyzer:
    """
    Analyzer kebakaran hutan/lahan.

    Output adalah estimasi sederhana FWI. Untuk operasional gunakan
    implementasi resmi BMKG / Indonesia Fire Danger Rating System (Fire-DRS).
    """

    def analyze(
        self,
        temperature_c: float,
        relative_humidity_pct: float,
        wind_speed_kmh: float,
        rainfall_24h_mm: float = 0.0,
        fuel_moisture_prev: float = 0.15,
    ) -> WildfireAssessment:
        T = float(temperature_c)
        RH = max(1.0, min(100.0, float(relative_humidity_pct)))
        W = max(0.0, float(wind_speed_kmh))
        rain = max(0.0, float(rainfall_24h_mm))

        # Saturation vapor pressure (kPa) - Tetens formula
        es = 0.6108 * math.exp(17.27 * T / (T + 237.3))
        ea = es * RH / 100.0
        vpd = max(0.0, es - ea)

        # FFMC simplified: drying when low RH and high T,
        # wetting by rainfall.
        # Formula sederhana: FFMC_proxy = 100 - RH*0.7 + (T-15)*1.5 - rain*1.2
        ffmc = 100 - RH * 0.7 + (T - 15) * 1.5 - rain * 1.2
        ffmc += W * 0.3
        ffmc = max(0.0, min(101.0, ffmc))

        # ISI (Initial Spread Index) proxy
        isi = math.exp(0.05 * W) * (ffmc / 100.0) ** 2 * 10.0
        # BUI proxy from drought factor
        dryness = max(0.0, 1.0 - rain / 30.0)
        bui = 60.0 * dryness * (1.0 - fuel_moisture_prev)

        # FWI proxy (Van Wagner 1987 form)
        fwi = math.sqrt(max(0.0, isi * bui)) * 0.6
        fwi = max(0.0, min(100.0, fwi))

        # Klasifikasi (CFFWIS)
        if fwi < 5:
            cls = "RENDAH"
        elif fwi < 11:
            cls = "SEDANG"
        elif fwi < 22:
            cls = "TINGGI"
        elif fwi < 38:
            cls = "SANGAT_TINGGI"
        else:
            cls = "EKSTREM"

        hazard_score = max(0.0, min(1.0, fwi / 50.0))

        explanation = (
            f"T={T:.1f} C, RH={RH:.0f}%, W={W:.1f} km/h, hujan 24h="
            f"{rain:.1f} mm. VPD = {vpd:.2f} kPa (semakin besar = udara "
            f"semakin haus uap air, vegetasi mudah terbakar). FFMC proxy "
            f"= {ffmc:.1f}, FWI proxy = {fwi:.1f} -> kelas {cls}. "
            f"Sistem FWI mengikuti Van Wagner (1987)."
        )

        return WildfireAssessment(
            temperature_c=T,
            relative_humidity_pct=RH,
            wind_speed_kmh=W,
            rainfall_24h_mm=rain,
            vpd_kpa=round(vpd, 3),
            ffmc_estimate=round(ffmc, 2),
            fwi_estimate=round(fwi, 2),
            risk_class=cls,
            hazard_score=hazard_score,
            explanation=explanation,
        )
