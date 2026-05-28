"""
EXTREME RAINFALL ANALYZER
=========================

Analisis hujan ekstrem dengan distribusi nilai ekstrim Gumbel/GEV dan
estimasi return period.

Sitasi:
    Gumbel (1958). Statistics of Extremes. Columbia University Press.
    Coles (2001). An Introduction to Statistical Modeling of Extreme Values.
        Springer.
    Koutsoyiannis (2004). Statistics of extremes and estimation of extreme
        rainfall I, II. Hydrological Sciences Journal 49.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Sequence
import math


@dataclass
class RainfallAssessment:
    annual_max_count: int
    location_param: float  # Gumbel mu
    scale_param: float  # Gumbel beta
    return_period_50yr_mm: float
    return_period_100yr_mm: float
    current_event_return_period: Optional[float]
    risk_class: str
    hazard_score: float
    explanation: str


class RainfallAnalyzer:
    """
    Fitting distribusi Gumbel pada seri annual maxima precipitation.
    Estimasi parameter dengan Method of Moments (sederhana, bisa diganti MLE).
    """

    EULER_MASCHERONI = 0.5772156649

    def analyze(
        self,
        annual_max_rainfall_mm: Sequence[float],
        current_event_mm: Optional[float] = None,
    ) -> RainfallAssessment:
        data = [float(x) for x in annual_max_rainfall_mm if x is not None]
        n = len(data)
        if n < 5:
            return RainfallAssessment(
                annual_max_count=n, location_param=float("nan"),
                scale_param=float("nan"), return_period_50yr_mm=float("nan"),
                return_period_100yr_mm=float("nan"),
                current_event_return_period=None,
                risk_class="DATA_TIDAK_CUKUP", hazard_score=0.0,
                explanation="Membutuhkan minimal 5 tahun data annual maxima.",
            )

        mean = sum(data) / n
        var = sum((x - mean) ** 2 for x in data) / max(1, n - 1)
        sd = math.sqrt(var) if var > 0 else 1e-9

        # Method of Moments untuk Gumbel:
        # beta = sd * sqrt(6) / pi
        # mu   = mean - gamma * beta
        beta = sd * math.sqrt(6.0) / math.pi
        mu = mean - self.EULER_MASCHERONI * beta

        # Return level: x_T = mu - beta * ln(-ln(1 - 1/T))
        def gumbel_quantile(T: float) -> float:
            return mu - beta * math.log(-math.log(1 - 1.0 / T))

        rl_50 = gumbel_quantile(50)
        rl_100 = gumbel_quantile(100)

        # Cek return period event saat ini
        current_T = None
        if current_event_mm is not None and beta > 0:
            # 1 - F(x) = exp(-exp(-(x-mu)/beta))
            z = (current_event_mm - mu) / beta
            try:
                # T = 1 / (1 - F(x))
                F = math.exp(-math.exp(-z))
                non_exceed = max(1e-9, min(1 - 1e-9, F))
                current_T = 1.0 / (1.0 - non_exceed)
            except Exception:
                current_T = None

        # Klasifikasi berdasarkan T
        if current_T is None:
            cls = "TIDAK_DINILAI"
            hazard = max(0.0, min(1.0, mean / 200.0))
        elif current_T < 5:
            cls = "WAJAR"
            hazard = 0.2
        elif current_T < 25:
            cls = "TIDAK_BIASA"
            hazard = 0.4
        elif current_T < 100:
            cls = "EKSTREM"
            hazard = 0.7
        else:
            cls = "SANGAT_EKSTREM"
            hazard = 0.95

        explanation = (
            f"Sample annual maxima = {n}. Parameter Gumbel (Method of Moments): "
            f"mu = {mu:.2f} mm, beta = {beta:.2f} mm. Return level 50 tahun = "
            f"{rl_50:.1f} mm; 100 tahun = {rl_100:.1f} mm. "
            + (f"Event saat ini ({current_event_mm:.1f} mm) memiliki return "
               f"period ~ {current_T:.1f} tahun -> {cls}." if current_T else f"")
            + " Distribusi Gumbel mengikuti Gumbel (1958)."
        )

        return RainfallAssessment(
            annual_max_count=n,
            location_param=round(mu, 3),
            scale_param=round(beta, 3),
            return_period_50yr_mm=round(rl_50, 2),
            return_period_100yr_mm=round(rl_100, 2),
            current_event_return_period=round(current_T, 2) if current_T else None,
            risk_class=cls,
            hazard_score=hazard,
            explanation=explanation,
        )
