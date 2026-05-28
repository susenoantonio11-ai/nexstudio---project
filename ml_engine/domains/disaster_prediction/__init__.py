"""
DISASTER PREDICTION ACCURACY & EARLY WARNING AI
================================================

Research-grade decision support platform untuk analisis prediksi bencana alam.

PENTING - DISCLAIMER ILMIAH:
Sistem ini ADALAH:
    research platform untuk analisis risiko, validasi model, edukasi.
Sistem ini BUKAN:
    pengganti BMKG, BNPB, USGS, atau lembaga peringatan dini resmi.
Untuk peringatan dini operasional selalu mengacu pada otoritas resmi.

Modul ini menganalisis 8 jenis bencana:
    1. Earthquake (Gutenberg-Richter, b-value, ETAS)
    2. Tsunami (Wells-Coppersmith, propagation modeling)
    3. Flood (Sen1Floods11, hidrologi rasional)
    4. Landslide (Mohr-Coulomb FOS, SHALSTAB)
    5. Wildfire (FWI, VPD, fuel moisture)
    6. Drought (SPI, SPEI, PDSI)
    7. Extreme Rainfall (Gumbel, GEV, IDF curve)
    8. Climate Risk (anomaly, trend, ENSO index)

Sitasi metodologi:
    Gutenberg & Richter (1944) Bulletin of the Seismological Society of America.
    Wells & Coppersmith (1994) BSSA, vol 84.
    Mohr-Coulomb dalam Terzaghi (1943) Theoretical Soil Mechanics.
    McKee et al (1993) Eighth Conference on Applied Climatology, AMS.
    Van Wagner (1987) Canadian Forest Service Forestry Technical Report 35.
"""

from .risk_score_engine import RiskScoreEngine, RiskComponents, RiskAssessment
from .warning_level import (
    WarningLevelClassifier,
    WarningLevel,
    WarningResult,
    WARNING_DISCLAIMER,
)
from .accuracy_benchmark import DisasterAccuracyBenchmark, BenchmarkReport
from .shap_explainer import SHAPExplainer, ExplanationReport

from .hazard_analyzers.earthquake import EarthquakeAnalyzer
from .hazard_analyzers.tsunami import TsunamiAnalyzer
from .hazard_analyzers.flood import FloodAnalyzer
from .hazard_analyzers.landslide import LandslideAnalyzer
from .hazard_analyzers.wildfire import WildfireAnalyzer
from .hazard_analyzers.drought import DroughtAnalyzer
from .hazard_analyzers.rainfall import RainfallAnalyzer
from .hazard_analyzers.climate import ClimateRiskAnalyzer

from .models.lstm_temporal import TemporalLSTMModel
from .models.xgboost_geospatial import GeospatialXGBoostModel
from .models.bayesian_risk import BayesianRiskModel
from .models.ensemble_pipeline import HybridEnsemblePipeline

__all__ = [
    "RiskScoreEngine",
    "RiskComponents",
    "RiskAssessment",
    "WarningLevelClassifier",
    "WarningLevel",
    "WarningResult",
    "WARNING_DISCLAIMER",
    "DisasterAccuracyBenchmark",
    "BenchmarkReport",
    "SHAPExplainer",
    "ExplanationReport",
    "EarthquakeAnalyzer",
    "TsunamiAnalyzer",
    "FloodAnalyzer",
    "LandslideAnalyzer",
    "WildfireAnalyzer",
    "DroughtAnalyzer",
    "RainfallAnalyzer",
    "ClimateRiskAnalyzer",
    "TemporalLSTMModel",
    "GeospatialXGBoostModel",
    "BayesianRiskModel",
    "HybridEnsemblePipeline",
]

__version__ = "1.0.0-research"
