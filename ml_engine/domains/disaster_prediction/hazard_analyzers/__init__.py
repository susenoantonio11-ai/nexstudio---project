"""Hazard analyzers untuk 8 jenis bencana."""
from .earthquake import EarthquakeAnalyzer
from .tsunami import TsunamiAnalyzer
from .flood import FloodAnalyzer
from .landslide import LandslideAnalyzer
from .wildfire import WildfireAnalyzer
from .drought import DroughtAnalyzer
from .rainfall import RainfallAnalyzer
from .climate import ClimateRiskAnalyzer

__all__ = [
    "EarthquakeAnalyzer",
    "TsunamiAnalyzer",
    "FloodAnalyzer",
    "LandslideAnalyzer",
    "WildfireAnalyzer",
    "DroughtAnalyzer",
    "RainfallAnalyzer",
    "ClimateRiskAnalyzer",
]
