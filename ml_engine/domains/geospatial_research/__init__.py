"""
GEOSPATIAL AI RESEARCH MODULE - Nexlytics
==========================================
End-to-end raster analysis untuk flood classification research.

Mendukung:
- GeoTIFF reading (rasterio dengan graceful degradation)
- Multi-band raster preprocessing (reproject, resample, clip, mask, stack)
- Spectral indices (NDVI, NDWI, MNDWI, NDBI, VV/VH ratio, slope)
- Flood classification (supervised: RF/XGBoost/SVM; threshold-based; change detection)
- Evaluation (IoU, Kappa, ROC-AUC, classification report)

Library backend: rasterio, rioxarray, xarray, numpy, scikit-learn.
Modul gracefully degrades ke synthetic raster (numpy array) jika rasterio belum terinstall,
sehingga developer bisa tetap test pipeline.

CRISP-DM compliance:
1. Business Understanding: project goal selection (classification/susceptibility/extent)
2. Data Understanding: RasterReader + metadata extraction
3. Data Preparation: RasterPreprocessor + SpectralIndexCalculator
4. Modeling: FloodClassifier orchestrator
5. Evaluation: FloodEvaluator (IoU, Kappa, classification metrics)
6. Deployment: GeoTIFF/CSV/GeoJSON export
"""
from .raster_reader import RasterReader
from .raster_preprocessor import RasterPreprocessor
from .spectral_indices import SpectralIndexCalculator
from .flood_classifier import FloodClassifier
from .threshold_classifier import ThresholdFloodClassifier
from .change_detection import ChangeDetector
from .flood_evaluator import FloodEvaluator
from .pipeline import FloodResearchPipeline

# Advanced features (graceful degradation)
from .unet_model import build_unet, is_torch_available
from .unet_trainer import UNetTrainer
from .gee_integration import GEEIntegration, is_ee_available, quick_fetch_jakarta_flood_imagery
from .sen1floods11_dataset import get_dataset_info, is_dataset_present
from .transfer_learning import TransferLearningManager, list_available_pretrained
from .geoserver_integration import GeoServerClient, quick_publish_flood_layer
from .time_series_flood import TimeSeriesFloodTracker, fetch_time_series_via_gee

__all__ = [
    "RasterReader",
    "RasterPreprocessor",
    "SpectralIndexCalculator",
    "FloodClassifier",
    "ThresholdFloodClassifier",
    "ChangeDetector",
    "FloodEvaluator",
    "FloodResearchPipeline",
    # Deep learning
    "build_unet",
    "is_torch_available",
    "UNetTrainer",
    # GEE
    "GEEIntegration",
    "is_ee_available",
    "quick_fetch_jakarta_flood_imagery",
    # Sen1Floods11 + transfer learning
    "get_dataset_info",
    "is_dataset_present",
    "TransferLearningManager",
    "list_available_pretrained",
    # GeoServer
    "GeoServerClient",
    "quick_publish_flood_layer",
    # Time-series
    "TimeSeriesFloodTracker",
    "fetch_time_series_via_gee",
]
