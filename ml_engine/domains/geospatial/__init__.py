"""
GEOSPATIAL DOMAIN
=================
Spatial data analysis modules for Nexlytics.

Components:
- SpatialAnalyzer: distance, density, basic statistics
- ChoroplethBuilder: aggregate metrics per region
- HeatmapGenerator: density grid for heatmap visualization
- SpatialClusterDetector: DBSCAN-based location clustering
- RegionPerformanceAnalyzer: compare regions on KPIs
- RouteAnalyzer: movement / trajectory analysis

Following CRISP-DM:
- Data Understanding: lat/lon validation, coordinate system check
- Data Preparation: projection, normalization, geocoding hints
- Modeling: spatial clustering, density estimation
- Evaluation: silhouette score, cluster validity
"""
from .spatial_analyzer import SpatialAnalyzer
from .choropleth_builder import ChoroplethBuilder
from .heatmap_generator import HeatmapGenerator
from .spatial_cluster_detector import SpatialClusterDetector
from .region_performance import RegionPerformanceAnalyzer
from .route_analyzer import RouteAnalyzer

__all__ = [
    "SpatialAnalyzer",
    "ChoroplethBuilder",
    "HeatmapGenerator",
    "SpatialClusterDetector",
    "RegionPerformanceAnalyzer",
    "RouteAnalyzer",
]
