"""
Backend mirror of the frontend NxDataSourceCatalog. Single source of truth
that the FastAPI router exposes via /api/research/data-sources.
"""
from typing import Any, Dict, List

DATA_SOURCE_REGISTRY: List[Dict[str, Any]] = [
    {"id": "jrc_global_surface_water", "name": "JRC Global Surface Water", "category": "flood", "type": "raster", "provider": "JRC", "spatial_resolution_m": 30, "requires_api_key": False, "status": "connected", "citation": "Pekel et al. (2016) Nature 540"},
    {"id": "sentinel1_grd", "name": "Sentinel-1 SAR GRD", "category": "satellite", "type": "raster", "provider": "ESA Copernicus", "spatial_resolution_m": 10, "requires_api_key": False, "status": "connected", "citation": "Torres et al. (2012)"},
    {"id": "sentinel2_l2a_sr", "name": "Sentinel-2 L2A SR", "category": "satellite", "type": "raster", "provider": "ESA Copernicus", "spatial_resolution_m": 10, "requires_api_key": False, "status": "connected", "citation": "Drusch et al. (2012)"},
    {"id": "modis_terra_mod09ga", "name": "MODIS Terra MOD09GA", "category": "satellite", "type": "raster", "provider": "NASA LP DAAC", "spatial_resolution_m": 500, "requires_api_key": False, "status": "connected", "citation": "Vermote (2015)"},
    {"id": "modis_ndvi", "name": "MODIS Vegetation Indices", "category": "environment", "type": "raster", "provider": "NASA LP DAAC", "spatial_resolution_m": 250, "requires_api_key": False, "status": "connected", "citation": "Didan (2015)"},
    {"id": "chirps_daily", "name": "CHIRPS Daily Precipitation", "category": "rainfall", "type": "raster", "provider": "UCSB CHC", "spatial_resolution_m": 5500, "requires_api_key": False, "status": "connected", "citation": "Funk et al. (2015)"},
    {"id": "gldas_noah_2_1", "name": "GLDAS-2.1 Noah", "category": "climate", "type": "raster", "provider": "NASA GES DISC", "spatial_resolution_m": 27750, "requires_api_key": False, "status": "connected", "citation": "Beaudoing & Rodell (2020)"},
    {"id": "noaa_climate", "name": "NOAA Climate Data", "category": "climate", "type": "tabular", "provider": "NOAA NCEI", "spatial_resolution_m": None, "requires_api_key": True, "status": "demo", "citation": "NOAA NCEI"},
    {"id": "openweather", "name": "OpenWeather Forecast", "category": "climate", "type": "tabular", "provider": "OpenWeatherMap", "spatial_resolution_m": None, "requires_api_key": True, "status": "demo", "citation": "OpenWeather (2024)"},
    {"id": "bmkg_indonesia", "name": "BMKG Indonesia", "category": "climate", "type": "tabular", "provider": "BMKG", "spatial_resolution_m": None, "requires_api_key": False, "status": "demo", "citation": "BMKG"},
    {"id": "srtm_30m", "name": "SRTM 30m DEM", "category": "satellite", "type": "raster", "provider": "NASA / USGS", "spatial_resolution_m": 30, "requires_api_key": False, "status": "connected", "citation": "Farr et al. (2007)"},
    {"id": "opentopo_dem", "name": "OpenTopography Global DEM", "category": "satellite", "type": "raster", "provider": "OpenTopography", "spatial_resolution_m": 30, "requires_api_key": True, "status": "demo", "citation": "OpenTopography"},
    {"id": "hydrosheds", "name": "HydroSHEDS Drainage", "category": "flood", "type": "vector", "provider": "WWF/USGS", "spatial_resolution_m": 90, "requires_api_key": False, "status": "connected", "citation": "Lehner & Grill (2013)"},
    {"id": "esa_worldcover_2021", "name": "ESA WorldCover 10m", "category": "environment", "type": "raster", "provider": "ESA", "spatial_resolution_m": 10, "requires_api_key": False, "status": "connected", "citation": "Zanaga et al. (2022)"},
    {"id": "esa_cci_landcover", "name": "ESA CCI Land Cover", "category": "environment", "type": "raster", "provider": "ESA CCI", "spatial_resolution_m": 300, "requires_api_key": False, "status": "connected", "citation": "ESA CCI Land Cover (2017)"},
    {"id": "soilgrids_250", "name": "SoilGrids 250m", "category": "environment", "type": "raster", "provider": "ISRIC", "spatial_resolution_m": 250, "requires_api_key": False, "status": "connected", "citation": "Hengl et al. (2017)"},
    {"id": "nasa_firms", "name": "NASA FIRMS Active Fire", "category": "wildfire", "type": "vector", "provider": "NASA LANCE", "spatial_resolution_m": 375, "requires_api_key": True, "status": "demo", "citation": "NASA FIRMS"},
    {"id": "usgs_earthquake", "name": "USGS Earthquake Catalog", "category": "earthquake", "type": "vector", "provider": "USGS", "spatial_resolution_m": None, "requires_api_key": False, "status": "connected", "citation": "USGS Earthquake Hazards Program"},
    {"id": "gdacs", "name": "GDACS Disaster Alert", "category": "disaster_alert", "type": "vector", "provider": "JRC + UN OCHA", "spatial_resolution_m": None, "requires_api_key": False, "status": "connected", "citation": "GDACS"},
    {"id": "reliefweb_disaster", "name": "ReliefWeb Disaster Reports", "category": "disaster_record", "type": "tabular", "provider": "UN OCHA", "spatial_resolution_m": None, "requires_api_key": False, "status": "connected", "citation": "ReliefWeb"},
    {"id": "bnpb_dibi", "name": "BNPB DIBI Indonesia", "category": "disaster_record", "type": "vector", "provider": "BNPB", "spatial_resolution_m": None, "requires_api_key": False, "status": "demo", "citation": "BNPB DIBI"},
    {"id": "osm_admin", "name": "OpenStreetMap Admin Boundary", "category": "boundary", "type": "vector", "provider": "OSM Foundation", "spatial_resolution_m": 10, "requires_api_key": False, "status": "connected", "citation": "OpenStreetMap contributors"},
    {"id": "bps_kabupaten", "name": "BPS Statistik Kabupaten", "category": "population", "type": "tabular", "provider": "BPS", "spatial_resolution_m": None, "requires_api_key": False, "status": "demo", "citation": "BPS Statistik Indonesia"},
]

_RESEARCH_STACKS = {
    "multi_province_flood_classification": ["jrc_global_surface_water","sentinel1_grd","modis_terra_mod09ga","gldas_noah_2_1","chirps_daily","srtm_30m","hydrosheds","esa_worldcover_2021","bnpb_dibi","osm_admin"],
    "flood_classification": ["jrc_global_surface_water","sentinel2_l2a_sr","sentinel1_grd","chirps_daily","srtm_30m","hydrosheds","bnpb_dibi"],
    "flood_extent_mapping": ["sentinel1_grd","sentinel2_l2a_sr","jrc_global_surface_water","hydrosheds"],
    "rainfall_risk_analysis": ["chirps_daily","gldas_noah_2_1","bmkg_indonesia","noaa_climate"],
    "forest_fire_risk": ["nasa_firms","modis_ndvi","chirps_daily","srtm_30m","esa_worldcover_2021","gldas_noah_2_1"],
    "climate_change_analysis": ["noaa_climate","chirps_daily","gldas_noah_2_1","modis_ndvi","bmkg_indonesia"],
    "earthquake_risk": ["usgs_earthquake","bmkg_indonesia","srtm_30m","soilgrids_250","osm_admin"],
    "tsunami_susceptibility": ["usgs_earthquake","srtm_30m","sentinel1_grd","osm_admin","bnpb_dibi"],
    "environmental_monitoring": ["modis_ndvi","esa_worldcover_2021","esa_cci_landcover","sentinel2_l2a_sr","soilgrids_250"],
    "land_cover_change": ["esa_worldcover_2021","esa_cci_landcover","sentinel2_l2a_sr","modis_terra_mod09ga"],
    "drought_analysis": ["chirps_daily","gldas_noah_2_1","modis_ndvi","soilgrids_250"],
    "custom_geo_research": [],
}

def list_sources() -> List[Dict[str, Any]]:
    return list(DATA_SOURCE_REGISTRY)

def get_source(source_id: str) -> Dict[str, Any]:
    for s in DATA_SOURCE_REGISTRY:
        if s["id"] == source_id: return s
    return {}

def list_by_category(category: str) -> List[Dict[str, Any]]:
    return [s for s in DATA_SOURCE_REGISTRY if s.get("category") == category]

def suggest_for_research(research_type: str) -> List[Dict[str, Any]]:
    return [get_source(i) for i in _RESEARCH_STACKS.get(research_type, []) if get_source(i)]

def stats() -> Dict[str, Any]:
    return {
        "total": len(DATA_SOURCE_REGISTRY),
        "connected": sum(1 for s in DATA_SOURCE_REGISTRY if s.get("status") == "connected"),
        "demo": sum(1 for s in DATA_SOURCE_REGISTRY if s.get("status") == "demo"),
        "offline": sum(1 for s in DATA_SOURCE_REGISTRY if s.get("status") == "offline"),
        "requiring_api_key": sum(1 for s in DATA_SOURCE_REGISTRY if s.get("requires_api_key")),
    }
