"""
GEE Service (Google Earth Engine adapter)
=========================================
Pulls per-province daily features from Google Earth Engine collections.
The service has TWO modes:

  REAL   — when `earthengine-api` is installed AND a service account is
           configured (GOOGLE_APPLICATION_CREDENTIALS env var pointing to
           a JSON key with EE access). It then issues genuine GEE queries.
  STUB   — when EE is not configured. It synthesizes a deterministic, well-
           shaped mock panel using the province bbox area + a sinusoidal
           seasonal rainfall pattern. The frontend pipeline still gets
           realistic-shaped data and the user can run the full thesis
           pipeline before activating GEE credentials.

This split lets the user *defend the pipeline* on a torch-free / GEE-free
laptop, then re-run identical code on a server that has both, with no
code change.

GEE setup (when ready):
  1. https://signup.earthengine.google.com/  (academic use is free)
  2. Create a service account in Google Cloud Console.
  3. Grant the service account access to the EE-enabled project.
  4. Download the JSON key, put it at  backend/.gee_service_account.json
  5. `pip install earthengine-api`
  6. `export GOOGLE_APPLICATION_CREDENTIALS=$(pwd)/backend/.gee_service_account.json`
  7. Restart backend — `gee_service.is_live()` will return True.

CITATIONS (Method Monitor)
  * Gorelick et al. (2017) Remote Sens. Environ. 202:18-27 — Google Earth
    Engine: Planetary-scale geospatial analysis for everyone.
  * Funk et al. (2015) Sci. Data 2:150066 — CHIRPS.
  * Vermote (2015) NASA LP DAAC — MOD09GA.
  * Beaudoing & Rodell (2020) — GLDAS-2.1 Noah.
"""
from __future__ import annotations

import math
import os
import time
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

try:
    import numpy as np
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import ee  # type: ignore
    HAS_EE = True
except ImportError:
    HAS_EE = False

from .indonesia_provinces import INDONESIA_PROVINCES


def is_live() -> bool:
    """True kalau GEE credentials available (service account ATAU ADC)."""
    if not HAS_EE: return False
    # Cek service account JSON
    cred_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if cred_path and os.path.exists(cred_path):
        return True
    # Cek ADC (Application Default Credentials)
    adc_path = os.path.expanduser("~/.config/gcloud/application_default_credentials.json")
    if os.path.exists(adc_path):
        return True
    return False


def _initialize_ee_once():
    if not HAS_EE: return False
    # Strategi 1: Service account JSON (kalau ada)
    try:
        cred_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if cred_path and os.path.exists(cred_path):
            # Check if it's service account or user ADC
            with open(cred_path) as f:
                content = f.read()
            if '"type": "service_account"' in content:
                creds = ee.ServiceAccountCredentials(None, cred_path)
                ee.Initialize(credentials=creds)
                return True
    except Exception:
        pass
    # Strategi 2: ADC (Application Default Credentials) via gcloud auth
    try:
        ee.Initialize()
        return True
    except Exception:
        pass
    # Strategi 3: Manual ADC path
    try:
        adc_path = os.path.expanduser("~/.config/gcloud/application_default_credentials.json")
        if os.path.exists(adc_path):
            import google.auth
            credentials, project = google.auth.default()
            ee.Initialize(credentials=credentials, project=project)
            return True
    except Exception:
        pass
    return False


# ---------------------------------------------------------------------------
# Stub generator — deterministic, climatically-plausible synthetic panel
# ---------------------------------------------------------------------------
def _synthesize_panel(province_ids: List[str], start: str, end: str) -> "pd.DataFrame":
    if not HAS_PANDAS:
        raise RuntimeError("pandas required even for stub generator")
    sd = pd.Timestamp(start)
    ed = pd.Timestamp(end)
    days = pd.date_range(sd, ed, freq="D")
    rng = np.random.default_rng(42)
    rows = []
    for pid in province_ids:
        prov = next((p for p in INDONESIA_PROVINCES if p["id"] == pid), None)
        if not prov: continue
        # bbox-area as a stand-in for catchment size
        w, s, e, n = prov["bbox"]
        area = max(0.05, (e - w) * (n - s))
        # latitude — wet/dry season phase-shift
        lat = (n + s) / 2.0
        for d in days:
            doy = d.dayofyear
            # Indonesia bimodal monsoon: peak Dec–Feb in southern, Aug in north
            phase = 1.0 if lat < -1.0 else (-1.0 if lat > 1.0 else 0.0)
            seasonal = 18 + 12 * math.sin(2 * math.pi * (doy / 365.0) + phase * math.pi / 6)
            shock = rng.exponential(scale=4.0) if rng.random() < 0.05 else 0.0
            chirps = max(0.0, seasonal + rng.normal(0, 6) + shock)
            ndwi = max(-0.4, min(0.6, 0.10 + 0.20 * math.sin(2 * math.pi * (doy / 365.0)) + rng.normal(0, 0.05)))
            soilmoi = max(0.05, min(0.55, 0.25 + 0.15 * math.sin(2 * math.pi * (doy / 365.0)) + rng.normal(0, 0.04)))
            elev = float(50 + 200 * (1 - area / 5))
            slope = float(2 + rng.normal(0, 0.5))
            twi = float(8 + rng.normal(0, 1.5))
            jrc_occ = float(min(95, 5 + 30 * (lat < 0) + rng.normal(0, 3)))
            lc_built = float(min(60, 5 + (1 - area / 5) * 30 + rng.normal(0, 1)))
            lc_water = float(max(0, jrc_occ * 0.5 + rng.normal(0, 1)))
            rows.append({
                "province_id": pid,
                "date": d,
                "chirps_precip": round(chirps, 3),
                "modis_ndwi": round(ndwi, 4),
                "gldas_soilmoi": round(soilmoi, 4),
                "srtm_elevation": round(elev, 1),
                "srtm_slope": round(slope, 3),
                "srtm_twi": round(twi, 3),
                "jrc_water_occurrence": round(jrc_occ, 1),
                "lc_built_up_pct": round(lc_built, 1),
                "lc_water_pct": round(lc_water, 1),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Real GEE pull (lazy — only used when is_live())
# ---------------------------------------------------------------------------
def _pull_real(province_ids: List[str], start: str, end: str) -> "pd.DataFrame":
    if not _initialize_ee_once():
        raise RuntimeError("GEE failed to initialize. Check service account.")
    rows = []
    for pid in province_ids:
        prov = next((p for p in INDONESIA_PROVINCES if p["id"] == pid), None)
        if not prov: continue
        bbox = prov["bbox"]
        region = ee.Geometry.Rectangle(bbox)
        # CHIRPS daily precipitation (mean over province)
        chirps = (ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
                  .filterDate(start, end)
                  .select("precipitation"))
        # MODIS Terra surface reflectance — derive NDWI = (b04-b02)/(b04+b02)
        modis = (ee.ImageCollection("MODIS/061/MOD09GA")
                 .filterDate(start, end)
                 .map(lambda im: im.normalizedDifference(["sur_refl_b04", "sur_refl_b02"]).rename("ndwi")))
        # GLDAS soil moisture 0-10 cm (3-hourly → daily mean)
        gldas = (ee.ImageCollection("NASA/GLDAS/V021/NOAH/G025/T3H")
                 .filterDate(start, end)
                 .select("SoilMoi0_10cm_inst"))
        # Reduce per day to province mean
        def per_day(date):
            d = ee.Date(date)
            d_end = d.advance(1, "day")
            r1 = chirps.filterDate(d, d_end).mean().reduceRegion(ee.Reducer.mean(), region, 5500).get("precipitation")
            r2 = modis.filterDate(d, d_end).mean().reduceRegion(ee.Reducer.mean(), region, 500).get("ndwi")
            r3 = gldas.filterDate(d, d_end).mean().reduceRegion(ee.Reducer.mean(), region, 27750).get("SoilMoi0_10cm_inst")
            return ee.Feature(None, {"date": d.format("YYYY-MM-dd"),
                                     "chirps_precip": r1, "modis_ndwi": r2, "gldas_soilmoi": r3})
        n_days = (pd.Timestamp(end) - pd.Timestamp(start)).days
        date_seq = [(pd.Timestamp(start) + pd.Timedelta(days=i)).strftime("%Y-%m-%d") for i in range(n_days)]
        # NOTE: real code typically uses ee.List + .map server-side; here we
        # keep the loop for clarity. For 38 provinces × 10 years that is
        # 138_000 calls — switch to image-collection .reduceRegions for prod.
        for ds in date_seq:
            try:
                feat = per_day(ee.Date(ds)).getInfo()["properties"]
                feat["province_id"] = pid
                rows.append(feat)
            except Exception:
                continue
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Public function used by the orchestrator
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Map tile generation · for frontend Leaflet rendering
# ---------------------------------------------------------------------------
DATASET_CATALOG = {
    "chirps_rainfall": {
        "label": "CHIRPS Daily Rainfall",
        "ee_collection": "UCSB-CHG/CHIRPS/DAILY",
        "band": "precipitation",
        "vis_params": {"min": 0, "max": 60, "palette": ["#ffffff", "#a0d8ef", "#3a8fb7", "#0a4d7f", "#0a2c5c"]},
        "unit": "mm/day",
        "description": "Climate Hazards InfraRed Precipitation with Stations"
    },
    "modis_ndvi": {
        "label": "MODIS NDVI 16-day",
        "ee_collection": "MODIS/061/MOD13Q1",
        "band": "NDVI",
        "vis_params": {"min": -2000, "max": 8000, "palette": ["#a52a2a", "#fee0a0", "#a8e6cf", "#0a8a0a", "#003c00"]},
        "unit": "scaled NDVI",
        "description": "MODIS Terra Vegetation Indices 250m"
    },
    "gldas_soilmoisture": {
        "label": "GLDAS Soil Moisture",
        "ee_collection": "NASA/GLDAS/V021/NOAH/G025/T3H",
        "band": "SoilMoi0_10cm_inst",
        "vis_params": {"min": 0, "max": 50, "palette": ["#fff5b8", "#a8d8ea", "#5588cc", "#0a3d80"]},
        "unit": "kg/m2",
        "description": "Global Land Data Assimilation System soil moisture 0-10cm"
    },
    "srtm_elevation": {
        "label": "SRTM Elevation",
        "ee_collection": "USGS/SRTMGL1_003",
        "band": "elevation",
        "vis_params": {"min": 0, "max": 3000, "palette": ["#0a4d7f", "#a0d8ef", "#88b87a", "#f4d27a", "#8b4513", "#ffffff"]},
        "unit": "meters",
        "description": "Shuttle Radar Topography Mission 1 arc-second",
        "static": True  # not time-dependent
    },
    "esa_worldcover": {
        "label": "ESA WorldCover Land Use",
        "ee_collection": "ESA/WorldCover/v200",
        "band": "Map",
        "vis_params": {"min": 10, "max": 100, "palette": ["#006400", "#ffbb22", "#ffff4c", "#f096ff", "#fa0000", "#b4b4b4", "#f0f0f0", "#0064c8", "#0096a0", "#00cf75", "#fae6a0"]},
        "unit": "class code",
        "description": "ESA WorldCover 10m land cover (2021)",
        "static": True
    }
}


def get_map_tile_url(dataset_id: str, start: str = None, end: str = None) -> Dict[str, Any]:
    """Returns a Leaflet-compatible tile URL template for the given dataset.

    REAL mode requires GEE auth · generates Image.getMapId from collection mean/median.
    STUB mode returns a placeholder OpenStreetMap-based URL that doesn't render the
    dataset but lets the frontend Leaflet map initialize plus show base layer.
    """
    if dataset_id not in DATASET_CATALOG:
        return {"status": "error", "message": f"Unknown dataset · {dataset_id}",
                "available": list(DATASET_CATALOG.keys())}

    spec = DATASET_CATALOG[dataset_id]

    if not is_live():
        # Stub mode · return OSM tile URL plus dataset metadata so UI still works
        return {
            "status": "success",
            "mode": "stub",
            "dataset_id": dataset_id,
            "label": spec["label"],
            "tile_url": "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
            "attribution": "OSM (GEE credentials not set, satellite layer unavailable)",
            "vis_params": spec["vis_params"],
            "unit": spec["unit"],
            "description": spec["description"],
            "note": "Stub mode active · set GOOGLE_APPLICATION_CREDENTIALS for real satellite tiles"
        }

    # REAL mode · query GEE for tile URL
    if not _initialize_ee_once():
        return {"status": "error", "message": "GEE failed to initialize"}

    try:
        if spec.get("static"):
            img = ee.Image(spec["ee_collection"]).select(spec["band"])
        else:
            # Date-range collection · take mean over period (or last available)
            sd = start or "2024-01-01"
            ed = end or "2024-12-31"
            collection = ee.ImageCollection(spec["ee_collection"]).select(spec["band"]).filterDate(sd, ed)
            img = collection.mean()

        # Clip to Indonesia bounding box for relevance
        indo_geom = ee.Geometry.Rectangle([94, -11, 142, 7])
        img = img.clip(indo_geom)

        map_id = img.getMapId(spec["vis_params"])
        tile_url = map_id["tile_fetcher"].url_format

        return {
            "status": "success",
            "mode": "live_gee",
            "dataset_id": dataset_id,
            "label": spec["label"],
            "tile_url": tile_url,
            "attribution": f"Google Earth Engine · {spec['ee_collection']}",
            "vis_params": spec["vis_params"],
            "unit": spec["unit"],
            "description": spec["description"],
            "date_range": {"start": start, "end": end} if not spec.get("static") else None
        }
    except Exception as e:
        return {"status": "error", "message": f"GEE tile generation failed · {e}"}


def export_geotiff(dataset_id: str, province_id: str = None,
                    start: str = None, end: str = None, scale: int = 1000) -> Dict[str, Any]:
    """Generate a downloadable GeoTIFF URL for the given dataset clipped to a province.
    Uses ee.Image.getDownloadURL which works for images under 32 MB. For larger
    exports, returns instructions for Export.image.toDrive() workflow.

    Naming convention matches uploaded sample: {DATASET}_{PROVINCE}_{START}_{END}.tif
    """
    if dataset_id not in DATASET_CATALOG:
        return {"status": "error", "message": f"Unknown dataset · {dataset_id}",
                "available": list(DATASET_CATALOG.keys())}
    spec = DATASET_CATALOG[dataset_id]

    # Get province geometry · default to whole Indonesia kalau province_id None
    province = None
    if province_id:
        province = next((p for p in INDONESIA_PROVINCES if p["id"] == province_id), None)
    if province:
        bbox = province["bbox"]  # [west, south, east, north]
        roi = ee.Geometry.Rectangle(bbox) if HAS_EE else None
        roi_label = province.get("name_id") or province.get("name_en") or province_id
    else:
        roi = ee.Geometry.Rectangle([94, -11, 142, 7]) if HAS_EE else None
        roi_label = "Indonesia"
        if province_id:
            return {"status": "error", "message": f"Unknown province · {province_id}. Valid IDs: " + ", ".join([p["id"] for p in INDONESIA_PROVINCES[:10]]) + ", ..."}

    if not is_live():
        return {
            "status": "error",
            "mode": "stub",
            "message": "GEE credentials required for GeoTIFF export. Setup ADC: gcloud auth application-default login"
        }
    if not _initialize_ee_once():
        return {"status": "error", "message": "GEE initialization failed"}

    try:
        sd = start or "2024-01-01"
        ed = end or "2024-12-31"
        if spec.get("static"):
            img = ee.Image(spec["ee_collection"]).select(spec["band"])
        else:
            collection = ee.ImageCollection(spec["ee_collection"]).select(spec["band"]).filterDate(sd, ed)
            img = collection.mean()
        img = img.clip(roi)

        # Try direct getDownloadURL for small exports
        filename_base = f"{dataset_id.upper()}_{(province_id or 'Indonesia').replace(' ', '')}_{sd[:4]}_{ed[:4]}"
        try:
            url = img.getDownloadURL({
                "scale": scale,
                "region": roi,
                "crs": "EPSG:4326",
                "format": "GEO_TIFF",
                "filePerBand": False,
                "name": filename_base
            })
            return {
                "status": "success",
                "mode": "direct_download",
                "filename": f"{filename_base}.tif",
                "download_url": url,
                "dataset": dataset_id,
                "province": province_id or "Indonesia",
                "date_range": {"start": sd, "end": ed},
                "scale_meters": scale,
                "note": "Direct download URL valid 1 hour. Click to download GeoTIFF."
            }
        except Exception as direct_err:
            # Fallback to Export.image.toDrive script
            return {
                "status": "success",
                "mode": "drive_export",
                "filename": f"{filename_base}.tif",
                "download_url": None,
                "dataset": dataset_id,
                "province": province_id or "Indonesia",
                "date_range": {"start": sd, "end": ed},
                "note": f"Image too large for direct download ({direct_err}). Use Export.image.toDrive script below.",
                "ee_script": _build_drive_export_script(spec, dataset_id, province_id, roi_label, sd, ed, scale)
            }
    except Exception as e:
        return {"status": "error", "message": f"Export failed · {e}"}


def _build_drive_export_script(spec, dataset_id, province_id, roi_label, sd, ed, scale):
    """Generate a GEE Code Editor JavaScript snippet user can paste."""
    return f"""// Paste into GEE Code Editor · https://code.earthengine.google.com/
var roi = ee.FeatureCollection('FAO/GAUL_SIMPLIFIED_500m/2015/level1')
  .filter(ee.Filter.eq('ADM1_NAME', '{roi_label}'))
  .geometry();
var img = {'ee.Image' if spec.get('static') else 'ee.ImageCollection'}('{spec["ee_collection"]}')
  .select('{spec["band"]}')
  {f'.filterDate(' + repr(sd) + ', ' + repr(ed) + ').mean()' if not spec.get('static') else ''}
  .clip(roi);
Map.centerObject(roi, 8);
Map.addLayer(img, {{min: {spec["vis_params"].get("min", 0)}, max: {spec["vis_params"].get("max", 100)}, palette: {spec["vis_params"].get("palette", [])}}}, '{spec["label"]}');
Export.image.toDrive({{
  image: img,
  description: '{dataset_id}_{province_id or "Indonesia"}_{sd[:4]}_{ed[:4]}',
  folder: 'NXLYTICS',
  region: roi,
  scale: {scale},
  crs: 'EPSG:4326',
  maxPixels: 1e10
}});
"""


def inspect_pixel(lat: float, lng: float, dataset_id: str, start: str = None, end: str = None) -> Dict[str, Any]:
    """Query pixel value at a specific lat/lng for the given dataset.
    Mimics GEE Code Editor's Inspector tool functionality."""
    if dataset_id not in DATASET_CATALOG:
        return {"status": "error", "message": f"Unknown dataset · {dataset_id}"}
    spec = DATASET_CATALOG[dataset_id]

    if not is_live():
        # Stub mode · return synthesized realistic value based on lat/lng
        synthetic_values = {
            "chirps_rainfall": round(15 + 25 * abs(math.sin(lat * 0.1 + lng * 0.05)), 2),
            "modis_ndvi": round(0.3 + 0.4 * (1 - abs(lat) / 11), 4),
            "gldas_soilmoisture": round(15 + 20 * abs(math.cos(lng * 0.05)), 2),
            "srtm_elevation": round(50 + 800 * abs(math.sin(lat * 0.3)), 1),
            "esa_worldcover": 30
        }
        return {
            "status": "success",
            "mode": "stub",
            "dataset_id": dataset_id,
            "value": synthetic_values.get(dataset_id, 0),
            "unit": spec["unit"],
            "lat": lat,
            "lng": lng,
            "note": "Synthesized stub value"
        }

    if not _initialize_ee_once():
        return {"status": "error", "message": "GEE initialization failed"}

    try:
        point = ee.Geometry.Point([lng, lat])
        if spec.get("static"):
            img = ee.Image(spec["ee_collection"]).select(spec["band"])
        else:
            sd = start or "2024-01-01"
            ed = end or "2024-12-31"
            collection = ee.ImageCollection(spec["ee_collection"]).select(spec["band"]).filterDate(sd, ed)
            img = collection.mean()
        value = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=point, scale=1000).getInfo()
        v = value.get(spec["band"], None)
        return {
            "status": "success",
            "mode": "live_gee",
            "dataset_id": dataset_id,
            "value": v,
            "unit": spec["unit"],
            "lat": lat,
            "lng": lng
        }
    except Exception as e:
        return {"status": "error", "message": f"Pixel query failed · {e}"}


def list_datasets() -> Dict[str, Any]:
    """Returns the full dataset catalog for frontend dropdown."""
    return {
        "status": "success",
        "live_mode": is_live(),
        "datasets": [
            {
                "id": k,
                "label": v["label"],
                "description": v["description"],
                "unit": v["unit"],
                "static": v.get("static", False)
            }
            for k, v in DATASET_CATALOG.items()
        ]
    }


def pull_panel(province_ids: List[str], start: str, end: str) -> Dict[str, Any]:
    """Returns a dict envelope with the dataframe + provenance metadata.
    Live mode talks to GEE; stub mode synthesises plausible data."""
    if not HAS_PANDAS:
        return {"status": "error", "message": "pandas/numpy unavailable"}
    t0 = time.perf_counter()
    try:
        if is_live():
            df = _pull_real(province_ids, start, end)
            mode = "live_gee"
        else:
            df = _synthesize_panel(province_ids, start, end)
            mode = "stub_synthetic"
    except Exception as e:
        return {"status": "error", "message": f"GEE pull failed: {e}",
                "duration_ms": int((time.perf_counter() - t0) * 1000)}

    return {
        "status": "success",
        "mode": mode,
        "rows": int(len(df)),
        "columns": list(df.columns),
        "data": df,
        "duration_ms": int((time.perf_counter() - t0) * 1000),
        "method_monitor": {
            "method": ("Live GEE pull (Gorelick et al., 2017)"
                       if mode == "live_gee"
                       else "Synthetic stub — climatologically plausible (DJF wet bias)"),
            "limitations": [
                "Real GEE pull issues per-day per-province queries; for 38 prov × 10 yr "
                "(~140 k calls) switch to image-collection batch export for production.",
                "Stub mode uses a deterministic seed (42) so results are reproducible.",
            ],
            "citations": [
                "Gorelick et al. (2017) Remote Sens. Environ. 202 — Google Earth Engine.",
                "Funk et al. (2015) Sci. Data 2:150066 — CHIRPS.",
                "Vermote (2015) NASA LP DAAC — MOD09GA.",
                "Beaudoing & Rodell (2020) — GLDAS-2.1 Noah.",
            ],
        },
    }
