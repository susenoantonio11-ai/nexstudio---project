"""
Google Earth Engine Integration
=================================
Wrapper untuk earthengine-api Python.

Mendukung:
- Authentication (service account atau interactive)
- Sentinel-1 SAR composite (VV, VH)
- Sentinel-2 surface reflectance composite (cloud-free median)
- Landsat-8 surface reflectance
- Server-side index computation (NDWI, MNDWI, NDVI)
- Export ke GeoTIFF via Google Drive atau direct download URL

Setup:
    pip install earthengine-api
    earthengine authenticate    # untuk personal use
    # ATAU
    service account JSON untuk production

Reference: https://developers.google.com/earth-engine/guides/python_install
"""
from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
import os
from datetime import datetime, timedelta


def _try_import_ee():
    try:
        import ee
        return ee
    except ImportError:
        return None


_EE = _try_import_ee()


def is_ee_available() -> bool:
    return _EE is not None


class GEEIntegration:
    """Google Earth Engine wrapper untuk Nexlytics flood research."""

    def __init__(
        self,
        service_account: Optional[str] = None,
        key_file: Optional[str] = None,
    ):
        self.service_account = service_account
        self.key_file = key_file
        self._initialized = False

    def is_available(self) -> bool:
        return is_ee_available()

    def initialize(self) -> Dict[str, Any]:
        """
        Authenticate dan initialize Earth Engine.

        Methods:
        1. Service account (production): provide service_account email + JSON key file
        2. Interactive (development): assumes `earthengine authenticate` was run
        """
        if not is_ee_available():
            return {
                "available": False,
                "status": "not_installed",
                "reason": "earthengine-api belum terinstall",
                "install_steps": [
                    "pip install earthengine-api",
                    "Untuk service account production: download key JSON dari Google Cloud Console",
                    "Untuk dev/personal: jalankan `earthengine authenticate` di terminal",
                ],
            }

        try:
            if self.service_account and self.key_file and os.path.exists(self.key_file):
                credentials = _EE.ServiceAccountCredentials(
                    self.service_account, self.key_file
                )
                _EE.Initialize(credentials)
                auth_method = "service_account"
            else:
                _EE.Initialize()
                auth_method = "interactive_or_default"
            self._initialized = True
            return {
                "available": True,
                "status": "initialized",
                "authentication_method": auth_method,
            }
        except Exception as e:
            return {
                "available": False,
                "status": "auth_failed",
                "error": str(e),
                "hint": "Run `earthengine authenticate` first, atau provide valid service account.",
            }

    def fetch_sentinel2_composite(
        self,
        bbox: Tuple[float, float, float, float],
        start_date: str,
        end_date: str,
        max_cloud_cover: float = 20.0,
    ) -> Dict[str, Any]:
        """
        Fetch Sentinel-2 surface reflectance median composite.

        Args:
            bbox: (min_x, min_y, max_x, max_y) dalam EPSG:4326
            start_date, end_date: ISO format 'YYYY-MM-DD'
            max_cloud_cover: skip image dengan cloud > X%

        Returns:
            ee.Image object jika berhasil + metadata.
        """
        if not self._initialized:
            init_result = self.initialize()
            if not init_result["available"]:
                return init_result

        try:
            geometry = _EE.Geometry.Rectangle(list(bbox))
            collection = (
                _EE.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                .filterBounds(geometry)
                .filterDate(start_date, end_date)
                .filter(_EE.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud_cover))
            )

            n_images = collection.size().getInfo()
            if n_images == 0:
                return {
                    "available": False,
                    "reason": f"Tidak ada Sentinel-2 image dengan cloud < {max_cloud_cover}% di periode {start_date} sampai {end_date}",
                }

            # Median composite untuk cloud-free image
            composite = collection.median().clip(geometry)

            return {
                "available": True,
                "source": "Sentinel-2 SR Harmonized",
                "ee_image": composite,
                "geometry": geometry,
                "n_source_images": n_images,
                "bands": ["B2 (blue)", "B3 (green)", "B4 (red)", "B8 (NIR)",
                          "B11 (SWIR1)", "B12 (SWIR2)"],
                "resolution_m": 10,
                "method_monitor": {
                    "selected_method": "Sentinel-2 median composite",
                    "why_chosen": (
                        "Median composite dari multiple acquisitions = cloud-free output. "
                        "Median lebih robust dari mean terhadap outlier (residual cloud)."
                    ),
                    "why_not_alternatives": [
                        {"alternative": "Single image", "reason_rejected": "Sering ada awan"},
                        {"alternative": "Mean composite", "reason_rejected": "Tidak robust ke outlier (cloud shadow)"},
                    ],
                },
            }
        except Exception as e:
            return {"available": False, "error": str(e)}

    def fetch_sentinel1_sar(
        self,
        bbox: Tuple[float, float, float, float],
        start_date: str,
        end_date: str,
        polarization: str = "VV",
        orbit_pass: str = "DESCENDING",
    ) -> Dict[str, Any]:
        """
        Fetch Sentinel-1 SAR median backscatter (dB).
        Critical untuk flood karena tembus awan.
        """
        if not self._initialized:
            init_result = self.initialize()
            if not init_result["available"]:
                return init_result

        try:
            geometry = _EE.Geometry.Rectangle(list(bbox))
            collection = (
                _EE.ImageCollection("COPERNICUS/S1_GRD")
                .filterBounds(geometry)
                .filterDate(start_date, end_date)
                .filter(_EE.Filter.listContains("transmitterReceiverPolarisation", polarization))
                .filter(_EE.Filter.eq("orbitProperties_pass", orbit_pass))
                .filter(_EE.Filter.eq("instrumentMode", "IW"))
                .select(polarization)
            )

            n_images = collection.size().getInfo()
            if n_images == 0:
                return {
                    "available": False,
                    "reason": f"Tidak ada Sentinel-1 SAR {polarization} di periode tersebut",
                }

            composite = collection.median().clip(geometry)
            return {
                "available": True,
                "source": f"Sentinel-1 SAR GRD ({polarization}, {orbit_pass})",
                "ee_image": composite,
                "geometry": geometry,
                "n_source_images": n_images,
                "polarization": polarization,
                "unit": "dB",
                "resolution_m": 10,
            }
        except Exception as e:
            return {"available": False, "error": str(e)}

    def compute_mndwi_server_side(self, ee_image: Any) -> Dict[str, Any]:
        """Compute MNDWI on Earth Engine server (no local processing)."""
        if not is_ee_available():
            return {"available": False}
        try:
            green = ee_image.select("B3")
            swir = ee_image.select("B11")
            mndwi = green.subtract(swir).divide(green.add(swir)).rename("MNDWI")
            return {
                "available": True,
                "ee_index": mndwi,
                "formula": "(B3 - B11) / (B3 + B11)",
                "interpretation": "MNDWI > 0 → water; > 0.3 → confident water",
            }
        except Exception as e:
            return {"available": False, "error": str(e)}

    def compute_ndwi_server_side(self, ee_image: Any) -> Dict[str, Any]:
        if not is_ee_available():
            return {"available": False}
        try:
            green = ee_image.select("B3")
            nir = ee_image.select("B8")
            ndwi = green.subtract(nir).divide(green.add(nir)).rename("NDWI")
            return {
                "available": True,
                "ee_index": ndwi,
                "formula": "(B3 - B8) / (B3 + B8)",
                "interpretation": "NDWI > 0.3 → water (McFeeters 1996)",
            }
        except Exception as e:
            return {"available": False, "error": str(e)}

    def compute_ndvi_server_side(self, ee_image: Any) -> Dict[str, Any]:
        if not is_ee_available():
            return {"available": False}
        try:
            nir = ee_image.select("B8")
            red = ee_image.select("B4")
            ndvi = nir.subtract(red).divide(nir.add(red)).rename("NDVI")
            return {
                "available": True,
                "ee_index": ndvi,
                "formula": "(B8 - B4) / (B8 + B4)",
            }
        except Exception as e:
            return {"available": False, "error": str(e)}

    def get_download_url(
        self,
        ee_image: Any,
        bbox: Tuple[float, float, float, float],
        scale: int = 10,
        crs: str = "EPSG:4326",
        name: str = "nexlytics_export",
    ) -> Dict[str, Any]:
        """
        Generate signed download URL untuk ee.Image.
        Limit: 32 MB per download — for larger, use export to Drive.
        """
        if not is_ee_available():
            return {"available": False}
        try:
            geometry = _EE.Geometry.Rectangle(list(bbox))
            url = ee_image.getDownloadURL({
                "scale": scale,
                "crs": crs,
                "region": geometry,
                "format": "GEO_TIFF",
                "name": name,
            })
            return {
                "available": True,
                "download_url": url,
                "warning_size_limit": "Max 32MB per request — pakai export_to_drive untuk area besar",
            }
        except Exception as e:
            return {"available": False, "error": str(e)}

    def export_to_drive(
        self,
        ee_image: Any,
        bbox: Tuple[float, float, float, float],
        description: str = "nexlytics_export",
        folder: str = "nexlytics",
        scale: int = 10,
    ) -> Dict[str, Any]:
        """
        Export ke Google Drive sebagai GeoTIFF.
        Untuk area besar — async task.
        """
        if not is_ee_available():
            return {"available": False}
        try:
            geometry = _EE.Geometry.Rectangle(list(bbox))
            task = _EE.batch.Export.image.toDrive(
                image=ee_image,
                description=description,
                folder=folder,
                fileNamePrefix=description,
                scale=scale,
                region=geometry,
                fileFormat="GeoTIFF",
                maxPixels=1e10,
            )
            task.start()
            return {
                "available": True,
                "task_id": task.id,
                "status": "started",
                "drive_folder": folder,
                "description": description,
                "monitor_url": f"https://code.earthengine.google.com/tasks?task_id={task.id}",
            }
        except Exception as e:
            return {"available": False, "error": str(e)}

    def get_image_info(self, ee_image: Any) -> Dict[str, Any]:
        """Get metadata server-side."""
        if not is_ee_available():
            return {"available": False}
        try:
            info = ee_image.getInfo()
            return {"available": True, "info": info}
        except Exception as e:
            return {"available": False, "error": str(e)}


# ==================================================================
# QUICK-USE FUNCTIONS
# ==================================================================
def quick_fetch_jakarta_flood_imagery(
    start_date: str = "2024-01-01",
    end_date: str = "2024-01-31",
) -> Dict[str, Any]:
    """
    Quick example: fetch Jakarta Sentinel-2 + Sentinel-1 untuk flood research.
    Use case: flood event Jakarta Januari 2024.
    """
    JAKARTA_BBOX = (106.7, -6.3, 106.9, -6.1)

    gee = GEEIntegration()
    init = gee.initialize()
    if not init["available"]:
        return init

    s2 = gee.fetch_sentinel2_composite(JAKARTA_BBOX, start_date, end_date)
    s1 = gee.fetch_sentinel1_sar(JAKARTA_BBOX, start_date, end_date)

    return {
        "study_area": "Jakarta",
        "bbox": JAKARTA_BBOX,
        "period": (start_date, end_date),
        "sentinel2": {
            "available": s2.get("available"),
            "n_images": s2.get("n_source_images"),
        },
        "sentinel1": {
            "available": s1.get("available"),
            "n_images": s1.get("n_source_images"),
        },
        "next_steps": [
            "Compute MNDWI server-side: gee.compute_mndwi_server_side(s2['ee_image'])",
            "Generate download URL: gee.get_download_url(image, bbox)",
            "Atau export ke Drive untuk area besar",
        ],
    }
