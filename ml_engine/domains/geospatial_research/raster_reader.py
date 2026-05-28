"""
RasterReader - GeoTIFF metadata extraction.
========================================
Baca file .tif yang diekspor dari Google Earth Engine atau sumber lain.
Gracefully degrade jika rasterio belum terinstall — return mock metadata.
"""
from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import os


class RasterReader:
    """Read GeoTIFF metadata dan pixel arrays."""

    def __init__(self):
        self._rio = self._try_import_rasterio()

    def _try_import_rasterio(self):
        try:
            import rasterio
            return rasterio
        except ImportError:
            return None

    def is_available(self) -> bool:
        return self._rio is not None

    def read_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Baca metadata GeoTIFF tanpa load full pixel array (memory-efficient).

        Returns lengkap:
        - crs (EPSG code), width, height, count (n_bands), resolution
        - bounding box, transform, nodata, dtype
        - per-band: min, max, mean, std (jika dapat dihitung cepat)
        """
        path = Path(file_path)
        if not path.exists():
            return {"available": False, "error": f"File tidak ditemukan: {file_path}"}

        if not self.is_available():
            return self._mock_metadata(file_path)

        try:
            with self._rio.open(file_path) as src:
                bounds = src.bounds
                transform = src.transform
                meta = {
                    "available": True,
                    "file_path": str(path),
                    "file_size_bytes": int(path.stat().st_size),
                    "driver": src.driver,
                    "crs": str(src.crs) if src.crs else None,
                    "crs_epsg": src.crs.to_epsg() if src.crs else None,
                    "width": int(src.width),
                    "height": int(src.height),
                    "n_bands": int(src.count),
                    "dtype": str(src.dtypes[0]) if src.dtypes else None,
                    "nodata": src.nodata,
                    "resolution_x": float(transform.a),
                    "resolution_y": float(abs(transform.e)),
                    "bounding_box": {
                        "min_x": float(bounds.left),
                        "min_y": float(bounds.bottom),
                        "max_x": float(bounds.right),
                        "max_y": float(bounds.top),
                    },
                    "transform": list(transform[:6]),
                    "tags": dict(src.tags()),
                    "bands": self._read_bands_summary(src),
                }
                return meta
        except Exception as e:
            return {"available": False, "error": str(e), "file_path": str(path)}

    def _read_bands_summary(self, src) -> List[Dict[str, Any]]:
        """Hitung statistik per band."""
        import numpy as np
        bands = []
        for i in range(1, src.count + 1):
            band_data = src.read(i, masked=True)
            valid = band_data.compressed() if hasattr(band_data, "compressed") else band_data[~np.isnan(band_data)]
            if len(valid) == 0:
                bands.append({"band_id": i, "n_valid_pixels": 0, "all_nodata": True})
                continue
            n_total = src.width * src.height
            n_nodata = n_total - len(valid)
            band_name = src.descriptions[i - 1] if src.descriptions and src.descriptions[i - 1] else f"band_{i}"
            bands.append({
                "band_id": i,
                "band_name": band_name,
                "min_value": float(valid.min()),
                "max_value": float(valid.max()),
                "mean_value": float(valid.mean()),
                "std_value": float(valid.std()),
                "n_valid_pixels": int(len(valid)),
                "n_nodata_pixels": int(n_nodata),
                "nodata_percentage": round((n_nodata / n_total) * 100, 2),
            })
        return bands

    def read_array(self, file_path: str, band_idx: int = 1) -> Optional[Any]:
        """Read full pixel array untuk satu band."""
        if not self.is_available():
            return None
        try:
            with self._rio.open(file_path) as src:
                return src.read(band_idx, masked=True)
        except Exception:
            return None

    def read_all_bands(self, file_path: str) -> Optional[Any]:
        """Read semua band sebagai 3D array (n_bands, height, width)."""
        if not self.is_available():
            return None
        try:
            with self._rio.open(file_path) as src:
                return src.read(masked=True)
        except Exception:
            return None

    def _mock_metadata(self, file_path: str) -> Dict[str, Any]:
        """Mock metadata untuk pengembangan tanpa rasterio."""
        return {
            "available": False,
            "fallback_mode": True,
            "warning": "rasterio not installed - returning mock metadata. Install: pip install rasterio",
            "file_path": file_path,
            "crs": "EPSG:4326",
            "crs_epsg": 4326,
            "width": 512,
            "height": 512,
            "n_bands": 4,
            "dtype": "float32",
            "nodata": -9999.0,
            "resolution_x": 0.0001,
            "resolution_y": 0.0001,
            "bounding_box": {
                "min_x": 106.7, "min_y": -6.3,
                "max_x": 106.9, "max_y": -6.1,
            },
            "bands": [
                {"band_id": 1, "band_name": "blue", "min_value": 0.05, "max_value": 0.45,
                 "mean_value": 0.15, "std_value": 0.07, "nodata_percentage": 2.1},
                {"band_id": 2, "band_name": "green", "min_value": 0.08, "max_value": 0.52,
                 "mean_value": 0.18, "std_value": 0.09, "nodata_percentage": 2.1},
                {"band_id": 3, "band_name": "red", "min_value": 0.06, "max_value": 0.58,
                 "mean_value": 0.16, "std_value": 0.10, "nodata_percentage": 2.1},
                {"band_id": 4, "band_name": "nir", "min_value": 0.10, "max_value": 0.60,
                 "mean_value": 0.30, "std_value": 0.12, "nodata_percentage": 2.1},
            ],
        }

    def histogram(self, file_path: str, band_idx: int = 1, n_bins: int = 50) -> Dict[str, Any]:
        """Hitung histogram pixel value untuk satu band."""
        if not self.is_available():
            return {"available": False, "warning": "rasterio not installed"}
        import numpy as np
        try:
            with self._rio.open(file_path) as src:
                data = src.read(band_idx, masked=True)
                valid = data.compressed() if hasattr(data, "compressed") else data[~np.isnan(data)]
                counts, edges = np.histogram(valid, bins=n_bins)
                return {
                    "available": True,
                    "band_id": band_idx,
                    "n_bins": n_bins,
                    "edges": edges.tolist(),
                    "counts": counts.tolist(),
                    "min_value": float(valid.min()),
                    "max_value": float(valid.max()),
                    "n_valid_pixels": int(len(valid)),
                }
        except Exception as e:
            return {"available": False, "error": str(e)}
