"""
RasterPreprocessor - reproject, resample, clip, mask, stack.
=============================================================
Preprocess pipeline untuk multi-band raster sebelum modeling.
"""
from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
import os
from pathlib import Path


class RasterPreprocessor:
    """Pipeline preprocessing raster: reproject, resample, clip, mask, stack."""

    def __init__(self):
        self._rio = None
        try:
            import rasterio
            self._rio = rasterio
        except ImportError:
            pass

    def is_available(self) -> bool:
        return self._rio is not None

    def reproject(
        self,
        input_path: str,
        output_path: str,
        target_crs: str = "EPSG:4326",
    ) -> Dict[str, Any]:
        """Reproject raster ke CRS target."""
        if not self.is_available():
            return self._stub("reproject", input_path, target_crs)

        try:
            from rasterio.warp import calculate_default_transform, reproject, Resampling
            with self._rio.open(input_path) as src:
                transform, width, height = calculate_default_transform(
                    src.crs, target_crs, src.width, src.height, *src.bounds
                )
                kwargs = src.meta.copy()
                kwargs.update({
                    "crs": target_crs, "transform": transform,
                    "width": width, "height": height,
                })
                with self._rio.open(output_path, "w", **kwargs) as dst:
                    for i in range(1, src.count + 1):
                        reproject(
                            source=self._rio.band(src, i),
                            destination=self._rio.band(dst, i),
                            src_transform=src.transform, src_crs=src.crs,
                            dst_transform=transform, dst_crs=target_crs,
                            resampling=Resampling.nearest,
                        )
            return {
                "status": "success", "operation": "reproject",
                "input": input_path, "output": output_path,
                "source_crs": str(src.crs) if src.crs else None,
                "target_crs": target_crs,
                "method_monitor": {
                    "selected_method": "GDAL warp + bilinear/nearest resampling",
                    "why_chosen": "Standar industri untuk reprojection raster, mendukung semua CRS yang terdaftar di EPSG",
                    "why_not_alternatives": [
                        {"alternative": "manual coordinate transformation", "reason_rejected": "Error-prone untuk skala besar"},
                    ],
                },
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def resample(
        self,
        input_path: str,
        output_path: str,
        target_resolution: float = 30.0,
    ) -> Dict[str, Any]:
        """Resample ke resolusi target (meter atau degree, tergantung CRS)."""
        if not self.is_available():
            return self._stub("resample", input_path, target_resolution)

        try:
            from rasterio.enums import Resampling
            with self._rio.open(input_path) as src:
                scale_x = src.transform.a / target_resolution
                scale_y = abs(src.transform.e) / target_resolution
                new_height = int(src.height * scale_y)
                new_width = int(src.width * scale_x)
                data = src.read(
                    out_shape=(src.count, new_height, new_width),
                    resampling=Resampling.bilinear,
                )
                new_transform = src.transform * src.transform.scale(
                    src.width / new_width, src.height / new_height
                )
                kwargs = src.meta.copy()
                kwargs.update({
                    "transform": new_transform,
                    "width": new_width, "height": new_height,
                })
                with self._rio.open(output_path, "w", **kwargs) as dst:
                    dst.write(data)
            return {
                "status": "success", "operation": "resample",
                "input": input_path, "output": output_path,
                "target_resolution": target_resolution,
                "new_dimensions": (new_height, new_width),
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def clip_by_bbox(
        self,
        input_path: str,
        output_path: str,
        bbox: Tuple[float, float, float, float],
    ) -> Dict[str, Any]:
        """Clip raster ke bounding box (min_x, min_y, max_x, max_y)."""
        if not self.is_available():
            return self._stub("clip_by_bbox", input_path, bbox)

        try:
            from rasterio.windows import from_bounds
            with self._rio.open(input_path) as src:
                window = from_bounds(*bbox, transform=src.transform)
                data = src.read(window=window)
                new_transform = src.window_transform(window)
                kwargs = src.meta.copy()
                kwargs.update({
                    "transform": new_transform,
                    "width": int(window.width), "height": int(window.height),
                })
                with self._rio.open(output_path, "w", **kwargs) as dst:
                    dst.write(data)
            return {
                "status": "success", "operation": "clip", "bbox": list(bbox),
                "input": input_path, "output": output_path,
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def stack_bands(
        self,
        band_paths: List[str],
        output_path: str,
        band_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Stack beberapa single-band raster jadi multi-band stack."""
        if not self.is_available():
            return self._stub("stack_bands", band_paths, None)

        try:
            import numpy as np
            arrays = []
            ref_meta = None
            for p in band_paths:
                with self._rio.open(p) as src:
                    arrays.append(src.read(1))
                    if ref_meta is None:
                        ref_meta = src.meta.copy()
            stacked = np.stack(arrays, axis=0)
            ref_meta["count"] = len(arrays)
            with self._rio.open(output_path, "w", **ref_meta) as dst:
                dst.write(stacked)
                if band_names:
                    dst.descriptions = tuple(band_names)
            return {
                "status": "success", "operation": "stack_bands",
                "n_bands_stacked": len(arrays),
                "band_names": band_names or [f"band_{i+1}" for i in range(len(arrays))],
                "output": output_path,
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def to_feature_matrix(
        self,
        raster_path: str,
        valid_only: bool = True,
    ) -> Dict[str, Any]:
        """
        Convert raster pixel jadi tabular feature matrix.
        Setiap row = 1 pixel, setiap column = 1 band.
        """
        if not self.is_available():
            return self._stub("to_feature_matrix", raster_path, None)

        try:
            import numpy as np
            with self._rio.open(raster_path) as src:
                data = src.read(masked=True)  # (n_bands, H, W)
                n_bands, H, W = data.shape
                # Reshape ke (H*W, n_bands)
                X = data.reshape(n_bands, -1).T
                if valid_only and hasattr(data, "mask"):
                    valid_mask = ~np.any(data.mask.reshape(n_bands, -1).T, axis=1) if data.mask.ndim > 0 else None
                    if valid_mask is not None:
                        X = X[valid_mask]
            return {
                "status": "success",
                "shape": list(X.shape),
                "n_pixels": int(X.shape[0]),
                "n_bands": int(X.shape[1]),
                "X_array": X,  # numpy array
                "method_explanation": (
                    "Konversi raster 3D (bands, height, width) menjadi tabular 2D (pixels, bands). "
                    "Setiap baris = 1 pixel, setiap kolom = 1 band/feature. "
                    "Format ini siap untuk model machine learning standar."
                ),
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def normalize(self, X, method: str = "minmax") -> Dict[str, Any]:
        """Normalize feature matrix per band."""
        try:
            import numpy as np
            X = np.asarray(X, dtype=float)
            if method == "minmax":
                mins = np.nanmin(X, axis=0)
                maxs = np.nanmax(X, axis=0)
                ranges = np.where(maxs - mins == 0, 1, maxs - mins)
                X_norm = (X - mins) / ranges
                stats = {"mins": mins.tolist(), "maxs": maxs.tolist()}
            elif method == "standard":
                means = np.nanmean(X, axis=0)
                stds = np.nanstd(X, axis=0)
                stds = np.where(stds == 0, 1, stds)
                X_norm = (X - means) / stds
                stats = {"means": means.tolist(), "stds": stds.tolist()}
            else:
                X_norm = X
                stats = {}
            return {
                "status": "success", "method": method,
                "X_normalized": X_norm, "stats": stats,
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _stub(self, operation: str, *args) -> Dict[str, Any]:
        return {
            "status": "stub",
            "operation": operation,
            "warning": "rasterio not installed - operation skipped. Install: pip install rasterio",
            "args": [str(a)[:80] for a in args],
        }
