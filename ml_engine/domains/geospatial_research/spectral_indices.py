"""
Spectral Index Calculator
=========================
Menghitung indeks geospasial dari multi-band imagery.

Mendukung:
- NDVI (Normalized Difference Vegetation Index)
- NDWI (McFeeters - Normalized Difference Water Index)
- MNDWI (Modified NDWI - Xu) — paling akurat untuk deteksi air
- NDBI (Normalized Difference Built-up Index)
- VV/VH ratio (untuk SAR Sentinel-1)
- Slope (dari DEM)
- Distance to river (dari river vector)
"""
from __future__ import annotations
from typing import Dict, Any, Optional
import numpy as np


class SpectralIndexCalculator:
    """Calculator untuk indeks geospasial standard."""

    # Reference: each formula source
    INDEX_FORMULAS = {
        "ndvi": "(NIR − Red) / (NIR + Red)",
        "ndwi": "(Green − NIR) / (Green + NIR)",
        "mndwi": "(Green − SWIR) / (Green + SWIR)",
        "ndbi": "(SWIR − NIR) / (SWIR + NIR)",
        "vv_vh_ratio": "VV / VH",
        "ndmi": "(NIR − SWIR) / (NIR + SWIR)",
    }

    def calculate_ndvi(self, red: np.ndarray, nir: np.ndarray) -> Dict[str, Any]:
        """
        NDVI = (NIR − Red) / (NIR + Red)
        Range: [-1, 1]. > 0.4 = vegetasi sehat. < 0 = air/bare soil.
        """
        red = np.asarray(red, dtype=float)
        nir = np.asarray(nir, dtype=float)
        denom = nir + red
        denom = np.where(denom == 0, np.nan, denom)
        ndvi = (nir - red) / denom
        return self._build_result("NDVI", ndvi, {
            "interpretation_thresholds": {
                "water": "< 0",
                "bare_soil": "0 to 0.2",
                "sparse_vegetation": "0.2 to 0.4",
                "healthy_vegetation": "> 0.4",
            },
            "purpose": "Mengidentifikasi vegetasi sehat. Berguna untuk konteks pre-flood (deteksi area hijau yang berubah jadi air).",
        })

    def calculate_ndwi(self, green: np.ndarray, nir: np.ndarray) -> Dict[str, Any]:
        """
        NDWI (McFeeters 1996) = (Green − NIR) / (Green + NIR)
        Air biasanya > 0.3. Sensitif terhadap built-up environment.
        """
        green = np.asarray(green, dtype=float)
        nir = np.asarray(nir, dtype=float)
        denom = green + nir
        denom = np.where(denom == 0, np.nan, denom)
        ndwi = (green - nir) / denom
        return self._build_result("NDWI", ndwi, {
            "interpretation_thresholds": {
                "non_water": "< 0",
                "weak_water_signal": "0 to 0.3",
                "water": "> 0.3",
            },
            "reference": "McFeeters (1996) - The use of the Normalized Difference Water Index (NDWI) in the delineation of open water features",
            "purpose": "Deteksi air. Klasik tapi suka false-positive di area built-up (gedung). MNDWI lebih akurat.",
        })

    def calculate_mndwi(self, green: np.ndarray, swir: np.ndarray) -> Dict[str, Any]:
        """
        MNDWI (Xu 2006) = (Green − SWIR) / (Green + SWIR)
        Lebih baik dari NDWI karena menekan reflektansi gedung.
        """
        green = np.asarray(green, dtype=float)
        swir = np.asarray(swir, dtype=float)
        denom = green + swir
        denom = np.where(denom == 0, np.nan, denom)
        mndwi = (green - swir) / denom
        return self._build_result("MNDWI", mndwi, {
            "interpretation_thresholds": {
                "non_water": "< 0",
                "water_likely": "0 to 0.3",
                "water_confirmed": "> 0.3",
            },
            "reference": "Xu (2006) - Modification of normalized difference water index (NDWI) to enhance open water features in remotely sensed imagery",
            "purpose": "Indeks PALING akurat untuk deteksi air. Lebih baik dari NDWI di area urban karena SWIR menekan reflektansi gedung.",
        })

    def calculate_ndbi(self, nir: np.ndarray, swir: np.ndarray) -> Dict[str, Any]:
        """NDBI = (SWIR − NIR) / (SWIR + NIR). Mendeteksi area built-up."""
        nir = np.asarray(nir, dtype=float)
        swir = np.asarray(swir, dtype=float)
        denom = swir + nir
        denom = np.where(denom == 0, np.nan, denom)
        ndbi = (swir - nir) / denom
        return self._build_result("NDBI", ndbi, {
            "interpretation_thresholds": {
                "natural_surface": "< 0",
                "mixed": "0 to 0.1",
                "built_up": "> 0.1",
            },
            "purpose": "Identifikasi area built-up (urban). Berguna untuk damage assessment pasca banjir di area perkotaan.",
        })

    def calculate_vv_vh_ratio(self, vv: np.ndarray, vh: np.ndarray) -> Dict[str, Any]:
        """
        VV/VH ratio dari SAR Sentinel-1.
        Air permukaan = nilai VV rendah & rasio VV/VH unik.
        """
        vv = np.asarray(vv, dtype=float)
        vh = np.asarray(vh, dtype=float)
        vh_safe = np.where(vh == 0, np.nan, vh)
        ratio = vv / vh_safe
        return self._build_result("VV/VH ratio", ratio, {
            "interpretation": (
                "VV polarization sangat sensitif terhadap permukaan air (sangat reflektif jika tenang). "
                "VV/VH ratio digunakan untuk membedakan air dari permukaan kasar lainnya."
            ),
            "purpose": "Deteksi air permukaan menggunakan SAR — bekerja siang/malam dan tembus awan (Sentinel-1)",
        })

    def calculate_slope(self, dem: np.ndarray, pixel_size_x: float = 30.0, pixel_size_y: float = 30.0) -> Dict[str, Any]:
        """
        Slope (in degrees) dari Digital Elevation Model.
        Slope penting untuk flood susceptibility — slope rendah = lebih mudah tergenang.
        """
        dem = np.asarray(dem, dtype=float)
        # Sobel-like gradient
        dz_dx = np.gradient(dem, axis=1) / pixel_size_x
        dz_dy = np.gradient(dem, axis=0) / pixel_size_y
        slope_rad = np.arctan(np.sqrt(dz_dx ** 2 + dz_dy ** 2))
        slope_deg = np.degrees(slope_rad)
        return self._build_result("Slope (degrees)", slope_deg, {
            "interpretation_thresholds": {
                "flat (high flood risk)": "< 2°",
                "gentle": "2° to 8°",
                "moderate": "8° to 30°",
                "steep (low flood risk)": "> 30°",
            },
            "purpose": "Slope kecil = drainase lambat = risiko banjir tinggi. Faktor penting untuk flood susceptibility mapping.",
        })

    def calculate_all_indices(
        self,
        bands: Dict[str, np.ndarray],
    ) -> Dict[str, Any]:
        """
        Hitung semua indeks yang mungkin dari bands yang tersedia.
        bands: dict mapping band_name → numpy array, misal:
            {"red": array, "nir": array, "green": array, "swir": array}
        """
        results = {}
        if "red" in bands and "nir" in bands:
            results["ndvi"] = self.calculate_ndvi(bands["red"], bands["nir"])
        if "green" in bands and "nir" in bands:
            results["ndwi"] = self.calculate_ndwi(bands["green"], bands["nir"])
        if "green" in bands and "swir" in bands:
            results["mndwi"] = self.calculate_mndwi(bands["green"], bands["swir"])
        if "nir" in bands and "swir" in bands:
            results["ndbi"] = self.calculate_ndbi(bands["nir"], bands["swir"])
        if "vv" in bands and "vh" in bands:
            results["vv_vh_ratio"] = self.calculate_vv_vh_ratio(bands["vv"], bands["vh"])
        if "dem" in bands or "elevation" in bands:
            dem = bands.get("dem", bands.get("elevation"))
            results["slope"] = self.calculate_slope(dem)

        return {
            "indices_calculated": list(results.keys()),
            "n_indices": len(results),
            "results": results,
        }

    def _build_result(self, name: str, array: np.ndarray, extra: Dict[str, Any]) -> Dict[str, Any]:
        valid = array[~np.isnan(array)] if array.dtype.kind == "f" else array
        return {
            "name": name,
            "array": array,  # 2D numpy array
            "shape": list(array.shape),
            "stats": {
                "min": float(valid.min()) if len(valid) else None,
                "max": float(valid.max()) if len(valid) else None,
                "mean": float(valid.mean()) if len(valid) else None,
                "std": float(valid.std()) if len(valid) else None,
                "n_valid": int(len(valid)),
                "n_nan": int(np.isnan(array).sum()) if array.dtype.kind == "f" else 0,
            },
            "formula": self.INDEX_FORMULAS.get(name.lower().replace(" (degrees)", "").replace("/", "_"), ""),
            **extra,
        }
