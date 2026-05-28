"""
ThresholdFloodClassifier - rule-based water/flood detection.
=============================================================
Untuk kasus tanpa label training, gunakan threshold pada indeks geospasial.
Cepat, deterministic, dan ilmiah-validated thresholds.
"""
from __future__ import annotations
from typing import Dict, Any, List, Optional
import numpy as np


class ThresholdFloodClassifier:
    """Klasifikasi banjir berbasis threshold pada indeks spektral."""

    DEFAULT_THRESHOLDS = {
        "ndwi": 0.3,    # > 0.3 = water (McFeeters, 1996)
        "mndwi": 0.0,   # > 0 = water (Xu, 2006); >0.3 = water confirmed
        "ndvi": 0.0,    # < 0 = water/non-vegetation
        "vv_db": -17.0, # SAR VV in dB; < -17 = water (Twele et al., 2016)
    }

    def classify_with_mndwi(
        self,
        mndwi: np.ndarray,
        threshold: float = 0.0,
    ) -> Dict[str, Any]:
        """Klasifikasi air dengan MNDWI threshold (paling akurat)."""
        flood_mask = (mndwi > threshold).astype(np.uint8)
        n_flooded = int(flood_mask.sum())
        n_total = int(flood_mask.size)
        return {
            "method": "MNDWI threshold",
            "threshold": threshold,
            "flood_mask": flood_mask,
            "n_flooded_pixels": n_flooded,
            "n_total_pixels": n_total,
            "flooded_percentage": round((n_flooded / n_total) * 100, 2) if n_total else 0,
            "method_monitor": {
                "selected_method": "MNDWI threshold",
                "why_chosen": (
                    "MNDWI (Modified NDWI) lebih akurat daripada NDWI klasik karena menekan "
                    "false-positive di area built-up (gedung). Threshold 0 sudah validated di literatur."
                ),
                "why_not_alternatives": [
                    {"alternative": "NDWI", "reason_rejected": "Lebih banyak false-positive di urban area"},
                    {"alternative": "RF supervised", "reason_rejected": "Membutuhkan labeled training data — threshold tidak butuh label"},
                ],
                "limitations": [
                    "Akurasi tergantung kualitas spektral imagery (cloud, shadow)",
                    "Threshold tunggal tidak adaptif terhadap kondisi pencahayaan",
                ],
                "reference": "Xu (2006). Modification of normalized difference water index (NDWI)",
            },
        }

    def classify_with_ndwi(
        self,
        ndwi: np.ndarray,
        threshold: float = 0.3,
    ) -> Dict[str, Any]:
        """Klasifikasi air dengan NDWI threshold."""
        flood_mask = (ndwi > threshold).astype(np.uint8)
        n_flooded = int(flood_mask.sum())
        n_total = int(flood_mask.size)
        return {
            "method": "NDWI threshold",
            "threshold": threshold,
            "flood_mask": flood_mask,
            "n_flooded_pixels": n_flooded,
            "n_total_pixels": n_total,
            "flooded_percentage": round((n_flooded / n_total) * 100, 2) if n_total else 0,
            "method_monitor": {
                "selected_method": "NDWI threshold",
                "why_chosen": "Klasik & sederhana untuk Sentinel-2/Landsat. Threshold > 0.3 untuk air.",
                "limitations": ["Bisa salah deteksi gedung sebagai air. Pakai MNDWI jika ada SWIR band."],
            },
        }

    def classify_with_sar_vv(
        self,
        vv_db: np.ndarray,
        threshold: float = -17.0,
    ) -> Dict[str, Any]:
        """
        SAR-based water detection: pixel dengan VV < -17 dB (sangat dark) = water.
        Reference: Twele et al. (2016) - Sentinel-1-based flood mapping.
        """
        flood_mask = (vv_db < threshold).astype(np.uint8)
        n_flooded = int(flood_mask.sum())
        n_total = int(flood_mask.size)
        return {
            "method": "SAR VV threshold",
            "threshold": threshold,
            "flood_mask": flood_mask,
            "n_flooded_pixels": n_flooded,
            "n_total_pixels": n_total,
            "flooded_percentage": round((n_flooded / n_total) * 100, 2) if n_total else 0,
            "method_monitor": {
                "selected_method": "SAR VV backscatter threshold",
                "why_chosen": (
                    "SAR (Sentinel-1) menembus awan & berfungsi siang/malam — ideal untuk flood "
                    "yang sering terjadi saat cuaca buruk. Air permukaan VV sangat rendah karena "
                    "specular reflection."
                ),
                "why_not_alternatives": [
                    {"alternative": "Optical (NDWI/MNDWI)", "reason_rejected": "Tidak bisa tembus awan — flood sering tertutup awan"},
                ],
                "reference": "Twele et al. (2016). Sentinel-1-based flood mapping: a fully automated processing chain.",
                "limitations": [
                    "Wind-roughened water bisa false-negative (VV tidak serendah air tenang)",
                    "Smooth surfaces (asphalt basah, tarmac) bisa false-positive",
                ],
            },
        }

    def classify_combined(
        self,
        mndwi: Optional[np.ndarray] = None,
        ndwi: Optional[np.ndarray] = None,
        vv_db: Optional[np.ndarray] = None,
        slope: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Combined rule: pixel = flood jika MULTIPLE indikator agree.
        Lebih konservatif tapi akurasi tinggi.
        """
        masks: List[np.ndarray] = []
        method_components: List[str] = []
        if mndwi is not None:
            masks.append(mndwi > self.DEFAULT_THRESHOLDS["mndwi"])
            method_components.append("MNDWI > 0")
        if ndwi is not None:
            masks.append(ndwi > self.DEFAULT_THRESHOLDS["ndwi"])
            method_components.append("NDWI > 0.3")
        if vv_db is not None:
            masks.append(vv_db < self.DEFAULT_THRESHOLDS["vv_db"])
            method_components.append("SAR VV < -17 dB")
        if slope is not None:
            # Flood susceptibility filter — skip slope curam (slope > 15° kurang mungkin banjir)
            masks.append(slope < 15)
            method_components.append("slope < 15°")

        if not masks:
            return {"error": "Tidak ada indikator yang disediakan"}

        # Pixel dianggap flood jika SEMUA indikator yang tersedia agree
        combined = np.logical_and.reduce(masks)
        flood_mask = combined.astype(np.uint8)
        n_flooded = int(flood_mask.sum())
        n_total = int(flood_mask.size)

        return {
            "method": "Combined threshold (consensus)",
            "components": method_components,
            "n_indicators": len(masks),
            "flood_mask": flood_mask,
            "n_flooded_pixels": n_flooded,
            "n_total_pixels": n_total,
            "flooded_percentage": round((n_flooded / n_total) * 100, 2) if n_total else 0,
            "method_monitor": {
                "selected_method": "Multi-indicator consensus",
                "why_chosen": (
                    f"Menggabungkan {len(masks)} indikator: {', '.join(method_components)}. "
                    f"Pixel dianggap flood hanya jika SEMUA indikator setuju → false-positive rate rendah."
                ),
                "limitations": [
                    "Konservatif — bisa miss flood sebenarnya",
                    "Threshold per indikator perlu validasi domain",
                ],
            },
        }
