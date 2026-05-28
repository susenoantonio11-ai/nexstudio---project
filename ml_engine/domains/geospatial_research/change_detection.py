"""
ChangeDetector - before/after flood comparison.
================================================
Mendeteksi area baru tergenang dengan membandingkan raster pre-flood vs post-flood.
"""
from __future__ import annotations
from typing import Dict, Any, Optional
import numpy as np


class ChangeDetector:
    """Before-after change detection untuk flood mapping."""

    def detect_water_change(
        self,
        before_water_mask: np.ndarray,
        after_water_mask: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Compare water mask pre-flood vs post-flood.

        Returns:
            new_flood_mask: pixel yang baru tergenang (after=1, before=0)
            persistent_water: water di kedua waktu
            receded_water: water sebelum tapi tidak setelah
        """
        before = before_water_mask.astype(bool)
        after = after_water_mask.astype(bool)

        new_flood = np.logical_and(after, ~before)         # tergenang baru
        persistent = np.logical_and(before, after)         # selalu air (sungai/danau)
        receded = np.logical_and(before, ~after)           # surut

        n_total = int(before.size)
        return {
            "method": "Pixel-wise water mask change detection",
            "n_total_pixels": n_total,
            "new_flood_mask": new_flood.astype(np.uint8),
            "persistent_water_mask": persistent.astype(np.uint8),
            "receded_water_mask": receded.astype(np.uint8),
            "summary": {
                "new_flood_pixels": int(new_flood.sum()),
                "new_flood_percentage": round(float(new_flood.sum()) / n_total * 100, 2),
                "persistent_water_pixels": int(persistent.sum()),
                "persistent_water_percentage": round(float(persistent.sum()) / n_total * 100, 2),
                "receded_water_pixels": int(receded.sum()),
                "receded_water_percentage": round(float(receded.sum()) / n_total * 100, 2),
            },
            "method_monitor": {
                "selected_method": "Pre-post water mask delta",
                "why_chosen": (
                    "Cara paling jelas mengisolasi NEW FLOOD: pixel air post-flood yang "
                    "TIDAK air pre-flood. Memisahkan permanent water (sungai/danau) dari flooding."
                ),
                "why_not_alternatives": [
                    {"alternative": "Image differencing", "reason_rejected": "Sensitif terhadap perubahan pencahayaan"},
                    {"alternative": "PCA-based change", "reason_rejected": "Tidak interpretable, sulit divalidasi"},
                ],
                "limitations": [
                    "Dependent pada akurasi water masks pre/post",
                    "Tidak menangkap flood yang sudah surut sebelum citra diambil",
                ],
            },
        }

    def detect_index_change(
        self,
        before_index: np.ndarray,
        after_index: np.ndarray,
        threshold: float = 0.2,
    ) -> Dict[str, Any]:
        """Detect significant change in spectral index (e.g., NDWI/MNDWI)."""
        delta = after_index - before_index
        increase_mask = (delta > threshold).astype(np.uint8)
        decrease_mask = (delta < -threshold).astype(np.uint8)
        return {
            "method": "Index delta thresholding",
            "threshold": threshold,
            "delta_array": delta,
            "significant_increase_mask": increase_mask,
            "significant_decrease_mask": decrease_mask,
            "summary": {
                "n_increased": int(increase_mask.sum()),
                "n_decreased": int(decrease_mask.sum()),
                "delta_mean": float(np.nanmean(delta)),
                "delta_std": float(np.nanstd(delta)),
            },
        }
