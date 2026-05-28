"""
Time-Series Flood Evolution Tracker
=====================================
Backend untuk track perubahan flood extent over time.

Workflow:
1. Fetch multi-date imagery (Sentinel-1/2 dari GEE atau lokal)
2. Apply flood detection per timestamp
3. Compute statistics per timestamp (area, %, peak time, recede time)
4. Generate frames untuk animation
5. Detect peak flood event + extent

Output: time-series JSON yang frontend bisa play sebagai animasi Leaflet.
"""
from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta


class TimeSeriesFloodTracker:
    """Track flood evolution across multiple timestamps."""

    def __init__(self):
        self.frames: List[Dict[str, Any]] = []

    def add_frame(
        self,
        timestamp: str,            # ISO format date
        flood_mask,                 # 2D numpy array atau path GeoTIFF
        bounds: Tuple[float, float, float, float],
        wms_url: Optional[str] = None,
        png_url: Optional[str] = None,
        method: str = "MNDWI threshold",
    ) -> Dict[str, Any]:
        """
        Tambah single frame ke time-series.

        Args:
            timestamp: ISO date e.g., '2024-01-15'
            flood_mask: numpy 2D atau path. Untuk array, akan dihitung statistik langsung
            bounds: (min_x, min_y, max_x, max_y) lat/lon
            wms_url: optional URL WMS dari GeoServer
            png_url: optional path/URL PNG overlay
        """
        try:
            import numpy as np
            if hasattr(flood_mask, "shape"):
                # numpy array
                mask = flood_mask
                n_total = int(mask.size)
                n_flooded = int(mask.sum())
            else:
                # path or other - try opening
                try:
                    import rasterio
                    with rasterio.open(flood_mask) as src:
                        mask = src.read(1)
                        n_total = int(mask.size)
                        n_flooded = int((mask == 1).sum())
                except Exception:
                    return {"success": False, "error": "Cannot read flood_mask"}

            flooded_pct = round((n_flooded / n_total) * 100, 2) if n_total else 0
            area_km2 = self._estimate_area_km2(n_flooded, bounds, mask.shape)

            frame = {
                "timestamp": timestamp,
                "frame_index": len(self.frames),
                "method": method,
                "bounds": list(bounds),
                "raster_shape": list(mask.shape),
                "stats": {
                    "n_flooded_pixels": n_flooded,
                    "n_total_pixels": n_total,
                    "flooded_percentage": flooded_pct,
                    "estimated_area_km2": area_km2,
                },
                "wms_url": wms_url,
                "png_url": png_url,
            }
            self.frames.append(frame)
            return {"success": True, "frame": frame}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_evolution_summary(self) -> Dict[str, Any]:
        """Hitung statistik agregat seluruh time-series."""
        if not self.frames:
            return {"available": False, "reason": "No frames added yet"}

        sorted_frames = sorted(self.frames, key=lambda f: f["timestamp"])
        flooded_pcts = [f["stats"]["flooded_percentage"] for f in sorted_frames]
        areas = [f["stats"]["estimated_area_km2"] for f in sorted_frames]
        timestamps = [f["timestamp"] for f in sorted_frames]

        # Peak flood
        peak_idx = max(range(len(flooded_pcts)), key=lambda i: flooded_pcts[i])
        # Lowest (baseline)
        min_idx = min(range(len(flooded_pcts)), key=lambda i: flooded_pcts[i])

        # Detect rising / receding phase
        phases = self._detect_phases(timestamps, flooded_pcts)

        return {
            "available": True,
            "n_frames": len(sorted_frames),
            "date_range": {
                "start": timestamps[0],
                "end": timestamps[-1],
            },
            "peak_flood": {
                "timestamp": timestamps[peak_idx],
                "flooded_percentage": flooded_pcts[peak_idx],
                "area_km2": areas[peak_idx],
                "frame_index": peak_idx,
            },
            "minimum_flood": {
                "timestamp": timestamps[min_idx],
                "flooded_percentage": flooded_pcts[min_idx],
                "area_km2": areas[min_idx],
            },
            "evolution_curve": [
                {"timestamp": t, "flooded_percentage": p, "area_km2": a}
                for t, p, a in zip(timestamps, flooded_pcts, areas)
            ],
            "phases": phases,
            "frames": sorted_frames,
            "method_monitor": {
                "selected_method": "Time-series flood evolution tracking",
                "why_chosen": (
                    "Banjir dinamis: muncul, mencapai puncak, lalu surut. "
                    "Single timestamp hanya snapshot — time-series memberikan FULL DAMAGE ASSESSMENT, "
                    "lokasi peak, durasi inundation, dan recovery rate per area."
                ),
                "use_cases": [
                    "Emergency response: prediksi peak time + lokasi terparah",
                    "Insurance assessment: durasi flooding per area",
                    "Climate research: pola temporal banjir multi-tahun",
                    "Infrastructure planning: identifikasi recurring flood zones",
                ],
                "limitations": [
                    "Frequency tergantung ketersediaan satellite (Sentinel-2 ~5 hari, Sentinel-1 ~6 hari)",
                    "Cloud cover bisa miss event puncak",
                    "Resolusi temporal kalah dari sensor in-situ",
                ],
            },
        }

    def _estimate_area_km2(
        self,
        n_flooded_pixels: int,
        bounds: Tuple[float, float, float, float],
        shape: Tuple[int, int],
    ) -> float:
        """Estimasi area km² dari pixel count + bounds (asumsi EPSG:4326)."""
        try:
            import math
            min_x, min_y, max_x, max_y = bounds
            H, W = shape
            # Approx: 1 degree lat ~111 km, 1 degree lon ~111*cos(lat) km
            mid_lat = (min_y + max_y) / 2
            deg_per_pixel_lat = (max_y - min_y) / H
            deg_per_pixel_lon = (max_x - min_x) / W
            km_per_pixel_lat = deg_per_pixel_lat * 111.0
            km_per_pixel_lon = deg_per_pixel_lon * 111.0 * math.cos(math.radians(mid_lat))
            area_per_pixel_km2 = km_per_pixel_lat * km_per_pixel_lon
            return round(n_flooded_pixels * area_per_pixel_km2, 2)
        except Exception:
            return 0.0

    def _detect_phases(
        self,
        timestamps: List[str],
        percentages: List[float],
    ) -> List[Dict[str, Any]]:
        """
        Detect rising / peak / receding phases.

        Naive approach: bandingkan dengan frame sebelumnya, label as:
        - 'rising' jika naik > 1%
        - 'receding' jika turun > 1%
        - 'stable' jika perubahan kecil
        """
        if len(percentages) < 2:
            return []

        phases = []
        for i in range(1, len(percentages)):
            delta = percentages[i] - percentages[i - 1]
            if delta > 1:
                phase = "rising"
            elif delta < -1:
                phase = "receding"
            else:
                phase = "stable"
            phases.append({
                "from": timestamps[i - 1],
                "to": timestamps[i],
                "delta_percentage": round(delta, 2),
                "phase": phase,
            })
        return phases

    def to_animation_json(self) -> Dict[str, Any]:
        """Format khusus untuk frontend animation playback."""
        summary = self.get_evolution_summary()
        if not summary.get("available"):
            return summary

        return {
            "type": "flood_time_series_animation",
            "version": "1.0",
            "n_frames": summary["n_frames"],
            "duration_ms_per_frame": 800,
            "auto_play": True,
            "loop": True,
            "frames": [
                {
                    "id": i,
                    "timestamp": f["timestamp"],
                    "label": _humanize_date(f["timestamp"]),
                    "flooded_percentage": f["stats"]["flooded_percentage"],
                    "area_km2": f["stats"]["estimated_area_km2"],
                    "wms_url": f.get("wms_url"),
                    "png_url": f.get("png_url"),
                    "bounds": f["bounds"],
                }
                for i, f in enumerate(summary["frames"])
            ],
            "evolution_curve": summary["evolution_curve"],
            "peak_flood": summary["peak_flood"],
            "minimum_flood": summary["minimum_flood"],
            "phases": summary["phases"],
        }


def _humanize_date(iso_date: str) -> str:
    """Convert ISO date jadi human-readable label."""
    try:
        d = datetime.fromisoformat(iso_date.split("T")[0])
        return d.strftime("%d %b %Y")
    except Exception:
        return iso_date


# ==================================================================
# QUICK-USE HELPERS
# ==================================================================
def fetch_time_series_via_gee(
    bbox: Tuple[float, float, float, float],
    start_date: str,
    end_date: str,
    interval_days: int = 7,
    method: str = "mndwi",
) -> Dict[str, Any]:
    """
    Quick: fetch multi-timestamp Sentinel-2 + apply MNDWI threshold.
    Returns time-series data siap untuk TimeSeriesFloodTracker.

    Note: requires GEE setup.
    """
    from .gee_integration import GEEIntegration, is_ee_available

    if not is_ee_available():
        return {
            "available": False,
            "reason": "earthengine-api belum terinstall",
            "install": "pip install earthengine-api && earthengine authenticate",
        }

    gee = GEEIntegration()
    init = gee.initialize()
    if not init.get("available"):
        return init

    # Generate intervals
    start = datetime.fromisoformat(start_date)
    end = datetime.fromisoformat(end_date)
    intervals = []
    current = start
    while current <= end:
        next_date = min(current + timedelta(days=interval_days), end)
        intervals.append((current.isoformat()[:10], next_date.isoformat()[:10]))
        current = next_date + timedelta(days=1)

    return {
        "available": True,
        "bbox": list(bbox),
        "n_intervals": len(intervals),
        "intervals": intervals,
        "method": method,
        "next_step": (
            "Iterate: untuk setiap interval, fetch Sentinel-2 composite + compute MNDWI, "
            "kemudian add_frame() ke TimeSeriesFloodTracker."
        ),
    }
