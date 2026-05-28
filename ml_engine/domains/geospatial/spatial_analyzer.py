"""
Spatial Analyzer - core geospatial functions.
"""
from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional
import math
import pandas as pd
import numpy as np


class SpatialAnalyzer:
    """Distance, density, and basic spatial statistics."""

    EARTH_RADIUS_KM = 6371.0

    def haversine_distance(
        self, lat1: float, lon1: float, lat2: float, lon2: float
    ) -> float:
        """Distance in km between two lat/lon points."""
        lat1_r, lat2_r = math.radians(lat1), math.radians(lat2)
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return self.EARTH_RADIUS_KM * c

    def validate_coordinates(self, df: pd.DataFrame, lat_col: str, lon_col: str) -> Dict[str, Any]:
        """CRISP-DM Data Understanding: validate coordinate columns."""
        issues = []
        n_total = len(df)

        if lat_col not in df.columns or lon_col not in df.columns:
            return {
                "valid": False,
                "error": f"Columns '{lat_col}' and '{lon_col}' not found",
            }

        lat = pd.to_numeric(df[lat_col], errors="coerce")
        lon = pd.to_numeric(df[lon_col], errors="coerce")

        n_missing = int((lat.isna() | lon.isna()).sum())
        n_out_of_range = int(
            ((lat < -90) | (lat > 90) | (lon < -180) | (lon > 180)).sum()
        )

        if n_missing > 0:
            issues.append({
                "type": "missing_coords",
                "severity": "warning",
                "count": n_missing,
                "description": f"{n_missing} rows have missing or non-numeric coordinates",
            })
        if n_out_of_range > 0:
            issues.append({
                "type": "out_of_range",
                "severity": "critical",
                "count": n_out_of_range,
                "description": f"{n_out_of_range} rows have coordinates outside Earth bounds",
            })

        return {
            "valid": n_out_of_range == 0,
            "n_total_rows": n_total,
            "n_valid_rows": int(lat.notna().sum() - n_out_of_range),
            "n_missing": n_missing,
            "n_out_of_range": n_out_of_range,
            "lat_range": [float(lat.min()), float(lat.max())] if lat.notna().any() else None,
            "lon_range": [float(lon.min()), float(lon.max())] if lon.notna().any() else None,
            "issues": issues,
        }

    def bounding_box(self, df: pd.DataFrame, lat_col: str, lon_col: str) -> Dict[str, float]:
        """Compute geographic bounding box."""
        return {
            "min_lat": float(df[lat_col].min()),
            "max_lat": float(df[lat_col].max()),
            "min_lon": float(df[lon_col].min()),
            "max_lon": float(df[lon_col].max()),
            "center_lat": float(df[lat_col].mean()),
            "center_lon": float(df[lon_col].mean()),
        }

    def density_grid(
        self,
        df: pd.DataFrame,
        lat_col: str,
        lon_col: str,
        n_bins: int = 30,
    ) -> Dict[str, Any]:
        """Bin points into a 2D grid for heatmap visualization."""
        bbox = self.bounding_box(df, lat_col, lon_col)
        lat_edges = np.linspace(bbox["min_lat"], bbox["max_lat"], n_bins + 1)
        lon_edges = np.linspace(bbox["min_lon"], bbox["max_lon"], n_bins + 1)

        lat_idx = np.digitize(df[lat_col].values, lat_edges) - 1
        lon_idx = np.digitize(df[lon_col].values, lon_edges) - 1
        lat_idx = np.clip(lat_idx, 0, n_bins - 1)
        lon_idx = np.clip(lon_idx, 0, n_bins - 1)

        grid = np.zeros((n_bins, n_bins), dtype=int)
        for li, lo in zip(lat_idx, lon_idx):
            grid[li, lo] += 1

        cells = []
        for i in range(n_bins):
            for j in range(n_bins):
                if grid[i, j] > 0:
                    cells.append({
                        "lat_min": float(lat_edges[i]),
                        "lat_max": float(lat_edges[i + 1]),
                        "lon_min": float(lon_edges[j]),
                        "lon_max": float(lon_edges[j + 1]),
                        "count": int(grid[i, j]),
                    })

        return {
            "n_bins": n_bins,
            "bounding_box": bbox,
            "max_count": int(grid.max()),
            "cells": cells,
            "method_summary": (
                f"Density grid with {n_bins}x{n_bins} bins. Max cell count = {int(grid.max())}. "
                f"Higher count cells indicate hotspot locations."
            ),
        }

    def spatial_distribution_stats(
        self,
        df: pd.DataFrame,
        lat_col: str,
        lon_col: str,
    ) -> Dict[str, Any]:
        """
        Compute spatial distribution statistics (centroid, dispersion, NN distance).
        Useful for understanding if data is uniform, clustered, or dispersed.
        """
        bbox = self.bounding_box(df, lat_col, lon_col)
        center_lat = bbox["center_lat"]
        center_lon = bbox["center_lon"]

        # Distance from each point to centroid
        distances = []
        for _, row in df.iterrows():
            d = self.haversine_distance(
                row[lat_col], row[lon_col], center_lat, center_lon
            )
            distances.append(d)
        distances = np.array(distances)

        return {
            "n_points": int(len(df)),
            "centroid_lat": center_lat,
            "centroid_lon": center_lon,
            "mean_distance_to_centroid_km": float(distances.mean()),
            "std_distance_to_centroid_km": float(distances.std()),
            "max_distance_to_centroid_km": float(distances.max()),
            "spatial_dispersion": self._classify_dispersion(distances.std(), distances.mean()),
        }

    def _classify_dispersion(self, std: float, mean: float) -> str:
        if mean == 0:
            return "single_point"
        cv = std / mean
        if cv < 0.3:
            return "concentrated"
        if cv < 0.7:
            return "moderate"
        return "dispersed"
