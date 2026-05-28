"""
Heatmap Generator
=================
Produces 2D density grid for heatmap rendering. Optionally weighted by metric.
"""
from __future__ import annotations
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np


class HeatmapGenerator:
    """Generate 2D heatmap data with optional weighting."""

    def generate(
        self,
        df: pd.DataFrame,
        lat_col: str,
        lon_col: str,
        weight_col: Optional[str] = None,
        n_bins: int = 30,
        gaussian_smoothing: bool = True,
    ) -> Dict[str, Any]:
        """
        Args:
            df: Dataframe with lat/lon
            weight_col: optional column to weight density (e.g., revenue)
            n_bins: grid resolution
            gaussian_smoothing: apply 3x3 Gaussian blur for smoother heatmap
        """
        df_clean = df.dropna(subset=[lat_col, lon_col])
        if len(df_clean) == 0:
            return {"error": "No valid coordinates"}

        lat_min, lat_max = float(df_clean[lat_col].min()), float(df_clean[lat_col].max())
        lon_min, lon_max = float(df_clean[lon_col].min()), float(df_clean[lon_col].max())

        lat_edges = np.linspace(lat_min, lat_max, n_bins + 1)
        lon_edges = np.linspace(lon_min, lon_max, n_bins + 1)

        weights = (
            pd.to_numeric(df_clean[weight_col], errors="coerce").fillna(0).values
            if weight_col and weight_col in df_clean.columns
            else np.ones(len(df_clean))
        )

        grid, _, _ = np.histogram2d(
            df_clean[lat_col].values,
            df_clean[lon_col].values,
            bins=[lat_edges, lon_edges],
            weights=weights,
        )

        if gaussian_smoothing:
            grid = self._gaussian_smooth(grid)

        # Normalize for visualization
        grid_max = grid.max() if grid.max() > 0 else 1
        normalized = grid / grid_max

        cells = []
        for i in range(n_bins):
            for j in range(n_bins):
                if grid[i, j] > 0:
                    cells.append({
                        "lat": float((lat_edges[i] + lat_edges[i + 1]) / 2),
                        "lon": float((lon_edges[j] + lon_edges[j + 1]) / 2),
                        "intensity": float(normalized[i, j]),
                        "raw_value": float(grid[i, j]),
                    })

        return {
            "n_bins": n_bins,
            "bounding_box": {
                "min_lat": lat_min, "max_lat": lat_max,
                "min_lon": lon_min, "max_lon": lon_max,
            },
            "weight_column": weight_col,
            "max_intensity_value": float(grid_max),
            "cells": cells,
            "smoothed": gaussian_smoothing,
            "method_explanation": (
                f"2D histogram with {n_bins}x{n_bins} bins"
                + (f", weighted by '{weight_col}'" if weight_col else "")
                + (", with 3x3 Gaussian smoothing for visual clarity" if gaussian_smoothing else "")
                + ". Intensity normalized to [0,1] for color mapping."
            ),
        }

    def _gaussian_smooth(self, grid: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        """Apply simple 3x3 Gaussian smoothing without scipy dependency."""
        kernel = np.array([
            [1, 2, 1],
            [2, 4, 2],
            [1, 2, 1],
        ]) / 16.0

        rows, cols = grid.shape
        smoothed = grid.copy()
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                smoothed[i, j] = float(np.sum(grid[i-1:i+2, j-1:j+2] * kernel))
        return smoothed
