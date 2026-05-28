"""
Spatial Cluster Detector
========================
DBSCAN-based density clustering of geographic points. Detects natural
neighborhoods and outlier locations.

Reference:
    Ester et al. (1996). A density-based algorithm for discovering clusters.
"""
from __future__ import annotations
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np


class SpatialClusterDetector:
    """Cluster locations using DBSCAN with haversine metric."""

    def cluster(
        self,
        df: pd.DataFrame,
        lat_col: str,
        lon_col: str,
        eps_km: float = 5.0,
        min_samples: int = 5,
    ) -> Dict[str, Any]:
        """
        Args:
            eps_km: max distance (km) between points in same cluster
            min_samples: minimum points to form a cluster
        """
        df_clean = df.dropna(subset=[lat_col, lon_col]).copy().reset_index(drop=True)
        if len(df_clean) < min_samples:
            return {
                "error": f"Need at least {min_samples} valid points; got {len(df_clean)}",
                "n_clusters": 0,
                "n_outliers": 0,
            }

        try:
            from sklearn.cluster import DBSCAN
        except ImportError:
            return {"error": "scikit-learn not installed"}

        # Convert to radians for haversine metric
        coords_rad = np.radians(df_clean[[lat_col, lon_col]].values)
        eps_rad = eps_km / 6371.0  # km to radians on earth

        model = DBSCAN(
            eps=eps_rad,
            min_samples=min_samples,
            metric="haversine",
            algorithm="ball_tree",
        )
        labels = model.fit_predict(coords_rad)

        df_clean["cluster"] = labels
        n_clusters = int(len(set(labels)) - (1 if -1 in labels else 0))
        n_outliers = int((labels == -1).sum())

        # Per-cluster summary
        clusters = []
        for cid in sorted(set(labels)):
            if cid == -1:
                continue
            members = df_clean[df_clean["cluster"] == cid]
            clusters.append({
                "cluster_id": int(cid),
                "n_points": int(len(members)),
                "centroid_lat": float(members[lat_col].mean()),
                "centroid_lon": float(members[lon_col].mean()),
                "spread_km": float(self._cluster_spread(members, lat_col, lon_col)),
            })

        clusters.sort(key=lambda c: c["n_points"], reverse=True)

        return {
            "method": "DBSCAN with haversine metric",
            "eps_km": eps_km,
            "min_samples": min_samples,
            "n_total_points": int(len(df_clean)),
            "n_clusters": n_clusters,
            "n_outliers": n_outliers,
            "outlier_rate": round(n_outliers / len(df_clean), 4),
            "clusters": clusters,
            "labels": labels.tolist(),
            "method_explanation": (
                f"DBSCAN detected {n_clusters} dense clusters "
                f"and {n_outliers} outlier points using eps={eps_km}km, min_samples={min_samples}. "
                f"Outliers are isolated locations that may represent rare/unusual sites. "
                f"Reference: Ester et al. (1996)."
            ),
            "method_monitor": {
                "selected_method": "DBSCAN",
                "why_chosen": (
                    "DBSCAN is density-based, doesn't require pre-specifying number of clusters, "
                    "naturally identifies outliers, and works well on geographic data with "
                    "varying density. The haversine metric correctly handles distances on Earth's surface."
                ),
                "why_not_alternatives": [
                    {"alternative": "K-Means", "reason_rejected": "Requires k upfront; can't identify outliers; assumes spherical clusters"},
                    {"alternative": "Hierarchical", "reason_rejected": "O(n²) memory, slow on large geographic datasets"},
                    {"alternative": "OPTICS", "reason_rejected": "More complex; DBSCAN sufficient for most cases"},
                ],
                "limitations": [
                    "eps and min_samples need tuning per domain",
                    "Struggles when clusters have very different densities",
                ],
            },
        }

    def _cluster_spread(self, members: pd.DataFrame, lat_col: str, lon_col: str) -> float:
        """Average distance of cluster members from their centroid."""
        if len(members) <= 1:
            return 0.0
        from .spatial_analyzer import SpatialAnalyzer
        sa = SpatialAnalyzer()
        center_lat = members[lat_col].mean()
        center_lon = members[lon_col].mean()
        distances = [
            sa.haversine_distance(row[lat_col], row[lon_col], center_lat, center_lon)
            for _, row in members.iterrows()
        ]
        return float(np.mean(distances))
