"""
Route / Movement Analyzer
==========================
Analyzes movement trajectories - sequences of geographic points over time.
Useful for fleet tracking, customer journey, delivery optimization.
"""
from __future__ import annotations
from typing import Dict, Any, List
import pandas as pd
import numpy as np


class RouteAnalyzer:
    """Analyze movement trajectories."""

    def analyze(
        self,
        df: pd.DataFrame,
        entity_column: str,
        timestamp_column: str,
        lat_col: str,
        lon_col: str,
    ) -> Dict[str, Any]:
        """
        Args:
            entity_column: column identifying each moving entity (e.g., 'vehicle_id')
            timestamp_column: time column for ordering
        """
        from .spatial_analyzer import SpatialAnalyzer
        sa = SpatialAnalyzer()

        df = df.copy()
        df[timestamp_column] = pd.to_datetime(df[timestamp_column], errors="coerce")
        df = df.dropna(subset=[lat_col, lon_col, timestamp_column, entity_column])
        df = df.sort_values([entity_column, timestamp_column])

        routes = []
        for entity, group in df.groupby(entity_column):
            if len(group) < 2:
                continue
            group = group.reset_index(drop=True)
            total_distance = 0.0
            segment_distances = []
            for i in range(1, len(group)):
                d = sa.haversine_distance(
                    group.loc[i - 1, lat_col], group.loc[i - 1, lon_col],
                    group.loc[i, lat_col], group.loc[i, lon_col],
                )
                total_distance += d
                segment_distances.append(d)

            duration_hours = (
                group[timestamp_column].max() - group[timestamp_column].min()
            ).total_seconds() / 3600

            avg_speed = total_distance / duration_hours if duration_hours > 0 else 0

            routes.append({
                "entity_id": str(entity),
                "n_waypoints": int(len(group)),
                "total_distance_km": round(float(total_distance), 2),
                "duration_hours": round(float(duration_hours), 2),
                "avg_speed_kmh": round(float(avg_speed), 2),
                "max_segment_km": round(float(max(segment_distances)), 2),
                "min_segment_km": round(float(min(segment_distances)), 2),
                "start_lat": float(group.iloc[0][lat_col]),
                "start_lon": float(group.iloc[0][lon_col]),
                "end_lat": float(group.iloc[-1][lat_col]),
                "end_lon": float(group.iloc[-1][lon_col]),
            })

        if not routes:
            return {"n_routes": 0, "error": "No valid routes (need >=2 waypoints per entity)"}

        return {
            "n_routes": len(routes),
            "routes": routes,
            "summary": {
                "total_distance_km": round(sum(r["total_distance_km"] for r in routes), 2),
                "avg_distance_per_route_km": round(np.mean([r["total_distance_km"] for r in routes]), 2),
                "avg_speed_kmh": round(np.mean([r["avg_speed_kmh"] for r in routes]), 2),
            },
            "method_explanation": (
                f"Analyzed {len(routes)} movement routes by computing haversine distance between "
                f"consecutive waypoints sorted by timestamp. Useful for fleet optimization, "
                f"delivery tracking, and customer journey analysis."
            ),
        }
