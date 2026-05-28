"""
Choropleth Builder
==================
Aggregates a numeric metric per region (column) and produces data structure
suitable for choropleth map rendering.
"""
from __future__ import annotations
from typing import Dict, Any, List
import pandas as pd
import numpy as np


class ChoroplethBuilder:
    """Build choropleth-ready aggregations."""

    AGG_METHODS = ["sum", "mean", "median", "count", "min", "max"]

    def build(
        self,
        df: pd.DataFrame,
        region_column: str,
        metric_column: str,
        agg_method: str = "sum",
    ) -> Dict[str, Any]:
        """
        Args:
            df: Source dataframe
            region_column: column with region/area names (e.g., 'province', 'state')
            metric_column: numeric column to aggregate (e.g., 'revenue', 'population')
            agg_method: sum / mean / median / count / min / max
        """
        if region_column not in df.columns:
            raise ValueError(f"Region column '{region_column}' not in dataframe")

        if agg_method not in self.AGG_METHODS:
            agg_method = "sum"

        if metric_column == "<count>" or agg_method == "count":
            grouped = df.groupby(region_column).size().reset_index(name="value")
            agg_method_used = "count"
        else:
            if metric_column not in df.columns:
                raise ValueError(f"Metric column '{metric_column}' not in dataframe")
            if not pd.api.types.is_numeric_dtype(df[metric_column]):
                # Coerce
                df = df.copy()
                df[metric_column] = pd.to_numeric(df[metric_column], errors="coerce")

            grouped = df.groupby(region_column)[metric_column].agg(agg_method).reset_index(name="value")
            agg_method_used = agg_method

        grouped = grouped.sort_values("value", ascending=False).reset_index(drop=True)

        regions = []
        for _, row in grouped.iterrows():
            regions.append({
                "region": str(row[region_column]),
                "value": float(row["value"]),
            })

        values = grouped["value"].values
        return {
            "agg_method": agg_method_used,
            "metric_column": metric_column,
            "region_column": region_column,
            "n_regions": int(len(regions)),
            "min_value": float(values.min()) if len(values) > 0 else 0,
            "max_value": float(values.max()) if len(values) > 0 else 0,
            "mean_value": float(values.mean()) if len(values) > 0 else 0,
            "median_value": float(np.median(values)) if len(values) > 0 else 0,
            "regions": regions,
            "top_5": regions[:5],
            "bottom_5": regions[-5:] if len(regions) > 5 else [],
            "method_explanation": (
                f"Aggregated '{metric_column}' by '{region_column}' using {agg_method_used}. "
                f"Result reveals which regions concentrate the metric, useful for "
                f"resource allocation and market prioritization."
            ),
        }
