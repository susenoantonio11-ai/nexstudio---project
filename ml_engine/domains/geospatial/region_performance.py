"""
Region Performance Analyzer
============================
Compare regions across multiple KPIs and rank them.
Useful for identifying best/worst-performing markets, store locations, etc.
"""
from __future__ import annotations
from typing import Dict, Any, List
import pandas as pd
import numpy as np


class RegionPerformanceAnalyzer:
    """Multi-KPI region comparison."""

    def analyze(
        self,
        df: pd.DataFrame,
        region_column: str,
        kpi_columns: List[str],
        kpi_higher_is_better: Dict[str, bool] = None,
    ) -> Dict[str, Any]:
        """
        Args:
            kpi_columns: list of numeric KPI columns to compare
            kpi_higher_is_better: {kpi_name: bool} per KPI direction
        """
        if kpi_higher_is_better is None:
            kpi_higher_is_better = {kpi: True for kpi in kpi_columns}

        # Aggregate per region
        agg = df.groupby(region_column).agg({
            **{kpi: "sum" for kpi in kpi_columns},
        }).reset_index()
        agg["transaction_count"] = df.groupby(region_column).size().values

        # Normalize each KPI to [0, 1] for composite scoring
        normalized = agg.copy()
        for kpi in kpi_columns:
            if kpi not in agg.columns:
                continue
            values = agg[kpi].astype(float)
            if values.max() == values.min():
                normalized[kpi + "_score"] = 0.5
            else:
                norm = (values - values.min()) / (values.max() - values.min())
                if not kpi_higher_is_better.get(kpi, True):
                    norm = 1 - norm
                normalized[kpi + "_score"] = norm

        # Composite score = mean of normalized KPIs
        score_cols = [c for c in normalized.columns if c.endswith("_score")]
        normalized["composite_score"] = normalized[score_cols].mean(axis=1)
        normalized = normalized.sort_values("composite_score", ascending=False).reset_index(drop=True)

        regions = []
        for _, row in normalized.iterrows():
            r = {
                "region": str(row[region_column]),
                "transaction_count": int(row["transaction_count"]),
                "composite_score": round(float(row["composite_score"]), 4),
                "rank": int(normalized.index[normalized[region_column] == row[region_column]][0]) + 1,
            }
            for kpi in kpi_columns:
                r[kpi] = float(row[kpi])
                r[kpi + "_normalized"] = round(float(row[kpi + "_score"]), 4)
            regions.append(r)

        # Identify outperformers and underperformers
        n = len(regions)
        top_quartile = regions[: max(1, n // 4)]
        bottom_quartile = regions[-max(1, n // 4):]

        return {
            "n_regions": n,
            "kpi_columns": kpi_columns,
            "regions_ranked": regions,
            "top_quartile": top_quartile,
            "bottom_quartile": bottom_quartile,
            "best_region": regions[0] if regions else None,
            "worst_region": regions[-1] if regions else None,
            "method_explanation": (
                f"Ranked {n} regions across {len(kpi_columns)} KPIs using min-max normalization "
                f"and composite score. Each KPI normalized to [0,1]; "
                f"composite = unweighted mean. Top quartile represents outperformers."
            ),
            "method_monitor": {
                "selected_method": "Min-max normalization + composite ranking",
                "why_chosen": (
                    "Min-max preserves relative differences between regions and scales each KPI "
                    "to comparable units. Composite mean treats KPIs equally; could be weighted "
                    "if business priorities differ."
                ),
                "why_not_alternatives": [
                    {
                        "alternative": "Z-score normalization",
                        "reason_rejected": "Assumes normal distribution; min-max is more robust for KPIs with skew"
                    },
                    {
                        "alternative": "Single-KPI ranking",
                        "reason_rejected": "Misses multi-dimensional performance picture"
                    },
                ],
                "limitations": [
                    "Equal weighting may not match business priority",
                    "Outliers in one KPI can distort min-max scaling",
                ],
            },
        }
