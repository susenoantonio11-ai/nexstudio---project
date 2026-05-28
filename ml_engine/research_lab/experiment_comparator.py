"""
Experiment Comparator
=====================
Compare multiple experiments side by side.
"""
from __future__ import annotations
from typing import Dict, Any, List


class ExperimentComparator:
    """Compare experiment runs and pick the best."""

    def compare(
        self,
        experiments: List[Dict[str, Any]],
        primary_metric: str,
        higher_is_better: bool = True,
    ) -> Dict[str, Any]:
        """
        Args:
            experiments: list of experiment records (from ExperimentTracker)
            primary_metric: key inside metrics dict to compare on
            higher_is_better: maximize or minimize
        """
        comparable = []
        for exp in experiments:
            metrics = exp.get("metrics", {})
            value = metrics.get(primary_metric)
            if value is None:
                continue
            comparable.append({
                "experiment_id": exp.get("experiment_id"),
                "name": exp.get("name"),
                "timestamp": exp.get("timestamp"),
                "model": exp.get("config", {}).get("selected_model"),
                "value": float(value),
                "metrics": metrics,
            })

        if not comparable:
            return {"best": None, "ranked": [], "comparison_table": []}

        comparable.sort(key=lambda r: r["value"], reverse=higher_is_better)
        best = comparable[0]
        worst = comparable[-1]
        improvement = best["value"] - worst["value"] if higher_is_better else worst["value"] - best["value"]

        return {
            "best": best,
            "ranked": comparable,
            "primary_metric": primary_metric,
            "higher_is_better": higher_is_better,
            "improvement_over_worst": round(float(improvement), 4),
            "comparison_table": [
                {
                    "rank": i + 1,
                    "name": e["name"],
                    "model": e["model"],
                    "value": round(e["value"], 4),
                    "experiment_id": e["experiment_id"],
                }
                for i, e in enumerate(comparable)
            ],
        }
