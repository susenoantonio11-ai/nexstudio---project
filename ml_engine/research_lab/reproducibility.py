"""
Reproducibility Manager
=======================
Ensures experiments can be replayed identically by managing:
- Random seeds (numpy, random, sklearn)
- Environment hash (Python + key library versions)
- Data hash (so we can detect if input data changed)
"""
from __future__ import annotations
from typing import Dict, Any
import os
import random
import numpy as np


class ReproducibilityManager:
    """Manage random state for reproducible experiments."""

    def set_seeds(self, seed: int = 42) -> Dict[str, Any]:
        """Set seeds across all common libraries."""
        random.seed(seed)
        np.random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        return {
            "seed": seed,
            "libraries_seeded": ["random", "numpy", "PYTHONHASHSEED"],
            "note": (
                "Seeds set. For full reproducibility, also pass random_state=seed to "
                "every sklearn estimator that accepts it. The sklearn Pipeline does "
                "this automatically when configured at the pipeline level."
            ),
        }

    def reproducibility_check(
        self,
        original_record: Dict[str, Any],
        current_data_hash: str,
        current_env: Dict[str, str],
    ) -> Dict[str, Any]:
        """Check if current state matches original experiment."""
        issues = []
        if original_record.get("data_hash") and original_record["data_hash"] != current_data_hash:
            issues.append({
                "type": "data_changed",
                "severity": "high",
                "description": (
                    f"Dataset hash differs. Original: {original_record.get('data_hash')}, "
                    f"Current: {current_data_hash}. Results will not match exactly."
                ),
            })

        original_env = original_record.get("environment", {})
        for key in ("python", "numpy", "sklearn"):
            orig = original_env.get(key)
            curr = current_env.get(key)
            if orig and curr and orig != curr:
                issues.append({
                    "type": "environment_changed",
                    "severity": "medium",
                    "description": f"{key} version changed: {orig} -> {curr}",
                })

        return {
            "reproducible": len(issues) == 0,
            "n_issues": len(issues),
            "issues": issues,
        }
