"""
Experiment Tracker
==================
Saves complete experiment state to local JSON files (or DB).
Each experiment has a unique ID and can be replayed for reproducibility.
"""
from __future__ import annotations
from typing import Dict, Any, List, Optional
from pathlib import Path
import json
import hashlib
import platform
import sys
from datetime import datetime
import uuid


class ExperimentTracker:
    """Track and persist experiment runs."""

    def __init__(self, lab_dir: str = "./research_lab"):
        self.lab_dir = Path(lab_dir)
        self.lab_dir.mkdir(parents=True, exist_ok=True)
        self.experiments_dir = self.lab_dir / "experiments"
        self.experiments_dir.mkdir(exist_ok=True)

    def save_experiment(
        self,
        name: str,
        config: Dict[str, Any],
        metrics: Dict[str, Any],
        method_monitor: List[Dict[str, Any]] = None,
        data_hash: Optional[str] = None,
        notes: str = "",
    ) -> Dict[str, Any]:
        """Persist an experiment record."""
        experiment_id = str(uuid.uuid4())[:8]
        record = {
            "experiment_id": experiment_id,
            "name": name,
            "timestamp": datetime.utcnow().isoformat(),
            "config": config,
            "metrics": metrics,
            "method_monitor": method_monitor or [],
            "data_hash": data_hash,
            "environment": self._capture_environment(),
            "notes": notes,
        }

        path = self.experiments_dir / f"{experiment_id}.json"
        with open(path, "w") as f:
            json.dump(record, f, indent=2, default=str)

        return record

    def load_experiment(self, experiment_id: str) -> Dict[str, Any]:
        path = self.experiments_dir / f"{experiment_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"Experiment {experiment_id} not found")
        with open(path) as f:
            return json.load(f)

    def list_experiments(self) -> List[Dict[str, Any]]:
        records = []
        for path in sorted(self.experiments_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                with open(path) as f:
                    rec = json.load(f)
                # Lightweight summary
                records.append({
                    "experiment_id": rec.get("experiment_id"),
                    "name": rec.get("name"),
                    "timestamp": rec.get("timestamp"),
                    "primary_metric": rec.get("metrics", {}).get("primary_metric"),
                    "primary_value": rec.get("metrics", {}).get("primary_value"),
                    "model": rec.get("config", {}).get("selected_model"),
                })
            except Exception:
                continue
        return records

    def hash_data(self, df) -> str:
        """Stable hash of dataframe contents for reproducibility check."""
        try:
            # Use pandas hash_pandas_object for stable hash
            from pandas.util import hash_pandas_object
            h = hash_pandas_object(df, index=True).values
            return hashlib.sha256(h.tobytes()).hexdigest()[:16]
        except Exception:
            # Fallback: hash shape + columns
            shape_str = f"{len(df)}x{len(df.columns)}_{','.join(df.columns)}"
            return hashlib.sha256(shape_str.encode()).hexdigest()[:16]

    def _capture_environment(self) -> Dict[str, str]:
        env = {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        }
        for pkg in ("numpy", "pandas", "sklearn"):
            try:
                mod = __import__(pkg)
                env[pkg] = getattr(mod, "__version__", "unknown")
            except ImportError:
                env[pkg] = "not_installed"
        return env
