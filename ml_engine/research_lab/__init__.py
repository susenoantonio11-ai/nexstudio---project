"""
Research Lab
============
Reproducible experiment tracking + comparison.

Features:
- Save complete experiment state (data hash, seeds, params, metrics, model)
- Compare multiple experiments
- Reproduce by replaying saved configuration
"""
from .experiment_tracker import ExperimentTracker
from .experiment_comparator import ExperimentComparator
from .reproducibility import ReproducibilityManager

__all__ = ["ExperimentTracker", "ExperimentComparator", "ReproducibilityManager"]
