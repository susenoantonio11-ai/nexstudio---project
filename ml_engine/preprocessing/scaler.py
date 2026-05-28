"""
Scaler step builder.
"""
from __future__ import annotations
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


def build_scaler_step(strategy: str = "standard"):
    """
    Strategies:
    - 'standard' : zero mean, unit variance. Most common; assumes ~normal distribution.
    - 'minmax'   : scale to [0, 1]. Used for neural networks, no outlier robustness.
    - 'robust'   : median + IQR. Best when outliers exist (Iglewicz & Hoaglin, 1993).
    - 'none'     : no scaling. Used when downstream model is tree-based.
    """
    if strategy == "standard":
        return StandardScaler()
    if strategy == "minmax":
        return MinMaxScaler()
    if strategy == "robust":
        return RobustScaler()
    if strategy == "none":
        return "passthrough"
    return StandardScaler()
