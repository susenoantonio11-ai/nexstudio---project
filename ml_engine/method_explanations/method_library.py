"""
Method Library - aggregator semua explanation files.
Provides unified lookup interface.
"""
from typing import Dict, Any, List, Optional
from ._models import MODEL_EXPLANATIONS
from ._metrics import METRIC_EXPLANATIONS
from ._preprocessing import PREPROCESSING_EXPLANATIONS
from ._misc import MISC_EXPLANATIONS
from ._plain_explanations import PLAIN_EXPLANATIONS
from ._geospatial import GEOSPATIAL_PLAIN_EXPLANATIONS


# Combine all categories into one lookup
METHOD_LIBRARY: Dict[str, Dict[str, Any]] = {
    **MODEL_EXPLANATIONS,
    **METRIC_EXPLANATIONS,
    **PREPROCESSING_EXPLANATIONS,
    **MISC_EXPLANATIONS,
}

# Merge plain-language fields into existing methods
for _method_key, _plain in PLAIN_EXPLANATIONS.items():
    if _method_key in METHOD_LIBRARY:
        METHOD_LIBRARY[_method_key]["plain_language"] = _plain
    else:
        METHOD_LIBRARY[_method_key] = {
            "name": _method_key.replace("_", " ").title(),
            "category": "statistics",
            "plain_language": _plain,
        }

# Merge geospatial methods (different format - already include name/category top-level)
for _method_key, _data in GEOSPATIAL_PLAIN_EXPLANATIONS.items():
    METHOD_LIBRARY[_method_key] = {
        "name": _data.get("name", _method_key),
        "category": _data.get("category", "geospatial"),
        "purpose": _data.get("purpose", ""),
        "plain_language": _data.get("plain_language", {}),
        "limitations": _data.get("limitations", []),
        "reference": _data.get("reference"),
        "vs_f1": _data.get("vs_f1"),
    }


# Aliases - common variations of method names
METHOD_ALIASES: Dict[str, str] = {
    # Models
    "random_forest_classifier": "random_forest",
    "random_forest_regressor": "random_forest",
    "rf": "random_forest",
    "rfc": "random_forest",
    "rfr": "random_forest",
    "gradient_boosting_classifier": "gradient_boosting",
    "gradient_boosting_regressor": "gradient_boosting",
    "gbm": "gradient_boosting",
    "gbc": "gradient_boosting",
    "gbr": "gradient_boosting",
    "linear_reg": "linear_regression",
    "linreg": "linear_regression",
    "ols": "linear_regression",
    "logistic_reg": "logistic_regression",
    "logreg": "logistic_regression",
    "k_means": "kmeans",
    "k-means": "kmeans",
    "isolationforest": "isolation_forest",

    # Preprocessing
    "standardscaler": "standardization",
    "standard_scaler": "standardization",
    "z_score": "standardization",
    "z-score": "standardization",
    "minmax": "minmax_scaling",
    "min_max": "minmax_scaling",
    "minmaxscaler": "minmax_scaling",
    "robustscaler": "robust_scaling",
    "robust_scaler": "robust_scaling",
    "onehot": "onehot_encoding",
    "one_hot": "onehot_encoding",
    "tfidf_vectorizer": "tfidf",
    "tf-idf": "tfidf",

    # Metrics
    "f1": "f1_score",
    "f1_macro": "f1_score",
    "f1_weighted": "f1_score",
    "auc": "roc_auc",
    "aucroc": "roc_auc",
    "average_precision": "pr_auc",
    "matthews_corrcoef": "mcc",
    "r2": "r_squared",
    "r2_score": "r_squared",
    "r²": "r_squared",
    "r²_(coefficient_of_determination)": "r_squared",
    "coefficient_of_determination": "r_squared",
    "neg_mean_squared_error": "rmse",
    "neg_root_mean_squared_error": "rmse",
    "mean_squared_error": "rmse",
    "mean_absolute_error": "mae",
    "neg_mean_absolute_error": "mae",
    "neg_mean_absolute_percentage_error": "mape",
    "sensitivity": "recall",
    "tpr": "recall",
    "true_positive_rate": "recall",
    "ppv": "precision",
    "positive_predictive_value": "precision",

    # Validation
    "kfold": "kfold_cv",
    "k_fold": "kfold_cv",
    "stratifiedkfold": "stratified_kfold",
    "stratified_k_fold": "stratified_kfold",
    "timeseriessplit": "time_series_split",

    # Outlier
    "iqr": "iqr_outlier",
    "zscore": "zscore_outlier",
    "z-score outlier": "zscore_outlier",

    # Drift
    "population_stability_index": "psi",
    "kolmogorov_smirnov": "ks_test",
}


def normalize_method_name(name: str) -> str:
    """Resolve aliases and normalize formatting (alias-aware, fuzzy-safe)."""
    if not name:
        return ""
    n = str(name).lower().strip()
    n = n.replace(" ", "_").replace("-", "_")

    if n in METHOD_LIBRARY:
        return n
    if n in METHOD_ALIASES:
        return METHOD_ALIASES[n]

    # Strict partial match: word boundary, only the longest matching key wins
    # (avoid "random" matching "random_forest")
    best = None
    best_len = 0
    n_padded = f"_{n}_"
    for key in METHOD_LIBRARY:
        key_padded = f"_{key}_"
        if (n == key
            or key_padded in n_padded
            or n.startswith(f"{key}_")
            or n.endswith(f"_{key}")):
            if len(key) > best_len:
                best = key
                best_len = len(key)
    if best:
        return best

    # Last resort: alias substring match
    for alias, target in METHOD_ALIASES.items():
        alias_padded = f"_{alias}_"
        if alias_padded in n_padded or alias == n:
            return target

    return n


def get_explanation(method_name: str) -> Optional[Dict[str, Any]]:
    """Lookup explanation for a method by name (alias-aware)."""
    normalized = normalize_method_name(method_name)
    return METHOD_LIBRARY.get(normalized)


def list_supported_methods() -> List[str]:
    """List all method keys available in library."""
    return sorted(METHOD_LIBRARY.keys())


def list_by_category(category: str) -> List[str]:
    """List methods filtered by category."""
    return sorted([
        k for k, v in METHOD_LIBRARY.items()
        if v.get("category", "").startswith(category)
    ])


def get_categories() -> List[str]:
    """List all unique categories."""
    return sorted(set(v.get("category", "") for v in METHOD_LIBRARY.values()))
