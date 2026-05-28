"""
Model Selector - Rule-based + meta-learning approach
=====================================================
Selects the most appropriate ML model based on:
- Target variable type (numerical, categorical, boolean)
- Time-series presence (datetime column)
- Dataset size
- Class balance (for classification)
- Use case context (forecasting, anomaly, etc)

Provides full reasoning for the Method Monitor including
why alternative models were rejected.

Reference: Hutter, Kotthoff & Vanschoren (2019) - AutoML
"""
from __future__ import annotations
from typing import Dict, List, Any


class ModelSelector:
    """Selects best ML model with explainable reasoning."""

    # Algorithm catalog with characteristics
    ALGORITHMS = {
        "random_forest_regressor": {
            "task": "regression",
            "library": "scikit-learn",
            "strengths": [
                "Handles non-linear relationships",
                "Robust to outliers",
                "No feature scaling required",
                "Provides feature importance"
            ],
            "weaknesses": ["Can overfit on small datasets", "Less interpretable than linear"],
            "min_rows": 50,
        },
        "linear_regression": {
            "task": "regression",
            "library": "scikit-learn",
            "strengths": ["Highly interpretable", "Fast", "Good baseline"],
            "weaknesses": ["Assumes linear relationship", "Sensitive to outliers"],
            "min_rows": 30,
        },
        "random_forest_classifier": {
            "task": "classification",
            "library": "scikit-learn",
            "strengths": [
                "Handles non-linear boundaries",
                "Robust to noise",
                "Multi-class out-of-the-box",
                "Provides feature importance"
            ],
            "weaknesses": ["Larger memory footprint"],
            "min_rows": 50,
        },
        "logistic_regression": {
            "task": "classification",
            "library": "scikit-learn",
            "strengths": ["Highly interpretable", "Fast", "Probability output"],
            "weaknesses": ["Linear decision boundary", "Requires scaling"],
            "min_rows": 30,
        },
        "isolation_forest": {
            "task": "anomaly_detection",
            "library": "scikit-learn",
            "strengths": [
                "Effective on high-dimensional data",
                "No assumption about distribution",
                "Linear time complexity"
            ],
            "weaknesses": ["Can struggle with high-density anomalies"],
            "min_rows": 100,
        },
        "arima": {
            "task": "forecasting",
            "library": "statsmodels",
            "strengths": [
                "Strong theoretical foundation",
                "Confidence intervals",
                "Works with limited data"
            ],
            "weaknesses": ["Requires stationary series", "Sensitive to outliers"],
            "min_rows": 50,
        },
        "prophet": {
            "task": "forecasting",
            "library": "prophet",
            "strengths": [
                "Handles seasonality automatically",
                "Robust to missing data",
                "Tunable trends"
            ],
            "weaknesses": ["Less good for short series", "Can over-smooth"],
            "min_rows": 100,
        },
        "kmeans": {
            "task": "clustering",
            "library": "scikit-learn",
            "strengths": ["Fast", "Scalable", "Simple to interpret"],
            "weaknesses": ["Requires k upfront", "Assumes spherical clusters"],
            "min_rows": 50,
        },
    }

    def select(
        self,
        profile: Dict[str, Any],
        target_column: str,
        objective: str = "auto",
        use_case: str = None,
    ) -> Dict[str, Any]:
        """
        Args:
            profile: Output dari DataProfiler
            target_column: Name of target variable Y
            objective: 'predict', 'detect_anomaly', 'forecast', 'cluster', 'auto'
            use_case: Optional business use case context

        Returns:
            {
                "selected_model": str,
                "model_type": str,
                "task": str,
                "reasoning": str,
                "alternatives_considered": [...],
                "hyperparameters": {...}
            }
        """
        target = self._find_column(profile, target_column)
        if target is None:
            return self._error_response(f"Target column '{target_column}' not found")

        n_rows = profile.get("n_rows", 0)
        has_datetime = self._has_datetime(profile)

        # Step 1: Decide task type
        task, task_reasoning = self._decide_task(
            target, has_datetime, objective, use_case
        )

        # Step 2: Pick best algorithm for that task
        candidates = [
            (name, meta) for name, meta in self.ALGORITHMS.items()
            if meta["task"] == task and n_rows >= meta["min_rows"]
        ]

        if not candidates:
            return self._error_response(
                f"No suitable algorithm for task '{task}' with {n_rows} rows"
            )

        # Score candidates
        scored = []
        for name, meta in candidates:
            score, reason = self._score_algorithm(name, meta, profile, target, n_rows)
            scored.append({
                "name": name,
                "score": score,
                "reason_for_score": reason,
                "meta": meta,
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        selected = scored[0]

        alternatives = [
            {
                "algorithm": s["name"],
                "score": round(s["score"], 3),
                "task": s["meta"]["task"],
                "library": s["meta"]["library"],
                "reason_rejected": (
                    f"Lower overall score than selected. {s['reason_for_score']}"
                ),
                "strengths": s["meta"]["strengths"],
                "weaknesses": s["meta"]["weaknesses"],
            }
            for s in scored[1:]
        ]

        # Build hyperparameters
        hparams = self._default_hyperparameters(selected["name"], n_rows, target)

        reasoning = (
            f"{task_reasoning} "
            f"Among algorithms suitable for this task, '{selected['name']}' was chosen "
            f"because: {selected['reason_for_score']} "
            f"Key strengths: {'; '.join(selected['meta']['strengths'][:2])}."
        )

        return {
            "selected_model": selected["name"],
            "model_type": task,
            "task": task,
            "library": selected["meta"]["library"],
            "reasoning": reasoning,
            "task_decision_reasoning": task_reasoning,
            "alternatives_considered": alternatives,
            "hyperparameters": hparams,
            "benefits": selected["meta"]["strengths"],
            "limitations": selected["meta"]["weaknesses"],
        }

    def _find_column(self, profile: Dict, name: str) -> Dict[str, Any]:
        for col in profile.get("columns", []):
            if col["name"] == name:
                return col
        return None

    def _has_datetime(self, profile: Dict) -> bool:
        return any(c["inferred_type"] == "datetime" for c in profile.get("columns", []))

    def _decide_task(
        self, target: Dict, has_datetime: bool, objective: str, use_case: str
    ) -> tuple:
        """Returns (task_name, reasoning_text)"""
        target_type = target["inferred_type"]
        n_unique = target["n_unique"]

        # Explicit objective overrides
        if objective == "detect_anomaly":
            return "anomaly_detection", (
                "User explicitly requested anomaly detection. "
                "Anomaly detection identifies data points that deviate significantly "
                "from the expected pattern, useful for fraud, fault, or outlier discovery."
            )

        if objective == "forecast" and has_datetime:
            return "forecasting", (
                "User requested forecasting and the dataset contains a datetime column. "
                "Forecasting predicts future values of a time-dependent variable using "
                "historical trends, seasonality, and patterns."
            )

        if objective == "cluster":
            return "clustering", (
                "User requested clustering. Clustering groups similar data points together "
                "without prior labels, revealing natural segments in the data."
            )

        # Auto mode - infer from target
        if target_type == "boolean" or (target_type == "categorical" and n_unique <= 20):
            return "classification", (
                f"The target variable '{target['name']}' has {n_unique} unique value(s) "
                f"and is of type '{target_type}'. This indicates a classification problem "
                f"where the model predicts a discrete category. "
                f"Reference: Hastie, Tibshirani & Friedman (2009) on supervised learning."
            )

        if target_type == "numerical":
            # Forecasting only when EXPLICITLY requested - regression handles
            # most numerical prediction tasks better when multiple features exist
            return "regression", (
                f"The target variable '{target['name']}' is continuous numerical. "
                f"Regression is the appropriate task for predicting continuous quantitative "
                f"outcomes such as prices, amounts, or measurements. Forecasting was not "
                f"selected because the user did not explicitly request future-period prediction."
            )

        if target_type == "categorical":
            return "classification", (
                f"The target is categorical with {n_unique} categories. "
                f"Multi-class classification will be applied."
            )

        # Fallback
        return "regression", (
            f"Could not confidently infer task type. Defaulting to regression. "
            f"User should verify this matches their objective."
        )

    def _score_algorithm(
        self, name: str, meta: Dict, profile: Dict, target: Dict, n_rows: int
    ) -> tuple:
        """Score how well algorithm fits this dataset."""
        score = 0.5  # base
        reasons = []

        # Bonus for ensemble methods on medium+ datasets
        if "random_forest" in name and n_rows > 200:
            score += 0.20
            reasons.append("Random Forest performs well on datasets >200 rows due to ensemble robustness")
        elif "random_forest" in name and n_rows <= 200:
            score += 0.05

        # Bonus for linear methods on small datasets (less overfitting)
        if "linear" in name or "logistic" in name:
            if n_rows < 200:
                score += 0.15
                reasons.append("Linear models avoid overfitting on small datasets")
            else:
                score += 0.05

        # Forecasting specific - prefer ARIMA as default since Prophet
        # requires extra installation. Boost ARIMA, slight penalty for Prophet.
        if name == "arima":
            score += 0.20
            reasons.append("ARIMA is the most reliable forecasting baseline available without extra dependencies")
        if name == "prophet" and n_rows > 365:
            # Only prefer Prophet when dataset clearly has seasonal patterns
            score += 0.10
            reasons.append("Prophet performs well with seasonal data spanning multiple cycles")
        elif name == "prophet":
            # Penalize Prophet for shorter series
            score -= 0.05

        # Anomaly detection
        if name == "isolation_forest":
            score += 0.10
            reasons.append("Isolation Forest is effective for unsupervised anomaly detection")

        # Cap score
        score = min(1.0, score)
        reason_text = ". ".join(reasons) if reasons else "Standard suitability for this task."

        return score, reason_text

    def _default_hyperparameters(self, algo_name: str, n_rows: int, target: Dict) -> Dict:
        """Sensible defaults that work for most datasets."""
        if "random_forest" in algo_name:
            return {
                "n_estimators": 100,
                "max_depth": None,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "n_jobs": -1,
                "random_state": 42,
            }
        if algo_name == "linear_regression":
            return {"fit_intercept": True}
        if algo_name == "logistic_regression":
            return {"max_iter": 1000, "random_state": 42}
        if algo_name == "isolation_forest":
            return {"contamination": "auto", "random_state": 42, "n_estimators": 100}
        if algo_name == "arima":
            return {"order": [1, 1, 1]}
        if algo_name == "prophet":
            return {"yearly_seasonality": "auto", "weekly_seasonality": "auto"}
        if algo_name == "kmeans":
            n_clusters = max(2, min(8, n_rows // 100))
            return {"n_clusters": n_clusters, "random_state": 42, "n_init": 10}
        return {}

    def _error_response(self, msg: str) -> Dict:
        return {
            "selected_model": None,
            "error": msg,
            "reasoning": msg,
            "alternatives_considered": [],
        }
