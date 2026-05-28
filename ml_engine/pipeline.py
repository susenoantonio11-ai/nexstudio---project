"""
NEXLYTICS End-to-End Pipeline
==============================
Orchestrates the entire ML pipeline:
1. Profile data
2. Detect target variable
3. Select model
4. Train model
5. Evaluate
6. Generate Method Monitor logs
7. Produce Nexa-formatted insight

This is the primary entry point used by the FastAPI backend.
"""
from __future__ import annotations
from typing import Dict, Any, Optional
import pandas as pd

from .profilers.data_profiler import DataProfiler
from .detectors.target_detector import TargetDetector
from .selectors.model_selector import ModelSelector
from .monitors.method_monitor import MethodMonitor
from .runners.regression_runner import RegressionRunner
from .runners.classification_runner import ClassificationRunner
from .runners.anomaly_runner import AnomalyRunner
from .runners.forecasting_runner import ForecastingRunner


class NexlyticsPipeline:
    """End-to-end orchestration with Method Monitor integration."""

    def __init__(self, goal: str = "Generate insight from dataset"):
        self.goal = goal
        self.monitor = MethodMonitor(goal=goal)
        self.profiler = DataProfiler()
        self.detector = TargetDetector()
        self.selector = ModelSelector()

    def run(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        objective: str = "auto",
        use_case: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute full pipeline. Returns complete result with all logs.
        """
        # ===== STEP 1: PROFILE =====
        step_id = self.monitor.start_step("profiling")
        profile = self.profiler.profile(df)
        self.monitor.log_step(
            step="profiling",
            selected_method="DataProfiler with heuristic type inference",
            why_chosen=(
                "Profiling is always the first step to understand data structure. "
                "Heuristic type inference is more robust than pure pandas dtype checking "
                "because it can detect datetime strings, boolean-like categoricals, and "
                "ID columns that would otherwise be missed."
            ),
            benefits=[
                "Detects column types beyond pandas dtype",
                "Computes missing value statistics",
                "Generates summary used by all downstream steps"
            ],
            limitations=[
                "Heuristic rules may misclassify edge cases",
                "Datetime detection limited to common formats"
            ],
            input_summary={"n_rows": profile["n_rows"], "n_columns": profile["n_columns"]},
            output_summary={
                "description": (
                    f"Profiled {profile['n_rows']} rows × {profile['n_columns']} columns. "
                    f"Type distribution: {profile['summary']['type_distribution']}"
                ),
                "type_distribution": profile["summary"]["type_distribution"],
            },
            step_id=step_id,
        )

        # ===== STEP 2: TARGET DETECTION =====
        step_id = self.monitor.start_step("target_detection")
        if target_column is None:
            detection = self.detector.detect(profile)
            chosen_target = detection["suggested_target"]
            why_chosen = detection["reasoning"]
            alternatives = [
                {"alternative": alt["column"], "reason_rejected": alt["reason"]}
                for alt in detection["alternatives"]
            ]
        else:
            chosen_target = target_column
            why_chosen = f"Target variable '{target_column}' was explicitly specified by the user."
            alternatives = []
            detection = {"suggested_target": target_column, "confidence": 1.0}

        self.monitor.log_step(
            step="target_detection",
            selected_method=chosen_target or "none",
            why_chosen=why_chosen,
            why_not_alternatives=alternatives,
            benefits=[
                "Allows the system to begin supervised learning automatically",
                "Reduces user friction by removing manual target specification"
            ],
            limitations=[
                "Heuristic-based; may miss domain-specific target conventions",
                "Confidence is approximate"
            ],
            output_summary={"target": chosen_target, "confidence": detection.get("confidence", 0)},
            step_id=step_id,
        )

        if not chosen_target:
            return self._partial_result("Could not determine target variable")

        # ===== STEP 3: MODEL SELECTION =====
        step_id = self.monitor.start_step("model_selection")
        selection = self.selector.select(
            profile=profile,
            target_column=chosen_target,
            objective=objective,
            use_case=use_case,
        )
        if selection.get("error"):
            return self._partial_result(selection["error"])

        self.monitor.log_step(
            step="model_selection",
            selected_method=selection["selected_model"],
            why_chosen=selection["reasoning"],
            why_not_alternatives=[
                {
                    "alternative": alt["algorithm"],
                    "reason_rejected": alt["reason_rejected"],
                }
                for alt in selection["alternatives_considered"]
            ],
            benefits=selection["benefits"],
            limitations=selection["limitations"],
            output_summary={
                "model": selection["selected_model"],
                "task": selection["task"],
                "library": selection["library"],
            },
            step_id=step_id,
        )

        # ===== STEP 4 + 5: TRAINING + EVALUATION =====
        step_id = self.monitor.start_step("training")
        runner = self._build_runner(selection)
        if runner is None:
            return self._partial_result(f"No runner available for {selection['selected_model']}")

        try:
            if selection["task"] == "anomaly_detection":
                result = runner.run(df=df)
            elif selection["task"] == "forecasting":
                date_col = self._find_datetime_column(profile)
                result = runner.run(df=df, target_column=chosen_target, date_column=date_col)
            else:
                result = runner.run(df=df, target_column=chosen_target)
        except Exception as e:
            return self._partial_result(f"Training failed: {e}")

        self.monitor.log_step(
            step="training",
            selected_method=f"Trained {selection['selected_model']}",
            why_chosen=(
                f"Training fits the selected model to the dataset. "
                f"An 80/20 train/test split is used to enable out-of-sample evaluation, "
                f"which is the standard ML evaluation protocol."
            ),
            benefits=["Produces a fitted model ready for inference"],
            limitations=["Default hyperparameters; tuning could improve results"],
            output_summary={
                "metrics": result.get("metrics", {}),
                "confidence_score": result.get("confidence_score", 0),
            },
            step_id=step_id,
        )

        # ===== STEP 6: GENERATE INSIGHT =====
        step_id = self.monitor.start_step("evaluation")
        insight, recommendation = self._generate_insight(result, selection, chosen_target)

        self.monitor.log_step(
            step="evaluation",
            selected_method="Insight generation from model output",
            why_chosen=(
                "After training, raw metrics are translated into actionable insight "
                "and recommendation in plain language. This serves the Explainable AI "
                "principle (Adadi & Berrada, 2018) by making model output accessible "
                "to non-technical users."
            ),
            benefits=[
                "Translates technical metrics into user-friendly language",
                "Provides actionable recommendations"
            ],
            output_summary={"insight_length": len(insight)},
            step_id=step_id,
        )

        # ===== ASSEMBLE FINAL RESULT =====
        nexa_output = self.monitor.to_nexa_format({
            "insight": insight,
            "recommendation": recommendation,
        })

        return {
            "status": "success",
            "goal": self.goal,
            "profile": profile,
            "target_detection": detection,
            "model_selection": selection,
            "model_result": result,
            "insight": insight,
            "recommendation": recommendation,
            "confidence_score": result.get("confidence_score", 0),
            "method_monitor": self.monitor.get_summary(),
            "nexa_output": nexa_output,
        }

    def _build_runner(self, selection: Dict[str, Any]):
        algo = selection["selected_model"]
        hparams = selection.get("hyperparameters", {})
        if "random_forest_regressor" in algo or algo == "linear_regression":
            return RegressionRunner(algorithm=algo, hyperparameters=hparams)
        if "random_forest_classifier" in algo or algo == "logistic_regression":
            return ClassificationRunner(algorithm=algo, hyperparameters=hparams)
        if algo == "isolation_forest":
            return AnomalyRunner(algorithm=algo, hyperparameters=hparams)
        if algo in ("arima", "prophet"):
            return ForecastingRunner(algorithm=algo, hyperparameters=hparams)
        return None

    def _find_datetime_column(self, profile: Dict) -> Optional[str]:
        for col in profile.get("columns", []):
            if col["inferred_type"] == "datetime":
                return col["name"]
        return None

    def _generate_insight(
        self, result: Dict, selection: Dict, target: str
    ) -> tuple:
        """Generate human-readable insight and recommendation."""
        task = selection["task"]
        metrics = result.get("metrics", {})

        if task == "regression":
            r2 = metrics.get("r2_test", 0)
            rmse = metrics.get("rmse_test", 0)
            if r2 > 0.7:
                quality = "strong"
            elif r2 > 0.4:
                quality = "moderate"
            else:
                quality = "weak"
            insight = (
                f"The {selection['selected_model']} model shows {quality} predictive performance "
                f"on '{target}' with R² = {r2:.3f} and RMSE = {rmse:.2f}. "
                f"This means the model explains approximately {r2*100:.0f}% of the variance "
                f"in the target variable on unseen data."
            )
            if r2 > 0.5:
                recommendation = (
                    f"The model is reliable enough for predictive use cases. Consider deploying it "
                    f"to forecast future '{target}' values. Top features by importance can be used "
                    f"to identify key drivers."
                )
            else:
                recommendation = (
                    f"Predictive accuracy is limited. Recommended next steps: collect more data, "
                    f"engineer additional features, or try alternative models."
                )

        elif task == "classification":
            f1 = metrics.get("f1", 0)
            acc = metrics.get("accuracy", 0)
            if f1 > 0.8:
                quality = "high accuracy"
            elif f1 > 0.6:
                quality = "moderate accuracy"
            else:
                quality = "low accuracy"
            insight = (
                f"The {selection['selected_model']} classifier achieved {quality} on '{target}' "
                f"with F1 = {f1:.3f} and accuracy = {acc:.3f}."
            )
            if f1 > 0.7:
                recommendation = (
                    f"Model is suitable for production classification tasks on '{target}'. "
                    f"Consider monitoring drift over time."
                )
            else:
                recommendation = (
                    f"Improve performance by addressing class imbalance, gathering more samples, "
                    f"or applying feature selection."
                )

        elif task == "anomaly_detection":
            n_anom = metrics.get("n_anomalies", 0)
            rate = metrics.get("anomaly_rate", 0) * 100
            insight = (
                f"Detected {n_anom} anomalies ({rate:.1f}% of records) using "
                f"{selection['selected_model']}. These outliers deviate substantially from "
                f"the typical pattern observed in the data."
            )
            recommendation = (
                f"Review the top anomalies in detail. They may indicate fraud, data entry errors, "
                f"or genuine business events that require investigation."
            )

        elif task == "forecasting":
            forecast = result.get("forecast", [])
            n_periods = len(forecast)
            insight = (
                f"Forecasted {n_periods} future periods of '{target}' using "
                f"{selection['selected_model']}. RMSE on historical fit = "
                f"{metrics.get('rmse', 0):.2f}."
            )
            recommendation = (
                f"Use the forecast for planning, but always combine with domain knowledge. "
                f"Re-train periodically as new data arrives to maintain accuracy."
            )

        else:
            insight = f"Analysis complete with {selection['selected_model']}."
            recommendation = "Review detailed metrics for next steps."

        return insight, recommendation

    def _partial_result(self, error_msg: str) -> Dict[str, Any]:
        return {
            "status": "error",
            "error": error_msg,
            "method_monitor": self.monitor.get_summary(),
        }
