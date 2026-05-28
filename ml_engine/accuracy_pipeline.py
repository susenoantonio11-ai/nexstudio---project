"""
ACCURACY PIPELINE - Top-level orchestrator
============================================
Combines EDA + Preprocessing + Splitting + Modeling + Evaluation + Monitoring
following CRISP-DM and ML best practices:

1. Business Understanding (user goal)
2. Data Understanding (EDA: quality, missing, outlier, imbalance, leakage, correlation)
3. Data Preparation (leak-safe Pipeline; preprocessing INSIDE CV folds)
4. Modeling (baseline → multi-model comparison → hyperparameter tuning)
5. Evaluation (CV + held-out test, with overfitting diagnosis)
6. Deployment monitoring (drift detection ready)

Every decision is logged with reasoning for the Method Monitor.
"""
from __future__ import annotations
from typing import Dict, Any, Optional, List
import time
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline

# EDA
from .eda.data_quality_checker import DataQualityChecker
from .eda.missing_analyzer import MissingAnalyzer
from .eda.outlier_detector import OutlierDetector
from .eda.imbalance_detector import ImbalanceDetector
from .eda.leakage_detector import LeakageDetector
from .eda.correlation_analyzer import CorrelationAnalyzer

# Preprocessing
from .preprocessing.pipeline_builder import PreprocessingPipelineBuilder
from .preprocessing.feature_engineer import FeatureEngineer

# Splitting
from .splitting.train_test_splitter import TrainTestSplitter
from .splitting.cv_strategy import CVStrategy

# Modeling
from .modeling.baseline_model import BaselineModel
from .modeling.model_comparator import ModelComparator
from .modeling.tuner import HyperparameterTuner

# Evaluation
from .evaluation.metric_selector import MetricSelector
from .evaluation.classification_eval import ClassificationEvaluator
from .evaluation.regression_eval import RegressionEvaluator
from .evaluation.overfitting_detector import OverfittingDetector

# Method monitoring
from .monitors.method_monitor import MethodMonitor

# Research lab
from .research_lab.experiment_tracker import ExperimentTracker
from .research_lab.reproducibility import ReproducibilityManager


class AccuracyPipeline:
    """
    The crown jewel: a leak-safe, CV-first ML pipeline that prioritizes
    GENERALIZATION over training accuracy.
    """

    def __init__(
        self,
        goal: str = "Build an accurate, generalizable predictor",
        random_seed: int = 42,
        track_experiments: bool = True,
        lab_dir: str = "./research_lab",
    ):
        self.goal = goal
        self.random_seed = random_seed
        self.track_experiments = track_experiments

        self.repro = ReproducibilityManager()
        self.repro.set_seeds(random_seed)

        self.tracker = ExperimentTracker(lab_dir) if track_experiments else None
        self.monitor = MethodMonitor(goal=goal)

    # -----------------------------------------------------------------
    # MAIN ENTRY POINT
    # -----------------------------------------------------------------
    def run(
        self,
        df: pd.DataFrame,
        target_column: str,
        task_type: str = "auto",
        datetime_column: Optional[str] = None,
        test_size: float = 0.2,
        n_cv_splits: int = 5,
        tune_hyperparameters: bool = True,
        experiment_name: str = None,
    ) -> Dict[str, Any]:
        """
        Execute the full leak-safe accuracy pipeline.
        """
        t0 = time.time()
        result: Dict[str, Any] = {
            "goal": self.goal,
            "started_at": datetime.utcnow().isoformat(),
            "config": {
                "target_column": target_column,
                "task_type": task_type,
                "datetime_column": datetime_column,
                "test_size": test_size,
                "n_cv_splits": n_cv_splits,
                "tune_hyperparameters": tune_hyperparameters,
                "random_seed": self.random_seed,
            },
        }

        # ---------- STEP 1: EDA ----------
        eda = self._run_eda(df, target_column, datetime_column)
        result["eda"] = eda

        # Detect task type if auto
        if task_type == "auto":
            task_type = self._infer_task_type(df[target_column])

        # Detect imbalance for classification
        is_imbalanced = False
        imbalance_severity = "balanced"
        if task_type == "classification":
            imb = eda["imbalance"]
            is_imbalanced = imb.get("is_imbalanced", False)
            imbalance_severity = imb.get("severity", "balanced")

        # ---------- STEP 2: METRIC SELECTION ----------
        metric_info = MetricSelector().select(
            task_type=task_type,
            n_classes=df[target_column].nunique() if task_type == "classification" else 0,
            is_imbalanced=is_imbalanced,
            imbalance_severity=imbalance_severity,
        )
        scoring = metric_info["primary_metric"]
        result["metric_selection"] = metric_info
        self.monitor.log_step(
            step="metric_selection",
            selected_method=metric_info["primary_metric_human"],
            why_chosen=metric_info["reasoning"],
            why_not_alternatives=[
                {"alternative": m, "reason_rejected": "Misleading for this task/balance"}
                for m in metric_info.get("avoid_metrics", [])
            ],
        )

        # ---------- STEP 3: FEATURE ENGINEERING ----------
        df_eng, fe_added = FeatureEngineer().transform(
            df, datetime_columns=[datetime_column] if datetime_column else None
        )
        result["feature_engineering"] = {"added_features": fe_added}
        self.monitor.log_step(
            step="feature_engineering",
            selected_method=f"Added {sum(len(a['features_added']) for a in fe_added)} features",
            why_chosen="Datetime decomposition is deterministic (no leakage) and helps tree models capture seasonality.",
        )

        # ---------- STEP 4: TRAIN/TEST SPLIT ----------
        # If we just dropped datetime via feature engineering, dont pass it again
        had_datetime = datetime_column is not None and datetime_column not in df_eng.columns
        split_result = TrainTestSplitter().split(
            df_eng,
            target_column=target_column,
            test_size=test_size,
            random_state=self.random_seed,
            task_type=task_type,
            datetime_column=None if had_datetime else datetime_column,
        )
        X_train = split_result["X_train"]
        X_test = split_result["X_test"]
        y_train = split_result["y_train"]
        y_test = split_result["y_test"]
        result["split"] = {
            "strategy": split_result["strategy"],
            "reasoning": split_result["reasoning"],
            "size_summary": split_result["size_summary"],
        }
        self.monitor.log_step(
            step="train_test_split",
            selected_method=split_result["strategy"],
            why_chosen=split_result["reasoning"],
        )

        # ---------- STEP 5: PREPROCESSING PIPELINE ----------
        # Drop columns flagged as leakage risk
        leakage_high_risk = [c["column"] for c in eda["leakage"].get("high_risk_columns", [])]
        identifier_cols = [
            c["column"] for c in eda["leakage"].get("recommendations", [])
            if "DROP" in c.get("action", "") and c.get("priority") == "high"
        ]
        cols_to_drop = list(set(leakage_high_risk + identifier_cols))

        preproc_info = PreprocessingPipelineBuilder().build(
            df=X_train,
            target_column=None,  # already split
            scaler_strategy=self._choose_scaler(eda),
            encoder_strategy="onehot",
            drop_columns=cols_to_drop,
        )
        preprocessor = preproc_info["pipeline"]
        result["preprocessing"] = {
            "numeric_cols": preproc_info["numeric_cols"],
            "categorical_cols": preproc_info["categorical_cols"],
            "dropped_cols": preproc_info["dropped_cols"],
            "scaler_strategy": preproc_info["config"]["scaler_strategy"],
        }

        # ---------- STEP 6: CV STRATEGY ----------
        cv_info = CVStrategy().select(
            y=y_train, task_type=task_type, n_splits=n_cv_splits,
            is_time_series=split_result["strategy"] == "chronological",
            random_state=self.random_seed,
        )
        cv_splitter = cv_info["splitter"]
        result["cv_strategy"] = {
            "strategy": cv_info["strategy"],
            "n_splits": cv_info["n_splits"],
            "reasoning": cv_info["reasoning"],
        }

        # ---------- STEP 7: BASELINE ----------
        baseline_result = BaselineModel().evaluate(
            X_train, y_train, task_type, cv_splitter, scoring
        )
        result["baseline"] = baseline_result
        self.monitor.log_step(
            step="baseline",
            selected_method=baseline_result.get("baseline_strategy"),
            why_chosen=baseline_result.get("description", ""),
        )

        # ---------- STEP 8: MODEL COMPARISON (multi-model CV) ----------
        comparator = ModelComparator()
        candidates = comparator.get_default_candidates(
            task_type=task_type, is_imbalanced=is_imbalanced
        )
        comparison_result = comparator.compare(
            candidates=candidates,
            preprocessor=preprocessor,
            X_train=X_train,
            y_train=y_train,
            cv_splitter=cv_splitter,
            scoring=scoring,
            task_type=task_type,
        )
        result["model_comparison"] = comparison_result
        best_model_name = comparison_result["best_model"]
        self.monitor.log_step(
            step="model_comparison",
            selected_method=best_model_name,
            why_chosen=comparison_result["reasoning"],
            why_not_alternatives=comparison_result["alternatives_considered"],
        )

        if not best_model_name:
            return {"status": "error", "error": "No successful model in comparison"}

        # Find the candidate dict for the winner
        winner_cand = next(c for c in candidates if c["name"] == best_model_name)

        # ---------- STEP 9: BUILD WINNER PIPELINE + (OPTIONAL) TUNING ----------
        winner_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", winner_cand["model"]),
        ])

        if tune_hyperparameters:
            tune_result = HyperparameterTuner().tune(
                pipeline=winner_pipeline,
                X_train=X_train,
                y_train=y_train,
                cv_splitter=cv_splitter,
                scoring=scoring,
                model_name=best_model_name,
                search_strategy="grid",
            )
            result["hyperparameter_tuning"] = {
                "performed": tune_result["tuning_performed"],
                "best_params": tune_result.get("best_params"),
                "best_cv_score": tune_result.get("best_score"),
                "reasoning": tune_result.get("reasoning"),
            }
            if tune_result["tuning_performed"]:
                final_pipeline = tune_result["best_pipeline"]
                self.monitor.log_step(
                    step="hyperparameter_tuning",
                    selected_method=f"GridSearchCV on {best_model_name}",
                    why_chosen=tune_result["reasoning"],
                )
            else:
                # Fit unchanged
                final_pipeline = winner_pipeline.fit(X_train, y_train)
        else:
            final_pipeline = winner_pipeline.fit(X_train, y_train)
            result["hyperparameter_tuning"] = {"performed": False, "reason": "Disabled by user"}

        # ---------- STEP 10: FINAL EVALUATION ON HELD-OUT TEST ----------
        y_pred_test = final_pipeline.predict(X_test)
        y_pred_train = final_pipeline.predict(X_train)

        if task_type == "classification":
            try:
                y_proba_test = final_pipeline.predict_proba(X_test)
            except Exception:
                y_proba_test = None
            test_eval = ClassificationEvaluator().evaluate(y_test, y_pred_test, y_proba_test)
            train_eval = ClassificationEvaluator().evaluate(y_train, y_pred_train)
            primary_test_score = test_eval.get("f1_macro" if metric_info["primary_metric"] == "f1_macro" else "f1")
            primary_train_score = train_eval.get("f1_macro" if metric_info["primary_metric"] == "f1_macro" else "f1")
        else:
            test_eval = RegressionEvaluator().evaluate(y_test, y_pred_test)
            train_eval = RegressionEvaluator().evaluate(y_train, y_pred_train)
            primary_test_score = test_eval["r2"]
            primary_train_score = train_eval["r2"]

        result["final_evaluation"] = {
            "test_set_metrics": test_eval,
            "train_set_metrics": train_eval,
        }

        # ---------- STEP 11: OVERFITTING DIAGNOSIS ----------
        cv_test_score = comparison_result["best_model_score"]
        cv_train_score = next(
            r["mean_train_score"] for r in comparison_result["results"]
            if r["name"] == best_model_name
        )
        overfit_diag = OverfittingDetector().diagnose(
            cv_train_score=cv_train_score,
            cv_test_score=cv_test_score,
            held_out_test_score=primary_test_score,
            scoring_metric=scoring,
        )
        result["overfitting_diagnosis"] = overfit_diag

        # ---------- STEP 12: METHOD MONITOR + NEXA ----------
        result["method_monitor"] = self.monitor.get_summary()

        # ---------- STEP 13: SAVE EXPERIMENT ----------
        if self.tracker:
            data_hash = self.tracker.hash_data(df)
            tracked = self.tracker.save_experiment(
                name=experiment_name or f"{best_model_name}_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}",
                config={
                    **result["config"],
                    "selected_model": best_model_name,
                    "scaler_strategy": preproc_info["config"]["scaler_strategy"],
                    "dropped_cols": preproc_info["dropped_cols"],
                },
                metrics={
                    "primary_metric": scoring,
                    "primary_value": primary_test_score,
                    "test_set": test_eval,
                    "cv_score_mean": cv_test_score,
                    "cv_score_std": comparison_result["best_model_std"],
                    "baseline_score": baseline_result.get("mean_cv_score"),
                    "improvement_over_baseline": (
                        round(primary_test_score - baseline_result.get("mean_cv_score", 0), 4)
                        if baseline_result.get("mean_cv_score") is not None else None
                    ),
                },
                method_monitor=result["method_monitor"]["steps"],
                data_hash=data_hash,
                notes=f"Goal: {self.goal}",
            )
            result["experiment_record"] = {
                "experiment_id": tracked["experiment_id"],
                "saved_to": str(self.tracker.experiments_dir / f"{tracked['experiment_id']}.json"),
            }

        result["total_duration_seconds"] = round(time.time() - t0, 2)
        result["status"] = "success"
        return result

    # -----------------------------------------------------------------
    # HELPERS
    # -----------------------------------------------------------------
    def _run_eda(self, df, target_column, datetime_column) -> Dict[str, Any]:
        return {
            "data_quality": DataQualityChecker().check(df),
            "missing": MissingAnalyzer().analyze(df),
            "outlier": OutlierDetector().detect(df),
            "imbalance": ImbalanceDetector().detect(df[target_column]) if target_column in df.columns else {},
            "leakage": LeakageDetector().detect(df, target_column, datetime_column),
            "correlation": CorrelationAnalyzer().analyze(df, target_column),
        }

    def _infer_task_type(self, target: pd.Series) -> str:
        if pd.api.types.is_numeric_dtype(target):
            if target.nunique() <= 20 or pd.api.types.is_bool_dtype(target):
                return "classification"
            return "regression"
        return "classification"

    def _choose_scaler(self, eda: Dict[str, Any]) -> str:
        """Pick scaler based on outlier presence."""
        outlier = eda.get("outlier", {})
        if outlier.get("summary", {}).get("any_outliers"):
            return "robust"
        return "standard"
