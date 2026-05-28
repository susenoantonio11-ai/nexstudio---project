"""
Method Monitor - THE CRITICAL EXPLAINABLE AI COMPONENT
========================================================
Logs every step of the ML pipeline with full reasoning, alternatives,
benefits, and goal alignment.

This is what makes Nexlytics differentiated: every decision the system
makes is traceable, explainable, and educational.

Reference: Adadi & Berrada (2018) - Peeking Inside the Black-Box: A Survey on XAI
"""
from __future__ import annotations
from typing import Dict, Any, List, Optional
from datetime import datetime
import time


def _get_builder():
    """Lazy import — avoid circular dependency."""
    try:
        from ml_engine.method_explanations import ExplanationBuilder, ExplanationMode
        return ExplanationBuilder(), ExplanationMode
    except Exception:
        return None, None


class MethodMonitor:
    """
    Tracks every step of the AI pipeline with full reasoning.
    Output is structured to feed both the database and the UI.

    Auto-enriches each step dengan mathematical explanation dari method_explanations library.
    """

    def __init__(
        self,
        experiment_id: int = None,
        goal: str = None,
        enrich_with_math: bool = True,
    ):
        self.experiment_id = experiment_id
        self.enrich_with_math = enrich_with_math
        self._builder = None
        self._mode_class = None
        if enrich_with_math:
            self._builder, self._mode_class = _get_builder()
        self.goal = goal or "Generate insight from dataset"
        self.logs: List[Dict[str, Any]] = []
        self._step_counter = 0
        self._step_start_times: Dict[int, float] = {}

    def start_step(self, step_name: str) -> int:
        """Begin tracking a new pipeline step. Returns step ID."""
        self._step_counter += 1
        step_id = self._step_counter
        self._step_start_times[step_id] = time.time()
        return step_id

    def log_step(
        self,
        step: str,
        selected_method: str,
        why_chosen: str,
        why_not_alternatives: List[Dict[str, str]] = None,
        benefits: List[str] = None,
        limitations: List[str] = None,
        input_summary: Dict[str, Any] = None,
        output_summary: Dict[str, Any] = None,
        intermediate_results: Dict[str, Any] = None,
        goal_alignment: str = None,
        step_id: int = None,
    ) -> Dict[str, Any]:
        """
        Log a completed pipeline step with full XAI metadata.

        Args:
            step: Step name (profiling, target_detection, model_selection, training, evaluation)
            selected_method: What was actually chosen
            why_chosen: Reasoning text
            why_not_alternatives: List of {alternative, reason_rejected}
            benefits: List of benefit strings
            limitations: List of known limitations
            input_summary: Brief input metadata
            output_summary: Brief output metadata
            intermediate_results: Anything else useful
            goal_alignment: How this step serves the user's goal
            step_id: From start_step (optional - for duration tracking)

        Returns:
            The log entry dict.
        """
        if step_id is None:
            step_id = self.start_step(step)

        duration_ms = int((time.time() - self._step_start_times.get(step_id, time.time())) * 1000)

        log_entry = {
            "step": step,
            "step_order": step_id,
            "selected_method": selected_method,
            "why_chosen": why_chosen,
            "why_not_alternatives": why_not_alternatives or [],
            "benefits": benefits or [],
            "limitations": limitations or [],
            "input_summary": input_summary or {},
            "output_summary": output_summary or {},
            "intermediate_results": intermediate_results or {},
            "goal_alignment": goal_alignment or self._derive_goal_alignment(step, selected_method),
            "duration_ms": duration_ms,
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Auto-enrich dengan mathematical explanation
        if self.enrich_with_math and self._builder is not None:
            try:
                self._builder.enrich_method_monitor_log(
                    log_entry,
                    mode=self._mode_class.BOTH,
                )
            except Exception:
                pass  # silently fail if explanation not available

        self.logs.append(log_entry)
        return log_entry

    def _derive_goal_alignment(self, step: str, method: str) -> str:
        """Generate default goal alignment text."""
        templates = {
            "profiling": (
                f"Profiling provides foundational understanding of the dataset structure, "
                f"which is necessary before any modeling. This serves the user's goal "
                f"of '{self.goal}' by ensuring all subsequent decisions are grounded "
                f"in factual data characteristics rather than assumptions."
            ),
            "target_detection": (
                f"Auto-detecting the target variable allows the system to immediately "
                f"orient itself toward the user's prediction goal '{self.goal}'. "
                f"Without a clear Y, no supervised learning is possible."
            ),
            "model_selection": (
                f"Choosing the right model directly impacts the quality of insights "
                f"that will be delivered for the goal '{self.goal}'. The selection "
                f"process matches model strengths to data characteristics."
            ),
            "training": (
                f"Training fits the chosen model to the user's specific dataset, "
                f"creating a personalized predictive engine for the stated goal."
            ),
            "evaluation": (
                f"Evaluation quantifies the trustworthiness of the trained model, "
                f"directly informing the confidence score that the user sees."
            ),
        }
        return templates.get(
            step,
            f"This step contributes to the overall goal: {self.goal}"
        )

    def get_logs(self) -> List[Dict[str, Any]]:
        """Get all logs."""
        return self.logs

    def get_summary(self) -> Dict[str, Any]:
        """Build complete summary suitable for UI."""
        return {
            "experiment_id": self.experiment_id,
            "goal": self.goal,
            "n_steps": len(self.logs),
            "total_duration_ms": sum(log["duration_ms"] for log in self.logs),
            "steps": self.logs,
            "decision_chain": [
                {
                    "step": log["step"],
                    "method": log["selected_method"],
                    "why": log["why_chosen"][:200] + "..." if len(log["why_chosen"]) > 200 else log["why_chosen"],
                }
                for log in self.logs
            ],
        }

    def to_nexa_format(self, final_result: Dict[str, Any] = None) -> Dict[str, str]:
        """
        Format the entire pipeline trace into Nexa AI Agent output structure:
        [DATA SUMMARY][DETECTED TARGET][SELECTED METHOD][WHY THIS][WHY NOT OTHERS]
        [MODEL RESULT][INSIGHT][RECOMMENDATION][CONFIDENCE]
        """
        # Find specific steps
        profiling = self._find_step("profiling")
        target_detection = self._find_step("target_detection")
        model_selection = self._find_step("model_selection")
        training = self._find_step("training")
        evaluation = self._find_step("evaluation")

        data_summary = (
            profiling["output_summary"].get("description", "Dataset profiled successfully.")
            if profiling else "No profiling performed."
        )
        detected_target = (
            target_detection["selected_method"] if target_detection else "Not detected"
        )
        selected_method = (
            model_selection["selected_method"] if model_selection else "Not selected"
        )
        why_this = (
            model_selection["why_chosen"] if model_selection else ""
        )
        why_not_others = (
            "; ".join([
                f"{alt.get('alternative', 'unknown')}: {alt.get('reason_rejected', 'rejected')}"
                for alt in (model_selection.get("why_not_alternatives", []) if model_selection else [])
            ])
        )

        model_result = ""
        confidence = 0.0
        if training:
            metrics = training.get("output_summary", {}).get("metrics", {})
            model_result = ", ".join([f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()])
            confidence = training.get("output_summary", {}).get("confidence_score", 0.0)

        insight = (final_result or {}).get("insight", "Analysis completed successfully.")
        recommendation = (final_result or {}).get("recommendation", "Review the model output.")

        return {
            "DATA_SUMMARY": data_summary,
            "DETECTED_TARGET_VARIABLE": detected_target,
            "SELECTED_METHOD": selected_method,
            "WHY_THIS_METHOD": why_this,
            "WHY_NOT_OTHER_METHODS": why_not_others,
            "MODEL_RESULT": model_result,
            "INSIGHT": insight,
            "RECOMMENDATION": recommendation,
            "CONFIDENCE_SCORE": f"{confidence:.2f}",
        }

    def _find_step(self, step_name: str) -> Dict[str, Any]:
        for log in self.logs:
            if log["step"] == step_name:
                return log
        return None
