"""
ExplanationBuilder
==================
Assembles full explanation per method dengan dukungan dua mode:
- BEGINNER: konsep, analogi, kata sederhana, contoh angka
- EXPERT: full math formulation, variables, optimization details

Output structured untuk konsumsi UI Method Monitor.
"""
from __future__ import annotations
from typing import Dict, Any, Optional
from enum import Enum

from .method_library import get_explanation, normalize_method_name
from .numerical_examples import NumericalExampleEngine


class ExplanationMode(str, Enum):
    BEGINNER = "beginner"
    EXPERT = "expert"
    BOTH = "both"


class ExplanationBuilder:
    """Build complete method explanation in requested mode."""

    def __init__(self):
        self.example_engine = NumericalExampleEngine()

    def build(
        self,
        method_name: str,
        mode: ExplanationMode = ExplanationMode.BOTH,
        why_chosen: Optional[str] = None,
        why_not_alternatives: Optional[list] = None,
    ) -> Dict[str, Any]:
        """
        Returns complete explanation in standardized format.

        Output schema:
        {
            "method": str,
            "category": str,
            "available": bool,
            "mode": str,
            "blocks": [
                {"label": "[METHOD]", "content": "..."},
                {"label": "[GOAL]", "content": "..."},
                {"label": "[WHY THIS METHOD]", "content": "..."},
                {"label": "[WHY NOT OTHER METHODS]", "content": "..."},
                {"label": "[CONCEPTUAL EXPLANATION]", "content": "..."},
                {"label": "[MATHEMATICAL FORMULATION]", "content": {...}},
                {"label": "[VARIABLE DESCRIPTION]", "content": [...]},
                {"label": "[CALCULATION PROCESS]", "content": [...]},
                {"label": "[NUMERICAL EXAMPLE]", "content": {...}},
                {"label": "[INTERPRETATION]", "content": "..."},
                {"label": "[LIMITATION]", "content": [...]},
            ]
        }
        """
        normalized = normalize_method_name(method_name)
        info = get_explanation(method_name)

        if not info:
            return {
                "method": method_name,
                "available": False,
                "reason": f"No explanation found for '{method_name}'. Add to method_library.",
                "mode": mode.value,
                "blocks": [],
            }

        blocks = []
        blocks.append(self._block("METHOD", info.get("name", method_name)))
        blocks.append(self._block("GOAL", info.get("purpose", "")))

        if why_chosen or info.get("why_chosen_template"):
            blocks.append(self._block(
                "WHY THIS METHOD",
                why_chosen or info.get("why_chosen_template", ""),
            ))

        if why_not_alternatives:
            content = "; ".join(
                f"{a.get('alternative', '?')}: {a.get('reason_rejected', '?')}"
                for a in why_not_alternatives
            )
            blocks.append(self._block("WHY NOT OTHER METHODS", content))
        elif info.get("why_not_others_template"):
            blocks.append(self._block("WHY NOT OTHER METHODS", info["why_not_others_template"]))

        # ===== PLAIN LANGUAGE LAYER (PRIORITY) =====
        # Selalu tampilkan plain explanation jika tersedia,
        # karena mengikuti spec user: "bahasa nyata, bukan simbol"
        plain = info.get("plain_language", {})

        if mode in (ExplanationMode.BEGINNER, ExplanationMode.BOTH):
            simple = info.get("how_it_works_simple")
            if simple:
                blocks.append(self._block("CONCEPTUAL EXPLANATION", simple))

        # REAL MATHEMATICAL EXPLANATION (plain language formula explanation)
        real_math = plain.get("real_math_explanation")
        if real_math:
            blocks.append(self._block("REAL MATHEMATICAL EXPLANATION", real_math, type_="prose"))

        # VARIABLE MEANING (plain language)
        variable_meaning = plain.get("variable_meaning")
        if variable_meaning:
            blocks.append(self._block("VARIABLE MEANING", variable_meaning, type_="prose"))

        # STEP-BY-STEP CALCULATION dengan angka konkret
        step_calc = plain.get("step_by_step_calculation")
        if step_calc:
            blocks.append(self._block("STEP-BY-STEP CALCULATION", step_calc, type_="steps"))

        # Expert/symbolic content (only if not duplicated by plain)
        if mode in (ExplanationMode.EXPERT, ExplanationMode.BOTH):
            math_form = info.get("math_formulation")
            if math_form:
                blocks.append(self._block("MATHEMATICAL FORMULATION (SYMBOLIC)", math_form, type_="math"))

            variables = info.get("variables")
            if variables:
                blocks.append(self._block("VARIABLE DESCRIPTION", variables, type_="list"))

            steps = info.get("calculation_steps")
            if steps:
                blocks.append(self._block("CALCULATION PROCESS", steps, type_="steps"))

        # NUMERICAL EXAMPLE (auto-computed if not in plain)
        if not step_calc:
            example = info.get("numerical_example") or self.example_engine.example_for(method_name)
            if example:
                blocks.append(self._block("NUMERICAL EXAMPLE", example, type_="example"))

        # RESULT INTERPRETATION (plain language)
        result_interp = plain.get("result_interpretation")
        if result_interp:
            blocks.append(self._block("RESULT INTERPRETATION", result_interp, type_="prose"))
        else:
            # Fallback to old interpretation field
            interp = info.get("interpretation")
            if interp:
                blocks.append(self._block("INTERPRETATION", interp))

        # BUSINESS / DATA SCIENCE MEANING (plain language)
        business = plain.get("business_meaning")
        if business:
            blocks.append(self._block("BUSINESS / DATA SCIENCE MEANING", business, type_="prose"))

        # Limitations (always show)
        limits = info.get("limitations")
        if limits:
            blocks.append(self._block("LIMITATION", limits, type_="list"))

        # Variant comparisons
        if "vs_linear" in info:
            blocks.append(self._block("VS LINEAR REGRESSION", info["vs_linear"]))
        if "vs_ridge" in info:
            blocks.append(self._block("VS RIDGE", info["vs_ridge"]))
        if "vs_minmax" in info:
            blocks.append(self._block("VS MIN-MAX SCALING", info["vs_minmax"]))
        if "vs_kmeans" in info:
            blocks.append(self._block("VS K-MEANS", info["vs_kmeans"]))
        if "vs_random_forest" in info:
            blocks.append(self._block("VS RANDOM FOREST", info["vs_random_forest"]))
        if "vs_mae" in info:
            blocks.append(self._block("VS MAE", info["vs_mae"]))
        if "vs_mean" in info:
            blocks.append(self._block("VS MEAN IMPUTATION", info["vs_mean"]))

        return {
            "method": info.get("name", method_name),
            "method_key": normalized,
            "category": info.get("category", "uncategorized"),
            "available": True,
            "mode": mode.value,
            "blocks": blocks,
        }

    def _block(
        self,
        label: str,
        content: Any,
        type_: str = "text",
    ) -> Dict[str, Any]:
        return {
            "label": f"[{label}]",
            "type": type_,  # text, list, steps, math, example
            "content": content,
        }

    def enrich_method_monitor_log(
        self,
        log_entry: Dict[str, Any],
        mode: ExplanationMode = ExplanationMode.BOTH,
    ) -> Dict[str, Any]:
        """
        Enrich a Method Monitor log entry with mathematical explanation.
        Looks up by 'selected_method' field.
        """
        method_name = log_entry.get("selected_method", "")
        explanation = self.build(
            method_name,
            mode=mode,
            why_chosen=log_entry.get("why_chosen"),
            why_not_alternatives=log_entry.get("why_not_alternatives"),
        )
        log_entry["math_explanation"] = explanation
        return log_entry
