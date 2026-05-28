"""
Mathematical & Statistical Explanation Layer
=============================================
Knowledge base of method explanations dengan format:
- Method Name
- Purpose
- Why Selected / Why Not Others (templated)
- How It Works (Conceptual) - Beginner mode
- Mathematical Formulation - Expert mode
- Variable Explanation
- Step-by-step Calculation Logic
- Output Interpretation
- Limitations / Assumptions

Setiap metode di Method Monitor di-enrich dengan field 'math_explanation'
otomatis melalui ExplanationBuilder.
"""
from .method_library import METHOD_LIBRARY, get_explanation, list_supported_methods
from .explanation_builder import ExplanationBuilder, ExplanationMode
from .numerical_examples import NumericalExampleEngine

__all__ = [
    "METHOD_LIBRARY",
    "get_explanation",
    "list_supported_methods",
    "ExplanationBuilder",
    "ExplanationMode",
    "NumericalExampleEngine",
]
