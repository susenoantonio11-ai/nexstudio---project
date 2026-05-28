"""Phase-2 Category 10 — Human-Centered Explainable AI (10 models)"""
from .base import AdvancedAIModel


class HumanUnderstandableReasoningAI(AdvancedAIModel):
    name="HumanUnderstandableReasoningAI"; model_id="human_understandable_reasoning_ai"; category="explainable"; domain="ai_intelligence"
    description="Translates internal AI reasoning ke bahasa manusia awam — bukan teknis-only."
    citations=["Miller (2019) Artificial Intelligence — Explanation in AI"]
    def run(self, p):
        return self._envelope({"output_style":"plain language + analogy + visual cue"}, confidence=0.80, uncertainty=0.20)


class ExplainableDecisionNarrator(AdvancedAIModel):
    name="ExplainableDecisionNarrator"; model_id="explainable_decision_narrator"; category="explainable"; domain="ai_intelligence"
    description="Narasi keputusan AI dalam format storyline (situation → reasoning → conclusion → recommendation)."
    def run(self, p):
        return self._envelope({"format":"storyline narrative","sections":["situation","reasoning","conclusion","recommendation"]}, confidence=0.78, uncertainty=0.22)


class VisualReasoningGenerator(AdvancedAIModel):
    name="VisualReasoningGenerator"; model_id="visual_reasoning_generator"; category="explainable"; domain="ai_intelligence"
    description="Generate visual explanation: SHAP plot, decision tree, sankey diagram, force plot."
    citations=["Lundberg & Lee (2017) NeurIPS — SHAP"]
    def run(self, p):
        return self._envelope({"visualizations":["shap_force","tree_path","sankey","heatmap"]}, confidence=0.82, uncertainty=0.18)


class InteractiveExplanationEngine(AdvancedAIModel):
    name="InteractiveExplanationEngine"; model_id="interactive_explanation_engine"; category="explainable"; domain="ai_intelligence"
    description="Interactive what-if: user ubah feature, lihat efek pada prediksi realtime."
    realtime_capable=True
    def run(self, p):
        return self._envelope({"controls":"per-feature slider","update":"live partial dependence"}, confidence=0.82, uncertainty=0.18)


class DecisionTransparencyAI(AdvancedAIModel):
    name="DecisionTransparencyAI"; model_id="decision_transparency_ai"; category="explainable"; domain="ai_intelligence"
    description="Transparansi setiap keputusan: input data + model version + reasoning trace + audit signature."
    citations=["EU AI Act (2024) — high-risk system transparency"]
    def run(self, p):
        return self._envelope({"transparency_fields":["input_hash","model_version","reasoning","auditor"]}, confidence=0.90, uncertainty=0.10)


class CognitiveInterpretationModel(AdvancedAIModel):
    name="CognitiveInterpretationModel"; model_id="cognitive_interpretation_model"; category="explainable"; domain="ai_intelligence"
    description="Cognitive model interpretation — peta hasil model ke mental model expert (heuristik domain)."
    citations=["Kahneman (2011) Thinking, Fast and Slow"]
    def run(self, p):
        return self._envelope({"interpretation_layer":"map AI output → expert heuristic"}, confidence=0.75, uncertainty=0.25)


class HumanRiskCommunicationAI(AdvancedAIModel):
    name="HumanRiskCommunicationAI"; model_id="human_risk_communication_ai"; category="explainable"; domain="ai_intelligence"
    description="Risk communication yang sesuai audience (executive / publik / teknis)."
    citations=["Fischhoff (1995) Risk communication: Five years of progress"]
    def run(self, p):
        audience = p.get("audience","executive")
        return self._envelope({"audience":audience,"tone":"non-technical, action-oriented"}, confidence=0.80, uncertainty=0.20)


class ScientificVisualizationReasoner(AdvancedAIModel):
    name="ScientificVisualizationReasoner"; model_id="scientific_visualization_reasoner"; category="explainable"; domain="ai_intelligence"
    description="Pilih visualisasi ilmiah optimal sesuai data type + audience + insight goal."
    citations=["Few (2009) Now You See It — visualization principles"]
    def run(self, p):
        return self._envelope({"chart_picker":"rule-based + intent recognition","types":["line","bar","heatmap","map","sankey"]}, confidence=0.82, uncertainty=0.18)


class ExecutiveSummaryAI(AdvancedAIModel):
    name="ExecutiveSummaryAI"; model_id="executive_summary_ai"; category="explainable"; domain="ai_intelligence"
    description="Generate executive summary 1-pager dari hasil analisis kompleks: insights + actions + risks."
    def run(self, p):
        return self._envelope({"sections":["situation","key_insights","recommendations","risks","next_actions"]}, confidence=0.82, uncertainty=0.18)


class EducationalExplainabilityEngine(AdvancedAIModel):
    name="EducationalExplainabilityEngine"; model_id="educational_explainability_engine"; category="explainable"; domain="ai_intelligence"
    description="Penjelasan model yang mendidik: cocok untuk training material + kuliah + dokumentasi."
    def run(self, p):
        return self._envelope({"audience":"student / educator","format":"step-by-step with example + analogy"}, confidence=0.82, uncertainty=0.18)


MODELS=[HumanUnderstandableReasoningAI,ExplainableDecisionNarrator,VisualReasoningGenerator,InteractiveExplanationEngine,
        DecisionTransparencyAI,CognitiveInterpretationModel,HumanRiskCommunicationAI,ScientificVisualizationReasoner,
        ExecutiveSummaryAI,EducationalExplainabilityEngine]
